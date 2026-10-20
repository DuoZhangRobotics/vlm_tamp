"""Small frozen RGB-D numerical pilot with six bounded real API calls."""

import base64
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .dataset import read_records, safe_path
from .experiment import Ledger
from .libero import ROOT, configure_libero
from .providers import ProviderResponse
from .rgbd_numericals import (
    blank_schema, compile_numericals, json_text, metric_depth, template, validate_blanks, world_points,
)


IMPLEMENTATION = [
    "src/vlm_tamp/rgbd_numericals.py", "src/vlm_tamp/pose_constraints.py",
    "src/vlm_tamp/rgbd_numeric_trial.py", "scripts/prepare_rgbd_numericals.py",
    "src/vlm_tamp/rgbd_numeric_evaluation.py",
    "src/vlm_tamp/rgb_role_choices.py", "src/vlm_tamp/rgb_role_points.py",
    "src/vlm_tamp/experiment.py",
]


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_config(path):
    config = json.loads(Path(path).read_text())
    if (config["max_requests"] != 6 or not 0 < config["max_budget_usd"] <= 0.1
            or config["max_retries"] != 0 or len(config["cases"]) != 6):
        raise ValueError("Numerical pilot is limited to six calls and ten cents")
    if config["model"] != "gpt-5.4-mini-2026-03-17" or config["service_tier"] != "default":
        raise ValueError("Numerical pilot must use the declared Mini snapshot and standard service")
    if len({case["id"] for case in config["cases"]}) != 6:
        raise ValueError("Duplicate numerical case ID")
    return config


def sdk_arguments(config, payload, image, prompt):
    allowed = {"instruction", "source_description", "target_description", "blanks", "coordinate_convention"}
    if set(payload) != allowed:
        raise ValueError("No inventory, metric answers, references, or simulator state in model payload")
    return {
        "model": config["model"], "instructions": prompt,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": json_text(payload)},
            {"type": "input_image", "detail": config["image_detail"],
             "image_url": "data:image/png;base64," + base64.b64encode(image).decode()},
        ]}],
        "text": {"format": {"type": "json_schema", "name": "RGBDNumericalBlanks",
                            "schema": blank_schema(), "strict": True}},
        "reasoning": {"effort": config["reasoning_effort"]},
        "service_tier": "default", "max_output_tokens": config["max_output_tokens"], "store": False,
    }


def prepare(config_path):
    config = read_config(config_path)
    frozen = ROOT / config["frozen_protocol"]
    if frozen.exists():
        raise FileExistsError("Do not replace an existing frozen numerical protocol")
    records, source = read_records(ROOT / config["dataset"])
    by_id = {record["example_id"]: record for record in records}
    output = ROOT / config["depth_dataset"]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh depth capture directory")
    output.mkdir(parents=True, exist_ok=True)
    # The selection and geometry policy are persisted before any rendering or model response.
    (output / "selection_before_capture.json").write_text(json_text(config) + "\n")
    configure_libero(gl="egl")
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.macros as macros

    captures = {}
    for case in config["cases"]:
        identifier = case["example_id"]
        if identifier in captures:
            continue
        record = by_id[identifier]
        template(record["model_inputs"]["instruction"], case["upright"])
        folder = output / identifier
        folder.mkdir()
        np.random.seed(0)
        random.seed(0)
        size = record["metadata"]["camera"]["height"]
        env = OffScreenRenderEnv(
            bddl_file_name=str(ROOT / "third_party/LIBERO" / record["metadata"]["bddl"]),
            camera_names=["agentview"], camera_heights=size, camera_widths=size, camera_depths=True,
        )
        try:
            env.seed(0)
            env.reset()
            obs = env.set_init_state(np.load(safe_path(source, record["metadata"]["initial_state_path"])))
            rgb, raw = obs["agentview_image"], obs["agentview_depth"]
            if macros.IMAGE_CONVENTION == "opengl":
                rgb, raw = rgb[::-1], raw[::-1]
            image = safe_path(source, record["model_inputs"]["image"])
            if not np.array_equal(rgb, np.asarray(Image.open(image))):
                raise ValueError("RGB-D capture must reproduce the frozen original RGB pixel-for-pixel")
            sim, cam = env.sim, env.sim.model.camera_name2id("agentview")
            near = float(sim.model.vis.map.znear * sim.model.stat.extent)
            far = float(sim.model.vis.map.zfar * sim.model.stat.extent)
            depth = metric_depth(raw, near, far)
            height, width = depth.shape
            focal = height/2 / np.tan(np.deg2rad(sim.model.cam_fovy[cam])/2)
            intrinsic = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1.]])
            transform = np.eye(4)
            transform[:3, :3] = sim.data.cam_xmat[cam].reshape(3, 3) @ np.diag([1., -1., -1.])
            transform[:3, 3] = sim.data.cam_xpos[cam]
            cloud = world_points(depth, intrinsic, transform)
            calibration = {"intrinsic": intrinsic.tolist(), "camera_to_world": transform.tolist(),
                           "near_m": near, "far_m": far, "frame": "world", "units": "meters",
                           "depth_convention": "camera optical-axis z, not Euclidean range",
                           "source": "camera configuration/calibration only"}
            np.save(folder / "depth_meters.npy", depth)
            np.save(folder / "world_points.npy", cloud)
            (folder / "rgb.png").write_bytes(image.read_bytes())
            (folder / "calibration.json").write_text(json_text(calibration) + "\n")
            captures[identifier] = {"rgb_matches_existing": True,
                                    "files": {name: file_sha(folder / name) for name in (
                                        "rgb.png", "depth_meters.npy", "world_points.npy", "calibration.json")}}
        finally:
            env.close()
        print("Captured RGB-D:", identifier, flush=True)
    requests = []
    for case in config["cases"]:
        payload = template(by_id[case["example_id"]]["model_inputs"]["instruction"], case["upright"])
        requests.append({"case_id": case["id"], "payload": payload,
                         "payload_sha256": hashlib.sha256(json_text(payload).encode()).hexdigest(),
                         "image_sha256": captures[case["example_id"]]["files"]["rgb.png"]})
    preserved = {}
    for pattern in ("**/predictions.jsonl", "**/summary.json"):
        for path in sorted((ROOT / "results").glob(pattern)):
            preserved[str(path.relative_to(ROOT))] = file_sha(path)
    protocol = {"version": "rgbd-numerical-1.0", "config": config,
                "dataset_sha256": file_sha(source / "records.jsonl"),
                "prompt_sha256": file_sha(ROOT / config["prompt"]),
                "schema_sha256": hashlib.sha256(json_text(blank_schema()).encode()).hexdigest(),
                "captures": captures, "requests": requests,
                "implementation_hashes": {path: file_sha(ROOT / path) for path in IMPLEMENTATION},
                "preserved_artifacts": preserved,
                "scope": "four previously queried scenes; six development cases, two explicit upright variants",
                "model_inventory": False, "simulator_answers_in_inference": False,
                "grasp_validation": "geometric proposals only; no IK/contact/execution success claimed"}
    frozen.write_text(json_text(protocol) + "\n")
    (output / "manifest.json").write_text(json_text({"captures": captures, "config": config}) + "\n")
    return protocol


def check_protocol(config_path):
    config = read_config(config_path)
    protocol = json.loads((ROOT / config["frozen_protocol"]).read_text())
    if config != protocol["config"]:
        raise ValueError("Numerical pilot settings changed after freezing")
    records, source = read_records(ROOT / config["dataset"])
    if file_sha(source / "records.jsonl") != protocol["dataset_sha256"]:
        raise ValueError("Source dataset changed")
    for path, sha in {**protocol["implementation_hashes"], **protocol["preserved_artifacts"]}.items():
        if file_sha(ROOT / path) != sha:
            raise ValueError("Frozen or preserved artifact changed: " + path)
    if file_sha(ROOT / config["prompt"]) != protocol["prompt_sha256"]:
        raise ValueError("Numerical prompt changed")
    if hashlib.sha256(json_text(blank_schema()).encode()).hexdigest() != protocol["schema_sha256"]:
        raise ValueError("Numerical schema changed")
    for identifier, capture in protocol["captures"].items():
        for name, sha in capture["files"].items():
            if file_sha(ROOT / config["depth_dataset"] / identifier / name) != sha:
                raise ValueError("RGB-D capture changed")
    return config, protocol, {r["example_id"]: r for r in records}


def run(config_path, output, *, allow_paid=False, dry_run=False, client=None):
    config, protocol, records = check_protocol(config_path)
    prompt = (ROOT / config["prompt"]).read_text()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve existing numerical results")
    if not dry_run and not allow_paid:
        raise PermissionError("Use --allow-paid only for the authorized bounded pilot")
    output.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(config)
    manifest = {"config": config, "model": config["model"], "provider": "openai", "offline_mock": client is not None,
                "dry_run": dry_run, "dataset": str(ROOT / config["dataset"] / "records.jsonl"),
                "frozen_protocol_sha256": file_sha(ROOT / config["frozen_protocol"]),
                "model_input_inventory": False, "depth_sent_to_vlm": False,
                "numerical_geometry_uses_depth": True, "segmentation_used": False,
                "dino_used": False, "reference_answers_sent": False, "prompt_text": prompt,
                "cases": config["cases"], "ledger": ledger.as_dict()}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json_text(manifest) + "\n")
    arguments = []
    for case, frozen in zip(config["cases"], protocol["requests"]):
        record = records[case["example_id"]]
        payload = template(record["model_inputs"]["instruction"], case["upright"])
        if payload != frozen["payload"]:
            raise ValueError("Instruction-derived request changed")
        folder = output / case["id"]
        folder.mkdir()
        image = (ROOT / config["depth_dataset"] / case["example_id"] / "rgb.png").read_bytes()
        args = sdk_arguments(config, payload, image, prompt)
        (folder / "request.json").write_text(json_text(args) + "\n")
        (folder / "payload.json").write_text(json_text(payload) + "\n")
        (folder / "original.png").write_bytes(image)
        arguments.append((case, payload, args, folder))
    if dry_run:
        return manifest
    if client is None:
        from openai import OpenAI
        client = OpenAI(base_url="https://api.openai.com/v1", timeout=config["timeout_seconds"], max_retries=0)
    rows = []
    for case, payload, args, folder in arguments:
        reservation = ((len((prompt+json_text(payload)+json_text(blank_schema())).encode())
                        +config["image_token_allowance"]+512)*config["pricing"]["input_per_million"]
                       +config["max_output_tokens"]*config["pricing"]["output_per_million"])/1e6
        row = {"case_id": case["id"], "example_id": case["example_id"], "status": "api_failure",
               "blanks": None, "numericals": None, "attempts": 0, "cost_usd": 0.}
        ledger.admit(reservation)
        row["attempts"] = 1
        start = time.perf_counter()
        try:
            response = client.responses.create(**args)
        except Exception as error:
            ledger.unknown_cost_reservation += reservation
            row["error_type"] = type(error).__name__
            rows.append(row)
            manifest["ledger"] = ledger.as_dict()
            manifest_path.write_text(json_text(manifest) + "\n")
            (output / "predictions.jsonl").write_text("".join(json_text(r)+"\n" for r in rows))
            break  # Preserve unknown charges and stop; never substitute a mock or retry.
        raw = response.model_dump(mode="json")
        (folder / "response.json").write_text(json_text(raw) + "\n")
        usage = response.usage
        row["cost_usd"] = ledger.account(ProviderResponse(response.output_text, raw,
                                                        usage.input_tokens if usage else 0,
                                                        usage.output_tokens if usage else 0), reservation)
        manifest["ledger"] = ledger.as_dict()
        manifest_path.write_text(json_text(manifest) + "\n")
        try:
            row["blanks"] = validate_blanks(json.loads(response.output_text))
            capture = ROOT / config["depth_dataset"] / case["example_id"]
            depth = np.load(capture / "depth_meters.npy", allow_pickle=False)
            calibration = json.loads((capture / "calibration.json").read_text())
            row["numericals"] = compile_numericals(row["blanks"], payload, depth, calibration, config["geometry"])
            row["status"] = "valid"
            (folder / "numericals.json").write_text(json_text(row["numericals"]) + "\n")
        except (ValueError, TypeError, KeyError) as error:
            row["status"], row["error_type"] = "invalid_output", type(error).__name__
        row["elapsed_seconds"] = time.perf_counter()-start
        rows.append(row)
        (output / "predictions.jsonl").write_text("".join(json_text(r)+"\n" for r in rows))
        print(case["id"], row["status"], "estimated $" + format(row["cost_usd"], ".5f"), flush=True)
    return manifest
