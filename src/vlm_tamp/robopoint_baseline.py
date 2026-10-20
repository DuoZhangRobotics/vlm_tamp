"""RGB-role adapter for the released RoboPoint model; references remain offline."""

import ast
import json
import math
import os
import time
from pathlib import Path

from .dataset import read_records, safe_path
from .libero import ROOT
from .rgb_evaluation import evaluate_rgb, load_references
from .rgb_role_choices import instruction_template
from .rgb_role_diagnostics import diagnose_role_points
from .rgb_role_points import compile_points, digest


PROMPT = (
    "Task instruction: {instruction}\n"
    "Pinpoint several points on the visible surface of the object described as: {role}. "
    "Preserve the instance and spatial qualifiers in that description. "
    "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...]. "
    "The coordinates should be between 0 and 1, indicating normalized pixel locations "
    "with a top-left origin."
)


def parse_points(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("Unclosed point response fence")
        text = "\n".join(lines[1:-1])
    try:
        points = ast.literal_eval(text)
    except (ValueError, SyntaxError) as error:
        raise ValueError("Response is not a literal point list") from error
    if not isinstance(points, (list, tuple)) or not points:
        raise ValueError("Response needs a nonempty point list")
    if len(points) == 2 and all(isinstance(value, (int, float)) for value in points):
        points = [points]
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2 or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1 for value in point
        ):
            raise ValueError("Points need finite normalized [x,y] coordinates")
    return [[float(value) for value in point] for point in points]


def prepare_protocol(path):
    dataset = ROOT / "data/rgb-role-generalization"
    records, root = read_records(dataset)
    protocol = {
        "dataset": str(dataset.relative_to(ROOT)),
        "dataset_sha256": digest((root / "records.jsonl").read_bytes()),
        "model_id": "wentao-yuan/robopoint-v1-vicuna-v1.5-13b",
        "model_revision": "508ed28687caeeb923e1d7d5905e37dccc16c991",
        "repository": "https://github.com/wentaoyuan/RoboPoint",
        "repository_commit": "206cb1e086fc3960a8673725a8540766655cd9c5",
        "references": "configs/rgb_role_generalization_references.json",
        "offline_roles": "configs/rgb_role_generalization_roles.json",
        "prompt_template": PROMPT,
        "quantization": "8bit; supported upstream option, not full-precision paper replication",
        "conversation_mode": "llava_v1",
        "do_sample": False, "max_new_tokens": 128,
        "point_selection": "First returned point; no mask-dependent selection or correction",
        "query_granularity": "One RGB query per instruction-derived role; 39 queries for 20 tasks",
        "input_inventory": False, "frozen_before_inference": True,
        "requests": {
            record["example_id"]: {
                "image_sha256": digest(safe_path(root, record["model_inputs"]["image"]).read_bytes()),
                "role_queries": {slot: PROMPT.format(instruction=record["model_inputs"]["instruction"], role=phrase)
                                 for slot, phrase in instruction_template(record["model_inputs"]["instruction"])["roles"].items()},
            } for record in records
        },
        "implementation_hashes": {
            name: digest((ROOT / "src/vlm_tamp" / name).read_bytes())
            for name in ("robopoint_baseline.py", "rgb_role_points.py", "rgb_role_choices.py",
                         "rgb_evaluation.py", "rgb_role_diagnostics.py", "rgb_grounding.py")
        },
    }
    path = Path(path)
    if path.exists():
        raise FileExistsError("Preserve the existing RoboPoint protocol")
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def run(protocol_path, checkpoint, output):
    protocol = json.loads(Path(protocol_path).read_text())
    for name, expected in protocol["implementation_hashes"].items():
        if digest((ROOT / "src/vlm_tamp" / name).read_bytes()) != expected:
            raise ValueError("Frozen baseline implementation changed")
    os.environ["HF_HUB_CACHE"] = str(ROOT / ".runtime/hf-cache")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    import torch
    from PIL import Image
    from robopoint.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from robopoint.conversation import conv_templates
    from robopoint.model.builder import load_pretrained_model
    from robopoint.mm_utils import tokenizer_image_token, process_images
    from robopoint.utils import disable_torch_init

    records, root = read_records(ROOT / protocol["dataset"])
    if digest((root / "records.jsonl").read_bytes()) != protocol["dataset_sha256"]:
        raise ValueError("Frozen baseline dataset changed")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh RoboPoint output directory")
    output.mkdir(parents=True, exist_ok=True)
    disable_torch_init()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    tokenizer, model, processor, _ = load_pretrained_model(
        str(checkpoint), None, protocol["model_id"].split("/")[-1], load_8bit=True,
        device_map="auto", token=False,
    )
    model.eval()
    manifest = {"provider": "local_robopoint", "model": protocol["model_id"], "offline_mock": False,
                "dataset": str(root / "records.jsonl"), "protocol": str(protocol_path),
                "model_revision": protocol["model_revision"], "quantization": protocol["quantization"],
                "model_input_inventory": False, "api_cost_usd": 0, "planned_tasks": len(records)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    with (output / "predictions.jsonl").open("w") as stream:
        for record in records:
            instruction = record["model_inputs"]["instruction"]
            requests = protocol["requests"][record["example_id"]]
            image_path = safe_path(root, record["model_inputs"]["image"])
            assert digest(image_path.read_bytes()) == requests["image_sha256"]
            image = Image.open(image_path).convert("RGB")
            image_tensor = process_images([image], processor, model.config)[0]
            points, raw = {}, {}
            start = time.perf_counter()
            for slot, question in requests["role_queries"].items():
                prefix = DEFAULT_IMAGE_TOKEN
                if model.config.mm_use_im_start_end:
                    prefix = DEFAULT_IM_START_TOKEN + prefix + DEFAULT_IM_END_TOKEN
                conversation = conv_templates[protocol["conversation_mode"]].copy()
                conversation.append_message(conversation.roles[0], prefix + "\n" + question)
                conversation.append_message(conversation.roles[1], None)
                input_ids = tokenizer_image_token(conversation.get_prompt(), tokenizer,
                                                  IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
                with torch.inference_mode():
                    generated = model.generate(input_ids, images=image_tensor.unsqueeze(0).half().cuda(),
                                               image_sizes=[image.size], do_sample=False,
                                               max_new_tokens=protocol["max_new_tokens"], use_cache=True)
                response = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
                raw[slot] = {"question": question, "text": response}
                try:
                    all_points = parse_points(response)
                    raw[slot]["all_points"] = all_points
                    points[slot] = all_points[0]
                except ValueError as error:
                    raw[slot]["parse_error"] = str(error)
                    points[slot] = None
            try:
                prediction = compile_points(points, instruction).model_dump(mode="json")
                status = "ok"
            except ValueError:
                prediction, status = None, "abstained_or_invalid"
            row = {"example_id": record["example_id"], "condition": "robopoint_rgb_roles", "status": status,
                   "points": points, "prediction": prediction, "raw_role_responses": raw,
                   "latency_seconds": time.perf_counter() - start}
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            rows.append(row)
            print(record["example_id"], status, flush=True)
    # Decode answer masks and roles only after local inference has ended.
    references = load_references(root, ROOT / protocol["references"])
    role_owners = json.loads((ROOT / protocol["offline_roles"]).read_text())["roles"]
    from .rgb_grounding import RGBPrediction
    metrics = []
    for record, row in zip(records, rows):
        prediction = RGBPrediction.model_validate(row["prediction"]) if row["prediction"] else None
        metric = evaluate_rgb(record, prediction, references[record["example_id"]], root)
        details = diagnose_role_points(record, row["points"], role_owners[record["example_id"]], root)
        metrics.append({"example_id": record["example_id"], **metric, "offline_role_diagnostics": details})
    (output / "metrics.jsonl").write_text("".join(json.dumps(metric) + "\n" for metric in metrics))
    summary = {"planned_tasks": len(records), "local_role_queries": sum(len(row["points"]) for row in rows),
               "compiled": sum(row["prediction"] is not None for row in rows),
               "semantic_complete_goals": sum(metric["semantic_goal"]["exact_match"] for metric in metrics),
               "correct_roles": sum(detail["diagnostic"] == "correct_instance" for metric in metrics
                                    for detail in metric["offline_role_diagnostics"]),
               "api_cost_usd": 0, "input_inventory": False, "quantization": protocol["quantization"],
               "robot_execution_tested": False, "query_granularity": protocol["query_granularity"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary
