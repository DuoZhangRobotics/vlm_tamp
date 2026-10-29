"""Run the pinned RoArtNet network on prepared clouds, without scoring labels."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vlm_tamp.roartnet_adapter import RoArtNetEstimator
import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def jobs(inputs, profiles, sensitivity):
    selected = []
    for profile in profiles:
        for item in inputs:
            selected.append((item, profile, 42))
    if sensitivity and "author-eval" in profiles:
        for item in inputs:
            if (item["kind"] == "local" and item["trial"] == "state020-r1-luna"
                    and item["camera"] == "frontview" and item["variant"] == "sam"
                    and item["observation"] in ["obs001", "obs018"]):
                selected.extend((item, "author-eval", seed) for seed in [43, 44])
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/roartnet-perception-20261025")
    parser.add_argument("--profiles", nargs="+", default=["author-eval", "checkpoint"])
    parser.add_argument("--kind", choices=["author", "local", "all"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-seed-sensitivity", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    items = json.loads((run / "inputs.json").read_text())
    items = [item for item in items if args.kind == "all" or item["kind"] == args.kind]
    tasks = jobs(items, args.profiles, not args.no_seed_sensitivity)
    if args.limit is not None:
        tasks = tasks[:args.limit]
    started = time.perf_counter()
    models = {}
    completed = []
    for index, (item, profile, seed) in enumerate(tasks):
        name = f"{item['id']}--{profile}--seed{seed}"
        output = run / "predictions" / (name + ".json")
        if output.is_file():
            saved = json.loads(output.read_text())
            if saved["status"] == "complete":
                completed.append({"id": name, "status": "complete", "cached": True})
                continue
        write_json(run / "inference-status.json", {"status": "running", "pid": os.getpid(),
            "current": name, "task_index": index, "task_count": len(tasks), "completed": completed,
            "elapsed_seconds": time.perf_counter() - started})
        if profile not in models:
            try:
                model = RoArtNetEstimator(ROOT / ".runtime/rpmart",
                    ROOT / ".runtime/models/RoArtNet/formal_microwave_2024_01_16_12_16_57", profile=profile)
            except Exception as error:
                write_json(run / "inference-status.json", {"status": "startup_failed", "pid": os.getpid(),
                    "current": name, "completed": completed, "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc()})
                raise
            model.torch.set_num_threads(4)
            models[profile] = model
            write_json(run / f"runtime-{profile}.json", {"profile": profile,
                "imports_seconds": model.import_seconds, "load_weights_seconds": model.model_load_seconds,
                "torch": model.torch.__version__, "cupy": model.cupy.__version__,
                "torch_cuda": model.torch.version.cuda, "gpu": model.torch.cuda.get_device_name(),
                "config": model.config, "inference_kwargs": model.kwargs,
                "source": "official inference function, unchanged models and native SHOT source",
                "native_labels_available_to_estimator": False})
        model = models[profile]
        row = {"id": name, "input_id": item["id"], "input_sha256": item["input_sha256"],
               "profile": profile, "seed": seed, "kind": item["kind"], "variant": item["variant"]}
        before = time.perf_counter()
        try:
            file = ROOT / item["input"]
            if hashlib.sha256(file.read_bytes()).hexdigest() != item["input_sha256"]:
                raise ValueError("Prepared input changed after manifest creation")
            with np.load(file, allow_pickle=False) as arrays:
                points = arrays["point_cloud"]
                c2w = arrays["camera_to_world"] if "camera_to_world" in arrays else None
            row["input_load_and_hash_seconds"] = time.perf_counter() - before
            row["prediction"] = model.predict(points, camera_to_world=c2w, seed=seed,
                                              author_raw_fixture=item["kind"] == "author")
            row["status"] = "complete"
        except Exception as error:
            row.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        row["case_wall_seconds"] = time.perf_counter() - before
        write_json(output, row)
        completed.append({"id": name, "status": row["status"]})
        print(json.dumps({k: row[k] for k in ["id", "status", "case_wall_seconds"]}
                         | ({"error": row["error"]} if "error" in row else {})), flush=True)
    write_json(run / "inference-status.json", {"status": "complete", "pid": os.getpid(),
        "completed": completed, "elapsed_seconds": time.perf_counter() - started,
        "failures": sum(row["status"] == "failed" for row in completed)})


if __name__ == "__main__":
    main()
