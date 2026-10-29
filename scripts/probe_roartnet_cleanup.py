"""Post-benchmark sensor-only component-cleanup ablation; no private labels."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vlm_tamp.point_cloud_components import largest_component
from vlm_tamp.roartnet_adapter import RoArtNetEstimator


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    run = ROOT / "results/roartnet-perception-20261025"
    dest = run / "cleanup"
    dest.mkdir(exist_ok=True)
    items = [x for x in json.loads((run / "inputs.json").read_text())
             if x["id"].startswith("state020-r1-luna-obs001-frontview-")]
    write(dest / "protocol.json", {"phase": "Exploratory after original 38 predictions; no baseline changes",
        "motivation": "Two-pixel mask expansion admitted disconnected background depth up to 3.48m, versus target maximum1.07m",
        "input_ids": [x["id"] for x in items], "profile": "author-eval", "seed": 42,
        "voxel_m": .005, "connection_m": .015, "selection": "Largest component by original point count",
        "native_information_used": False})
    model = RoArtNetEstimator(ROOT / ".runtime/rpmart",
        ROOT / ".runtime/models/RoArtNet/formal_microwave_2024_01_16_12_16_57", profile="author-eval")
    model.torch.set_num_threads(4)
    rows = []
    for item in items:
        with np.load(ROOT / item["input"], allow_pickle=False) as data:
            points, transform = data["point_cloud"], data["camera_to_world"]
        started = time.perf_counter()
        mask, diagnostics = largest_component(points)
        elapsed = time.perf_counter() - started
        output = dest / (item["id"] + ".npz")
        np.savez_compressed(output, point_cloud=points[mask], camera_to_world=transform)
        prediction = model.predict(points[mask], camera_to_world=transform, seed=42)
        row = {"input_id": item["id"], "input_sha256": item["input_sha256"],
               "cleaned_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
               "cleanup": diagnostics, "cleanup_seconds": elapsed, "prediction": prediction}
        rows.append(row)
        write(dest / "predictions.json", rows)
        print(json.dumps({"id": item["id"], "retained": int(mask.sum()), "removed": int((~mask).sum()),
                          "cleanup_seconds": elapsed}), flush=True)
    write(dest / "completion.json", {"cases": len(rows), "native_references_read": False})


if __name__ == "__main__":
    main()
