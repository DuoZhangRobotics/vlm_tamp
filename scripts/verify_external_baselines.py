"""Recheck actual external model outputs and preserve the earlier RGB evidence."""

import json
import subprocess

import numpy as np
from PIL import Image

from verify_rgb_role_points import PageParser
from vlm_tamp.dataset import read_records, safe_path
from vlm_tamp.libero import ROOT
from vlm_tamp.rgb_evaluation import evaluate_rgb, load_references
from vlm_tamp.rgb_grounding import RGBPrediction
from vlm_tamp.rgb_role_diagnostics import diagnose_role_points
from vlm_tamp.rgb_role_points import compile_points, digest
from vlm_tamp.robopoint_baseline import parse_points


def main():
    protocol = json.loads((ROOT / "configs/frozen_robopoint_native_cap.json").read_text())
    for name, expected in protocol["implementation_hashes"].items():
        assert digest((ROOT / "src/vlm_tamp" / name).read_bytes()) == expected
    records, root = read_records(ROOT / protocol["dataset"])
    assert digest((root / "records.jsonl").read_bytes()) == protocol["dataset_sha256"]
    references = load_references(root, ROOT / protocol["references"])
    roles = json.loads((ROOT / protocol["offline_roles"]).read_text())["roles"]
    trial = ROOT / "results/robopoint-native-cap"
    rows = [json.loads(line) for line in (trial / "predictions.jsonl").read_text().splitlines()]
    metrics = [json.loads(line) for line in (trial / "metrics.jsonl").read_text().splitlines()]
    assert len(rows) == len(metrics) == len(records) == 20
    queries, correct, role_correct = 0, 0, 0
    for record, row, saved in zip(records, rows, metrics):
        assert record["example_id"] == row["example_id"]
        expected = protocol["requests"][row["example_id"]]
        assert digest(safe_path(root, record["model_inputs"]["image"]).read_bytes()) == expected["image_sha256"]
        points = {}
        for slot, raw in row["raw_role_responses"].items():
            assert raw["question"] == expected["role_queries"][slot]
            parsed = parse_points(raw["text"])
            assert parsed == raw["all_points"]
            points[slot] = parsed[0]
            queries += 1
        assert points == row["points"]
        compiled = compile_points(points, record["model_inputs"]["instruction"]).model_dump(mode="json")
        assert compiled == row["prediction"]
        prediction = RGBPrediction.model_validate(compiled)
        metric = evaluate_rgb(record, prediction, references[row["example_id"]], root)
        assert json.loads(json.dumps(metric)) == {key: value for key, value in saved.items()
                                                 if key not in {"example_id", "offline_role_diagnostics"}}
        details = diagnose_role_points(record, points, roles[row["example_id"]], root)
        assert details == saved["offline_role_diagnostics"]
        correct += metric["semantic_goal"]["exact_match"]
        role_correct += sum(detail["diagnostic"] == "correct_instance" for detail in details)
    assert queries == 39 and correct == 7 and role_correct == 23
    api_root = ROOT / "results/external-component-api"
    ledger = json.loads((api_root / "ledger.json").read_text())
    prompts = outputs = 0
    for index in range(ledger["api_attempts"]):
        args = json.loads((api_root / f"request-{index:02d}.json").read_text())
        raw = json.loads((api_root / f"response-{index:02d}.json").read_text())
        assert args["model"] == raw["model"] == "gpt-4o-2024-11-20"
        assert args["max_tokens"] <= 1536 and args["store"] is False
        prompts += raw["usage"]["prompt_tokens"]
        outputs += raw["usage"]["completion_tokens"]
    cost = (prompts * 2.5 + outputs * 10) / 1e6
    assert abs(cost - ledger["estimated_usage_cost_usd"]) < 1e-12
    assert cost < 0.10 and ledger["unknown_reserved_usd"] == 0
    source = json.loads((ROOT / "results/external-baseline-audit/source-status.json").read_text())
    for repo in source["repositories"]:
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / repo["path"], text=True).strip() == repo["commit"]
    native = json.loads((ROOT / "results/rekep-component-summary.json").read_text())
    metadata = json.loads((ROOT / "results/rekep-keypoint-smoke/metadata.json").read_text())
    record = next(record for record in records if record["example_id"] == metadata["example_id"])
    matches = {}
    for index in (20, 23, 24):
        y, x = metadata["pixels_yx"][index]
        matches[str(index)] = [identifier for identifier, ref in record["reference"]["visual"].items()
                              if np.asarray(Image.open(safe_path(root, ref["mask"])))[y, x] > 0]
    assert matches == {"20": ["plate_1"], "23": ["akita_black_bowl_1"], "24": []}
    assert native["generated_code_executed"] is False and native["parsed_python_functions"] == 3
    report = ROOT / "results/external-baselines-report"
    page = PageParser()
    page.feed((report / "index.html").read_text())
    assert page.articles == 20 and len(page.images) == 62
    for name in page.images:
        assert (report / name).is_file()
    verification = {"actual_RoboPoint_role_queries_checked": queries, "RoboPoint_complete_goals": correct,
                    "RoboPoint_correct_roles": role_correct, "actual_API_calls_checked": ledger["api_attempts"],
                    "actual_API_estimated_usage_usd": cost, "native_rekep_offline_point_matches": matches,
                    "native_full_robot_reproduction_claimed": False,
                    "provided_oracle_inventory": False, "report_images_checked": len(page.images),
                    "checks_passed": True}
    (report / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
