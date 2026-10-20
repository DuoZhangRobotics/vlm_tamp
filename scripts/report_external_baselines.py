"""Source availability and measured first runs; heterogeneous components stay separate."""

import json
import shutil
from collections import Counter

from report_rgb_choices import escaped, pretty
from report_rgb_role_points import point_overlay
from vlm_tamp.dataset import read_records, safe_path
from vlm_tamp.libero import ROOT


def main():
    output = ROOT / "results/external-baselines-report"
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    records, root = read_records(ROOT / "data/rgb-role-generalization")
    by_id = {record["example_id"]: record for record in records}
    local = ROOT / "results/robopoint-native-cap"
    rows = [json.loads(line) for line in (local / "predictions.jsonl").read_text().splitlines()]
    metrics = [json.loads(line) for line in (local / "metrics.jsonl").read_text().splitlines()]
    robo_summary = json.loads((local / "summary.json").read_text())
    mini_summary = json.loads((ROOT / "results/rgb-role-generalization-mini/summary.json").read_text())
    mini_rows = {row["example_id"]: row for row in (
        json.loads(line) for line in (ROOT / "results/rgb-role-generalization-mini/predictions.jsonl").read_text().splitlines())}
    mini_metrics = {metric["example_id"]: metric for metric in (
        json.loads(line) for line in (ROOT / "results/rgb-role-generalization-mini/metrics.jsonl").read_text().splitlines())}
    tasks, transitions = [], Counter()
    for index, (row, metric) in enumerate(zip(rows, metrics)):
        record = by_id[row["example_id"]]
        old = mini_metrics[row["example_id"]]["semantic_goal"]["exact_match"]
        new = metric["semantic_goal"]["exact_match"]
        transitions["both_correct" if old and new else "RoboPoint_only" if new else "Mini_only" if old else "both_incorrect"] += 1
        prefix = f"task-{index:02d}"
        source = safe_path(root, record["model_inputs"]["image"])
        shutil.copyfile(source, assets / f"{prefix}-rgb.png")
        point_overlay(source, row["points"], assets / f"{prefix}-robopoint.png")
        point_overlay(source, mini_rows[row["example_id"]]["points"], assets / f"{prefix}-mini.png")
        tasks.append({"example_id": row["example_id"], "instruction": record["model_inputs"]["instruction"],
                      "robo_metric": metric, "mini_metric": mini_metrics[row["example_id"]],
                      "raw_role_responses": row["raw_role_responses"], "asset_prefix": prefix})
    rekep = json.loads((ROOT / "results/rekep-component-summary.json").read_text())
    keypoints = json.loads((ROOT / "results/rekep-keypoint-smoke/metadata.json").read_text())
    vilain = json.loads((ROOT / "results/vilain-component-smoke-fixed/result.json").read_text())
    ledger = json.loads((ROOT / "results/external-component-api/ledger.json").read_text())
    # Include component usage in future project-wide paid-history scans.
    (ROOT / "results/external-component-api/manifest.json").write_text(json.dumps({
        "provider": "openai", "model": ledger["model"], "offline_mock": False,
        "model_input_inventory": False, "dataset": None,
        "experiment_scope": "Ported ReKep/ViLaIn component smoke tests; no full robot reproduction",
        "ledger": {"api_attempts": ledger["api_attempts"],
                   "estimated_usage_cost_usd": ledger["estimated_usage_cost_usd"],
                   "unknown_usage_reserved_usd": ledger["unknown_reserved_usd"],
                   "budget_usd": ledger["budget_usd"]},
    }, indent=2) + "\n")
    availability = json.loads((ROOT / "results/external-baseline-audit/source-status.json").read_text())
    shutil.copyfile(ROOT / "results/rekep-keypoint-smoke/numbered.png", assets / "rekep-numbered.png")
    shutil.copyfile(ROOT / "results/vilain-component-smoke-fixed/rgb.png", assets / "vilain-rgb.png")
    summary = {"robo_summary": robo_summary, "mini_summary": mini_summary, "paired_transitions": dict(transitions),
               "tasks": tasks, "rekep": rekep, "keypoints": keypoints, "vilain": vilain,
               "api_ledger": ledger, "availability": availability,
               "native_full_robot_systems_reproduced": False,
               "scope": "Adapted RoboPoint image-role benchmark plus ported ReKep and ViLaIn grounding component smoke tests. No cross-paper robot success ranking."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    code = "\n\n".join(path.read_text() for path in sorted((ROOT / rekep["directory"]).glob("*_constraints.txt")))
    pieces = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Existing grounding methods: first runs</title><style>body{font:16px system-ui;line-height:1.5;max-width:1500px;margin:auto;padding:24px;background:#111826;color:#edf2ff}article,.panel{background:#1b2638;padding:20px;margin:20px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:16px}img{width:100%;height:auto;max-width:768px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #405169;text-align:left}a{color:#79d5ff}details{margin:12px 0}</style></head><body>',
        '<h1>Existing grounding methods: code availability and first runs</h1><p>Three public implementations were checked and their available grounding components tried. GroundedPlanBench still advertises Code (Coming Soon). These results do not reproduce complete robot systems or establish cross-paper performance rankings.</p>',
        '<table><tr><th>Work</th><th>Available source</th><th>What was tried</th></tr><tr><td><a href="https://github.com/wentaoyuan/RoboPoint">RoboPoint</a></td><td>Code and pretrained weights</td><td>Twenty shared RGB scenes, 39 separate role queries; 8-bit 13B model, greedy decoding, native 1024-token limit</td></tr><tr><td><a href="https://github.com/huangwl18/ReKep">ReKep</a></td><td>Simulator demo and grounding modules; real-world perception is omitted from release</td><td>One RGB-D scene, predicted RGB masks, native keypoint proposer and one live constraint query</td></tr><tr><td><a href="https://github.com/omron-sinicx/ViLaIn-TAMP/tree/jeremy_dev">ViLaIn-TAMP</a></td><td>Interpreter on public jeremy_dev branch; main has data/examples</td><td>Ported detection → initial state → goals on one public cooking scene</td></tr><tr><td><a href="https://groundedplanning.github.io/GroundedPlanning/">GroundedPlanBench</a></td><td>Paper/project available; official code link says Coming Soon</td><td>No runnable benchmark or annotated data release found</td></tr></table>',
        f'<h2>Shared RGB role comparison</h2><table><tr><th>Model/interface</th><th>Complete goal agreement</th><th>Correct individual roles</th></tr><tr><td>Saved GPT-5.4 Mini, one JSON per task</td><td>{mini_summary["semantic_complete_goals"]}/20</td><td>25/39</td></tr><tr><td>RoboPoint adapter, first returned point per role</td><td>{robo_summary["semantic_complete_goals"]}/20</td><td>{robo_summary["correct_roles"]}/39</td></tr></table>',
        '<div class="panel">The images, language parser, goal compiler and mask-based scorer are shared. RoboPoint uses one question per role, 8-bit weights and greedy decoding; this is an adapter benchmark, not its original evaluation protocol. No image candidates or oracle inventory enter the model. All returned point lists remain saved; the first point is selected without consulting masks. The initial 128-token setup run was truncated and is retained separately, excluded from this comparison.</div>',
        f'<p>Paired outcomes: {escaped(dict(transitions))}. The one-task score difference is too small for a model ranking. API smoke tests used {ledger["api_attempts"]} actual calls / ${ledger["estimated_usage_cost_usd"]:.6f} estimated usage; RoboPoint local inference adds no API charge.</p>',
        '<h2>ReKep component test</h2><div class="grid"><section><img src="assets/rekep-numbered.png" alt="Native ReKep keypoints from predicted RGB segments and sensor depth"></section><section>',
        f'<p>The native proposer produced {keypoints["num_keypoints"]} keypoints. The live query produced {rekep["num_stages"]} stages and {rekep["parsed_python_functions"]} parseable constraint functions. GPT-4o-2024-11-20 replaces the retired chatgpt-4o-latest alias. No generated code or robot trajectory was executed.</p>',
        '<p>Offline inspection: grasp keypoint 23 belongs to the bowl. The draft calls keypoint 24 the plate, but that pixel is background; keypoint 20 belongs to the actual plate. Valid Python and in-range indices therefore do not establish correct task grounding. This is one ported RGB-D/predicted-mask example, not the native OmniGibson setup or a measured overall accuracy.</p>',
        '</section></div><details><summary>Actual generated constraint draft</summary><pre>' + escaped(code) + '</pre></details>',
        '<h2>ViLaIn interpreter test</h2><div class="grid"><section><img src="assets/vilain-rgb.png" alt="Public native ViLaIn cooking scene"></section><section><p>The pipeline receives RGB and its predefined domain/category vocabulary. It uses its own detected object boxes, with no supplied fixed-object boxes or reference inventory. The port removes an unused ROS import, bypasses a private lab proxy, substitutes GPT-4o for the local Qwen detector, and handles an empty fixture list.</p><p>It generates an object list, initial state and goal. The initial state incorrectly asserts that the cucumber is already on the plate; the held-back reference places it on the tray. It also uses robot symbol a_bot without declaring it among the detected objects. This stricter RGB-only adaptation is not a faithful full-system reproduction.</p></section></div>',
        '<details><summary>Actual generated specification and adaptation record</summary><pre>' + pretty(vilain) + '</pre></details>',
        '<h2>All twenty shared RGB scenes</h2>',
    ]
    for task in tasks:
        prefix = task["asset_prefix"]
        pieces.extend([
            f'<article><h3>{escaped(task["instruction"])}</h3><p>Mini complete goal: {task["mini_metric"]["semantic_goal"]["exact_match"]}. RoboPoint complete goal: {task["robo_metric"]["semantic_goal"]["exact_match"]}.</p><div class="grid"><section><h4>Original RGB</h4><img src="assets/{prefix}-rgb.png" alt="Original shared RGB"></section><section><h4>Mini points</h4><img src="assets/{prefix}-mini.png" alt="Saved Mini points"></section><section><h4>RoboPoint selected first points</h4><img src="assets/{prefix}-robopoint.png" alt="First points selected from actual RoboPoint outputs"></section></div>',
            '<details><summary>Raw RoboPoint responses</summary><pre>' + pretty(task["raw_role_responses"]) + '</pre></details>',
            '<details><summary>Independent role and final-goal checks</summary><pre>' + pretty(task["robo_metric"]) + '</pre></details></article>',
        ])
    pieces.append('</body></html>')
    (output / "index.html").write_text("\n".join(pieces))
    print(json.dumps({"RoboPoint_complete_goals": robo_summary["semantic_complete_goals"],
                      "Mini_complete_goals": mini_summary["semantic_complete_goals"],
                      "new_api_usage_usd": ledger["estimated_usage_cost_usd"], "report": str(output / "index.html")}, indent=2))


if __name__ == "__main__":
    main()
