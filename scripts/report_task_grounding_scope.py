"""Inspect ordinary pick/place reuse and new middle-drawer grounding; no API calls."""

import argparse
import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_landmark_trial import file_sha
from vlm_tamp.rgbd_numericals import json_text, pixel_index


DEFAULT_CONFIG = "configs/task_grounding_scope.json"
DEFAULT_TRIAL = "results/task-grounding-scope-luna"
DEFAULT_OUTPUT = "results/task-grounding-scope-report"
HISTORICAL_TRIAL = "results/rgbd-dino-luna-comparison"
HISTORICAL_REPORT = "results/rgbd-dino-luna-report/summary.json"


def local_path(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    by_id = {row["case_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate saved case IDs")
    return by_id


def response_text(raw):
    return "".join(item["text"] for block in raw.get("output", []) if block.get("type") == "message"
                   for item in block.get("content", []) if item.get("type") == "output_text")


def response_context(folder, row):
    path = Path(folder) / "response.json"
    if not path.exists():
        return {"status": row.get("status", "not_run"), "selection_scored": False,
                "incomplete_details": None, "output_text": None}
    raw = json.loads(path.read_text())
    return {"status": raw["status"], "selection_scored": raw["status"] == "completed"
            and row["status"] == "valid", "incomplete_details": raw.get("incomplete_details"),
            "output_text": response_text(raw)}


def copy_asset(source, assets, name):
    """Copy exact source bytes and record their provenance for saved-result replay."""
    source, assets = Path(source), Path(assets)
    if Path(name).name != name:
        raise ValueError("Asset names must stay inside the report")
    destination = assets / name
    destination.write_bytes(source.read_bytes())
    return {"source": str(source), "asset": "assets/" + name, "sha256": file_sha(source)}


def points_picture(source, blanks, destination):
    colors = ((255, 70, 60), (40, 235, 130), (255, 205, 45), (65, 180, 255), (225, 85, 255))
    with Image.open(source) as original:
        image = original.convert("RGB")
    draw = ImageDraw.Draw(image)
    keys = ("source_uv", "handle_uv", "face_a_uv", "face_b_uv", "face_c_uv")
    for key, color in zip(keys, colors):
        uv = (blanks or {}).get(key)
        if uv is None:
            continue
        try:
            x, y = pixel_index(uv, (image.height, image.width))
        except (ValueError, TypeError):
            continue
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline="black", width=2)
        label = key.removesuffix("_uv")
        tx, ty = min(x + 9, image.width - len(label) * 7 - 2), max(0, y - 13)
        box = draw.textbbox((tx, ty), label)
        draw.rectangle((box[0] - 2, box[1] - 2, box[2] + 2, box[3] + 2), fill="black")
        draw.text((tx, ty), label, fill=color)
    image.save(destination)


def aggregate_drawer(tasks):
    scored = [task for task in tasks if task["response_context"]["selection_scored"]]
    metrics = [task["metrics"] for task in scored]
    boolean_keys = sorted({key for metric in metrics for key, value in metric.items()
                           if isinstance(value, bool)})
    return {"cases": len(tasks), "status_counts": dict(Counter(task["status"] for task in tasks)),
            "response_status_counts": dict(Counter(task["response_context"]["status"] for task in tasks)),
            "scored_selection_cases": len(scored),
            "unscored_status_counts": dict(Counter(task["status"] for task in tasks
                                                   if not task["response_context"]["selection_scored"])),
            "boolean_metric_counts": {key: sum(metric.get(key) is True for metric in metrics)
                                      for key in boolean_keys},
            "api_attempts": sum(task["raw_row"].get("attempts", 0) for task in tasks),
            "estimated_usage_cost_usd": sum(task["raw_row"].get("cost_usd", 0.) for task in tasks)}


def drawer_cloud_data(rgb, cloud, numeric):
    colors = ("#ff463c", "#28eb82", "#ffcd2d", "#41b4ff", "#e155ff")
    anchors = []
    for (key, landmark), color in zip((numeric or {}).get("landmarks", {}).items(), colors):
        if landmark["status"] == "measured":
            anchors.append({"name": key.removesuffix("_uv"), "position": landmark["position_m"], "color": color})
    center = np.mean([anchor["position"] for anchor in anchors], axis=0) if anchors else np.array([0., 0., .8])
    points = cloud[::7, ::7].reshape(-1, 3)
    palette = rgb[::7, ::7].reshape(-1, 3)
    keep = np.isfinite(points).all(axis=1) & (np.linalg.norm(points - center, axis=1) < .65)
    lines = []
    for relation in (numeric or {}).get("relations", []):
        if relation["type"] == "point_on_line":
            p = np.asarray(relation["line_point_world_m"])
            axis = np.asarray(relation["axis_world"])
            lines.append({"a": p.tolist(), "b": (p + axis * .12).tolist(), "color": "#28eb82"})
    return {"points": np.round(points[keep], 6).tolist(), "colors": palette[keep].tolist(),
            "center": center.tolist(), "anchors": anchors, "lines": lines}


def _json_details(title, value):
    return "<details><summary>" + html.escape(title) + "</summary><pre>" + html.escape(
        json_text(value)) + "</pre></details>"


def _links(task):
    return " · ".join('<a href="' + html.escape(value["asset"], quote=True) + '">' +
                      html.escape(name.replace("_", " ")) + "</a>"
                      for name, value in task.get("artifacts", {}).items())


def render(summary):
    from report_rgbd_numericals import VIEWER

    esc = html.escape
    ledger = summary["ledger"]
    parts = ["<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
             "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
             "<title>Task grounding: pick/place and the middle drawer</title>",
             "<style>body{margin:0;background:#f2f5fa;color:#182638;font:16px/1.5 system-ui}"
             "main{max-width:1220px;margin:auto;padding:25px}h1{font-size:30px}h2{font-size:23px}"
             ".panel,article{background:white;border:1px solid #d4dde9;border-radius:12px;padding:20px;margin:18px 0}"
             ".note{border-left:5px solid #bd751c;background:#fff9e9;padding:14px}.grid{display:grid;"
             "grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}.grid img{width:100%;height:auto}"
             "figure{margin:0}figcaption{font-size:14px}table{border-collapse:collapse;width:100%}"
             "th,td{text-align:left;padding:9px;border-bottom:1px solid #dbe1eb;vertical-align:top}"
             ".scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6fa;"
             "padding:12px;font-size:13px}summary{cursor:pointer}a{color:#1465a4}.muted{color:#576575}"
             ".pill{display:inline-block;border-radius:20px;background:#e8eef8;padding:3px 9px;margin:2px}"
             "canvas.cloud{width:100%;max-width:700px;height:auto;border-radius:8px;touch-action:none}"
             "</style></head><body><main><h1>Task grounding: pick/place and the middle drawer</h1>",
             "<p>Ordinary pick/place has no added task-specific constraint along the path. "
             "Drawer opening adds a line constraint during the pulling stage, after grasping the handle.</p>",
             '<div class="note"><strong>Grounding experiment:</strong> A selected handle and an estimated "'
             "drawer axis do not yet define a verified robot trajectory. The TCP attachment, handle grasp, "
             "opening distance, IK and collision checks remain unresolved. The drawer plane is a geometric "
             "estimate; simulator joint values are used only for evaluation.</div>",
             '<div class="panel"><h2>Current scope and cost</h2><p>' + esc(summary["model"]) +
             "; reasoning disabled. " + str(summary["planned_drawer_cases"]) + " planned drawer cases; " +
             str(ledger["api_attempts"]) + " actual API requests. " +
             str(summary["ordinary_pick_place"]["cases"]) + " saved ordinary pick/place answers reused.</p>",
             "<p>New drawer usage estimate: <strong>$" + format(ledger["estimated_usage_cost_usd"], ".6f") +
             "</strong>. Overall authorized ceiling: <strong>$2.00</strong>. Reusing the pick/place "
             "answers makes zero additional API calls. Provider invoices determine the final charge.</p>",
             '<p><a href="summary.json">Full report data</a> · <a href="verification.json">Replay verification</a> · '
             '<a href="assets/config.json">Configuration</a> · <a href="assets/protocol.json">Frozen protocol</a> · '
             '<a href="assets/prompt.txt">Exact drawer prompt</a> · <a href="assets/manifest.json">Run manifest</a></p>',
             _json_details("Usage ledger", ledger), "</div>",
             '<div class="panel"><h2>Ordinary pick/place: no path constraint</h2><p>'
             "Four ordinary instructions per encoder are retained. Earlier upright variants are excluded "
             "from this scope. Placement goals and grasp proposals remain; object-axis estimates and "
             "upright path relations are removed from these derived specifications.</p>",
             '<div class="scroll"><table><tr><th>Encoder</th><th>Cases</th><th>Valid forms</th>'
             '<th>Correct source + target</th><th>Path relations</th></tr>']
    for condition, counts in summary["ordinary_pick_place"]["condition_summaries"].items():
        parts.append("<tr><td>" + esc(condition) + "</td><td>" + str(counts["cases"]) +
                     "</td><td>" + str(counts["valid_forms"]) + "</td><td>" +
                     str(counts["semantic_complete_goals"]) + "/" + str(counts["cases"]) +
                     "</td><td>0</td></tr>")
    parts += ["</table></div><p>Correct source + target measures object selection. It does not certify "
              "a stable grasp or successful placement.</p></div>"]
    for task in summary["ordinary_pick_place"]["tasks"]:
        parts += ['<article class="pick-place"><h3>' + esc(task["case_id"]) + "</h3><p>" +
                  esc(task["instruction"]) + '</p><span class="pill">No task-specific path constraint</span>',
                  "<p>Source + target correct: <strong>" + str(task["metrics"]["semantic_complete_goal"]) +
                  "</strong>. These are reused saved selections.</p>", '<p>' + _links(task) + '</p>',
                  _json_details("Derived specification", task["numericals"]), "</article>"]
    parts += ['<div class="panel"><h2>Middle drawer: constrained pull</h2><p>'
              "The two input conditions use either the full RGB image with direct pixel blanks, "
              "or the same RGB image plus numbered DINOv3 candidates. Both select the middle drawer, "
              "handle and three drawer-front points. Measured depth lifts those points into world "
              "coordinates. A fitted front normal supplies the candidate pulling direction.</p>",
              "<p>The line relation restricts the selected handle tracking point to move along that "
              "direction during <strong>pull</strong>. No drawer line constraint applies during approach "
              "or retreat. 'Open' supplies no numerical travel distance; that goal remains unresolved.</p>",
              '<div class="scroll"><table><tr><th>Input condition</th><th>Planned / scored</th>'
              '<th>Drawer + handle correct</th><th>Constraint template correct</th>'
              '<th>Numerical candidate correct</th><th>Saved form status</th><th>Paid attempts</th><th>Estimated cost</th></tr>']
    for condition, counts in summary["drawer_condition_summaries"].items():
        scores = counts["boolean_metric_counts"]
        denominator = counts["scored_selection_cases"]
        cells = [str(scores.get(key, 0)) + "/" + str(denominator) if denominator else "unscored"
                 for key in ("role_binding_correct", "constraint_template_correct", "numerical_candidate_correct")]
        parts.append("<tr><td>" + esc(condition) + "</td><td>" + str(counts["cases"]) + "/" +
                     str(denominator) + "</td><td>" + "</td><td>".join(cells) +
                     "</td><td>" + esc(json.dumps(counts["status_counts"])) + "</td><td>" +
                     str(counts["api_attempts"]) + "</td><td>$" +
                     format(counts["estimated_usage_cost_usd"], ".6f") + "</td></tr>")
    parts += ["</table></div><p>Valid JSON, selecting the correct drawer, estimating the correct axis, "
              "and executing a motion are separate outcomes.</p>",
              '<div class="note"><strong>Candidate proposal screening:</strong> The four DINOv3 bundles '
              "contain no candidate on the required middle-drawer handle. Their API requests were skipped "
              "after an offline development audit; they are unscored selections, not VLM abstentions. "
              "This audit uses held-back simulator labels to avoid spending on a known missing candidate. "
              "A deployed system would need an independent perception or uncertainty check for that gate. "
              "Candidate parameters and selected points were not repaired using reference labels.</div></div>"]
    for task in summary["drawer_tasks"]:
        parts += ['<article class="drawer"><h3>' + esc(task["case_id"]) + "</h3><p>" +
                  esc(task["instruction"]) + '</p><span class="pill">' + esc(task["condition"]) +
                  '</span><span class="pill">' + esc(task["status"]) + '</span><div class="grid">']
        for name, caption in (("original_rgb", "Original RGB sent to Luna"),
                              ("numbered_rgb", "Numbered RGB sent only in the candidate condition"),
                              ("selected_points", "Selected source, handle and drawer-front points"),
                              ("depth", "Measured depth used by the geometry compiler")):
            if name in task["artifacts"]:
                if task["status"] == "proposal_unavailable" and name in {"original_rgb", "numbered_rgb"}:
                    caption = "Prepared " + ("original RGB" if name == "original_rgb" else "numbered RGB") + "; no API call"
                parts.append('<figure><img src="' + esc(task["artifacts"][name]["asset"], quote=True) +
                             '" alt="' + esc(caption, quote=True) + '"><figcaption>' + esc(caption) +
                             "</figcaption></figure>")
        if task.get("cloud_data") is not None:
            scene = "scene-" + task["case_id"]
            caption = ("Estimated pull direction in green." if task["cloud_data"]["lines"] else
                       "Visible depth surfaces; no accepted pull-line estimate.")
            parts += ['</div><p>' + caption + ' Drag to rotate; scroll to zoom.</p>',
                      '<canvas class="cloud" data-scene="' + esc(scene, quote=True) + '"></canvas>',
                      '<script type="application/json" id="' + esc(scene, quote=True) + '">' +
                      json_text(task["cloud_data"]) + '</script><div>']
        parts += ["</div><p>" + _links(task) + "</p>", '<div class="scroll"><table>'
                  '<tr><th>Offline metric</th><th>Result</th></tr>']
        if not task["response_context"]["selection_scored"]:
            parts.append("<tr><td>Selection evaluation</td><td>unscored: " + esc(task["status"]) + "</td></tr>")
        for key, value in (task["metrics"].items() if task["response_context"]["selection_scored"] else []):
            if isinstance(value, (dict, list)):
                continue
            rendered = "unresolved / not evaluated" if value is None else str(value)
            parts.append("<tr><td>" + esc(key.replace("_", " ")) + "</td><td>" + esc(rendered) + "</td></tr>")
        parts += ["</table></div>", _json_details("Accepted form and normalized pixels", {
            "choices": task["raw_row"].get("choices"), "blanks": task["raw_row"].get("blanks")}),
            _json_details("Typed numerical specification", task["raw_row"].get("numericals")),
            _json_details("Response completion and text", task["response_context"]),
            _json_details("Offline candidate coverage audit", task.get("offline_proposal_coverage")),
            _json_details("Full offline metrics", task["metrics"]), "</article>"]
    parts += ['<div class="panel"><h2>Evaluation and provenance</h2><p>'
              "RGB and selected pixels are model inputs. Calibrated depth supplies geometry after "
              "selection. Simulator masks, handle geometry and joint parameters are held back for "
              "offline scoring. They never supply the model answers or repair a selected point.</p>",
              "<p>Axis error is compared with the held-back joint axis; handle distance uses the "
              "held-back handle surface. Formula checks only test the line residual and Jacobian. "
              "They do not verify semantic selection, robot attachment, IK, collision freedom or "
              "execution. A single task with several initial states does not establish generalization "
              "or an encoder ranking. These four observations share the same drawer pose and joint axis; "
              "other scene objects vary.</p>", _json_details("Provenance and limitations", summary["provenance"]),
              "</div></main><script>" + VIEWER + "</script></body></html>"]
    return "\n".join(parts)


def build_report(config_path=DEFAULT_CONFIG, trial_path=DEFAULT_TRIAL, output_path=DEFAULT_OUTPUT):
    """Assemble the report from the authoritative versioned scope trial."""
    from report_rgbd_numericals import depth_picture
    from vlm_tamp.task_grounding_scope import check_protocol, grade_drawer

    config, protocol, records = check_protocol(local_path(config_path))
    trial, output = local_path(trial_path), local_path(output_path)
    if output.resolve() in {(ROOT / "results/rgbd-dino-luna-report").resolve(),
                            (ROOT / "results/rgbd-dino-comparison-report").resolve()}:
        raise ValueError("Preserve the previous reports")
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    manifest = json.loads((trial / "manifest.json").read_text())
    rows = read_rows(trial / "predictions.jsonl")
    pick_place = json.loads((trial / "pick_place.json").read_text())
    if isinstance(pick_place, dict):
        pick_place = pick_place["tasks"]
    pick_tasks = []
    history = json.loads((ROOT / HISTORICAL_REPORT).read_text())
    old_tasks = {task["id"]: task for task in history["tasks"]}
    for row in pick_place:
        identifier = row["case_id"]
        old = old_tasks[identifier]
        artifacts = {}
        folder = ROOT / HISTORICAL_TRIAL / identifier
        for name in ("request.json", "response.json", "payload.json", "original.png", "numbered.png"):
            if (folder / name).exists():
                artifacts[name.replace(".", "_")] = copy_asset(folder / name, assets, identifier + "-" + name)
        task = {**row, "instruction": row.get("instruction", old["instruction"]),
                "metrics": row.get("metrics", old["metrics"]), "artifacts": artifacts,
                "source_trial": HISTORICAL_TRIAL, "source_summary_sha256": file_sha(ROOT / HISTORICAL_REPORT)}
        (assets / (identifier + "-derived-result.json")).write_text(json_text(task) + "\n")
        task["artifacts"]["derived_result"] = {"asset": "assets/" + identifier + "-derived-result.json"}
        pick_tasks.append(task)
    pick_groups = {}
    for condition in sorted({task["condition"] for task in pick_tasks}):
        group = [task for task in pick_tasks if task["condition"] == condition]
        pick_groups[condition] = {"cases": len(group), "valid_forms": sum(task["status"] == "valid" for task in group),
                                  "semantic_complete_goals": sum(task["metrics"]["semantic_complete_goal"]
                                                                 for task in group), "path_relations": 0}
    drawer_tasks = []
    for case in config["cases"]:
        identifier = case["id"]
        row = rows[identifier]
        folder = trial / identifier
        record = records[case["example_id"]]
        metrics = grade_drawer(record, ROOT, case, row)
        payload = json.loads((folder / "payload.json").read_text())
        artifacts = {}
        for name, label in (("original.png", "original_rgb"), ("numbered.png", "numbered_rgb"),
                            ("request.json", "request"), ("response.json", "response"),
                            ("payload.json", "payload"), ("numericals.json", "numerical_specification")):
            if (folder / name).exists():
                artifacts[label] = copy_asset(folder / name, assets, identifier + "-" + name)
        if "original_rgb" in artifacts:
            name = identifier + "-points.png"
            points_picture(folder / "original.png", row.get("blanks"), assets / name)
            artifacts["selected_points"] = {"asset": "assets/" + name}
        capture = ROOT / config.get("depth_dataset", "data/rgbd-drawer-pilot") / case["example_id"]
        if (capture / "depth_meters.npy").exists():
            name = identifier + "-depth.png"
            depth_picture(np.load(capture / "depth_meters.npy", allow_pickle=False), assets / name)
            artifacts["depth"] = {"asset": "assets/" + name}
        for source, label in ((capture / "calibration.json", "calibration"),
                              (capture / "evaluation/reference.json", "offline_reference")):
            if source.exists():
                artifacts[label] = copy_asset(source, assets, identifier + "-" + label + ".json")
        cloud_data = None
        if (capture / "world_points.npy").exists() and (folder / "original.png").exists():
            with Image.open(folder / "original.png") as image:
                rgb = np.asarray(image.convert("RGB"))
            cloud_data = drawer_cloud_data(rgb, np.load(capture / "world_points.npy", allow_pickle=False),
                                           row.get("numericals"))
        task = {**case, "case_id": identifier, "instruction": payload["instruction"], "raw_row": row,
                "status": row["status"], "metrics": metrics, "artifacts": artifacts,
                "response_context": response_context(folder, row), "cloud_data": cloud_data,
                "offline_proposal_coverage": (protocol.get("offline_proposal_coverage", {}).get(case["example_id"])
                                               if case["condition"] == "dinov3_candidates" else None)}
        (assets / (identifier + "-result.json")).write_text(json_text(task) + "\n")
        task["artifacts"]["result"] = {"asset": "assets/" + identifier + "-result.json"}
        drawer_tasks.append(task)
    for path, name in ((local_path(config_path), "config.json"),
                       (ROOT / config["frozen_protocol"], "protocol.json"),
                       (trial / "manifest.json", "manifest.json"),
                       (trial / "pick_place.json", "pick_place.json"),
                       (ROOT / HISTORICAL_REPORT, "historical-luna-summary.json")):
        copy_asset(path, assets, name)
    (assets / "prompt.txt").write_text(manifest["prompt_text"])
    summary = {"model": config["model"], "config": config, "frozen_protocol_sha256": file_sha(
        ROOT / config["frozen_protocol"]), "planned_drawer_cases": len(config["cases"]),
        "drawer_tasks": drawer_tasks,
        "drawer_condition_summaries": {condition: aggregate_drawer([task for task in drawer_tasks
                                                                   if task["condition"] == condition])
                                       for condition in sorted({case["condition"] for case in config["cases"]})},
        "ordinary_pick_place": {"cases": len(pick_tasks), "new_api_calls": 0, "tasks": pick_tasks,
                                 "condition_summaries": pick_groups,
                                 "source_summary": HISTORICAL_REPORT,
                                 "source_summary_sha256": file_sha(ROOT / HISTORICAL_REPORT)},
        "ledger": manifest["ledger"], "stopped": manifest.get("stopped"),
        "prompt_text": manifest["prompt_text"], "robot_execution_tested": False,
        "joint_space_compiled": False, "encoder_ranking_established": False,
        "provenance": {"reference_answers_in_inference": False, "provided_inventory": False,
                       "pick_place_reused_without_new_api_calls": True,
                       "drawer_joint_ground_truth_used_only_for_evaluation": True,
                       "opening_distance_unresolved": True, "tcp_attachment_unresolved": True,
                       "ik_tested": False, "grasp_stability_verified": False,
                       "robot_execution_tested": False, "browser_rendering_tested": False,
                       "development_proposal_gate_uses_offline_reference_labels": True,
                       "drawer_observations_share_pose_and_joint_axis": True,
                       "preserved_artifacts": len(protocol.get("preserved_artifacts", {}))}}
    (output / "summary.json").write_text(json_text(summary) + "\n")
    (output / "index.html").write_text(render(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--trial", default=DEFAULT_TRIAL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build_report(args.config, args.trial, args.output)
    print(json_text({key: summary[key] for key in ("model", "planned_drawer_cases", "drawer_condition_summaries", "ledger")}))


if __name__ == "__main__":
    main()
