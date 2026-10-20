"""Report actual LIBERO predicates, keeping development outside evaluation."""

import html
import json
import os
from collections import Counter
from pathlib import Path


FAILURE_STAGES = ("perception", "program", "grasp", "tracking", "planning", "execution", "api", "budget")


def _episode(path):
    path = Path(path)
    file = path / "episode.json" if path.is_dir() or path.suffix != ".json" else path
    if not file.exists():
        return {"episode_id": file.parent.name, "success": False, "status": "not_run",
                "failure_stage": "not_run", "detail": "No completed episode artifact", "artifact_dir": str(file.parent)}
    row = json.loads(file.read_text())
    if not isinstance(row, dict) or type(row.get("success")) is not bool:
        raise ValueError("Every episode must record its actual boolean LIBERO success predicate")
    row = dict(row)
    row.setdefault("episode_id", file.parent.name)
    row["artifact_dir"] = str(file.parent)
    row.setdefault("status", "success" if row["success"] else "failed")
    row.setdefault("failure_stage", None if row["success"] else "execution")
    return row


def _link(path, output, label):
    path = Path(path)
    if not path.exists():
        return ""
    relative = os.path.relpath(path.resolve(), output.resolve()).replace(os.sep, "/")
    from urllib.parse import quote
    return '<a href="' + html.escape(quote(relative, safe="/")) + '">' + html.escape(label) + "</a>"


def _assets(row, output):
    folder = Path(row["artifact_dir"])
    links = [_link(folder / "episode.json", output, "Episode record")]
    explicit = row.get("video") or row.get("video_path")
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = folder / path
        links.append(_link(path, output, "Video"))
    else:
        for path in sorted(folder.glob("*.mp4")):
            links.append(_link(path, output, "Video"))
    for path, label in ((folder / "program/output_raw.txt", "Generated functions"),
                        (folder / "program/response.json", "API response"),
                        (folder / "perception/numbered.png", "Numbered keypoints"),
                        (folder / "perception/keypoints.npz", "Keypoint data"),
                        (folder / "initial/actions.json", "Executed actions"),
                        (folder / "trajectory.json", "Trajectory"),
                        (folder / "initial/metadata.json", "Observation metadata")):
        links.append(_link(path, output, label))
    for path in sorted(folder.glob("stage*-grasps/grasp_overlay.png")):
        links.append(_link(path, output, "Grasp overlay " + path.parent.name))
    for path in sorted(folder.glob("stage*-grasps/grasp_candidates.npz")):
        links.append(_link(path, output, "Candidates " + path.parent.name))
    for path in sorted(folder.glob("stage*-grasps/grasp_selection.json")):
        links.append(_link(path, output, "Grasp selection " + path.parent.name))
    for path in sorted(folder.glob("stage*-plan.json")):
        links.append(_link(path, output, path.stem))
    if not explicit:
        links.append(_link(folder / "initial/rollout.mp4", output, "Video"))
    return " · ".join(link for link in links if link)


def _thumbnails(row, output):
    from urllib.parse import quote
    folder = Path(row["artifact_dir"])
    images = [folder / "perception/numbered.png"] + sorted(folder.glob("stage*-grasps/grasp_overlay.png"))
    result = []
    for path in images:
        if path.exists():
            relative = os.path.relpath(path.resolve(), output.resolve()).replace(os.sep, "/")
            uri = html.escape(quote(relative, safe="/"))
            caption = "Numbered RGB keypoints" if path.name == "numbered.png" else path.parent.name + " grasp candidates"
            result.append('<figure><a href="' + uri + '"><img loading="lazy" src="' + uri
                          + '" alt="' + html.escape(caption) + '"></a><figcaption>' + html.escape(caption)
                          + "</figcaption></figure>")
    return "".join(result)


def build_report(episode_dirs, output_dir, expected_episodes=10, *, ledger_path=None,
                 expected_episode_paths=None, development_dirs=None):
    """Use explicit planned paths when supplied; missing runs count as failures."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    observed = [_episode(path) for path in episode_dirs]
    dev = [_episode(path) for path in (development_dirs or [])]
    dev.extend(row for row in observed if row["episode_id"].startswith("dev-") or row.get("development") is True)
    observed = [row for row in observed if row["episode_id"] not in {item["episode_id"] for item in dev}]
    if isinstance(expected_episodes, (list, tuple)):
        expected_episode_paths = expected_episodes
        expected_episodes = len(expected_episode_paths)
    if type(expected_episodes) is not int or expected_episodes < 1:
        raise ValueError("The evaluation denominator must be the positive planned episode count")
    if expected_episode_paths is not None:
        if len(expected_episode_paths) != expected_episodes:
            raise ValueError("Explicit planned episodes and denominator disagree")
        evaluation = [_episode(path) for path in expected_episode_paths]
    else:
        evaluation = observed
        if len(evaluation) > expected_episodes:
            raise ValueError("More evaluation records than planned episodes")
        evaluation.extend({"episode_id": f"missing-planned-{index + 1}", "success": False,
                           "status": "not_run", "failure_stage": "not_run", "artifact_dir": "",
                           "detail": "Planned episode has no record"}
                          for index in range(len(evaluation), expected_episodes))
    if len({row["episode_id"] for row in evaluation}) != expected_episodes:
        raise ValueError("Evaluation episode IDs must be unique")
    successes = sum(row["success"] for row in evaluation)
    failures = Counter(row.get("failure_stage") or "execution" for row in evaluation if not row["success"])
    by_task = {}
    for row in evaluation:
        task = row.get("task_type", "unknown")
        stats = by_task.setdefault(task, {"planned": 0, "successes": 0})
        stats["planned"] += 1
        stats["successes"] += int(row["success"])
    ledger = json.loads(Path(ledger_path).read_text()) if ledger_path and Path(ledger_path).exists() else None
    summary = {"experiment": "ReKep + GraspGen on LIBERO", "claim": "documented RGB-D adaptation",
               "success_criterion": "actual LIBERO success predicate", "evaluation_planned": expected_episodes,
               "evaluation_successes": successes, "evaluation_success_rate": successes / expected_episodes,
               "evaluation_physical_grasp_attempts": sum(int(row.get("physical_grasp_attempts", 0)) for row in evaluation),
               "development_count": len(dev), "failures": dict(failures), "by_task": by_task,
               "evaluation": evaluation, "development": dev, "api_ledger": ledger,
               "provided_inventory": False, "simulator_reference_data_used_in_inference": False}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    def rows(values):
        result = []
        for row in values:
            result.append("<tr><td>" + html.escape(str(row["episode_id"])) + "</td><td>"
                          + html.escape(str(row.get("task_type", ""))) + "</td><td>"
                          + ("Success" if row["success"] else html.escape(str(row.get("status", "failed"))))
                          + "</td><td>" + html.escape(str(row.get("failure_stage") or "")) + "</td><td>"
                          + html.escape(str(row.get("detail", row.get("failure_detail", "")))) + "</td><td>"
                          + html.escape(str(row.get("control_steps", 0))) + "</td><td>"
                          + str(row.get("physical_grasp_attempts", 0)) + "</td><td>"
                          + (_assets(row, output) if row["artifact_dir"] else "") + "</td></tr>")
            if row["artifact_dir"]:
                thumbnails = _thumbnails(row, output)
                if thumbnails:
                    result.append('<tr><td colspan="8"><div class="images">' + thumbnails + "</div></td></tr>")
        return "\n".join(result)

    budget = "API ledger has not been created; no paid request is reported."
    if ledger:
        budget = (f"Cumulative conservative API spend: ${ledger.get('cumulative_conservative_spend_usd', 0):.6f} / "
                  f"${ledger.get('budget_usd', 2):.2f}; prior spend ${ledger.get('prior_conservative_spend_usd', 0):.6f}. "
                  f"Pending or uncertain calls retain their reservation. Attempts: {ledger.get('api_attempts', 0)}.")
    columns = "<tr><th>Episode</th><th>Task</th><th>Result</th><th>Failure stage</th><th>Detail</th><th>Steps</th><th>Physical grasps</th><th>Artifacts</th></tr>"
    failure_text = ", ".join(f"{key}: {value}" for key, value in sorted(failures.items())) or "None"
    document = ("<!doctype html><html lang=en><meta charset=utf-8><title>ReKep + GraspGen on LIBERO</title>"
                "<style>body{font:16px system-ui;max-width:1500px;margin:36px auto;padding:0 20px;color:#19212a}"
                "table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border:1px solid #dce2e7;"
                "text-align:left;vertical-align:top}th{background:#eef2f5}a{color:#0969a2}p{line-height:1.5}"
                ".images{display:flex;gap:14px;flex-wrap:wrap}figure{margin:0}figure img{width:280px;height:auto;"
                "border:1px solid #dce2e7}figcaption{font-size:12px;color:#596773}</style>"
                "<h1>ReKep + GraspGen on LIBERO</h1><p>Documented adaptation using RGB-D perception, "
                "generated numerical stage functions, detected grasps, ReKep optimization and physical Panda execution.</p>"
                f"<p><strong>Official task success: {successes}/{expected_episodes} "
                f"({100 * successes / expected_episodes:.1f}%).</strong> Every planned evaluation episode is counted. "
                "Development runs are listed separately. Valid programs or optimizer solutions do not constitute task success.</p>"
                "<p>" + html.escape(budget) + "</p><p>Failure counts: " + html.escape(failure_text) + "</p>"
                "<h2>Evaluation</h2><table>" + columns + rows(evaluation) + "</table>"
                "<h2>Development</h2><table>" + columns + rows(dev) + "</table>"
                '<p><a href="summary.json">Machine-readable results</a></p></html>')
    (output / "index.html").write_text(document)
    return summary
