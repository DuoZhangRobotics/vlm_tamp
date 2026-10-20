"""Inspect the versioned frontal-camera drawer retry without changing prior reports."""

import argparse
import html
import json
from contextlib import contextmanager

import report_task_grounding_scope as historical_report
from report_task_grounding_scope import local_path
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_landmark_trial import file_sha
from vlm_tamp.rgbd_numericals import json_text


DEFAULT_CONFIG = "configs/task_grounding_frontview.json"
DEFAULT_TRIAL = "results/task-grounding-frontview-luna"
DEFAULT_OUTPUT = "results/task-grounding-frontview-report"
PREVIOUS_REPORT = "results/task-grounding-scope-report/summary.json"
PREVIOUS_TRIAL = "results/task-grounding-scope-luna"


@contextmanager
def protocol_binding(frontview):
    """Bind only the new builder invocation; frozen source files stay unchanged."""
    import vlm_tamp.task_grounding_scope as scope

    previous = scope.check_protocol, scope.grade_drawer
    scope.check_protocol, scope.grade_drawer = frontview.check_protocol, frontview.grade_drawer
    try:
        yield
    finally:
        scope.check_protocol, scope.grade_drawer = previous


def camera_comparison(summary):
    previous = json.loads((ROOT / PREVIOUS_REPORT).read_text())
    old = {task["case_id"]: task for task in previous["drawer_tasks"]}
    pairs = []
    for task in summary["drawer_tasks"]:
        before = old[task["case_id"]]
        pairs.append({"case_id": task["case_id"], "condition": task["condition"],
                      "example_id": task["example_id"], "previous_status": before["status"],
                      "current_status": task["status"],
                      "previous_selection_scored": before["response_context"]["selection_scored"],
                      "current_selection_scored": task["response_context"]["selection_scored"],
                      "previous_metrics": before["metrics"], "current_metrics": task["metrics"]})
    return {"source_summary": PREVIOUS_REPORT, "source_summary_sha256": file_sha(ROOT / PREVIOUS_REPORT),
            "previous_condition_summaries": previous["drawer_condition_summaries"],
            "previous_ledger": previous["ledger"], "paired_case_outcomes": pairs,
            "changed_intervention": "Fixed frontal camera and its resulting RGB/depth/candidate observations",
            "comparison_scope": "A development camera intervention, not an encoder or model ranking",
            "earlier_outputs_preserved": True, "previous_trial": PREVIOUS_TRIAL}


def _camera_panel(summary):
    comparison = summary["camera_intervention_comparison"]
    parts = ['<div class="panel"><h2>Why the camera changed</h2><p>'
             "The earlier oblique-view trial selected the right prismatic-pull template but did not "
             "resolve a valid pull line. This retry uses an explicitly chosen frontal camera view of "
             "the cabinet. The camera intervention changes visible RGB, measured depth and landmark "
             "proposals. It does not establish a model or encoder ranking.</p>",
             '<div class="scroll"><table><tr><th>Observation</th><th>Input</th><th>Planned / scored</th>'
             '<th>Drawer + handle correct</th><th>Template correct</th><th>Numerical candidate correct</th></tr>']
    for label, groups in (("Previous oblique view", comparison["previous_condition_summaries"]),
                          ("Current frontal view", summary["drawer_condition_summaries"])):
        for condition, counts in groups.items():
            n = counts["scored_selection_cases"]
            scores = counts["boolean_metric_counts"]
            cells = [str(scores.get(key, 0)) + "/" + str(n) if n else "unscored"
                     for key in ("role_binding_correct", "constraint_template_correct", "numerical_candidate_correct")]
            parts.append("<tr><td>" + html.escape(label) + "</td><td>" + html.escape(condition) +
                         "</td><td>" + str(counts["cases"]) + "/" + str(n) + "</td><td>" +
                         "</td><td>".join(cells) + "</td></tr>")
    parts += ["</table></div><p>Unrequested candidate conditions remain unscored. A correct numerical "
              "candidate is still an abstract handle-tracking estimate; grasp attachment and opening "
              "distance remain unresolved.</p>",
              '<p><a href="assets/previous-oblique-summary.json">Preserved oblique-view summary</a> · '
              '<a href="assets/camera-comparison.json">Matched case comparison</a></p></div>']
    return "\n".join(parts)


def _proposal_panel(summary):
    candidates = [task for task in summary["drawer_tasks"] if task["condition"] == "dinov3_candidates"]
    missing = sum(not task["offline_proposal_coverage"]["required_handle_covered"] for task in candidates)
    calls = sum(task["raw_row"]["attempts"] for task in candidates)
    return ('<div class="note"><strong>Candidate proposal screening:</strong> ' + str(missing) +
            " of " + str(len(candidates)) + " frontal-view bundles lack a candidate on the required "
            "middle-drawer handle. Candidate API attempts: " + str(calls) +
            ". Missing-proposal cases are skipped and unscored; they are not VLM abstentions. "
            "An offline development audit uses held-back simulator masks to screen known missing "
            "proposals. A deployed system would need an independent perception or uncertainty check "
            "for that gate. Reference points are not added to proposals or provided to the model.</div>")


def build_report(config_path=DEFAULT_CONFIG, trial_path=DEFAULT_TRIAL, output_path=DEFAULT_OUTPUT):
    import vlm_tamp.task_grounding_frontview as frontview

    output = local_path(output_path)
    if output.resolve() == (ROOT / historical_report.DEFAULT_OUTPUT).resolve():
        raise ValueError("Preserve the previous oblique-view report")
    config, _, _ = frontview.check_protocol(local_path(config_path))
    if config.get("depth_dataset") != config["dataset"]:
        raise ValueError("Use the frontal capture dataset for the report's depth and 3D views")
    with protocol_binding(frontview):
        summary = historical_report.build_report(config_path, trial_path, output_path)
    summary["camera_intervention_comparison"] = camera_comparison(summary)
    summary["provenance"].update(camera_intervention="Explicit fixed frontal cabinet camera",
                                 camera_intervention_is_model_or_encoder_ablation=False,
                                 previous_oblique_trial_preserved=True)
    (output / "summary.json").write_text(json_text(summary) + "\n")
    historical_report.copy_asset(ROOT / PREVIOUS_REPORT, output / "assets", "previous-oblique-summary.json")
    (output / "assets/camera-comparison.json").write_text(json_text(summary["camera_intervention_comparison"]) + "\n")
    page = historical_report.render(summary)
    start = '<div class="note"><strong>Candidate proposal screening:</strong>'
    opening = page.index(start)
    ending = page.index("</div>", opening) + len("</div>")
    page = page[:opening] + _proposal_panel(summary) + page[ending:]
    page = page.replace("<title>Task grounding: pick/place and the middle drawer</title>",
                        "<title>Frontal-view drawer grounding</title>")
    page = page.replace("<h1>Task grounding: pick/place and the middle drawer</h1>",
                        "<h1>Frontal-view drawer grounding</h1>")
    marker = '<div class="panel"><h2>Ordinary pick/place: no path constraint</h2>'
    page = page.replace(marker, _camera_panel(summary) + "\n" + marker)
    (output / "index.html").write_text(page)
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
