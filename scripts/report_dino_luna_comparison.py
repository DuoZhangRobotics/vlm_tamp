"""Report the Luna landmark trial without altering the frozen Mini report."""

import argparse
import html
import json
from collections import Counter
from contextlib import contextmanager

import report_dino_landmark_comparison as historical_report
from vlm_tamp.libero import ROOT
from vlm_tamp.rgbd_landmark_trial import file_sha
from vlm_tamp.rgbd_numericals import json_text


DEFAULT_CONFIG = "configs/rgbd_dino_luna_comparison.json"
DEFAULT_TRIAL = "results/rgbd-dino-luna-comparison"
DEFAULT_OUTPUT = "results/rgbd-dino-luna-report"
MINI_REPORT = ROOT / "results/rgbd-dino-comparison-report/summary.json"


def read_response_context(trial, identifier, row):
    context = historical_report.read_response_context(trial, identifier, row)
    if row["status"] == "api_failure" and context["status"] == "not_run":
        context.update(status="api_failure", raw_choice_context="no_saved_response",
                       unscored_selection_reason="api_failure")
    return context


@contextmanager
def protocol_binding(callback):
    """Use a new protocol validator while preserving the frozen implementation."""
    previous = historical_report.check_protocol
    previous_context = historical_report.read_response_context
    historical_report.check_protocol = callback
    historical_report.read_response_context = lambda trial, identifier, row: _context_with_failures(
        previous_context(trial, identifier, row), row)
    try:
        yield
    finally:
        historical_report.check_protocol = previous
        historical_report.read_response_context = previous_context


def _context_with_failures(context, row):
    if row["status"] == "api_failure" and context["status"] == "not_run":
        context.update(status="api_failure", raw_choice_context="no_saved_response",
                       unscored_selection_reason="api_failure")
    return context


def comparison_data(summary, history, source=MINI_REPORT):
    current, previous = summary["config"], history["config"]
    shared_keys = ("dataset", "depth_dataset", "candidate_datasets", "conditions", "cases",
                   "prompt", "segmentation", "proposal")
    for key in shared_keys:
        if current[key] != previous[key]:
            raise ValueError("Historical Mini inputs differ: " + key)
    if summary["prompt_text"] != history["prompt_text"]:
        raise ValueError("Historical Mini prompt differs")
    for key in summary["candidate_metadata"]:
        if summary["candidate_metadata"][key] != history["candidate_metadata"][key]:
            raise ValueError("Historical Mini candidates differ: " + key)
    old_tasks = {task["id"]: task for task in history["tasks"]}
    paired = []
    for task in summary["tasks"]:
        old = old_tasks[task["id"]]
        paired.append({
            "id": task["id"], "condition": task["condition"], "instruction": task["instruction"],
            "luna_response_status": task["response_context"]["status"],
            "mini_response_status": old["response_context"]["status"],
            "luna_form_status": task["status"], "mini_form_status": old["status"],
            "luna_goal_correct": task["metrics"]["semantic_complete_goal"],
            "mini_goal_correct": old["metrics"]["semantic_complete_goal"],
            "luna_selection_scored": task["response_context"]["selection_diagnostic_applicable"],
            "mini_selection_scored": old["response_context"]["selection_diagnostic_applicable"],
            "luna_cost_usd": task["raw_row"].get("cost_usd", 0.),
            "mini_cost_usd": old["raw_row"].get("cost_usd", 0.),
        })
    changed = {key: {"mini": previous.get(key), "luna": current.get(key)} for key in
               ("model", "reasoning_effort", "max_output_tokens", "max_budget_usd", "pricing")
               if previous.get(key) != current.get(key)}
    return {
        "source": str(source), "source_sha256": file_sha(source), "model": history["model"],
        "planned_cases": history["planned_cases"], "ledger": history["ledger"],
        "condition_summaries": history["condition_summaries"],
        "incomplete_comparison": history["incomplete_comparison"], "stopped": history["stopped"],
        "matched_input_fields": list(shared_keys), "candidate_metadata_matched": True,
        "paired_case_statuses": paired, "changed_trial_controls": changed,
        "model_ranking_established": False,
        "comparison_limitation": "Mini is incomplete; model, reasoning, output limit and budget differ. "
                                 "This trial compares practical configurations, not isolated model quality.",
    }


def _table(summary, comparison):
    esc = html.escape
    rows = ["<div class=\"panel\"><h2>Luna and the previous Mini trial</h2>",
            "<p>The same fixed candidate bundles, prompt, compiler and grader are used. "
            "Every configured case remains in its denominator. Mini had incomplete and unrequested cases; "
            "Luna also changes reasoning effort and output limit. These results do not establish a model ranking.</p>",
            "<div class=\"scroll\"><table><tr><th>Configuration</th><th>Encoder</th>"
            "<th>Valid / planned</th><th>Complete goals / planned</th><th>Correct anchors</th>"
            "<th>Paid attempts</th><th>Estimated cost</th></tr>"]
    for label, groups in (("Mini, medium reasoning (historical)", comparison["condition_summaries"]),
                          ("Luna, no reasoning", summary["condition_summaries"])):
        for encoder, counts in groups.items():
            rows.append(f"<tr><td>{esc(label)}</td><td>{esc(encoder)}</td>"
                        f"<td>{counts['valid_forms']}/{counts['cases']}</td>"
                        f"<td>{counts['semantic_complete_goals']}/{counts['cases']}</td>"
                        f"<td>{counts['correct_anchors']}/{counts['anchor_denominator']}</td>"
                        f"<td>{counts['api_attempts']}</td>"
                        f"<td>${counts['estimated_usage_cost_usd']:.5f}</td></tr>")
    rows += ["</table></div><p>Anchor and goal totals include operational failures in the planned denominator. "
             "The case table distinguishes unscored selections from complete accepted forms.</p>",
             "<details><summary>Changed trial controls</summary><pre>" +
             esc(json.dumps(comparison["changed_trial_controls"], indent=2)) + "</pre></details>",
             "<div class=\"scroll\"><table><tr><th>Case</th><th>Mini response / form</th>"
             "<th>Luna response / form</th><th>Mini goal</th><th>Luna goal</th></tr>"]
    for pair in comparison["paired_case_statuses"]:
        old = str(pair["mini_goal_correct"]) if pair["mini_selection_scored"] else "unscored"
        new = str(pair["luna_goal_correct"]) if pair["luna_selection_scored"] else "unscored"
        rows.append(f"<tr><td>{esc(pair['id'])}</td>"
                    f"<td>{esc(pair['mini_response_status'])} / {esc(pair['mini_form_status'])}</td>"
                    f"<td>{esc(pair['luna_response_status'])} / {esc(pair['luna_form_status'])}</td>"
                    f"<td>{old}</td><td>{new}</td></tr>")
    rows += ["</table></div><p><a href=\"assets/historical-mini-summary.json\">Historical Mini summary</a> · "
             "<a href=\"assets/mini-luna-comparison.json\">Matched case comparison</a></p></div>"]
    return "\n".join(rows)


def build_report(config_path=DEFAULT_CONFIG, trial_path=DEFAULT_TRIAL, output_path=DEFAULT_OUTPUT):
    from vlm_tamp.rgbd_luna_landmark_trial import check_protocol

    output = historical_report.local_path(output_path)
    if output.resolve() == (ROOT / historical_report.DEFAULT_OUTPUT).resolve():
        raise ValueError("Preserve the historical Mini report")
    with protocol_binding(check_protocol):
        summary = historical_report.build_report(config_path, trial_path, output_path)
    history = json.loads(MINI_REPORT.read_text())
    comparison = comparison_data(summary, history)
    summary["encoder_pair_controls"] = [value.replace("same Mini model", "same Luna model and no reasoning")
                                         for value in summary["encoder_pair_controls"]]
    summary["comparison_scope"] = "four development scenes; six cases per encoder with Luna; no population ranking"
    summary["historical_mini_comparison"] = comparison
    summary["reasoning_effort"] = summary["config"]["reasoning_effort"]
    summary["incomplete_comparison"] = any(task["response_context"]["status"] != "completed"
                                          or task["status"] != "valid" for task in summary["tasks"])
    summary["model_ranking_established"] = False
    trial = historical_report.local_path(trial_path)
    actual = Counter(json.loads(path.read_text()).get("model") for path in trial.glob("*/response.json"))
    summary["actual_response_models"] = dict(actual)
    (output / "summary.json").write_text(json_text(summary) + "\n")
    (output / "assets/historical-mini-summary.json").write_bytes(MINI_REPORT.read_bytes())
    (output / "assets/mini-luna-comparison.json").write_text(json_text(comparison) + "\n")
    page = (output / "index.html").read_text()
    replacements = {
        "<title>DINO landmark comparison</title>": "<title>Luna: DINO landmark comparison</title>",
        "<h1>Numbered landmarks with depth: DINOv2 versus DINOv3</h1>":
            "<h1>Luna selects numbered depth landmarks: DINOv2 versus DINOv3</h1>",
        "Mini fills landmark-ID and method blanks.": "GPT-6 Luna fills landmark-ID and method blanks with reasoning disabled.",
        "Proposal coverage before Mini selection": "Proposal coverage before Luna selection",
        "Mini supplied null or no usable form": "Luna supplied null or no usable form",
        "Original RGB sent to Mini": "Original RGB sent to Luna",
        "Numbered RGB sent to Mini": "Numbered RGB sent to Luna",
        "Exact numbered candidate image sent to Mini": "Exact numbered candidate image sent to Luna",
    }
    for old, new in replacements.items():
        if old not in page:
            raise ValueError("Historical renderer changed: " + old)
        page = page.replace(old, new)
    old_panel = ('<div class="panel"><strong>Matched inputs, incomplete comparison:</strong> DINOv2 and DINOv3 '
                 'use the same four scenes, six instructions each, masks, proposal settings, prompt, Mini snapshot, '
                 'and grader. All planned cases count, including missing, not-run and invalid replies. Output-token '
                 'truncation and budget admission prevented complete answers for every case. The scores include '
                 'those operational failures; they do not establish an encoder winner or population ranking.</div>')
    if old_panel not in page:
        raise ValueError("Historical renderer comparison panel changed")
    state = "incomplete" if summary["incomplete_comparison"] else "all planned replies complete and valid"
    new_panel = ('<div class="panel"><strong>Matched Luna inputs:</strong> DINOv2 and DINOv3 use the same four '
                 'scenes, six instructions each, masks, proposal settings, prompt, GPT-6 Luna configuration, '
                 'and grader. All 12 planned cases count. Current completion: ' + state +
                 '. These development scenes do not establish an encoder winner or population ranking.</div>')
    page = page.replace(old_panel, new_panel + "\n" + _table(summary, comparison))
    old_caveat = ('Comparing it with these partial runs changes the proposer, numbered-image input, answer format, '
                  'and inference workload simultaneously; that comparison does not isolate the feature backbone. '
                  'DINOv2 versus DINOv3 has matched inputs and settings, but unequal answer completion and '
                  'unrequested cases prevent an encoder-quality conclusion.')
    if old_caveat not in page:
        raise ValueError("Historical renderer direct-XY caveat changed")
    page = page.replace(old_caveat, 'Comparing it with this Luna run changes the proposer, numbered-image input, '
                        'answer format, model, reasoning and inference workload simultaneously; that comparison '
                        'does not isolate the feature backbone. DINOv2 versus DINOv3 uses matched Luna settings '
                        'on four development scenes; this small trial does not establish a population ranking.')
    (output / "index.html").write_text(page)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--trial", default=DEFAULT_TRIAL)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build_report(args.config, args.trial, args.output)
    print(json_text({key: summary[key] for key in ("model", "cases", "condition_summaries", "ledger")}))


if __name__ == "__main__":
    main()
