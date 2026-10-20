"""Report completed bowl-development attempts without pooling benchmark scores.

Reads saved observations and result records only. No simulator, model inference
or API calls are made. Regeneration updates only the explicitly chosen report.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]

OUTCOMES = {
    "approach_paths_rejected": "Approach blocked before grasping",
    "no_feasible_grasp": "No feasible grasp",
    "filter_only": "Grasp filtering only",
    "closed_gripper": "Gripper closed; lift unverified",
    "lift_path_rejected": "Gripper closed; lift path blocked",
    "grasp_not_retained": "Lift did not establish retention",
    "grasp_lift_success": "Lift and hold verified",
    "placement_plan_rejected": "Lift verified; placement plan blocked",
    "placement_path_rejected": "Lift verified; placement path blocked",
    "released_without_task_success": "Released without task completion",
    "task_success": "Bowl placed successfully",
    "error": "Attempt stopped with an error",
}

VARIANTS = {
    "raw": "Raw model pose",
    "antipodal_half_turn": "Panda symmetry half-turn",
    "asset_contact_aligned": "Measured finger alignment (+6 mm)",
    "asset_contact_aligned_half_turn": "Finger alignment + symmetry half-turn",
}


def read(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def href(path, report_dir):
    relative = os.path.relpath(Path(path).resolve(), report_dir.resolve()).replace(os.sep, "/")
    return html.escape(quote(relative, safe="/"), quote=True)


def link(path, label, report_dir):
    if not Path(path).is_file():
        return ""
    return f'<a href="{href(path, report_dir)}">{html.escape(str(label))}</a>'


def outcome_text(row):
    if "Actual robot posture failed" in str(row["failure"]):
        return "Actual arm posture failed clearance"
    if "Physical controller did not reach" in str(row["failure"]):
        return "Controller missed a checked waypoint"
    return OUTCOMES.get(row["status"], row["status"].replace("_", " "))


def collect_attempt(folder):
    summary = read(folder / "summary.json")
    if not summary or summary.get("status") == "running":
        return None
    for key in ("official_task_success", "grasp_lift_success"):
        if type(summary.get(key)) is not bool:
            raise ValueError(f"{folder.name}: missing actual boolean {key}")
    if (summary.get("task_index"), summary.get("initial_state"), summary.get("seed")) != (8, 1, 0):
        raise ValueError("This report is scoped to bowl task 8, initial state 1, seed 0")
    selection = read(folder / "grasp-selection.json")
    calibration = read(folder / "calibration.json")
    metadata = read(folder / "opened/metadata.json") or read(folder / "initial/metadata.json")
    fields = read(folder / "collision-fields.json")
    material = fields.get("exact_current_robot_visual_material") or {}
    density = read(folder / "tool-density.json")
    chosen = read(folder / "chosen-grasp.json")
    retention = read(folder / "lift-retention.json")
    evidence = read(folder / "cached-evidence.json")
    execution_checks = read(folder / "approach-execution-clearance.json")
    if not isinstance(execution_checks, list):
        execution_checks = []
    grasp_functions = read(folder / "cached-grasp-function-residuals.json")
    command_path = read(folder / "approach-command-poses.json")
    actions = read(folder / "initial/actions.json")
    if not isinstance(actions, list):
        actions = []
    motion_commands = sum(bool(item.get("control")) and item["control"][-1] == 0 for item in actions)
    closing_commands = sum(bool(item.get("control")) and item["control"][-1] > 0 for item in actions)
    # This development runner attempts at most one grasp. Later +1 commands
    # hold the same grasp, so count presence rather than closing-command groups.
    recorded_grasp_attempts = int(closing_commands > 0)
    if summary.get("physical_grasp_attempts") != recorded_grasp_attempts:
        raise ValueError(f"{folder.name}: physical grasp count disagrees with recorded closing controls")
    success_events = [item.get("evaluation_task_success") for item in actions]
    if any(type(value) is not bool for value in success_events):
        raise ValueError(f"{folder.name}: action records require actual boolean LIBERO predicates")
    recorded_success = any(success_events)
    if summary["official_task_success"] != recorded_success:
        raise ValueError(f"{folder.name}: official success disagrees with recorded predicate events")
    last_action = {} if not actions else {key: actions[-1].get(key) for key in
        ("step", "position_error_m", "orientation_error_rad", "posture_error_rad",
         "target_pose_world_xyzw", "eef_pose_world_xyzw", "completion_tolerances",
         "reached_criterion", "nullspace_kp", "controller_target_reached")}
    controller_settings = dict(summary.get("controller_configuration") or {})
    if last_action.get("completion_tolerances"):
        controller_settings["completion_tolerances"] = last_action["completion_tolerances"]
    if last_action.get("nullspace_kp") is not None:
        controller_settings["nullspace_kp"] = last_action["nullspace_kp"]
    poses = {(item["index"], item["variant"]) for item in selection.get("feasible_candidates", [])}
    approaches = []
    source_hashes = {"summary.json": fingerprint(folder / "summary.json")}
    for file in sorted(folder.glob("approach-*.json")):
        record = read(file)
        match = re.fullmatch(r"approach-(\d+)-(.+)\.json", file.name)
        if not match:
            continue  # Execution/state checks are separate from candidate paths.
        arm = [item["collision"] for item in record.get("solution_rejections", [])
               if item.get("reason") == "arm_collision_or_unknown" and "collision" in item]
        best = min(arm, key=lambda item: (item["blocked_points"], item["maximum_violation_m"])) if arm else None
        approaches.append({"file": str(file.resolve()), "index": int(match[1]), "variant": match[2],
                           "accepted": bool(record.get("accepted")), "failure": record.get("failure"),
                           "sample_index": record.get("sample_index"),
                           "validated_samples": len(record.get("checks", [])),
                           "best_arm_collision": best, "tool_collision": record.get("collision")})
        source_hashes[file.name] = fingerprint(file)
    for name in ("grasp-selection.json", "calibration.json", "collision-fields.json",
                 "chosen-grasp.json", "lift-retention.json", "cached-evidence.json", "initial/actions.json",
                 "approach-execution-clearance.json", "protocol.json",
                 "cached-grasp-function-residuals.json", "approach-command-poses.json"):
        if (folder / name).is_file():
            source_hashes[name] = fingerprint(folder / name)
    video = Path(summary.get("video") or folder / "initial/rollout.mp4")
    if not video.is_absolute():
        video = folder / video
    return {
        "attempt": folder.name, "artifact_dir": str(folder.resolve()), "status": summary["status"],
        "failure": summary.get("failure"), "task_index": summary.get("task_index"),
        "initial_state": summary.get("initial_state"), "seed": summary.get("seed"),
        "control_steps": summary.get("control_steps", 0),
        "motion_hold_commands": motion_commands, "closing_commands": closing_commands,
        "last_action_tracking": last_action, "controller_configuration": controller_settings or None,
        "executed_posture_checks": execution_checks,
        "executed_posture_failed_checks": sum(item.get("accepted") is False for item in execution_checks),
        "cached_grasp_function_diagnostics": grasp_functions or None,
        "command_pose_path_mode": command_path.get("mode"),
        "physical_grasp_attempts": recorded_grasp_attempts,
        "lift_verified": summary["grasp_lift_success"], "official_task_success": recorded_success,
        "official_success_control_events": sum(success_events), "outcomes_verified_against_actions": bool(actions),
        "new_api_calls": summary.get("new_api_calls"), "feasible_ik_solutions": summary.get("feasible_solutions", 0),
        "unique_feasible_pose_variants": len(poses), "pose_variants_checked": selection.get("attempted_pose_variants"),
        "rejection_counts": summary.get("rejection_counts", {}), "approaches": approaches,
        "selected_approach_pose": next(({"index": item["index"], "variant": item["variant"]}
                                        for item in approaches if item["accepted"]), None),
        "chosen_pose": None if not chosen else {key: chosen.get(key) for key in
            ("index", "variant", "solution_index", "tcp_pose_world_xyzw", "keypoint_distance_m", "score")},
        "retention_evidence": retention or None, "cached_grounding_evidence": evidence,
        "development_configuration": {
            "cameras": list(metadata.get("cameras", {})), "dense_tool_samples": density.get("surface_samples"),
            "contact_face_samples": density.get("contact_face_samples"),
            "tool_sampling_limit": density.get("coverage_limit"),
            "candidate_subset": summary.get("development_candidate_subset"),
            "collision_margin_m": fields.get("collision_margin_m"),
            "coarse_spacing_m": fields.get("coarse_spacing_m"), "fine_spacing_m": fields.get("fine_spacing_m"),
            "closed_material_components": material.get("accepted_topologically_closed_components",
                                                       material.get("watertight_components")),
            "cad_seam_components": material.get("cad_seam_components", 0),
            "cad_seam_pairs": material.get("cad_self_intersection_pairs", 0),
            "ignored_material_components": material.get("ignored_components"),
            "material_policy": material.get("query"),
            "measured_finger_alignment_m": selection.get("asset_contact_alignment_m"),
            "configured_joint_friction_loss_nm": calibration.get("arm_joint_friction_loss_nm"),
            "base_to_tcp_translation_m": None if not calibration.get("graspgen_base_to_tcp") else
                [item[3] for item in calibration["graspgen_base_to_tcp"][:3]],
        },
        "video": str(video.resolve()) if video.is_file() else None,
        "elapsed_seconds": summary.get("elapsed_seconds"), "source_files_sha256": source_hashes,
    }


def render_attempt(row, output):
    folder = Path(row["artifact_dir"])
    conf = row["development_configuration"]
    yes = lambda value: "Yes" if value else "No"
    stats = (("Gripper closing commands", row["closing_commands"]),
             ("Lift verified", yes(row["lift_verified"])),
             ("Task finished", yes(row["official_task_success"])),
             ("Executed postures checked", len(row["executed_posture_checks"])))
    cards = "".join(f"<div><strong>{html.escape(str(value))}</strong><span>{label}</span></div>" for label, value in stats)
    media = []
    for file, caption in ((folder / "target-region/agentview-overlay.png", "RGB-selected bowl region"),
                          (folder / "target-region/frontview-overlay.png", "Predicted region from another view")):
        if file.is_file():
            media.append(f'<figure><a href="{href(file, output)}"><img loading="lazy" src="{href(file, output)}" alt="{caption}"></a><figcaption>{caption}</figcaption></figure>')
    if not media and (folder / "opened/agentview.png").is_file():
        file = folder / "opened/agentview.png"
        media.append(f'<figure><img loading="lazy" src="{href(file, output)}" alt="Initial calibrated RGB view"><figcaption>Initial calibrated RGB view</figcaption></figure>')
    if row["video"]:
        media.append(f'<figure><video controls preload="none" src="{href(row["video"], output)}"></video><figcaption>Actual recorded simulation; {row["control_steps"]} control steps</figcaption></figure>')
    table_rows = []
    for item in row["approaches"]:
        best = item["best_arm_collision"] or item["tool_collision"] or {}
        result = "Passed" if item["accepted"] else (item["failure"] or "Rejected").replace("_", " ")
        table_rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in
            (item["index"], VARIANTS.get(item["variant"], item["variant"]), result,
             "—" if item["sample_index"] is None else item["sample_index"],
             best.get("blocked_unknown_points", "—"), best.get("blocked_occupied_points", "—"))) + "</tr>")
    links = [link(folder / name, label, output) for name, label in
             (("summary.json", "Attempt record"), ("grasp-selection.json", "Every grasp rejection"),
              ("collision-fields.json", "Saved collision policy"), ("calibration.json", "Robot calibration"),
              ("lift-retention.json", "Lift evidence"), ("chosen-grasp.json", "Chosen grasp"),
              ("rekep-placement-plan.json", "ReKep placement plan"),
              ("approach-execution-clearance.json", "Actual posture clearance checks"),
              ("approach-command-poses.json", "Commanded FK pose path"),
              ("cached-grasp-function-residuals.json", "Cached grasp-function diagnostics"),
              ("protocol.json", "Saved protocol and source hashes"),
              ("source-snapshot/scripts/try_rekep_bowl_v3.py", "Saved runner for this attempt"))]
    links.extend(link(Path(item["file"]), f'Approach {item["index"]}: {item["variant"]}', output)
                 for item in row["approaches"])
    chosen = row["chosen_pose"]
    selection = "No approach path passed validation."
    if chosen:
        selection = f'Chosen grasp: {chosen["index"]}, {VARIANTS.get(chosen["variant"], chosen["variant"])}.'
    elif row["selected_approach_pose"]:
        candidate = row["selected_approach_pose"]
        selection = (f'Approach selected candidate {candidate["index"]}, '
                     f'{VARIANTS.get(candidate["variant"], candidate["variant"])}. '
                     'No gripper closure was recorded.')
    failure = "" if not row["failure"] else f'<p class="detail">Recorded stop: {html.escape(str(row["failure"]))}</p>'
    motion = (f'<p>Executed {row["control_steps"]} control commands, including '
              f'{row["motion_hold_commands"]} arm motion/hold commands and {row["closing_commands"]} closing commands.</p>')
    if row["status"] == "error" and row["last_action_tracking"].get("position_error_m") is not None:
        motion += (f'<p>At the stop, position tracking error was '
                   f'{row["last_action_tracking"]["position_error_m"] * 1000:.1f} mm and orientation error '
                   f'{row["last_action_tracking"]["orientation_error_rad"]:.3f} rad.</p>')
        if row["last_action_tracking"].get("posture_error_rad") is not None:
            motion += (f'<p>Maximum joint-posture error at the stop: '
                       f'{row["last_action_tracking"]["posture_error_rad"]:.4f} rad.</p>')
    controller = row["controller_configuration"]
    if controller:
        settings = []
        for key, label in (("waypoint_steps", "Waypoint step budget"), ("servo_scale", "Correction scale"),
                           ("nullspace_kp", "Joint-posture gain"), ("closure_steps", "Closure step budget")):
            if key in controller:
                settings.append(f'{label}: {controller[key]}')
        limits = controller.get("completion_tolerances", {})
        for key, label, scale, unit in (("position_tolerance", "Position tolerance", 1000, "mm"),
                                       ("orientation_tolerance", "Orientation tolerance", 1, "rad"),
                                       ("posture_tolerance", "Joint-posture tolerance", 1, "rad")):
            if key in limits:
                settings.append(f'{label}: {limits[key] * scale:g} {unit}')
        motion += '<p class="configuration">Recorded controller settings: ' + html.escape("; ".join(settings)) + '.</p>'
    actual_checks = row["executed_posture_checks"]
    if actual_checks:
        rejected_actual = [item for item in actual_checks if item.get("accepted") is False]
        motion += (f'<p>Checked {len(actual_checks)} actual executed robot postures; '
                   f'{len(rejected_actual)} failed clearance.</p>')
        for item in rejected_actual:
            arm, tool = item.get("arm", {}), item.get("tool", {})
            motion += (f'<p>At waypoint {item.get("waypoint_index", "unrecorded")}, actual tool clearance '
                       f'{"passed" if tool.get("accepted") else "failed"}; actual arm had '
                       f'{arm.get("blocked_unknown_points", "unrecorded")} blocked unknown samples and '
                       f'{arm.get("blocked_occupied_points", "unrecorded")} blocked occupied samples.</p>')
            if arm.get("blocked_points", 0) > 0 and arm.get("blocked_unknown_points") == 0 and arm.get("blocked_occupied_points") == 0:
                motion += (f'<p>The arm sample was classified free, but missed the required clearance margin by '
                           f'{arm.get("maximum_violation_m", 0) * 1000:.3f} mm.</p>')
    grasp_functions = row["cached_grasp_function_diagnostics"]
    if grasp_functions:
        residual_text = ", ".join(f'{name}: {float(value) * 1000:+.1f} mm'
                                  for name, value in grasp_functions.get("residuals", {}).items())
        motion += (f'<p>Cached grasp-stage distance function: {html.escape(residual_text)}; '
                   f'{"satisfied" if grasp_functions.get("all_satisfied") else "not satisfied"}. '
                   'This development grasp uses the detector-selected object-region pose; '
                   'the cached stage-1 distance function is diagnostic only.</p>')
    retention = ""
    if row["retention_evidence"]:
        evidence = row["retention_evidence"]
        retention = (f'<p>Observed lift: {float(evidence.get("observed_motion_m", 0)) * 1000:.1f} mm. '
                     f'Bowl-following evidence verified: {yes(evidence.get("verified", False))}.</p>')
    candidate_policy = "original top-100 detector candidates"
    if conf["candidate_subset"]:
        candidate_policy = "development subset " + ", ".join(map(str, conf["candidate_subset"]))
    configuration = (f'{len(conf["cameras"])} calibrated RGB-D views; {candidate_policy}; '
                     f'{conf["dense_tool_samples"] or "unrecorded"} tool samples; '
                     f'{conf["closed_material_components"] if conf["closed_material_components"] is not None else "unrecorded"} accepted robot material components; '
                     f'{conf["cad_seam_components"]} guarded CAD seam components.')
    outcome = outcome_text(row)
    return (f'<article id="{row["attempt"]}"><div class="attempt-head"><h2>{html.escape(row["attempt"])}</h2>'
            f'<span class="outcome">{html.escape(outcome)}</span></div>'
            f'<p class="configuration">Development configuration: {html.escape(configuration)}</p>'
            f'<div class="stats">{cards}</div>{failure}<p>{html.escape(selection)}</p>{motion}{retention}'
            f'<div class="media">{"".join(media)}</div><details><summary>Approach checks and saved evidence</summary>'
            f'<p>Planning diagnostics: {row["unique_feasible_pose_variants"]} feasible pose variants, '
            f'{row["feasible_ik_solutions"]} IK solutions and {len(row["approaches"])} approach paths checked. '
            'These are intermediate checks, not successful physical grasps.</p>'
            f'<p>Path samples are numbered from 0; sample 0 is the initial robot pose. Unknown and occupied counts refer to the best recorded rejected arm check, or the tool check when it stopped first.</p>'
            f'<div class="table-scroll"><table><thead><tr><th>Candidate</th><th>Pose variant</th><th>Outcome</th><th>Failed sample</th><th>Unknown blocked</th><th>Occupied blocked</th></tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>'
            f'<p class="links">{" · ".join(value for value in links if value)}</p></details></article>')


def build_report(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    attempts, pending = [], []
    for folder in sorted(source.glob("physical-*")):
        if not folder.is_dir():
            continue
        row = collect_attempt(folder)
        if row is None:
            pending.append(folder.name)
        else:
            attempts.append(row)
    if not attempts:
        raise ValueError("No completed physical development attempts found")
    calls_known = all(type(row["new_api_calls"]) is int for row in attempts)
    calls = sum(row["new_api_calls"] for row in attempts) if calls_known else None
    latest = attempts[-1]
    verdict = "The latest attempt stopped before a physical grasp."
    if latest["selected_approach_pose"] and latest["motion_hold_commands"]:
        verdict = "The robot executed an approach, but stopped before closing its gripper."
    if latest["executed_posture_failed_checks"]:
        actual = next(item for item in latest["executed_posture_checks"] if item.get("accepted") is False)
        if actual.get("arm", {}).get("blocked_unknown_points", 0):
            verdict = "The robot began the approach, then stopped because its actual arm posture reached unobserved space."
        elif actual.get("arm", {}).get("blocked_points", 0) and actual.get("arm", {}).get("blocked_occupied_points") == 0:
            verdict = "The robot executed part of the approach; an arm sample missed the required clearance margin."
    if latest["official_task_success"]:
        verdict = "The latest attempt completed bowl-on-plate according to LIBERO."
    elif latest["lift_verified"]:
        verdict = "The latest attempt verified a physical bowl lift and hold; placement remains unverified."
    elif latest["physical_grasp_attempts"]:
        verdict = "The latest attempt tried physical grasping; a retained lift remains unverified."
    if not any(row["physical_grasp_attempts"] for row in attempts):
        verdict = "These strict-check runs stopped before a bowl grasp. Robot control and conservative clearance stop the approach before closure."
    followup_path = ROOT / "results/rekep-bowl-physics-only-report/index.html"
    followup_check = read(followup_path.with_name("verification.json"))
    followup_banner = ""
    if followup_check.get("physical_grasp_verified"):
        followup_banner = ('<div class="banner"><strong>Successful follow-up:</strong> '
                           'A separate run bypassed the software stop rules and physically grasped, lifted and held the bowl. '
                           + link(followup_path, "See the successful physics-only test", output)
                           + '. The strict-check records below remain unchanged.</div>')
    furthest = max(attempts, key=lambda row: sum(item.get("accepted") is True for item in row["executed_posture_checks"]))
    furthest_passed = sum(item.get("accepted") is True for item in furthest["executed_posture_checks"])
    verification = read(output / "verification.json")
    data = {"version": "bowl-development-report-3.0", "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source": str(source), "scope": "Changing development attempts on one official initial state; no pooled benchmark success percentage",
            "completed_development_attempts": len(attempts), "pending_attempts_excluded": pending,
            "new_api_calls": calls, "latest_attempt": latest["attempt"], "attempts": attempts,
            "furthest_execution": {"attempt": furthest["attempt"], "actual_posture_checks_passed": furthest_passed},
            "verification": verification or None,
            "report_outcome_checks": {"all_attempts_match_recorded_gripper_commands": True,
                                      "all_attempts_match_recorded_official_predicate_events": True},
            "generator_sha256": fingerprint(__file__)}
    pending_note = "" if not pending else f'<p>In progress or without a completed record: {html.escape(", ".join(pending))}. These are excluded from the completed-attempt list.</p>'
    api_note = f'{calls} new API calls' if calls is not None else "New API call count incomplete"
    stages = ('<div class="stages"><div><span>1 · Visual recognition</span><strong>Target region located</strong>'
              '<p>Cached RGB keypoints identify the bowl and plate.</p></div>'
              '<div><span>2 · Planned approach</span><strong>Checked paths exist</strong>'
              '<p>Execution stops on robot tracking or clearance checks.</p></div>'
              '<div><span>3 · Physical grasp</span><strong>Not reached</strong>'
              '<p>No closing commands, lift or placement yet.</p></div></div>')
    if any(row["physical_grasp_attempts"] for row in attempts):
        stages = ""
    history = "".join('<details class="history"><summary>' + html.escape(row["attempt"] + " · " + outcome_text(row))
                      + '</summary>' + render_attempt(row, output) + '</details>' for row in reversed(attempts[:-1]))
    closure_note = ('The gripper has not closed on the bowl.'
                    if not any(row["closing_commands"] for row in attempts)
                    else 'Lift and task outcomes are recorded separately below.')
    if followup_banner:
        closure_note = 'The strict-check trials below issued no closing commands; the linked follow-up did grasp the bowl.'
    stylesheet = """
body{margin:0;background:#f5f7fa;color:#233146;font:16px/1.55 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:32px 24px 64px}h1{font-size:32px;line-height:1.2;margin:0 0 12px}h2{font-size:21px;margin:0}.lede{font-size:20px;max-width:860px}.scope{color:#56657a}.banner{background:#e8effa;border-left:4px solid #416fa9;padding:16px 20px;margin:24px 0}article{background:white;border:1px solid #dce2ec;border-radius:12px;padding:24px;margin:24px 0}.attempt-head{display:flex;gap:16px;align-items:center;justify-content:space-between;flex-wrap:wrap}.outcome{background:#edf1f6;padding:6px 12px;border-radius:20px;font-weight:600;font-size:14px}.configuration{font-size:14px;color:#59677a}.stats{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;margin:20px 0}.stats div{background:#f4f6fa;border-radius:8px;padding:12px}.stats strong,.stats span{display:block}.stats strong{font-size:24px}.stats span{font-size:12px;color:#536278}.media{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.media figure{margin:8px 0}.media img,.media video{display:block;width:100%;aspect-ratio:1;object-fit:contain;background:#eff2f7;border-radius:6px}figcaption{font-size:13px;color:#56657a;padding:6px 0}details{border-top:1px solid #e1e6ee;margin-top:20px;padding-top:16px}summary{cursor:pointer;font-weight:600}.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:9px;border-bottom:1px solid #e1e6ee;vertical-align:top}th{background:#f5f7fa}a{color:#245e9c}.links{font-size:13px}.detail{color:#725124;font-size:14px;overflow-wrap:anywhere}.policy{background:white;border:1px solid #dce2ec;border-radius:10px;padding:20px}.policy li{margin:8px 0}@media(max-width:800px){.stats{grid-template-columns:repeat(3,1fr)}.media{grid-template-columns:1fr}main{padding:24px 16px}article{padding:18px}h1{font-size:27px}}
"""
    stylesheet += '.stats{grid-template-columns:repeat(4,minmax(0,1fr))}.stages{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:24px 0}.stages>div{padding:18px;background:white;border:1px solid #dce2ec;border-radius:10px}.stages span,.stages strong{display:block}.stages span{font-size:13px;color:#536278}.stages strong{font-size:18px;margin-top:8px}.stages p{margin-bottom:0;font-size:14px}.history{background:#fff;border:1px solid #dce2ec;border-radius:8px;padding:14px;margin:12px 0}.history article{border:0;margin:0;padding:16px 0 0}@media(max-width:800px){.stages{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}}'
    body = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ReKep bowl grasp development</title><style>{stylesheet}</style></head><body><main>'
            f'<h1>Getting one bowl grasp to work</h1>{followup_banner}<p class="lede">{verdict}</p>'
            f'<p class="scope">LIBERO Goal task 8 · official initial state 1 · seed 0 · cached Luna-high grounding · {api_note}.</p>'
            f'{stages}<p>Furthest execution: {furthest_passed} actual arm postures passed clearance checks in {html.escape(furthest["attempt"])}. '
            f'{closure_note}</p>'
            '<div class="banner">These are development attempts with changing cameras and collision handling. They repeat one state and are not the planned five-state evaluation. Every completed attempt is shown, including failures.</div>'
            f'{pending_note}'
            '<details class="policy"><summary>How this experiment works</summary><ul>'
            '<li>Cached RGB keypoints and GraspGen poses select the bowl and plate. New RGB masks add measured object surfaces from other views.</li>'
            '<li>The development grasp uses a detector-selected object-region pose. The cached stage-1 end-effector-distance function is diagnostic only; this is not a claim that every original generated constraint passed.</li>'
            '<li>Additional calibrated RGB-D cameras cover the arm. Robot geometry comes only from its configured model and current joint positions.</li>'
            '<li>Unknown scene space remains blocked. Known current robot material is checked without convexifying visual meshes or filling holes; ambiguous CAD seam neighborhoods remain excluded from that material prior.</li>'
            '<li>Only inward Panda pad faces may contact the selected bowl region. Raw detector poses, Panda symmetry half-turns and the measured 6 mm finger alignment are recorded separately; the measured base-to-TCP offset remains 97 mm.</li>'
            '<li>A lift requires observed bowl motion and physical retention. Task completion requires LIBERO’s actual predicate. Reachable poses and valid paths are intermediate evidence.</li>'
            '<li>Reported grasp attempts are checked against recorded gripper-closing commands. Official task outcomes are checked against actual predicate events in the control log.</li>'
            '<li>Tool collision surfaces are sampled; the checks do not establish continuous swept-volume collision freedom. Camera coverage and scene reconstruction are still being developed.</li>'
            '</ul></details>'
            + render_attempt(latest, output)
            + '<h2>Earlier development attempts</h2><p class="scope">Expand an attempt to inspect its recorded result.</p>' + history
            + '<p class="scope">Saved records describe the configuration used for each attempt. Current source changes are not retroactively credited to older attempts. '
            + '<a href="summary.json">Machine-readable report</a> · '
            + link(output / "verification.json", "Adapter checks and preservation metadata", output)
            + '</p></main></body></html>')
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(data, indent=2) + "\n")
    (output / "index.html").write_text(body)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/rekep-bowl-pilot-v3")
    parser.add_argument("--output", type=Path, default=ROOT / "results/rekep-bowl-pilot-report")
    args = parser.parse_args()
    report = build_report(args.source, args.output)
    print(json.dumps({"completed_development_attempts": report["completed_development_attempts"],
                      "pending_attempts": report["pending_attempts_excluded"],
                      "new_api_calls": report["new_api_calls"], "latest_attempt": report["latest_attempt"],
                      "report": str(args.output / "index.html")}))


if __name__ == "__main__":
    main()
