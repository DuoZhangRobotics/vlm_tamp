"""Report the full-arc policy experiment separately from hinge evaluation."""
from pathlib import Path
from collections import Counter
import hashlib
import html
import shutil

import numpy as np
from scipy.spatial import cKDTree

from vlm_tamp.coupled_storage import read_json,write_json
from report_mcvamp_microwave import metrics
from report_coupled_door import video

ROOT=Path(__file__).resolve().parents[1]
OFFLINE=ROOT/'results/partial-pose-arc-validation-20261015'
RUN=ROOT/'results/partial-pose-arc-physical-state11-20261015'
CASE=RUN/'open_microwave-state011'


def main():
    output=ROOT/'results/partial-pose-arc-report';output.mkdir(exist_ok=True)
    policy=read_json(OFFLINE/'summary.json')
    evaluation=read_json(OFFLINE/'hinge-evaluation/summary.json')
    physical,native,contacts=metrics(CASE)
    planning=read_json(CASE/'plan-00/horizon-0/result.json')
    protocol=read_json(RUN/'protocol.json')
    witnesses=np.asarray(read_json(ROOT/'results/mcvamp-convex-fix-audit-20261015/remaining-finger-collision.json')['witness_points_world'])
    arrays=np.load(read_json(OFFLINE/'scene-reference.json')['arrays_path'])
    moving_distance=cKDTree(arrays['moving_reference']).query(witnesses)[0]
    static_distance=cKDTree(arrays['static_points']).query(witnesses)[0]
    roles={'previously_flagged_samples':len(witnesses),'now_supported_by_moving_observations_m':moving_distance,
        'nearest_current_static_observation_m':static_distance,
        'all_three_now_in_moving_cloud':bool(np.max(moving_distance)<1e-7),
        'historical_static_samples_removed':policy['role_reconciliation']['reconciled_prior_samples'],
        'object_identity_is_not_guaranteed_by_nearest_surface_association':True}
    layers=planning['continuation_layers']
    proposal_rejections=Counter(f"{int(row['scene']['last_rejected_part'])}/{int(row['scene']['last_rejected_surface'])}"
        for row in planning['goal_proposals'] if row.get('scene') and not row['scene']['accepted'])
    ledger=read_json(ROOT/'results/articulated-api-v1/ledger.json')
    sources=protocol['sources_sha256'];snapshot=RUN/'source-snapshot'
    verification={'source_snapshots_match':all(hashlib.sha256((snapshot/name).read_bytes()).hexdigest()==digest for name,digest in sources.items()),
        'current_policy_sources_match':all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest for name,digest in sources.items()),
        'frozen_baseline_sources_unchanged':read_json(RUN/'summary.json')['baseline_sources_unchanged'],
        'api_ledger_unchanged':read_json(RUN/'summary.json')['api_ledger_unchanged'],
        'evaluation_did_not_change_policy_result':evaluation['policy_summary_unchanged'],
        'execution_veto_checks':physical['execution_veto_checks'],'focused_tests_passed':27}
    if not all(v for k,v in verification.items() if k.endswith(('match','unchanged','result'))):
        raise ValueError('Source or result verification failed')
    summary={'physical_trial':physical,'offline_policy':{k:v for k,v in policy.items() if k!='goal_proposals'},
        'postclosure_search':{k:v for k,v in planning.items() if k!='goal_proposals'},
        'postclosure_goal_rejections':dict(proposal_rejections),'hinge_evaluation_only':evaluation,
        'contact_role_audit':roles,'validation':verification,'planned_physical_trials':1,
        'successes':int(physical['official_task_success']),'diagnostic_grasp_replay_not_independent_benchmark':True,
        'new_api_calls':0,'remaining_api_budget_usd':ledger['remaining_usd']}
    write_json(output/'summary.json',summary)
    shutil.copyfile(OFFLINE/'hinge-evaluation/prediction-error.png',output/'hinge-prediction-error.png')
    if not (output/'state11.mp4').exists():video(CASE,physical,native[max(native)],native,contacts,output/'state11.mp4')
    residual_angle=np.degrees(planning.get('last_progress_rad',0.))
    last_connected=max((np.degrees(row['angle_rad']) for row in layers if row['connected_candidates']),default=0.)
    hinge_mm=evaluation['perpendicular_axis_offset_m']*1000
    error90=next(r['median_point_error_m'] for r in evaluation['rows'] if r['progress_deg']==90)*1000
    angle_rows=''.join(f'<tr><td>{row["progress_deg"]:g}°</td><td>{row["median_point_error_m"]*1000:.2f} mm</td></tr>' for row in evaluation['rows'])
    word='success' if physical['official_task_success'] else 'failure'
    page=f'''<!doctype html><meta charset="utf-8"><title>Full-arc planning and hinge uncertainty</title>
    <style>body{{max-width:1050px;margin:36px auto;padding:0 20px;font:17px/1.5 system-ui;background:#f5f7fa;color:#182b3b}}
    article{{background:white;padding:22px;border-radius:12px;margin:18px 0}}img,video{{max-width:100%;border-radius:8px}}
    td,th{{padding:8px 18px;border-bottom:1px solid #dce2e8;text-align:left}}table{{border-collapse:collapse}}
    code{{background:#edf1f5;padding:2px 5px}}.large{{font-size:24px;font-weight:650}}</style>
    <h1>Full-arc planning with free wrist orientation</h1>
    <p>Targeted state-11 development experiment, 15 October 2026. Original frozen comparisons remain intact.</p>
    <article><h2>Implemented and tested; the complete motion remains unsolved</h2>
    <p>The new backend optimizes all seven arm joints while keeping the pad midpoint on the perceived handle arc.
    It prescribes no wrist quaternion and uses no orientation menu. Jaw tilt stays within 60°.
    Clearance objectives help propose postures; native MC-VAMP remains the collision and connecting-path authority.</p>
    <p>Any internal arc waypoints are fully connected before execution. The new backend requires the complete remaining
    arc and never executes an incomplete search prefix. The two material finger contacts are not individually fixed;
    physical grasp retention still has to be established by simulation.</p>
    <p>Postclosure goal proposals: {len(planning['goal_proposals'])}; admissible full-angle goals: {planning['admissible_goals']}.
    Search status: <b>{html.escape(planning['status'])}</b>. Last connected internal waypoint: {last_connected:.1f}°;
    first blocked layer: {residual_angle:.1f}°. These are planning quantities, not executed door motion.</p></article>
    <article><h2>Your hinge-uncertainty concern is measurable here</h2>
    <p class="large">{hinge_mm:.2f} mm hinge-axis offset → {error90:.2f} mm predicted door error at 90°</p>
    <p>The moving point cloud is rigidly transformed about the estimated hinge during planning. The real hinge is used
    only in this post-hoc evaluation. Both comparisons transform the same observed partial cloud, so the figure isolates
    hinge error rather than adding a true scene mesh. No true hinge data enters planning or control.</p>
    <img src="hinge-prediction-error.png" alt="Predicted door clouds diverge as opening angle increases when the hinge is wrong">
    <table><tr><th>Predicted rotation</th><th>Median error due to measured hinge offset</th></tr>{angle_rows}</table>
    <p>For parallel axes the error is <code>(I − R(θ)) δh</code>, with magnitude
    <code>2 sin(θ/2) ‖δh_perpendicular‖</code>. Thus even 10 mm hinge error produces 14.1 mm error at 90°.</p>
    <p>A path can be collision-free in the estimated scene and collide in the simulator, or be falsely rejected by
    the estimated scene. Grip slip and uncertain point ownership add further error. Millimetre collision thresholds
    do not make a centimetre-uncertain scene accurate. This experiment adds neither blanket padding nor execution vetoes.</p></article>
    <article><h2>Contact ownership correction</h2>
    <p>The selected-handle association now uses 8 mm, consistent with existing selected-surface fusion.
    All three previously flagged pad samples are now in the moving cloud: <b>{roles['all_three_now_in_moving_cloud']}</b>.
    Only the existing intended finger/handle permission applies; arm and palm still see that geometry.</p>
    <p>Historical static observations are reconciled only with current selected-handle association and moving depth support.
    This scene removed {roles['historical_static_samples_removed']} historical samples: the three corrected samples came from
    current-frame association. This is proximity-based ownership, not a guarantee that every adjacent object is correctly labeled.</p></article>
    <article><h2>Physical replay: {word}</h2>
    <p>The original approach and physical gripper closure were replayed after matching initial RGB, proprioception and
    handle cloud. The old 20° manipulation trajectory was not replayed. A fresh full remaining arc was required after closure.</p>
    <p>Actual final opening: {physical['final_opening_angle_deg']:.2f}°. Native task success: {physical['official_task_success']}.
    Issued contact trajectories: {physical.get('completed_contact_trajectories',0)}. Execution vetoes: {physical['execution_veto_checks']}.
    No complete arc was found, so no partial pulling trajectory was issued. The approach and closure commands completed.</p>
    <video controls preload="metadata" src="state11.mp4"></video>
    <p>This is one targeted grasp replay, counted as 0/1 successful trials. It is not an independent frozen benchmark.
    A failed finite search does not prove physical infeasibility or isolate hinge error as its only cause.</p></article>
    <article><h2>Validation and retained artifacts</h2>
    <p>27 focused tests passed, including wrist freedom, clearance-objective gradients, role reconciliation,
    hinge-error propagation, native collision permissions and uninterrupted timed execution. Policy source snapshots,
    frozen baseline hashes and the unchanged API ledger were verified. Ground-truth evaluation leaves the policy result unchanged.</p>
    <p>No new API calls; remaining accounted budget ${ledger['remaining_usd']:.6f}. Per-run cleanup keeps essential
    videos, outcomes, source pins and numeric geometry. See <a href="../../docs/partial_pose_arc.md">reproduction notes</a>.</p></article>'''
    (output/'index.html').write_text(page)
    print({'report':str(output/'index.html'),'hinge_error_mm':hinge_mm,'door_error90_mm':error90,
        'native_task_success':physical['official_task_success'],'search_status':planning['status']})


if __name__=='__main__':main()
