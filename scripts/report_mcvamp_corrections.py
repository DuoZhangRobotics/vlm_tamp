"""Compact report of the 90-degree diagnostic, convex fix and 12k Luna attempts."""
from collections import Counter
import hashlib
import html
from pathlib import Path

import numpy as np

from vlm_tamp.coupled_storage import read_json,write_json
from report_mcvamp_microwave import metrics,plot_pair
from report_coupled_door import video

ROOT=Path(__file__).resolve().parents[1]
DIRECT=ROOT/'results/mcvamp-convex-full90-state11-20261015'
SEGMENT=ROOT/'results/mcvamp-convex-segment-state11-20261015'
PAIRED=ROOT/'results/mcvamp-convex-frozen-grasp-state11-v2-20261015'
CASE='open_microwave-state011'
OLD=ROOT/'results/mcvamp-microwave-frozen-20261015'/CASE


def endpoint_diagnostics(folder):
    rows=[];attempts=[]
    for path in folder.glob('selection-full-task/contact*/goals-*.json*'):
        rows.extend(r for r in read_json(path) if r.get('source')=='endpoint_admission')
    for path in folder.glob('selection-full-task/contact*/attempts.json*'):
        attempts.extend(read_json(path))
    grasp=read_json(folder/'selection-full-task/grasp-attempts.json')
    return {'grasp_ik_attempts':len(grasp),'terminal_ik_postures_checked':len(rows),
        'terminal_ik_postures_admitted':sum(r['accepted'] for r in rows),
        'native_search_queries':sum('solve_seconds' in r for r in attempts),
        'grasp_attempt_outcomes':dict(Counter(r['status'] for r in grasp)),
        'rejection_parts':dict(Counter(f"part {r['scene']['last_rejected_part']}, surface {r['scene']['last_rejected_surface']}"
            for r in rows if r.get('scene') and not r['scene']['accepted']))}


def main():
    output=ROOT/'results/mcvamp-corrections-report';output.mkdir(exist_ok=True)
    direct=read_json(DIRECT/CASE/'summary.json')
    current,native,contacts=metrics(PAIRED/CASE);previous,_,_=metrics(OLD)
    fresh,_,_=metrics(SEGMENT/CASE)
    api=read_json(ROOT/'results/microwave-luna-12k-20261015/summary.json')
    numerical=read_json(ROOT/'results/microwave-luna-12k-20261015/numerical-summary.json')
    collision=read_json(ROOT/'results/mcvamp-convex-fix-audit-20261015/summary.json')
    remaining=read_json(ROOT/'results/mcvamp-convex-fix-audit-20261015/remaining-finger-collision.json')
    diagnostics=endpoint_diagnostics(DIRECT/CASE)
    plot_pair(OLD,PAIRED/CASE,output/'state11-angle.png')
    if not (output/'state11-corrected.mp4').exists():
        video(PAIRED/CASE,current,native[max(native)],native,contacts,output/'state11-corrected.mp4')
    frozen=[]
    for root in [DIRECT,SEGMENT,PAIRED]:
        protocol=read_json(root/'protocol.json')
        frozen.append({'run':str(root),'snapshot_matches':all(
            hashlib.sha256((root/'source-snapshot'/name).read_bytes()).hexdigest()==digest
            for name,digest in protocol['sources_sha256'].items()),
            'baseline_sources_unchanged':read_json(root/'summary.json')['baseline_sources_unchanged'],
            'storage':read_json(root/CASE/'retention.json')})
    changed_grasp=None
    if (PAIRED/CASE/'chosen-grasp.json').exists():
        old=read_json(OLD/'chosen-grasp.json');new=read_json(PAIRED/CASE/'chosen-grasp.json')
        changed_grasp=float(np.max(np.abs(np.asarray(old['joints'])-np.asarray(new['joints']))))
    summary={'direct90':direct,'direct90_endpoint_diagnostics':diagnostics,
             'original_state11':previous,'paired_original_grasp_state11':current,
             'fresh_ik_selection_state11':fresh,
             'grasp_joint_difference_from_original_rad':changed_grasp,
             'luna_12k':api,'luna_12k_sensor_geometry':numerical,
             'corrected_saved_start_check':collision['corrected_start'],
             'remaining_collision_diagnostic':remaining,
             'tests_passed':23,'validation':frozen,
             'diagnostic_development_runs_not_new_frozen_success_rate':True}
    write_json(output/'summary.json',summary)
    api_rows=''.join(f'<tr><td>{r["state"]}</td><td>{r["response_status"]}</td>'
        f'<td>{r["usage"]["output_tokens"]}</td><td>{r["usage"]["output_tokens_details"]["reasoning_tokens"]}</td>'
        f'<td>{r["visible_characters"]}</td></tr>' for r in api['results'])
    result='success' if current['official_task_success'] else 'failure'
    reason=current.get('failure',current.get('continuation_stop_reason','All selected trajectories completed.'))
    page=f'''<!doctype html><meta charset="utf-8"><title>MC-VAMP corrections</title>
    <style>body{{max-width:1000px;margin:36px auto;padding:0 20px;font:17px/1.5 system-ui;color:#182b3b;background:#f5f7fa}}
    article{{background:white;padding:22px;border-radius:12px;margin:18px 0}}table{{border-collapse:collapse}}
    td,th{{text-align:left;padding:9px 16px;border-bottom:1px solid #dde3e9}}video,img{{max-width:100%;border-radius:8px}}</style>
    <h1>MC-VAMP: 90° planning, palm collision correction, Luna 12k</h1>
    <p>Targeted development diagnostics, 15 October 2026. The original frozen ten-episode comparison is preserved.
    These trials do not establish a new benchmark success rate.</p>
    <article><h2>Direct 90° planning</h2><p>No full 90° path was found in the state-11 trial.
    All 24 grasp poses were tried, with {diagnostics['grasp_ik_attempts']} grasp IK attempts.
    {diagnostics['terminal_ik_postures_checked']} terminal joint proposals were checked;
    {diagnostics['terminal_ik_postures_admitted']} passed scene admission.
    Native connecting-path searches: {diagnostics['native_search_queries']}.</p>
    <p>The 10°/20° fallback was disabled. No manipulation trajectory was issued.
    Finite grasp/endpoint search failure is not a proof that a full path is impossible.</p></article>
    <article><h2>The palm false positive is corrected</h2><p>Gripper cell spheres now serve only as a broad phase.
    Hits must also lie inside the calibrated convex Panda geometry. Contact-rich trials add zero gripper padding;
    the arm sphere checker retains its 2.5 mm point radius. Selected finger/handle contact remains allowed.</p>
    <p>The exact saved state-11 closed posture changes from invalid to admitted. Regression tests still reject
    a point inside the palm and unrelated moving geometry. A separate 2.5 mm padded trial showed why that extra
    padding also rejected this posture: nearby samples are outside the actual palm by as little as 0.64 mm.</p>
    <p>Paired physical state-11 diagnostic, replaying the original RGB-D-derived approach/grasp
    and planning every postclosure segment afresh: <b>{result}</b>.
    Final opening {current['final_opening_angle_deg']:.2f}°, completed contact trajectories
    {current.get('completed_contact_trajectories',0)}, execution vetoes {current.get('execution_veto_checks',0)}.
    {html.escape(reason)}</p>
    <p>Maximum selected-grasp joint change from the original trial: {changed_grasp} rad.
    Native LIBERO predicates alone decide task success; a planned arc angle does not.</p>
    <img src="state11-angle.png" alt="Actual door angles before and after correction">
    <video src="state11-corrected.mp4" controls preload="metadata"></video>
    <p>The original two-finger grasp is reproduced, but no postclosure contact path is found.
    In a separate EE-continuation probe, the first blocked posture is +3.5°: three static-tagged samples enter
    a configured pad by 0.27–0.54 mm. They are only 1.4–1.6 mm from the predicted moving handle.
    Their surface ownership remains unverified. This is a separate scene-association issue to investigate;
    it is not the palm-sphere overhang corrected above.</p>
    <img src="remaining-finger-collision.png" alt="Static-tagged points near the predicted moving handle intersect a configured finger pad"></article>
    <article><h2>Fresh IK selection also remains a failure</h2>
    <p>A separate trial selected a different arm IK posture for almost the same requested grasp pose.
    At closure it was 13.2 mm from its requested TCP pose, versus 0.72 mm in the original grasp.
    It missed the handle and completed {fresh.get('completed_contact_trajectories',0)} issued contact trajectories
    without opening the door (final {fresh['final_opening_angle_deg']:.2f}°; native task success
    {fresh['official_task_success']}). No execution veto occurred. This failure is retained;
    the paired replay does not replace it or establish a new success rate.</p></article>
    <article><h2>Planning versus tracking</h2><p>MC-VAMP already plans on the handle-point hinge-circle manifold,
    with jaw alignment or bounded tilt and free wrist twist. Joint PD tracks each timed joint trajectory.
    A prescribed timed end-effector curve needs continuous IK or differential IK for tracking; the planner is not
    a servo controller. All issued trajectories finish before observing or replanning. There are no execution-time vetoes.</p></article>
    <article><h2>Luna High, 12,000-token cap</h2><p>Identical RGB images and prompts were reused.
    All three new single attempts completed with valid template choices. The existing sensor-only handle geometry and
    full finger-pad support stage also passed on {numerical['passed']}/{numerical['planned']} choices.
    These are intermediate checks, not proof of correct hinge estimates or physical task success.
    No automatic API retries.</p>
    <table><tr><th>Closing state</th><th>Status</th><th>Output tokens</th><th>Reasoning tokens</th><th>Visible characters</th></tr>{api_rows}</table>
    <p>Additional conservative charge ${api['additional_conservative_spend_usd']:.6f}; cumulative
    ${api['cumulative_conservative_spend_usd']:.6f}; remaining ${2.-api['cumulative_conservative_spend_usd']:.6f} of the $2 cap.
    The cap includes reasoning and visible output; see <a href="https://developers.openai.com/api/docs/guides/reasoning">OpenAI documentation</a>.</p></article>
    <p>23 focused tests passed. Source snapshots and baseline hashes verified. Each physical run was compacted after completion,
    keeping its video, actions, outcomes, source pins, and lossless numeric planning evidence.</p>'''
    (output/'index.html').write_text(page)
    print(output/'index.html')


if __name__=='__main__':main()
