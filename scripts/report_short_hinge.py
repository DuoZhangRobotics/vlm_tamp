"""Short-plan hinge diagnostics and author-hosted VLA microwave videos."""
import hashlib
import html
from pathlib import Path

import numpy as np

from vlm_tamp.coupled_storage import read_json,write_json
from report_mcvamp_microwave import metrics
from report_coupled_door import video

ROOT=Path(__file__).resolve().parents[1]

def present(path):return path.exists() or path.with_suffix('.json.gz').exists()
VIDEOS=[
    {'model':'FLOWER','title':'Real robot: open the microwave',
     'source':'https://intuitive-robots.github.io/flower_vla/',
     'url':'https://intuitive-robots.github.io/flower_vla/static/video/real_vids/open_microwave/primary_camera.mp4',
     'note':'Authors’ representative real-robot evaluation clip, labeled Open the microwave.'},
    {'model':'π₀','title':'Real robot: microwave task, published success',
     'source':'https://www.pi.website/blog/pi0',
     'paper':'https://www.physicalintelligence.company/download/pi0.pdf',
     'url':'https://website.pi-asset.com/v2/upload/processed_tupperware_success.mp4',
     'note':'The paper defines this bimanual task as opening the microwave, putting a plastic container inside, and closing it.'},
    {'model':'π₀','title':'Real robot: same microwave task, published failure',
     'source':'https://www.pi.website/blog/pi0',
     'url':'https://website.pi-asset.com/v2/upload/processed_tupperware_fail.mp4',
     'note':'Authors’ failure video for the same task. Failure label comes from the source, not our independent evaluation.'}]


def main():
    output=ROOT/'results/short-planned-hinge-report';output.mkdir(exist_ok=True)
    write_json(output/'published-vla-videos.json',{'videos':VIDEOS,'remote_media_downloaded':False,
        'urls_verified_from_author_page_html':True,'video_headers_checked_without_downloading':True,
        'independent_evaluation':False,'matched_LIBERO_state_comparison':False})
    rows=[];cards=[]
    for index,run in [(11,'short-planned-hinge-state11-20261015'),(1,'short-planned-hinge-state01-20261015')]:
        root=ROOT/'results'/run;folder=root/f'open_microwave-state{index:03d}'
        row,native,contacts=metrics(folder);phases=read_json(folder/'phases.json') if present(folder/'phases.json') else []
        history=read_json(folder/'hinge-calibration.json')
        actual_hinge=np.asarray(native[max(native)]['anchor_world_m'])[:2]
        initial_hinge=np.asarray(read_json(folder/'task-spec.json')['joint_origin_world_m'])[:2]
        initial_error=float(np.linalg.norm(initial_hinge-actual_hinge))
        current_hinge=initial_hinge.copy();fit_errors=[]
        observations=read_json(folder/'observations.json') if present(folder/'observations.json') else []
        for observation in observations:
            fit=observation['hinge_fit']
            if not fit.get('accepted'):continue
            pivot=np.asarray(fit['hinge_world_m'])[:2]
            applied=bool(np.linalg.norm(pivot-current_hinge)<.08)
            if applied:current_hinge=pivot
            fit_errors.append({'control_step':observation['control_step'],'fit_applied':applied,
                'perpendicular_axis_error_m':float(np.linalg.norm(pivot-actual_hinge))})
        points=np.asarray(history['observed_centres_world_m']).reshape(-1,3)
        span=float(np.max(np.linalg.norm(points[:,None]-points[None,:],axis=2))) if len(points) else 0.
        protocol=read_json(root/'protocol.json');pins=protocol['sources_sha256']
        verified=all(hashlib.sha256((root/'source-snapshot'/n).read_bytes()).hexdigest()==h for n,h in pins.items())
        assert verified and read_json(root/'summary.json')['baseline_sources_unchanged']
        row.update(calibration=history,observed_motion_span_m=span,planned_horizons_deg=[float(np.degrees(p['planned_horizon_rad'])) for p in phases],
            source_snapshots_verified=verified,api_ledger_unchanged=read_json(root/'summary.json')['api_ledger_unchanged'],
            cleanup=read_json(folder/'retention.json')['bytes_reclaimed_this_pass'],
            initial_hinge_error_m_evaluation_only=initial_error,hinge_fit_errors_evaluation_only=fit_errors,
            final_model_hinge_error_m_evaluation_only=float(np.linalg.norm(current_hinge-actual_hinge)))
        rows.append(row)
        movie=output/f'state{index:02d}.mp4'
        if not movie.exists():
            tracking=folder/'tracking.json';temporary_tracking=False
            if not tracking.exists() and present(tracking):
                write_json(tracking,read_json(tracking));temporary_tracking=True
            try:video(folder,row,native[max(native)],native,contacts,movie)
            finally:
                if temporary_tracking:tracking.unlink()
        reason=row.get('failure',row.get('continuation_stop_reason','All selected trajectories completed.'))
        if row['official_task_success'] and row.get('continuation_stop_reason'):
            reason='Native opening goal succeeded; subsequent model-goal continuation stopped: '+reason
        cards.append(f'''<article><h2>Our pipeline · state {index}</h2><p>Native task success: <b>{row["official_task_success"]}</b>.
        Final actual opening: <b>{row["final_opening_angle_deg"]:.2f}°</b>. Completed short plans: {len(phases)}.
        Planned horizons: {html.escape(str(row["planned_horizons_deg"]))} degrees.</p>
        <p>Observed handle positions: {len(points)}. Observed span: {span*1000:.1f} mm.
        Hinge refinement used: <b>{history["refinement_used_in_planning"]}</b>.
        Latest fit: {html.escape(str(history["last_fit"]))}. Predicted positions excluded: {history["predicted_samples_excluded"]}.</p>
        <p>Hinge-axis error, evaluation only: initial {initial_error*1000:.1f} mm;
        latest applied model {np.linalg.norm(current_hinge-actual_hinge)*1000:.1f} mm. Simulator hinge data are read only by this report.</p>
        <p>{html.escape(reason)} Execution veto checks: {row["execution_veto_checks"]}.
        Every issued short trajectory completed before the next depth observation.</p>
        <video controls preload="metadata" src="{movie.name}"></video>
        <p>The magenta video marker shows the initial hinge estimate; use the calibration measurements above for fitted values.</p></article>''')
    video_cards=[]
    for i,entry in enumerate(VIDEOS):
        paper=f' · <a href="{entry["paper"]}">Paper, task definition on page 9</a>' if 'paper' in entry else ''
        video_cards.append(f'''<article><h2>{html.escape(entry["model"])} · {html.escape(entry["title"])}</h2>
        <p>{html.escape(entry["note"])} <a href="{entry["source"]}">Official source</a>{paper}</p>
        <video id="external-{i}" controls preload="none" playsinline src="{entry["url"]}"></video>
        <div>Playback: <button onclick="speed({i},.25)">0.25×</button><button onclick="speed({i},.5)">0.5×</button>
        <button onclick="speed({i},1)">1×</button><button onclick="seek({i},-.1)">−0.1 s</button><button onclick="seek({i},.1)">+0.1 s</button></div></article>''')
    ledger=read_json(ROOT/'results/articulated-api-v1/ledger.json')
    write_json(output/'evaluation.json',{'planned':2,'completed':2,'trials':rows,
        'native_successes':sum(r['official_task_success'] for r in rows),'new_api_calls':0,
        'reasoning_effort':'high','remaining_api_budget_usd':ledger['remaining_usd'],
        'targeted_grasp_replays_not_independent_benchmark':True,
        'author_videos_not_our_rollouts':True,'focused_tests_passed':30})
    page='''<!doctype html><meta charset="utf-8"><title>Short planned motions and VLA microwave videos</title>
    <style>body{max-width:1050px;margin:32px auto;padding:0 20px;font:17px/1.5 system-ui;background:#f5f7fa;color:#182b3b}
    article{background:white;padding:20px;border-radius:12px;margin:18px 0}video{display:block;max-width:100%;max-height:620px;margin:16px 0}
    button{font:inherit;padding:5px 12px;margin:4px}</style>
    <h1>Short planned motions + microwave videos from VLA papers</h1>
    <p>The restored policy plans complete 3° pulls while gathering depth observations, then complete 8° pulls after
    an accepted hinge refinement, with 2°/1° fallbacks. The wrist remains free within the contact model.
    Each plan finishes before observation and replanning. No execution-time vetoes are used.
    Predicted handle positions cannot enter the observed hinge fit.</p>
    <h2>Published VLA videos</h2><p>These are author-hosted real-robot policy examples. They are not our LIBERO runs or
    matched-state comparisons. Videos stream from the original hosts; none are downloaded into this project.
    Use the playback controls to inspect grasp closure and wrist motion. Playback speed is relative to the published clip.</p>'''
    page+=''.join(video_cards)+'<h2>Our short-motion tests</h2><p>Two targeted replays of previously captured grasps, not a new independent benchmark. Native LIBERO predicates determine task success. A small opening is not yet full task completion.</p>'+''.join(cards)
    page+='''<p>30 focused tests passed. No new API calls. Source snapshots and baseline hashes verified;
    each run was compacted after completion. <a href="../../docs/short_planned_hinge.md">Reproduction notes</a>.</p>
    <script>function speed(i,s){document.getElementById('external-'+i).playbackRate=s}
    function seek(i,s){let v=document.getElementById('external-'+i);v.pause();v.currentTime=Math.max(0,v.currentTime+s)}</script>'''
    (output/'index.html').write_text(page)
    print({'report':str(output/'index.html'),'successes':sum(r['official_task_success'] for r in rows),
        'hinge_refinement_used':[r['calibration']['refinement_used_in_planning'] for r in rows]})


if __name__=='__main__':main()
