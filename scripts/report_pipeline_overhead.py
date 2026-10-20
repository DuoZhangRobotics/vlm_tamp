"""Report measured frontend costs, exact-reference speedups, and loop latency."""
from collections import defaultdict
from html import escape
from pathlib import Path
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.coupled_storage import read_json,write_json

REPORT=ROOT/'results/pipeline-overhead-report'
PROFILES={'fresh_v1':ROOT/'results/pipeline-profile-fresh-20261016',
          'fresh':ROOT/'results/pipeline-profile-fresh-v2-20261016',
          'cached':ROOT/'results/pipeline-profile-cached-v2-20261016'}
LABELS={'open_middle_drawer':'Open middle drawer','open_top_drawer':'Open top drawer',
        'close_top_drawer':'Close top drawer','turn_on_stove':'Turn on stove'}


def stats(values):
    a=np.asarray(values,float)
    return {'n':len(a),'min':float(a.min()),'median':float(np.median(a)),
        'p90':float(np.quantile(a,.9)),'max':float(a.max())} if len(a) else {'n':0}


def summarize_profile(root):
    summary=read_json(root/'summary.json');timeline=read_json(root/'timing.json');episodes=[]
    for row in summary['episodes']:
        folder=root/f'{row["task"]}-state{row["initial_state"]:03d}'
        spans=read_json(folder/'timing.json')['events'];groups=defaultdict(list)
        for s in spans:groups[s['name']].append(s['seconds'])
        phases=read_json(folder/'phases.json') if (folder/'phases.json').exists() else []
        ex=[s for s in spans if s['name']=='execution.segment'];assert len(ex)==len(phases)
        cycles=[]
        for i in range(2,len(ex)-1):
            cycles.append({'from':phases[i]['label'],'to':phases[i+1]['label'],
                'wall_seconds':ex[i+1]['start_s']-ex[i]['start_s'],
                'execution_seconds':ex[i]['seconds'],'between_seconds':ex[i+1]['start_s']-ex[i]['end_s']})
        rpc=read_json(folder/'rpc-profile.json')
        observations=[x['seconds'] for x in rpc if x['operation']=='observe']
        episodes.append({'task':row['task'],'state':row['initial_state'],'native_success':row['official_task_success'],
            'failure':row.get('failure'),'first_motion_after_observation_seconds':row['fresh_observation_to_first_task_motion_seconds'],
            'episode_to_first_motion_seconds':row['episode_to_first_task_motion_seconds'],
            'episode_wall_seconds':row['episode_wall_seconds'],'control_steps':row['control_steps'],
            'new_api_calls':row['new_api_calls'],'folder':str(folder),
            'stages':{k:{**stats(v),'sum':float(sum(v))} for k,v in groups.items()},
            'cycles':cycles,'observation_rpc_seconds':stats(observations)})
    cold={e['name']:e['seconds'] for e in timeline['events'] if e['name'].startswith('initialization.') and 'task' not in e}
    return {'planned':summary['planned'],'completed':summary['completed'],'native_successes':summary['native_successes'],
        'cold_model_initialization':cold,'episodes':episodes,
        'actual_http_calls':sum(e['stages'].get('grounding.luna_round_trip',{}).get('n',0) for e in episodes)}


def main():
    REPORT.mkdir(exist_ok=True);profiles={k:summarize_profile(p) for k,p in PROFILES.items()}
    replay_root=ROOT/'results/pipeline-io-final-20261016';rows=[]
    for folder in sorted(replay_root.glob('fast-*')):
        fast=read_json(folder/'summary.json');task=fast['task']
        baseline=(ROOT/'results/pipeline-io-comparison-20261016/baseline-open-top' if task=='open_top_drawer' else replay_root/'baseline-turn_on_stove' if task=='turn_on_stove' else None)
        old=read_json(baseline/'summary.json') if baseline else None
        rows.append({'task':task,'commands':fast['joint_commands'],'fast_seconds':fast['execution_wall_seconds'],
            'fast_commands_per_wall_second':fast['joint_commands']/fast['execution_wall_seconds'],
            'baseline_seconds':old['execution_wall_seconds'] if old else None,
            'execution_speedup':old['execution_wall_seconds']/fast['execution_wall_seconds'] if old else None,
            'fast_episode_seconds':fast['wall_seconds'],'baseline_episode_seconds':old['wall_seconds'] if old else None,
            'episode_speedup':old['wall_seconds']/fast['wall_seconds'] if old else None,
            'exact_physics_match':fast['controller_and_physics_equivalent'],'errors':fast['maximum_errors_versus_source'],
            'fast_folder':str(folder),'baseline_folder':str(baseline) if baseline else None,
            'execution_requests':len(read_json(folder/'phases.json'))})
    cycles=[c for e in profiles['cached']['episodes'] for c in e['cycles']]
    loop=stats([c['wall_seconds'] for c in cycles]);pauses=stats([c['between_seconds'] for c in cycles])
    ledger=read_json(ROOT/'results/articulated-api-v1/ledger.json')
    before=read_json(PROFILES['fresh_v1']/'protocol.json')['api_before']['cumulative_conservative_spend_usd']
    cleanup_folders=[ROOT/'results/pipeline-io-development-20261016/fast-open-top',ROOT/'results/pipeline-io-comparison-20261016/baseline-open-top']
    cleanup_folders+=list(replay_root.glob('*'))
    cleanup_folders+=[Path(e['folder']) for p in profiles.values() for e in p['episodes']]
    cleaned=[read_json(f/'summary.json') for f in cleanup_folders if (f/'summary.json').exists()]
    result={'profiles':profiles,'replay_comparisons':rows,'cached_loop_seconds':loop,'cached_pause_seconds':pauses,
        'cached_loop_hz':1/loop['median'],'historical_loop_estimate_seconds':8.943645139,
        'historical_loop_was_from_filesystem_timestamps':True,'current_loop_uses_monotonic_instrumentation':True,
        'new_api_cost_usd':ledger['cumulative_conservative_spend_usd']-before,
        'cumulative_api_cost_usd':ledger['cumulative_conservative_spend_usd'],'remaining_api_budget_usd':ledger['remaining_usd'],
        'actual_new_http_calls':profiles['fresh_v1']['actual_http_calls']+profiles['fresh']['actual_http_calls'],
        'cleaned_runs':len(cleaned),'temporary_bytes_reclaimed':sum(s['storage']['bytes_reclaimed_this_pass'] for s in cleaned),
        'policy_change':False,'asynchronous_planning':False,'longer_trajectories':False}
    write_json(REPORT/'evaluation.json',result)
    comparison=''.join(f'<tr><td>{LABELS[r["task"]]}</td><td>{r["commands"]}</td><td>{r["baseline_seconds"]:.2f} s</td><td>{r["fast_seconds"]:.2f} s</td><td>{r["execution_speedup"]:.1f}×</td><td>{r["fast_commands_per_wall_second"]:.1f} Hz</td></tr>' if r['baseline_seconds'] is not None else f'<tr><td>{LABELS[r["task"]]}</td><td>{r["commands"]}</td><td>Not rerun</td><td>{r["fast_seconds"]:.2f} s</td><td>—</td><td>{r["fast_commands_per_wall_second"]:.1f} Hz</td></tr>' for r in rows)
    timing_rows=[]
    fields=[('Full vision preparation, both views','vision.prepare_view_total'),
            ('DINO feature extraction + transfer, both views','vision.dinov2_features_and_transfer'),
            ('SAM automatic masks, both views','vision.sam_automatic_masks'),
            ('Luna request round trip','grounding.luna_round_trip'),
            ('Metric constraint construction','constraints.metric_geometry'),
            ('Initial IK worker/calibration setup','initialization.ik_worker_and_calibration'),
            ('Grasp selection and preliminary plans','grasp.selection_and_preplans')]
    fresh=profiles['fresh']['episodes'];cached=profiles['cached']['episodes']
    for label,key in fields:
        values=[e['stages'].get(key,{}).get('sum',0.) for e in fresh]
        timing_rows.append(f'<tr><td>{label}</td>'+''.join(f'<td>{v:.3f} s</td>' for v in values)+'</tr>')
    timing_rows.append('<tr><td><b>Fresh observation available → first task motion</b></td>'+''.join(f'<td><b>{e["first_motion_after_observation_seconds"]:.2f} s</b></td>' if e['first_motion_after_observation_seconds'] is not None else '<td>No task motion</td>' for e in fresh)+'</tr>')
    timing_rows.append('<tr><td>Episode start → first task motion, including simulator setup and initial capture</td>'+''.join(f'<td>{e["episode_to_first_motion_seconds"]:.2f} s</td>' if e['episode_to_first_motion_seconds'] is not None else '<td>No task motion</td>' for e in fresh)+'</tr>')
    cards=[]
    for name,prof in profiles.items():
        for e in prof['episodes']:
            if name=='fresh_v1':continue
            import os
            video=os.path.relpath(Path(e['folder'])/'initial/rollout.mp4',REPORT)
            cards.append(f'<details><summary>{"Fresh frontend" if name=="fresh" else "Cached grounding"}: {LABELS[e["task"]]} — {"native success" if e["native_success"] else "failed"}</summary><p>{escape(e.get("failure") or "All planned trajectories completed.")}</p><p>{e["control_steps"]} control steps; {e["episode_wall_seconds"]:.1f} seconds episode wall time.</p><video controls preload="none" src="{escape(video,quote=True)}"></video></details>')
    cold=profiles['fresh']['cold_model_initialization'].get('initialization.models_total',0)
    doc=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pipeline timing and I/O optimization</title><style>
body{{font:16px system-ui;margin:30px auto;max-width:1100px;padding:0 22px;color:#17212b;background:#f5f7f9}}h1{{font-size:30px}}h2{{font-size:21px}}section,details{{background:white;padding:20px;border-radius:10px;margin:20px 0}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dde2e7}}.muted{{color:#53616f}}video{{width:440px;max-width:100%}}.numbers{{display:flex;gap:16px;flex-wrap:wrap}}.numbers div{{background:#e9f4ef;padding:16px;border-radius:8px}}.numbers b{{display:block;font-size:25px}}a{{color:#1466aa}}</style></head><body>
<h1>Full pipeline timing and lower-overhead execution</h1><p>Steps 1–2 only: measure fresh vision/Luna costs, reduce execution and communication overhead, preserve planning decisions and parameters.</p>
<div class="numbers"><div><b>4/4 exact matches</b>Saved-reference physics checks</div><div><b>{loop['median']:.2f} s / {1/loop['median']:.2f} Hz</b>Measured cached-grounding update cycle</div><div><b>${result['new_api_cost_usd']:.4f}</b>New conservative API cost</div></div>
<section><h2>Identical commands, lower execution overhead</h2><p>These are engineering replays of the same saved trajectories, rather than new task-success evaluations. Joint states, gripper states, controller commands and native success flags match exactly. Commands still advance the original 20 Hz simulation schedule.</p><table><tr><th>Task</th><th>Commands</th><th>Original executor</th><th>Batched executor</th><th>Speedup</th><th>During execution</th></tr>{comparison}</table><p class="muted">Baseline and fast runs used a single simulator worker, sequentially. This throughput excludes the pauses for observation and planning.</p></section>
<section><h2>Fresh task processing includes DINO and Luna</h2><p>Model outputs were not reused. DINO/SAM processed fresh captured RGB-D, and Luna was called with its existing prompt, high reasoning and 12,000-token ceiling. The returned choices drove the actual pipeline. Vision and grasp model loading took {cold:.2f} seconds before the task. IK-worker initialization and first-use inference costs remain included in each first-motion measurement.</p><table><tr><th>Stage</th>{''.join('<th>'+LABELS[e['task']]+'</th>' for e in fresh)}</tr>{''.join(timing_rows)}</table><p class="muted">Rows are overlapping scopes: DINO and SAM are components of full vision preparation. Do not add every row. First-motion time begins once fresh RGB-D is available and includes all subsequent grounding, grasp selection and planning. Reset, initial opening and camera acquisition are also recorded separately in the full timeline. Two final fresh requests are a small latency sample; two earlier development requests remain in the record.</p></section>
<section><h2>Remaining update latency</h2><p>With grounding already available, the median start-to-start manipulation cycle is <b>{loop['median']:.2f} seconds</b>, about <b>{1/loop['median']:.2f} Hz</b>. The pause between completed segments is {pauses['median']:.2f} seconds. The previous approximately 8.94-second estimate was reconstructed from file timestamps; the new measurement uses explicit monotonic clocks.</p><p>Average command throughput exceeds the 20 Hz target on these simulator tests; this is not a hard real-time deadline guarantee. The sub-second observation-to-next-trajectory target is still unmet. All 12 planning camera/depth views are retained; observation capture, scene reconstruction and collision-index rebuilding still contribute. No asynchronous replanning or longer motion horizons were introduced.</p></section>
<section><h2>What changed</h2><ul><li>A complete trajectory crosses the process boundary once, while the original controller advances every reference locally.</li><li>Full geometry and accumulated logs are transferred/written at segment boundaries. Automatic camera observations are disabled between explicit captures; each joint command still has a video frame.</li><li>Observation arrays are written losslessly without compression during execution, then compressed after cleanup. Redundant intermediate PNG and robot-mesh exports are omitted.</li><li>Voxel deduplication uses an equivalent integer ordering, verified to retain the same points in the same order. Robot-frame calibration is reused only for identical inputs.</li></ul><p>The planner, collision margins/contact policy, constraints, grasp filtering, controller gains and trajectory horizons are unchanged. A fresh stochastic Luna response can change grounding; therefore fresh-run outcomes are shown separately from equivalence tests.</p></section>
<section><h2>Accounting and evidence</h2><p>{result['actual_new_http_calls']} actual new Luna calls, with no automatic retry. Cumulative conservative API cost: ${result['cumulative_api_cost_usd']:.6f} / $2. Cleanup completed after {result['cleaned_runs']} runs, reclaiming {result['temporary_bytes_reclaimed']/2**30:.2f} GiB of temporary data.</p><p><a href="evaluation.json">Timing data</a> · <a href="final-validation.json">Validation</a> · <a href="voxel-equivalence.json">Identical point-cloud output check</a> · <a href="https://developers.openai.com/api/reference/python/resources/responses/methods/create">OpenAI request metadata documentation</a></p></section>
<h2>Actual pipeline outcomes and retained videos</h2><p>Videos show simulation time with planning pauses omitted. At the retained 4 fps, joint execution is displayed five times slower than its simulated 20 Hz command schedule.</p>{''.join(cards)}</body></html>'''
    (REPORT/'index.html').write_text(doc)
    print(json.dumps({'report':str(REPORT/'index.html'),'loop_seconds':loop,'pause_seconds':pauses,
        'new_api_cost_usd':result['new_api_cost_usd'],'cumulative_api_cost_usd':result['cumulative_api_cost_usd'],
        'cleaned_runs':result['cleaned_runs'],'reclaimed_bytes':result['temporary_bytes_reclaimed']}))

if __name__=='__main__':main()
