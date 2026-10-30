"""Show grasp sampling, planner timing and physical evidence in Live Preview."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path

import numpy as np
from vlm_tamp.coupled_storage import read_json,write_json

ROOT=Path(__file__).resolve().parents[1]


def audit(folder):
    result=read_json(folder/'result.json')
    events=[json.loads(line) for line in (folder/'tool-events.jsonl').read_text().splitlines()]
    grasp=next(folder.glob('attempt01/*-grasp'))
    profile=read_json(grasp/'profiling.json')
    generation=read_json(grasp/'generation/generation-timing.json')
    search=read_json(grasp/'selection/search-summary.json') if (grasp/'selection/search-summary.json').exists() else {}
    queries=[]
    if (grasp/'selection/approach-queries.jsonl').exists():
        queries=[json.loads(line) for line in (grasp/'selection/approach-queries.jsonl').read_text().splitlines()]
    physical=read_json(folder/'attempt01/initial/evaluation_contact_diagnostics.json')['steps']
    contacts=Counter();peaks={};angles=[]
    for step in physical:
        j=next((j for j in step.get('evaluation_only_articulation_state',[]) if 'microjoint' in j['joint_name']),None)
        if j:angles.append([step['step'],abs(float(np.degrees(j['position'])))])
        for c in step['robot_involved_contacts']:
            if all(c['robot_flags']):continue
            key=' / '.join(c['geom_names']);contacts[key]+=1
            peaks[key]=max(peaks.get(key,0),c['normal_force_n'])
    motion=[e['response']['result'] for e in events if e['request']['operation']=='execute' and e['response']['ok']]
    close=next((e['response']['result'] for e in events if e['request']['operation']=='gripper' and e['response']['ok']),{})
    measured=None;target=None
    if motion:
        metadata=read_json(next(folder.glob('attempt01/004-observation/metadata.json*')))
        measured=np.asarray(metadata['eef_pose_world_xyzw'])[:3]
        target=np.asarray(read_json(grasp/'selection/selected.json')['pose_world_xyzw'])[:3]
    summary={'evaluation_only':True,'excluded_from_policy_input':True,
        'grasp_call_seconds':next(e['wall_seconds'] for e in events if e['request']['operation']=='plan_grasp'),
        'generation':generation,'phases':profile['phases'],
        'hand_filter_seconds':search.get('search_timing_seconds',{}).get('hand_scene_filter'),
        'ik_queries':profile['grasp_ik_query_count'],'ik_total_seconds':profile['grasp_ik_total_seconds'],
        'path_queries':len(queries),'path_total_seconds':sum(q['total_seconds'] for q in queries),
        'native_solve_seconds':sum(q.get('solve_seconds',0.) for q in queries),
        'extra_validation_seconds':sum(q.get('resampling_seconds',0.) for q in queries),
        'closure_evidence':close.get('grasp_evidence'),
        'probe_evidence':motion[-1].get('motion_effect') if len(motion)>1 else None,
        'approach_endpoint_error_m':float(np.linalg.norm(target-measured)) if measured is not None else None,
        'angle_by_control':angles,'native_success':read_json(folder/'attempt01/summary.json')['official_task_success'],
        'robot_scene_contacts':[{'pair':k,'contact_records':v,'peak_normal_force_n':peaks[k]} for k,v in contacts.most_common()],
        'error':result['error']}
    write_json(folder/'physical-and-timing-audit.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',required=True)
    args=parser.parse_args();study=(ROOT/args.study).resolve();cards=[];summaries={}
    for case in read_json(study/'protocol.json')['cases']:
        folder=study/case
        if not (folder/'result.json').exists():continue
        a=audit(folder);summaries[case]=a
        old=ROOT/'results/clean-start-600-20261029/trials'/f'clean-v1-frozen-{case}-600'
        oldprofile=read_json(sorted(old.glob('attempt01/*-grasp/profiling.json'))[0])
        phases=a['phases'];g=a['generation'];closure=a['closure_evidence'] or {}
        rows=[('Historical hybrid grasp call',oldprofile['elapsed_seconds']),
            ('Small GraspGen grasp call',a['grasp_call_seconds']),
            ('Model loading (part of generation)',g['model_initialization_seconds']),
            ('Generation, including loading',phases.get('generation',0)),
            ('Scene preparation',phases.get('scene_preparation',0)),
            ('IK, including first-query setup',a['ik_total_seconds']),
            ('Path calls, including wrapper overhead',a['path_total_seconds']),
            ('Native planner solve (inside path calls)',a['native_solve_seconds']),
            ('Extra dense validation (inside path calls)',a['extra_validation_seconds'])]
        table=''.join(f'<tr><td>{html.escape(k)}</td><td>{v:.3f} s</td></tr>' for k,v in rows)
        if a.get('hand_filter_seconds') is not None:
            table+=f'<tr><td>Hand/scene filter before IK</td><td>{a["hand_filter_seconds"]*1000:.3f} ms</td></tr>'
        error='' if not a['error'] else '<p>'+html.escape(a['error'])+'</p>'
        gap=closure.get('pad_gap_mm');gap_text='not measured' if gap is None else f'{max(0.,gap):.2f} mm'
        grasp_status=('Grasp failed: fingers closed empty.' if closure.get('status')=='near_closed_without_bar_width_support'
            else 'Closure finished with separated pads; see the motion evidence below.' if gap is not None
            else 'No grasp executed.')
        motion=a.get('probe_evidence') or {}
        hand=motion.get('hand_rotation_deg');part=motion.get('observed_region_rotation_deg')
        motion_text=(f'Hand motion: {hand:.2f}°. Observed part motion: {part:.2f}°.' if hand is not None and part is not None
            else 'Part motion was not measured in this probe.')
        endpoint='not executed' if a['approach_endpoint_error_m'] is None else f'{a["approach_endpoint_error_m"]*1000:.1f} mm'
        contacts=''.join(f'<li>{html.escape(r["pair"])}: peak {r["peak_normal_force_n"]:.1f} N</li>' for r in a['robot_scene_contacts'][:4])
        cards.append(f'''<article><h2>{case}</h2>{error}
<p><b>IK calls: {a['ik_queries']}; path calls: {a['path_queries']}.</b> Eight learned poses available; stop at first executable approach.</p>
<table>{table}</table><p>Gap after closure: <b>{gap_text}</b>. Approach endpoint error: <b>{endpoint}</b>.</p>
<p><b>{html.escape(grasp_status)}</b> Native task success: {a['native_success']}.</p>
<p>{motion_text}</p>
<p>This is a grasp + 10° probe test. A completed RPC is not a verified grasp or a full-task evaluation.</p>
<details><summary>Offline physical contacts</summary><ul>{contacts}</ul></details>
<video controls preload="metadata" src="{case}/attempt01/initial/rollout.mp4"></video>
<p><a href="{case}/physical-and-timing-audit.json">Timing and physical evidence</a> ·
<a href="{case}/tool-events.jsonl">Every input/output</a> · <a href="{case}/result.json">Result</a></p></article>''')
    write_json(study/'comparison.json',summaries)
    completed=len(cards)==len(read_json(study/'protocol.json')['cases'])
    refresh='' if completed else '<meta http-equiv="refresh" content="15">'
    page=f'''<!doctype html><html><head><meta charset="utf-8">{refresh}<title>Small GraspGen test</title>
<style>body{{font:17px system-ui;margin:32px auto;max-width:1450px;background:#f3f5f8;color:#182230;padding:0 20px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:20px}}article{{background:white;padding:24px;border-radius:12px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td{{border-bottom:1px solid #ddd;padding:7px}}td:last-child{{text-align:right;white-space:nowrap}}video{{width:100%;margin-top:16px}}h1{{font-size:28px}}h2{{font-size:21px}}a{{color:#0759a8}}</style></head><body>
<h1>Small GraspGen pipeline: timing and physical outcomes</h1>
<p>Eight native learned poses → fixed robot/TCP conversion → closest valid returned IK → one approach query → physical pinch and 10° probe.</p>
<p>Same public RGB selections and initial sensors as the historical cases. No frontal/oblique poses, height relocation or depth sweep. Model stays loaded between cases. GPU timing and sampling can vary; this is a component test, not a fresh Luna evaluation.</p>
<p>Planner checks are trusted; the current version has no additional dense validation. Environment collision policy for this run: <b>{html.escape(read_json(study/'protocol.json')['environment_collision_policy'])}</b>. No added collision padding.</p>
<p>Timings overlap as labeled: model loading is inside generation; native solving is inside path calls. Historical runs each loaded a fresh model; the later cases here reuse it.</p>
<p><a href="protocol.json">Protocol</a> · <a href="comparison.json">All measurements</a></p>
<div class="grid">{''.join(cards)}</div></body></html>'''
    (study/'watch.html').write_text(page)


if __name__=='__main__':main()
