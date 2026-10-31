"""Physical continuation tests from recorded public acquisition requests.

No Luna calls, hidden object geometry or successful path replay enters planning.
Only the initial public region/skill requests are replayed; subsequent directions
come from the new tool's measured-contact result. These are local skill tests,
not fresh-policy task-success evaluations.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import time

import numpy as np

from serve_subscription_robot import RobotService
from vlm_tamp.coupled_storage import read_json,write_json,compact_public_observations


def render_report(output):
    output=Path(output);summary=read_json(output/'summary.json')
    calls=[s for r in summary['records'] for s in r['steps'] if s['operation']=='continue_push']
    mean=lambda key:sum(s['timing_seconds'].get(key,0.) for s in calls)/max(1,len(calls))
    text='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Continued finger push validation</title><style>
body{font:16px system-ui;max-width:1050px;margin:32px auto;padding:0 20px;color:#203249;background:#f3f6fa}
section{background:white;padding:20px;margin:20px 0;border:1px solid #d6dfeb;border-radius:8px}
table{border-collapse:collapse;width:100%}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}
video{width:100%;max-height:430px}a{color:#1462ad}.scroll{overflow:auto}</style>
<h1>Continued finger push validation</h1>'''
    text+=f'<p><b>{summary["passed"]}/{summary["completed"]} matched physical replays passed; {len(calls)} continuation strokes.</b> Each replay acquires fingertip contact, then makes two 30 mm direction-only requests.</p>'
    text+='<p>These test local skill repeatability, not fresh Luna task success. Initial regions come from a recorded public request; subsequent directions follow measured part rotation. No simulator object geometry is used for planning.</p>'
    text+='<p><a href="../luna-push-continuation-rerun-20261031/watch.html">Fresh Luna trials, task success and memory/I/O</a> · <a href="summary.json">Full timing measurements</a> · <a href="protocol.json">Replay protocol</a></p>'
    text+='<section><h2>Mean time per 30 mm continuation</h2><table>'
    for title,key in [('Total','total'),('Scene reconstruction','perception_seconds'),('Contact search','contact_search_seconds'),
        ('Stroke planning, including IK and planner collision checks','stroke_planning_seconds'),
        ('Execution and image capture','execution_and_capture_seconds'),('Tracker initialization','tracker_initialization_seconds'),
        ('Measured current-robot point filtering','current_robot_filter_seconds'),('Object motion tracking','motion_tracking_seconds'),
        ('Moving-part geometry update','moving_geometry_update_seconds'),('Other bookkeeping','other_seconds')]:
        text+=f'<tr><td>{title}</td><td>{mean(key):.3f} s</td></tr>'
    text+='</table><p>No new collision margin or dense post-planner validation. The stationary collision index is shared; native scene-template history remains bounded to two entries. Feedback updates the rigid moving part. Unseen changes elsewhere are not reconstructed during continuation.</p></section>'
    for row in summary['records']:
        folder=f'replay{row["repetition"]:02d}'
        text+=f'<section><h2>Replay {row["repetition"]}: {"passed" if row["continuation_passed"] else "failed"}</h2><div class="scroll"><table><tr><th>Operation</th><th>Result</th><th>Seconds</th><th>Hand progress</th><th>Object progress</th><th>Builds / searches</th></tr>'
        for s in row['steps']:
            if s['operation'] not in {'push','continue_push'}:continue
            hand=s.get('measured_contact_progress_m');progress=s.get('observed_progress')
            hand='—' if hand is None else f'{hand*1000:.2f} mm'
            motion='—' if progress is None else f'{progress*1000:.2f} mm' if s['operation']=='continue_push' else f'{np.degrees(progress):.2f}°'
            text+=f'<tr><td>{s["operation"]}</td><td>{html.escape(s["stop_reason"] or "")}</td><td>{s["seconds"]:.2f}</td><td>{hand}</td><td>{motion}</td><td>{s["full_scene_builds"]} / {s["contact_searches"]}</td></tr>'
        text+=f'</table></div><p><a href="{folder}/tool-events.jsonl">Requests and responses</a></p><video controls preload="none" src="{folder}/attempt01/initial/rollout.mp4"></video></section>'
    text+='<p>Requested hand travel is 30 mm; measured travel must be within the existing 3 mm IK position tolerance. This is a reported endpoint result, not an extra motion-validation pass. Actual measurements for each stroke are shown above.</p>'
    (output/'watch.html').write_text(text+'\n')


def run(source,output,repetitions=3):
    source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    events=[json.loads(line) for line in (source/'tool-events.jsonl').read_text().splitlines()]
    grounding=next(e['request']['arguments'] for e in events if e['request']['operation']=='ground')
    acquired=next(e for e in events if e['request']['operation']=='push'
        and not e['request']['arguments'].get('preview',False)
        and e['response'].get('result',{}).get('stop_reason')=='requested_progress_reached')
    original_robot=read_json(source/'agent-workspace/robot_data/attempt01/obs001/robot.json')
    protocol=read_json(source/'protocol.json');case=protocol['case']
    write_json(output/'protocol.json',{'scope':'recorded public input skill repeatability; no fresh Luna task-success claim',
        'source':str(source),'case':case,'repetitions':repetitions,'continuation_distance_m':.03,
        'numerical_source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__).resolve().parents[1]/'src/vlm_tamp/contact_skill_service.py',
             Path(__file__).resolve().parents[1]/'src/vlm_tamp/contact_skills.py']}})
    records=[]
    for number in range(1,repetitions+1):
        folder=output/f'replay{number:02d}';workspace=folder/'agent-workspace';folder.mkdir();workspace.mkdir()
        service=RobotService(folder,workspace,state_index=case['state_index'],attempts=1,
            task=case['task'],stop_on_success=True,surface_contact_tools=True,maximum_control_steps=600)
        row={'repetition':number,'steps':[]};calls=[]
        def call(op,args):
            began=time.perf_counter();reply=service.dispatch(op,args)
            event={'sequence':service.calls,'request':{'operation':op,'arguments':args},
                'response':{'ok':True,'result':reply},'wall_seconds':time.perf_counter()-began}
            calls.append(event)
            with (folder/'tool-events.jsonl').open('a') as stream:
                stream.write(json.dumps(event,default=lambda value:value.tolist() if hasattr(value,'tolist') else str(value))+'\n')
            brief={'operation':op,'seconds':event['wall_seconds'],'stop_reason':reply.get('stop_reason'),
                'control_steps':service.state['control_steps'],'continue_push_available':reply.get('continue_push_available'),
                'scene_reused':reply.get('scene_reused'),'preview_reused':reply.get('preview_reused'),
                'full_scene_builds':reply.get('full_scene_builds'),'contact_searches':reply.get('contact_searches'),
                'commanded_contact_distance_m':reply.get('commanded_contact_distance_m'),
                'measured_contact_progress_m':reply.get('measured_contact_progress_m'),
                'observed_progress':reply.get('observed_progress'),'timing_seconds':reply.get('timing_seconds')}
            row['steps'].append(brief);write_json(folder/'progress.json',row)
            print(json.dumps({'replay':number,**{k:v for k,v in brief.items() if k!='timing_seconds'}}),flush=True)
            return reply
        try:
            initial=call('reset',{})
            row['initial_robot_max_error_rad']=float(np.max(np.abs(np.asarray(service.state['metadata']['joint_positions'])-original_robot['joint_positions'])))
            if row['initial_robot_max_error_rad']>1e-8:raise ValueError('Initial robot state differs from the recorded case')
            call('ground',grounding)
            request={**acquired['request']['arguments'],'observation_id':service.latest_observation['observation_id'],
                'joint_model_epoch':service.epoch,'budget_seconds':120.}
            preview=call('push',{**request,'preview':True})
            if not preview.get('robot_valid_plan_available'):raise ValueError('Recorded acquisition did not find a contact in this replay')
            result=call('push',{**request,'preview':False})
            if not result.get('preview_reused'):raise ValueError('Identical preview was not reused')
            for _ in range(2):
                if not result.get('continue_push_available'):raise ValueError('No retained contact after acquisition/continuation')
                result=call('continue_push',{'direction_world':result['continuation']['reference_direction_world'],
                    'distance_m':.03,'budget_seconds':60.})
            continued=[r for r in row['steps'] if r['operation']=='continue_push']
            row['continuation_passed']=len(continued)==2 and all(r['full_scene_builds']==r['contact_searches']==0
                and r['stop_reason'] in {'requested_stroke_complete','stroke_executed','native_task_success'}
                and abs((r['measured_contact_progress_m'] or 0)-.03)<=.003 for r in continued)
        except Exception as error:row.update(error=f'{type(error).__name__}: {error}',continuation_passed=False)
        finally:
            service.finish_attempt('diagnostic_complete')
            write_json(folder/'summary.json',row)
            if row.get('continuation_passed'):
                write_json(folder/'public-retention.json',compact_public_observations(workspace,finished=True,run=folder))
            records.append(row);write_json(output/'summary.json',{'scope':'skill repeatability, not fresh Luna task success',
                'completed':len(records),'passed':sum(r['continuation_passed'] for r in records),'records':records})
            render_report(output)
        if not row['continuation_passed']:break
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repetitions',type=int,default=3);args=parser.parse_args()
    rows=run(args.source,args.output,args.repetitions)
    raise SystemExit(0 if len(rows)==args.repetitions and all(r['continuation_passed'] for r in rows) else 1)
