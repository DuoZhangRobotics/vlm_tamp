"""Report native scene collision admission without hiding rejected attempts."""
import argparse
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path

from vlm_tamp.coupled_storage import read_json,write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--studies',nargs='+',required=True)
    args=parser.parse_args();output=(ROOT/args.output).resolve();output.mkdir(exist_ok=True)
    rows=[];cards=[]
    def link(path,label):
        return f'<a href="{html.escape(os.path.relpath(path,output))}">{html.escape(label)}</a>'
    for study_name in args.studies:
        study=ROOT/study_name;protocol=read_json(study/'protocol.json')
        for case in protocol['cases']:
            folder=study/case
            if not (folder/'result.json').exists():continue
            result=read_json(folder/'result.json')
            events=[json.loads(x) for x in (folder/'tool-events.jsonl').read_text().splitlines()]
            grasp=next(folder.glob('attempt01/*-grasp'),None)
            attempts=read_json(grasp/'selection/grasp-attempts.json') if grasp and (grasp/'selection/grasp-attempts.json').exists() else []
            profile=read_json(grasp/'profiling.json') if grasp else {}
            generation=read_json(grasp/'generation/generation-timing.json') if grasp else {}
            scene=read_json(grasp/'selection/approach-initial-scene.json')['record'] if grasp else {}
            queries=[json.loads(x) for x in (grasp/'selection/approach-queries.jsonl').read_text().splitlines()] if grasp and (grasp/'selection/approach-queries.jsonl').exists() else []
            statuses=Counter(r['status'] for r in attempts);details=[]
            for attempt in attempts:
                checks=attempt.get('ik_endpoint_checks',[])
                blocked=Counter((c[k].get('blocking_robot_part'),c[k].get('blocking_surface'))
                    for c in checks for k in ['open_aperture','closed_aperture'] if not c[k]['accepted'])
                def part_label(part):
                    return {'gripper0_hand_collision':'palm','gripper0_finger1_collision':'finger 1 body',
                        'gripper0_finger2_collision':'finger 2 body'}.get(part,
                        str(part).replace('robot0_link','arm link ').replace('_collision',''))
                why='; '.join(f'{part_label(part)} → {surface}' for part,surface in blocked)
                details.append({'pose_rank':attempt['pose_index']+1,'status':attempt['status'],
                    'grasp_score':attempt.get('graspgen_score'),'ik_branches_checked':len(checks),'blocking_pairs':why})
            motion=[e for e in events if e['request']['operation']=='execute' and e['response']['ok']]
            closure=next((e['response']['result'].get('grasp_evidence') for e in events
                if e['request']['operation']=='gripper' and e['response']['ok']),None)
            seconds=next((e['wall_seconds'] for e in events if e['request']['operation']=='plan_grasp'),None)
            frozen=None
            if protocol.get('frozen_grasp_cache_source') and grasp:
                source=Path(protocol['frozen_grasp_cache_source'])/case
                archived=next(source.glob('attempt01/*-grasp/generation/grasp_candidates.npz'))
                frozen=hashlib.sha256(archived.read_bytes()).hexdigest()==hashlib.sha256((grasp/'generation/grasp_candidates.npz').read_bytes()).hexdigest()
            row={'case':case,'study':study.name,'grasp_seed':protocol.get('grasp_seed',0),
                'grasp_call_seconds':seconds,'phases':profile.get('phases',{}),'generation':generation,
                'ik_queries':profile.get('grasp_ik_query_count',0),'path_queries':len(queries),
                'path_total_seconds':sum(q.get('total_seconds',0.) for q in queries),
                'native_solve_seconds':sum(q.get('solve_seconds',0.) for q in queries),
                'extra_dense_validation_seconds':sum(q.get('resampling_seconds',0.) for q in queries),
                'scene_policy':scene.get('environment_collision_policy'),
                'obstacles_passed_to_mcvamp':{k:scene.get(k) for k in ['static_points','moving_noncontact_points','selected_contact_points']},
                'grasp_status_counts':dict(statuses),'candidate_details':details,
                'executed_trajectories':len(motion),'closure_evidence':closure,
                'frozen_poses_exactly_match':frozen,'error':result['error'],
                'replay_input_match':result.get('replay_input_match'),
                'source_integrity':read_json(study/'source-integrity.json')}
            physical_path=folder/'physical-and-timing-audit.json'
            physical=read_json(physical_path) if physical_path.exists() else None
            if physical:
                row['physical_outcome']={k:physical[k] for k in ['approach_endpoint_error_m',
                    'closure_evidence','probe_evidence','robot_scene_contacts','native_success']}
            rows.append(row)
            timing='Not run' if seconds is None else f'{seconds:.2f} s'
            state=('No grasp motion: all examined candidates rejected.' if not motion and attempts else
                'Stopped before grasp planning.' if not attempts else 'Physical attempt executed; inspect the measured evidence.')
            names={'endpoint_rejected':'Endpoint collision','no_ik_solution':'No valid IK returned','selected':'Selected'}
            table=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [d['pose_rank'],
                f'{d["grasp_score"]:.3f}' if d['grasp_score'] is not None else '—',
                names.get(d['status'],d['status']),d['ik_branches_checked'],d['blocking_pairs']])+'</tr>' for d in details)
            links=' · '.join([link(folder/'result.json','Result'),link(folder/'tool-events.jsonl','All inputs / outputs')])
            if grasp:links+=' · '+link(grasp/'selection/grasp-attempts.json','Every endpoint check')
            physical_text=''
            if physical:
                blocked_contacts=[r for r in physical['robot_scene_contacts'] if 'gripper0_hand' in r['pair'] or 'robot0_link' in r['pair']]
                physical_text=f'<p>Measured approach error: <b>{physical["approach_endpoint_error_m"]*1000:.2f} mm</b>. Recorded arm/palm environment contacts: <b>{sum(r["contact_records"] for r in blocked_contacts)}</b>. Pad gap after closure: {max(0.,physical["closure_evidence"]["pad_gap_mm"]):.2f} mm. Grasp support: {html.escape(physical["closure_evidence"]["status"])}.</p>'
                links+=' · '+link(study/'watch.html','Physical replay and video')
            counts=row['obstacles_passed_to_mcvamp']
            count_text=('Not constructed: reset stopped before planning.' if not grasp else
                f'{counts["static_points"]:,} stationary; {counts["moving_noncontact_points"]:,} moving noncontact; {counts["selected_contact_points"]:,} selected contact.')
            cards.append(f'''<article><h2>{case} · seed {row['grasp_seed']}</h2>
<p class="source">{html.escape(study.name)}</p><p><b>{state}</b></p>
<p>Grasp call: <b>{timing}</b> · IK queries: {row['ik_queries']} · path queries: {len(queries)} · extra dense validation: {row['extra_dense_validation_seconds']:.3f} s.</p>
<p>Obstacle points passed to MC-VAMP: {html.escape(count_text)} Saved poses identical: {frozen if frozen is not None else 'new batch or no planning'}.</p>
<p>{html.escape(result['error'] or '')}</p>
{physical_text}
<table><tr><th>Pose rank</th><th>Learned score</th><th>Outcome</th><th>IK branches checked</th><th>Rejected robot/surface pairs</th></tr>{table}</table>
<p>{links}</p></article>''')
    write_json(output/'comparison.json',rows)
    physical_intro=''
    for r in rows:
        if r.get('physical_outcome'):
            p=r['physical_outcome'];physical_intro+=f'<p><b>Fresh-batch result:</b> one approach admitted; path call {r["path_total_seconds"]:.3f} s, measured arrival error {p["approach_endpoint_error_m"]*1000:.2f} mm. The fingers closed empty; this is not a successful grasp.</p>'
    page='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Arm and palm scene checks</title><style>body{font:16px system-ui;background:#f3f5f8;color:#172535;max-width:1300px;margin:30px auto;padding:0 18px}article{background:white;border-radius:12px;padding:22px;margin:18px 0}h1{font-size:29px}h2{font-size:22px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}a{color:#075ba5}.source{color:#667;font-size:13px}.policy{background:#dfeee4;padding:20px;border-radius:12px}</style></head><body>
<h1>Arm and palm checked; intended finger-pad contact allowed</h1>
<div class="policy">Live tools now require observed-scene collision checking. MC-VAMP receives the obstacle clouds and checks arm, palm and proximal fingers. Only the selected pad/target pair is exempt by default. An explicit palm-contact skill permits only that palm/target pair. No extra collision padding or dense validation pass was added.</div>
<p>Eight learned poses per batch. One GPU IK request per pose can return parallel branches; the closest scene-valid branch gets at most one path query. These are numerical component tests using recorded public regions, not fresh Luna evaluations.</p>
<p>The original advisory runs passed empty scene clouds to MC-VAMP. This report shows the obstacle counts actually passed after the fix. A rejected small batch does not prove that all grasps or all IK branches are infeasible.</p>
<p>Two early replay resets stopped on one-level RGB rendering differences (85 values in an unused view, then 15 in the selected view). Both records are retained below. These numerical tests consume metric arrays and fixed public region coordinates; all metric inputs must match exactly, while RGB differences are recorded. No fresh visual actor is being evaluated.</p>
<p><a href="comparison.json">All measurements and source-integrity checks</a></p>'''+physical_intro+''.join(cards)+'</body></html>'
    (output/'watch.html').write_text(page)


if __name__=='__main__':main()
