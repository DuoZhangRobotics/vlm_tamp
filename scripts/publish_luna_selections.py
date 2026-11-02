"""Publish the selection viewer; optionally add an operator-only hinge reference."""
import argparse
import hashlib
import json
from pathlib import Path

from live_study_dashboard import publish
from luna_selection_history import selection_history
from report_contact_skills import read, read_events


def export_evaluator_reference(study,run,joint_name):
    records=selection_history(study,run,read_events(run/'tool-events.jsonl'))
    controls={r['control_step'] for r in records if r['control_step'] is not None}
    sources=sorted(run.glob('attempt*/initial/evaluation_contact_diagnostics.json'))
    if len(sources)!=1:return {'run':run.name,'reference_exported':False,'reason':'requires one unambiguous evaluation episode'}
    source=sources[0];states={}
    for sample in read(source,{}).get('steps',[]):
        if sample.get('step') not in controls:continue
        joint=next((j for j in sample.get('evaluation_only_articulation_state',[]) if j.get('joint_name')==joint_name),None)
        if joint:
            states[str(sample['step'])]={'joint_origin_world_m':joint['anchor_world_m'],'joint_axis_world':joint['axis_world']}
    result={'evaluation_only':True,'never_supplied_to_actor':True,'joint_name':joint_name,
        'attempt':int(source.parent.parent.name.removeprefix('attempt')),
        'source':str(source.relative_to(run)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'states':states,'scope':'Per-capture simulator reference, for human inspection only; not part of robot input or grounding.'}
    (run/'selection-evaluation-reference.json').write_text(json.dumps(result,indent=2)+'\n')
    return {'run':run.name,'reference_exported':True,'reference_captures':len(states)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',required=True,type=Path)
    parser.add_argument('--evaluator-joint',help='Explicit native joint name for an optional human-only comparison; never inferred or sent to the actor')
    args=parser.parse_args();study=args.study.resolve();summary=[]
    if args.evaluator_joint:
        for run in sorted((study/'trials').iterdir()):
            if (run/'tool-events.jsonl').exists():summary.append(export_evaluator_reference(study,run,args.evaluator_joint))
    publication=publish(study,force=True)
    print(json.dumps({'references':summary,'publication':publication}))


if __name__=='__main__':main()
