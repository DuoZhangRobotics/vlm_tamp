"""Summarize larger-output trials without feeding observer evidence to the actor."""
import argparse
import json
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.local_agent import write_json


def repetition(text):
    paragraphs=[re.sub(r'\s+',' ',p.lower()).strip() for p in re.split(r'\n\s*\n',text)]
    paragraphs=[p for p in paragraphs if len(p)>=80]
    seen=set();duplicates=[]
    for p in paragraphs:
        if p in seen:duplicates.append(p)
        seen.add(p)
    total=sum(len(p) for p in paragraphs)
    return {'long_paragraphs':len(paragraphs),'repeated_long_paragraphs':len(duplicates),
        'repeated_character_fraction':sum(len(p) for p in duplicates)/total if total else None,
        'definition':'Paragraphs >=80 characters after lowercasing and whitespace normalization; repeated occurrences match an earlier paragraph exactly.',
        'interpretation':'Descriptive text repetition, not a direct measure of internal reasoning quality.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--geometry',action='store_true');args=parser.parse_args();root=args.run.resolve()
    rows=json.loads((root/'study/results.json').read_text());analysis=[]
    source=ROOT/'results/robot-io-contract-20261024/qwen-retest'
    for row in rows:
        old=json.loads((source/'responses'/row['id']/'call001/response.json').read_text())
        path=root/'study/responses'/row['id']/'call001/response.json'
        current=json.loads(path.read_text()) if path.exists() else {}
        text=lambda r:r.get('reasoning',r.get('reasoning_content','')) or ''
        analysis.append({'id':row['id'],'old_repetition':repetition(text(old)),
                         'new_repetition':repetition(text(current)),
                         'prefix_comparison':row.get('reasoning_prefix_comparison')})
    write_json(root/'reasoning-analysis.json',analysis)
    if args.geometry:
        geometry=[]
        for row in rows:
            record={'id':row['id'],'private_observer_only':True,'supplied_to_model':False,
                    'contract_valid':row['contract_valid'],'geometry_accepted':False}
            if not row['contract_valid']:
                record['skip_reason']='No complete valid ground request; no geometry fitting attempted.'
            else:
                import numpy as np
                from validate_robot_io_geometry import load_case,choices,template,shared_ground_handle
                cameras,metadata,provenance=load_case(row['source_private']['case'])
                regions=row['arguments']['regions'];started=time.monotonic()
                try:
                    mechanism,cloud,mask,anchors,evidence=shared_ground_handle(cameras,choices(regions),template(regions['camera']),metadata)
                    native=json.loads((ROOT/'results/astra-luna-comparison-20261023/grounding-diagnostics-private.json').read_text())
                    reference=next(x for x in native if x['trial']==row['source_private']['case'])
                    delta=np.asarray(mechanism.spec['joint_origin_world_m'])-reference['native_hinge_anchor_world_m']
                    record.update(geometry_accepted=True,model=mechanism.spec,fit_diagnostics=evidence,
                        native_hinge_line_error_mm=float(np.linalg.norm(delta[:2])*1000),provenance=provenance)
                except Exception as error:record['error']=f'{type(error).__name__}: {error}'
                record['geometry_seconds']=time.monotonic()-started
            geometry.append(record)
        write_json(root/'geometry-private.json',geometry)
    print(json.dumps({'responses_analyzed':len(analysis),'repetition':[(x['id'],x['new_repetition']['repeated_character_fraction']) for x in analysis]}))


if __name__=='__main__':main()
