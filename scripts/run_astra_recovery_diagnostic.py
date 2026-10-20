"""Matched Astra/Luna choices on two saved failure observations; no new tools."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src')]
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.model_diagnostic_budget import DiagnosticLedger,reservation
from vlm_tamp.rekep_libero.api import request_arguments
from vlm_tamp.rekep_libero.articulated_grounding import response_text
from vlm_tamp.recovery_decision import parse_decision

OUT=ROOT/'results/astra-recovery-diagnostic-20261018'
SOURCE=ROOT/'results/vlm-recovery-frozen-20261018'
CASES=[('closing15_late','close_bottom_drawer-state015-vlm','round-04'),
       ('closing17_contact','close_bottom_drawer-state017-vlm','round-03')]
MODELS=['gpt-6-luna','gpt-6-astra']

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-paid',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if not args.resume:
        OUT.mkdir(exist_ok=False)
        sources=list((ROOT/'src/vlm_tamp').rglob('*.py'))+[Path(__file__),ROOT/'configs/prompts/strategy_feedback_v2.txt']
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        for name in hashes:
            dest=OUT/'source-snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        requests=[]
        for key,episode,round_id in CASES:
            source=SOURCE/episode/'decisions'/round_id;dest=OUT/'inputs'/key;dest.mkdir(parents=True)
            for name in ('feedback-sheet.png','prompt.txt','feedback.json','decision.json','tool-result.json'):
                shutil.copyfile(source/name,dest/name)
            requests.extend({'case':key,'source_episode':episode,'source_round':round_id,'model':model} for model in MODELS)
        write_json(OUT/'protocol.json',{'source_sha256':hashes,'cases':CASES,'models':MODELS,'reasoning':'high',
            'max_output_tokens':12000,'prior_Luna_ceiling':32000,'same_prompt_image_history_tools':True,
            'model_only_difference_between_new_paired_requests':True,'budget_usd':2.,'automatic_retries':0,
            'study':'saved decision-point diagnostic conditioned on a Luna-generated prefix; not an episode success benchmark',
            'physical_followup':'Replay prior choices to reconstruct the decision point, validate matching observations, then execute the new choice with unchanged tools. No simulator state restoration or evaluation truth for control.'})
        write_json(OUT/'manifest.json',{'requests':requests})
    protocol=read_json(OUT/'protocol.json')
    if not all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protocol['source_sha256'].items()):
        raise ValueError('Frozen diagnostic sources changed')
    ledger=DiagnosticLedger(OUT/'api');rows=[]
    entries=read_json(OUT/'manifest.json')['requests'];planned_reservations=[]
    for entry in entries:
        source=OUT/'inputs'/entry['case']
        arguments=request_arguments(source/'feedback-sheet.png',(source/'prompt.txt').read_text(),12000)
        arguments['model']=entry['model'];planned_reservations.append(reservation(arguments))
    if sum(planned_reservations)>2.:raise ValueError('All four worst-case reservations must fit the shared cap')
    write_json(OUT/'budget-preflight.json',{'maximum_total_usd':sum(planned_reservations),'request_reservations':planned_reservations})
    if args.allow_paid:
        from vlm_tamp.cli import load_credentials
        load_credentials('/home/duo/.codex/keys/vlm-tamp.env')
    for entry in entries:
        target=OUT/'decisions'/entry['case']/entry['model'];target.mkdir(parents=True,exist_ok=True)
        source=OUT/'inputs'/entry['case']
        arguments=request_arguments(source/'feedback-sheet.png',(source/'prompt.txt').read_text(),12000)
        arguments['model']=entry['model'];started=time.perf_counter()
        record={**entry,'folder':str(target),'completed':False,'decision_valid':False}
        try:
            raw=ledger.perform(arguments,entry['case']+'-'+entry['model'],allow_paid=args.allow_paid)
            write_json(target/'response.json',raw);text=response_text(raw);(target/'response.txt').write_text(text)
            record.update(response_status=raw.get('status'),usage=raw.get('usage'),completed=True)
            if raw.get('status')!='completed':raise ValueError('Incomplete response; no automatic retry')
            decision=parse_decision(text);write_json(target/'decision.json',decision)
            record.update(decision_valid=True,decision=decision)
        except Exception as error:
            record.update(error_type=type(error).__name__,error=str(error))
        record['round_trip_seconds']=time.perf_counter()-started
        write_json(target/'summary.json',record);rows.append(record)
        write_json(OUT/'summary.json',{'planned_requests':len(entries),'completed_requests':len(rows),
            'results':rows,'api':ledger.snapshot()})
        print(json.dumps(record),flush=True)

if __name__=='__main__':main()
