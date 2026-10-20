"""One user-authorized 12k/high attempt per previously truncated RGB request."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from vlm_tamp.cli import load_credentials
from vlm_tamp.coupled_storage import write_json
from vlm_tamp.rekep_libero.api import request_arguments
from vlm_tamp.rekep_libero.articulated_grounding import ArticulatedLedger,parse_choices,response_text

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/coupled-planning-resumed-20261015/frozen-evaluation/runs'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--states',nargs='+',type=int,default=[21,31,41])
    args=parser.parse_args()
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    ledger=ArticulatedLedger(ROOT/'results/articulated-api-v1')
    load_credentials('/home/duo/.codex/keys/vlm-tamp.env')
    before=json.loads(ledger.path.read_text())['cumulative_conservative_spend_usd']
    protocol_path=output/'protocol.json'
    if protocol_path.exists():
        protocol=json.loads(protocol_path.read_text())
        if protocol['states']!=args.states:raise ValueError('Frozen 12k request manifest changed')
        before=protocol['cumulative_spend_before_usd']
    else:
        if (output/'summary.json').exists():
            prior=json.loads((output/'summary.json').read_text())
            before=prior['cumulative_conservative_spend_usd']-prior['additional_conservative_spend_usd']
        write_json(protocol_path,{'states':args.states,'cumulative_spend_before_usd':before,
            'max_output_tokens':12000,'reasoning_effort':'high','budget_usd':2.,
            'same_rgb_and_prompt':True,'automatic_api_retries':0})
    results=[]
    for index in args.states:
        label=f'close_microwave-state{index:03d}'
        source=BASE/label/label/'grounding'
        folder=output/label;folder.mkdir(exist_ok=True)
        for name in ['role-choice-sheet.png','prompt.txt']:
            destination=folder/name
            if destination.exists() and destination.read_bytes()!=(source/name).read_bytes():
                raise ValueError('Existing RGB request inputs changed')
            shutil.copyfile(source/name,destination)
        sheet_hash=hashlib.sha256((source/'role-choice-sheet.png').read_bytes()).hexdigest()
        prompt=(source/'prompt.txt').read_text()
        entry={'state':index,'source':str(source),'image_sha256':sheet_hash,
               'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
               'model':'gpt-6-luna','reasoning_effort':'high','max_output_tokens':12000,
               'same_rgb_and_prompt_as_6k':True,'automatic_retries':0}
        try:
            raw=ledger.perform(request_arguments(folder/'role-choice-sheet.png',prompt,12000),
                f'microwave-luna-12k-v1-{label}-{sheet_hash[:16]}',allow_paid=True)
            write_json(folder/'response.json',raw)
            text=response_text(raw);(folder/'response.txt').write_text(text)
            entry.update(response_status=raw.get('status'),incomplete_details=raw.get('incomplete_details'),
                         usage=raw.get('usage'),visible_characters=len(text))
            if raw.get('status')!='completed':raise ValueError('Response incomplete; no automatic retry')
            choices=parse_choices(text);write_json(folder/'choices.json',choices)
            entry.update(status='completed_valid_choices')
        except Exception as error:
            entry.update(status='failed',failure_type=type(error).__name__,failure=str(error))
        results.append(entry);write_json(folder/'summary.json',entry)
        current=json.loads(ledger.path.read_text())
        summary={'planned':len(args.states),'completed':len(results),'results':results,
                 'cumulative_conservative_spend_usd':current['cumulative_conservative_spend_usd'],
                 'additional_conservative_spend_usd':current['cumulative_conservative_spend_usd']-before,
                 'budget_usd':2.,'api_attempts':current['api_attempts']}
        write_json(output/'summary.json',summary)
        print(json.dumps(entry),flush=True)


if __name__=='__main__':main()
