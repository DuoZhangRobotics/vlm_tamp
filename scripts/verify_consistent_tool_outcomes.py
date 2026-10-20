"""Derive native outcomes without rewriting the original runner aggregate."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.tool_event_audit import native_attempt_outcome

RUN=ROOT/'results/shared-tools-transfer-20261020'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source=RUN/'summary.json';original=json.loads(source.read_text());verified=copy.deepcopy(original)
    corrections=[];evidence=[]
    for row in verified['agent_cases']:
        folder=RUN/'luna'/f'state{row["state"]:03d}'
        native=native_attempt_outcome(folder)
        assert native['native_evidence_available'] and native['native_attempts_recorded']==1
        actions_path=folder/'attempt01/initial/actions.json'
        actions=json.loads(actions_path.read_text())
        action_success=any(a['evaluation_task_success'] for a in actions)
        assert action_success==native['native_success']
        private=folder/'evaluation-private.json'
        episode=json.loads(private.read_text())['attempts'][0]
        assert episode['control_steps']==len(actions)
        if action_success:
            first=next(i for i,a in enumerate(actions) if a['evaluation_task_success'])
            assert len(actions)-first-1==0
        if row['native_success']!=native['native_success']:
            corrections.append({'state':row['state'],'runner_value':row['native_success'],
                'native_value':native['native_success'],
                'reason':'Postprocessing failed on a malformed operation; missing aggregate report was misclassified as task failure.'})
        row.update(native)
        row['report_repaired_without_motion']=row['state']==31
        evidence.append({'state':row['state'],'evaluation_private_sha256':sha(private),
            'actions_sha256':sha(actions_path),'native_success':action_success,
            'control_steps':len(actions)})
    assert len(verified['agent_cases'])==4 and verified['agent_attempts_started']==4
    verified['agent_native_successes']=sum(r['native_success'] for r in verified['agent_cases'])
    verified['verification']={'source_summary_sha256':sha(source),'original_summary_preserved':True,
        'native_actions_and_evaluation_agree':True,'corrections':corrections,'evidence':evidence,
        'physical_episodes_rerun':False,'post_success_motion':False}
    output=RUN/'verified-summary.json'
    text=json.dumps(verified,indent=2)+'\n'
    if output.exists():assert output.read_text()==text,'Existing verified record differs'
    else:output.write_text(text)
    assert sha(source)==verified['verification']['source_summary_sha256']
    print(json.dumps({'agent_native_successes':verified['agent_native_successes'],
        'planned':4,'corrected_postprocessing_cases':len(corrections),'output':str(output)}))


if __name__=='__main__':
    main()
