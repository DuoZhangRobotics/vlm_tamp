"""Check same-thread subscription continuation without a physical episode."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from vlm_tamp.subscription_agent import (
    clean_environment,codex_arguments,codex_resume_arguments,verify_isolation)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=(ROOT/args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    work=Path(tempfile.mkdtemp(prefix='vlm-resume-smoke-'))
    thread=None;records=[]
    try:
        isolation=verify_isolation(work,[ROOT/'scripts/serve_subscription_robot.py'])
        prompts=[
            'This is a transport smoke test, with no robot or simulation. Remember the test phrase ORCHID37. Write canary.txt containing BLUE, then finish. Use no network or files outside this workspace.',
            'Continue the same transport smoke test. Append the test phrase from the previous turn to canary.txt, then finish. Use no network or files outside this workspace.'
        ]
        for number,prompt in enumerate(prompts,1):
            cmd=(codex_arguments(work,out/f'last{number}.txt',ephemeral=False) if thread is None
                 else codex_resume_arguments(work,out/f'last{number}.txt',thread))
            with (out/f'events{number}.jsonl').open('w') as log,(out/f'stderr{number}.log').open('w') as err:
                result=subprocess.run(cmd,input=prompt,text=True,cwd=work,env=clean_environment(),
                    stdout=log,stderr=err,timeout=180)
            assert result.returncode==0,f'Continuation smoke turn {number} failed'
            events=[json.loads(x) for x in (out/f'events{number}.jsonl').read_text().splitlines()]
            identity=next(e['thread_id'] for e in events if e.get('type')=='thread.started')
            if thread is not None:assert thread==identity,'Resume changed the thread'
            thread=identity;records.append({'turn':number,'thread_id':thread})
        content=(work/'canary.txt').read_text()
        assert 'BLUE' in content and 'ORCHID37' in content
        shutil.copyfile(work/'canary.txt',out/'canary.txt')
        (out/'result.json').write_text(json.dumps({'passed':True,'same_thread':thread,
            'turns':records,'prior_context_recalled':True,'isolation_passed':isolation['passed'],
            'physical_episodes':0,'separately_billed_api_calls':0},indent=2)+'\n')
        print(json.dumps({'passed':True,'same_thread':thread,'physical_episodes':0}),flush=True)
    finally:
        shutil.rmtree(work)


if __name__=='__main__':
    main()
