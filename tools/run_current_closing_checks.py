"""Run exactly two current-code trials through the existing monitored runner."""
import json
import argparse
import html
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from vlm_tamp.coupled_storage import read_json,write_json
from run_restriction_study import source_hashes

_last_publish=0.


def gpu_ready():
    """Test startup before paying for an actor or consuming an episode budget."""
    probes=[('CUDA',sys.executable,
        'import ctypes,sys; c=ctypes.CDLL("libcuda.so.1").cuInit(0); print("CUDA initialization",c,flush=True); sys.exit(0 if c==0 else 1)'),
        ('EGL',os.environ['VLM_TAMP_SIM_PYTHON'],
        'import mujoco; c=mujoco.GLContext(16,16); c.make_current(); print("EGL context ready",flush=True); c.free()')]
    for stage,python,code in probes:
        try:
            result=subprocess.run([python,'-u','-c',code],env=os.environ.copy(),
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=30)
        except subprocess.TimeoutExpired:
            return {'ready':False,'stage':stage,'reason':'Initialization timed out; probe stopped','checked_at':time.time()}
        if result.returncode:
            return {'ready':False,'stage':stage,'reason':result.stdout[-1500:],
                'returncode':result.returncode,'checked_at':time.time()}
    return {'ready':True,'checked_at':time.time()}


def render(study,force=False):
    """Read child reports without two publishers writing the same detail file."""
    global _last_publish
    if not force and time.monotonic()-_last_publish<60:return
    manifest=read_json(study/'study-manifest.json');done={r['run']:r for r in manifest['completed_runs']}
    rows=[];videos=[];health=[]
    for entry in manifest['plan']['baseline']:
        index=entry['state_index'];prefix=f'state{index:03d}'
        run=study/prefix/'trials'/f'baseline-r1-close_microwave-state{index:03d}-600'
        state=read_json(study/prefix/'watch-status.json') if (study/prefix/'watch-status.json').exists() else {}
        timing=read_json(run/'timing-breakdown.json') if (run/'timing-breakdown.json').exists() else {}
        episode=read_json(run/'live-episode-status.json') if (run/'live-episode-status.json').exists() else {}
        result=done.get(entry['run'],{})
        outcome=(('Success' if result.get('official_task_success') else 'Failed') if result else
            'Interrupted' if state.get('status_text')=='Infrastructure interruption' else
            'Running' if state.get('running') else 'Not started' if manifest.get('problem') else 'Queued')
        fmt=lambda v:f'{v:.1f}' if isinstance(v,(float,int)) else '—'
        rows.append(f'<tr><td>Closing {index}</td><td>{outcome}</td><td>{result.get("control_steps",episode.get("control_steps","—"))}</td>'
            f'<td>{fmt(timing.get("actor_wall_seconds"))}</td><td>{fmt(timing.get("service_wall_seconds"))}</td>'
            f'<td>{fmt(timing.get("outside_service_seconds"))}</td><td><a href="{prefix}/watch.html">Live details and full profiling</a></td></tr>')
        movie=run/'attempt01/initial/rollout.mp4'
        if movie.exists():videos.append(f'<h2>Closing {index}</h2><video controls preload="none" src="{movie.relative_to(study)}"></video>')
        if state.get('health_text'):health.append(f'<p>State {index}: {html.escape(state["health_text"])}</p>')
    text='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Current closing checks</title>
<style>body{font:16px system-ui;max-width:1100px;margin:28px auto;padding:0 16px;color:#203249}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}video{max-width:100%;width:512px}.scroll{overflow:auto}button{font:inherit;padding:8px}</style>
<h1>Current-code closing checks</h1><p>Fresh Luna/high, no inherited memory, 600 controls per episode. Two trials only.</p>
<p><button onclick="location.reload()">Refresh overview</button> No automatic page reload.</p>'''
    text+=f'<p>{len(done)} / 2 completed; {sum(bool(r.get("official_task_success")) for r in done.values())} native successes.</p>'
    if manifest.get('phase')=='waiting_for_gpu':
        readiness=read_json(study/'gpu-readiness.json') if (study/'gpu-readiness.json').exists() else {}
        text+='<p>Waiting for GPU initialization. No Luna episode is running. Checks every five minutes; the two trials start automatically after CUDA and EGL initialize.</p>'
        if readiness.get('reason'):text+='<p>Last result: '+html.escape(readiness['reason'])+'</p>'
    if manifest.get('problem'):text+='<p>Stopped: '+html.escape(manifest['problem']['reason'])+'</p>'
    text+='<div class="scroll"><table><tr><th>Case</th><th>Outcome</th><th>Controls</th><th>Total seconds</th><th>Robot tools</th><th>Luna/transport/analysis</th><th>Inspect</th></tr>'+''.join(rows)+'</table></div>'
    text+=''.join(health)+''.join(videos)
    (study/'watch.html').write_text(text+'\n');_last_publish=time.monotonic()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'results/luna-current-closing-checks-20261031')
    parser.add_argument('--wait-for-gpu',action='store_true')
    args=parser.parse_args();study=args.study.resolve()
    existing=study.exists()
    study.mkdir(exist_ok=True);(study/'trials').mkdir(exist_ok=True)
    cases=[(17,1),(49,2)]
    entries=[{'task':'close_microwave','state_index':state,'repeat':repeat,
        'run':f'baseline-r{repeat}-close_microwave-state{state:03d}-600'} for state,repeat in cases]
    frozen=source_hashes()
    manifest=read_json(study/'study-manifest.json') if existing else {'study_directory':str(study),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':frozen,'phase':'baseline','completed_runs':[],'completed':False,
        'plan':{'baseline':entries,'transfer':[],'model':'gpt-6-luna','reasoning':'high',
            'control_steps':600,'episode_seconds':1800,'clean_start':True,'prior_notebooks_supplied':False}}
    if existing:
        if manifest.get('completed'):return
        if manifest['source_sha256']!=frozen:raise RuntimeError('Runtime changed before continuation')
        if not manifest['completed_runs'] and manifest.get('phase')!='waiting_for_gpu':raise RuntimeError('Preserve an unfinished trial; do not restart it automatically')
    manifest.pop('problem',None)
    write_json(study/'study-manifest.json',manifest);write_json(study/'plan.json',manifest['plan'])
    supervisor={'pid':os.getpid(),'status':'running','started_at':time.time()}
    write_json(study/'supervisor-state.json',supervisor)
    child=None
    try:
        if args.wait_for_gpu:
            manifest.update(phase='waiting_for_gpu',active_run=None)
            write_json(study/'study-manifest.json',manifest);render(study,force=True)
            while True:
                if source_hashes()!=frozen:raise RuntimeError('Runtime changed while queued; preserve this study instead of mixing versions')
                readiness=gpu_ready();write_json(study/'gpu-readiness.json',readiness);render(study,force=True)
                if readiness['ready']:break
                # Background launcher only; there is no actor, robot motion or
                # episode deadline while the shared GPU is unavailable.
                for _ in range(30):time.sleep(10)
            manifest['phase']='baseline';write_json(study/'study-manifest.json',manifest)
        for entry in entries:
            if entry['run'] in {r['run'] for r in manifest['completed_runs']}:continue
            if source_hashes()!=frozen:raise RuntimeError('Current runtime changed; stop without mixing versions')
            state=entry['state_index'];nested=study/f'state{state:03d}';nested.mkdir(exist_ok=True)
            internal=f'baseline-r1-close_microwave-state{state:03d}-600'
            if (nested/'trials'/internal).exists():raise RuntimeError('Do not overwrite or repeat an existing trial')
            run_link=study/'trials'/entry['run']
            if not run_link.is_symlink():run_link.symlink_to(nested/'trials'/internal,target_is_directory=True)
            for name in ['active-process.json','resource-monitor.json']:
                link=study/name
                if link.is_symlink() or link.exists():link.unlink()
                link.symlink_to(nested/name)
            # The archived run has its own internal name. The overview uses an
            # explicit alias while retaining the actual process/liveness data.
            manifest['active_run']=entry['run'];write_json(study/'study-manifest.json',manifest)
            command=[sys.executable,str(ROOT/'scripts/supervise_luna_dashboard_study.py'),
                '--study',str(nested),'--stage','first','--first-run',internal,'--log-tag','current-closing-check']
            with (study/f'state{state:03d}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                print(json.dumps({'started':entry['run'],'pid':child.pid}),flush=True)
                while child.poll() is None:
                    # Publish an alias with the real actor's PID/command.
                    active_path=nested/'active-process.json'
                    if active_path.exists():
                        active=read_json(active_path);active['run']=entry['run']
                        link=study/'active-process.json'
                        if link.is_symlink():link.unlink()
                        write_json(link,active)
                    render(study)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:pass
            if child.returncode:raise RuntimeError(f'State {state} monitor stopped with code {child.returncode}; inspect its log')
            inner=read_json(nested/'study-manifest.json')
            if len(inner['completed_runs'])!=1:raise RuntimeError('Expected exactly one completed trial')
            completed={**inner['completed_runs'][0],'run':entry['run'],'scheduled_run':entry['run']}
            manifest['completed_runs'].append(completed);manifest['active_run']=None
            write_json(study/'study-manifest.json',manifest);render(study,force=True)
            print(json.dumps({'completed':entry['run'],'success':completed.get('official_task_success'),
                'controls':completed.get('control_steps')}),flush=True)
        manifest.update(completed=True,baseline_completed=True,phase='complete',finished_at=time.time())
        supervisor.update(status='complete',finished_at=time.time())
    except BaseException as error:
        manifest['problem']={'type':type(error).__name__,'reason':str(error)}
        supervisor.update(status='stopped',reason=str(error),finished_at=time.time())
        raise
    finally:
        if child is not None and child.poll() is None:child.terminate()
        write_json(study/'study-manifest.json',manifest)
        write_json(study/'supervisor-state.json',supervisor)
        render(study,force=True)


if __name__=='__main__':main()
