"""Run the declared Luna study sequentially, with local resource telemetry.

The first full regression is an I/O check. The remaining stage runs regressions
and clean practice. Transfer waits for analyst review of learner memory.
Sustained host pressure interrupts the infrastructure, never scores a task.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
SAMPLE_SECONDS=15


def write(path,value):
    path=Path(path);temp=path.with_name(path.name+'.pending')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def pressure(kind):
    rows={}
    for line in Path('/proc/pressure',kind).read_text().splitlines():
        name,*parts=line.split();rows[name]={k:float(v) for k,v in (p.split('=') for p in parts)}
    return rows


def cgroup_path():
    row=next(line for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    return Path('/sys/fs/cgroup')/row.split('::',1)[1].lstrip('/')


def sample(group):
    io=pressure('io');memory=pressure('memory');pids=[];totals={}
    for value in (group/'cgroup.procs').read_text().split():
        pid=int(value)
        try:
            counters={k:int(v) for k,v in (line.split(':',1) for line in Path(f'/proc/{pid}/io').read_text().splitlines())}
            name=Path(f'/proc/{pid}/comm').read_text().strip()
            args=Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace').split('\0')
            entrypoint=name
            if name.startswith('python'):
                if '-m' in args and args.index('-m')+1<len(args):entrypoint=args[args.index('-m')+1]
                else:entrypoint=next((Path(x).name for x in args[1:] if x.endswith('.py')),name)
            state=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]
            wait=Path(f'/proc/{pid}/wchan').read_text().strip()
        except (OSError,ValueError,IndexError):continue
        pids.append({'pid':pid,'name':name,'entrypoint':entrypoint,'state':state,'wait_channel':wait,'io':counters})
        for k,v in counters.items():totals[k]=totals.get(k,0)+v
    return {'time':time.time(),'io_full_avg10':io.get('full',{}).get('avg10',0),
            'io_some_avg10':io.get('some',{}).get('avg10',0),
            'memory_full_avg10':memory.get('full',{}).get('avg10',0),
            'job_memory_gib':int((group/'memory.current').read_text())/1024**3,
            'job_memory_peak_gib':int((group/'memory.peak').read_text())/1024**3,
            'process_io_sum':totals,'processes':pids,
            'counter_scope':'Current cgroup processes; sums can fall when processes exit. rchar/wchar include IPC, not only disk I/O.'}


def pressure_stop(samples):
    """Stop on recurring observed server strain, not on a task/planner failure."""
    tail=samples[-3:]
    if len(tail)<3:return None
    if all(s['io_full_avg10']>=10 for s in tail):return 'sustained_host_io_pressure'
    if all(s['memory_full_avg10']>=5 for s in tail):return 'sustained_host_memory_pressure'
    return None


def mark_interruption(study,reason):
    """Update the small status records after stopping the process group."""
    active_path=study/'active-process.json'
    active=json.loads(active_path.read_text()) if active_path.exists() else {}
    manifest_path=study/'study-manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    done={r['run'] for r in manifest.get('completed_runs',[])}
    if active.get('run') and active['run'] not in done and (study/'trials'/active['run']).is_dir():
        write(study/'trials'/active['run']/'operator-interruption.json',
              {'reason':reason,'shutdown_requested':True,'excluded_from_completed_task_outcomes':True})
    active.update(status='interrupted',reason=reason);write(active_path,active)
    status_path=study/'watch-status.json'
    if status_path.exists():
        status=json.loads(status_path.read_text());status.update(running=False,checked_at=time.time(),
            status_text='Infrastructure interruption',message=reason,active_run=None)
        write(status_path,status)


def run(study,stages):
    local=Path(tempfile.mkdtemp(prefix='vlm-luna-resource-monitor-'));group=cgroup_path()
    state={'pid':os.getpid(),'started_at':time.time(),'status':'running','local_directory':str(local),
           'sample_interval_seconds':SAMPLE_SECONDS,'stages':stages,
           'stop_policy':'Three consecutive samples with host full I/O pressure >=10% or full memory pressure >=5%. Infrastructure interruption; no automatic retries.'}
    write(study/'resource-monitor.json',{'latest_sample':str(local/'latest.json'),'samples':str(local/'samples.jsonl'),
                                       'supervisor':str(local/'supervisor.json'),'policy':state['stop_policy']})
    write(study/'supervisor-state.json',state);write(local/'supervisor.json',state)
    history=[];child=None;interrupted=None
    def stop(signum,frame):
        nonlocal interrupted
        interrupted='supervisor_signal_'+str(signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for stage,limit in stages:
            state.update(stage=stage,limit=limit,stage_started_at=time.time());write(local/'supervisor.json',state)
            label=stage+('-first' if limit else '-rest' if stage=='baseline' else '')
            command=[sys.executable,str(ROOT/'scripts/run_luna_pipeline_validation.py'),'--study',str(study),'--stage',stage]
            if limit:command+=['--limit',str(limit)]
            with (study/(label+'-driver.log')).open('x') as stream:
                child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state['driver_pid']=child.pid;write(local/'supervisor.json',state)
                while child.poll() is None:
                    row=sample(group);history.append(row);history=history[-3:]
                    write(local/'latest.json',row)
                    with (local/'samples.jsonl').open('a') as samples:samples.write(json.dumps(row)+'\n')
                    interrupted=interrupted or pressure_stop(history)
                    state.update(heartbeat_at=time.time());write(local/'supervisor.json',state)
                    if interrupted:
                        state.update(status='infrastructure_interruption',reason=interrupted)
                        write(local/'supervisor.json',state)
                        try:os.killpg(child.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                        try:child.wait(timeout=20)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                        break
                    try:child.wait(timeout=SAMPLE_SECONDS)
                    except subprocess.TimeoutExpired:pass
                code=child.returncode
            if interrupted or code:
                state.update(status='infrastructure_interruption' if interrupted else 'driver_error',
                             reason=interrupted,returncode=code,finished_at=time.time())
                write(local/'supervisor.json',state);write(study/'supervisor-state.json',state)
                mark_interruption(study,interrupted or f'driver_exit_{code}; inspect driver log')
                return 1
        state.update(status='memory_scope_review' if stages[-1][0]=='practice' else 'complete',finished_at=time.time())
        write(local/'supervisor.json',state);write(study/'supervisor-state.json',state)
        return 0
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True,type=Path)
    p.add_argument('--stage',choices=['first','remaining','transfer'],default='first')
    a=p.parse_args();study=a.study.resolve()
    if not study.is_relative_to(ROOT/'results'):raise ValueError('Use a study under results/')
    study.mkdir(parents=True,exist_ok=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3' or os.environ.get('MUJOCO_EGL_DEVICE_ID')!='3':raise ValueError('GPU 3 only')
    stages={'first':[('baseline',1)],'remaining':[('baseline',None),('practice',None)],'transfer':[('transfer',None)]}[a.stage]
    raise SystemExit(run(study,stages))


if __name__=='__main__':main()
