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
import threading
import queue
import time

ROOT=Path(__file__).resolve().parents[1]
SAMPLE_SECONDS=2
DETAIL_SECONDS=15
MAX_JOB_MEMORY_BYTES=32*1024**3


def write(path,value):
    path=Path(path);temp=path.with_name(path.name+'.pending')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def pressure(kind):
    return pressure_file(Path('/proc/pressure',kind))


def pressure_file(path):
    rows={}
    for line in Path(path).read_text().splitlines():
        name,*parts=line.split();rows[name]={k:float(v) for k,v in (p.split('=') for p in parts)}
    return rows


def cgroup_path():
    row=next(line for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    return Path('/sys/fs/cgroup')/row.split('::',1)[1].lstrip('/')


def nfs_counters():
    result={};mount=None
    for line in Path('/proc/self/mountstats').read_text().splitlines():
        if line.startswith('device '):
            mount=next((p for p in ['/common/users','/common/home'] if ' mounted on '+p+' ' in line),None)
            if mount:result[mount]={}
        elif mount and ':' in line:
            operation,values=line.strip().split(':',1)
            if operation in {'READ','WRITE','COMMIT'}:
                result[mount][operation]=[int(x) for x in values.split()]
    return result


def process_details(group):
    """Optional diagnostics: /proc PID inspection must never block the watchdog."""
    pids=[];totals={}
    for value in (group/'cgroup.procs').read_text().split():
        pid=int(value)
        try:
            counters={k:int(v) for k,v in (line.split(':',1) for line in Path(f'/proc/{pid}/io').read_text().splitlines())}
            name=Path(f'/proc/{pid}/comm').read_text().strip()
            # Avoid reading target address spaces (including cmdline) here.
            fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
            state=fields[0];parent=int(fields[1])
            wait=Path(f'/proc/{pid}/wchan').read_text().strip()
        except (OSError,ValueError,IndexError):continue
        pids.append({'pid':pid,'ppid':parent,'name':name,'state':state,'wait_channel':wait,'io':counters})
        for k,v in counters.items():totals[k]=totals.get(k,0)+v
    return {'time':time.time(),'process_io_sum':totals,'processes':pids,'nfs_counters':nfs_counters()}


def sample(group):
    started=time.monotonic();io=pressure('io');memory=pressure('memory')
    own={k:int(v) for k,v in (line.split() for line in (group/'memory.stat').read_text().splitlines())}
    events={k:int(v) for k,v in (line.split() for line in (group/'memory.events').read_text().splitlines())}
    own_pressure=pressure_file(group/'memory.pressure')
    pool=Path('/sys/fs/cgroup/user.slice');headroom=None
    if (pool/'memory.high').exists():
        high=(pool/'memory.high').read_text().strip()
        if high!='max':
            stats={k:int(v) for k,v in (line.split() for line in (pool/'memory.stat').read_text().splitlines())}
            clean=max(0,stats.get('inactive_file',0)-stats.get('file_dirty',0)-stats.get('file_writeback',0))
            headroom=(int(high)-int((pool/'memory.current').read_text())+clean)/1024**3
    available=next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))/1024**2
    return {'time':time.time(),'io_full_avg10':io.get('full',{}).get('avg10',0),
            'io_some_avg10':io.get('some',{}).get('avg10',0),
            'memory_full_avg10':memory.get('full',{}).get('avg10',0),
            'job_memory_full_avg10':own_pressure.get('full',{}).get('avg10',0),
            'job_memory_max_events':events.get('max',0),
            'job_memory_oom_kills':events.get('oom_kill',0),
            'job_memory_gib':int((group/'memory.current').read_text())/1024**3,
            'job_memory_peak_gib':int((group/'memory.peak').read_text())/1024**3,
            'job_anon_and_shmem_gib':(own.get('anon',0)+own.get('shmem',0))/1024**3,
            'shared_headroom_including_clean_inactive_file_gib':headroom,
            'host_available_memory_gib':available,
            'health_read_seconds':time.monotonic()-started}


def capacity_status(row,limit):
    # Subtract only already-charged anonymous/tmpfs memory. Subtracting clean
    # file cache as well would double-count it as available shared headroom.
    needed=max(0,limit/1024**3-row.get('job_anon_and_shmem_gib',0))+2.
    shared=row.get('shared_headroom_including_clean_inactive_file_gib')
    ready=(row['host_available_memory_gib']>=needed and (shared is None or shared>=needed)
           and row['io_full_avg10']<10 and row['memory_full_avg10']<5)
    return {'waiting':not ready,'additional_headroom_needed_gib':needed,
            'shared_headroom_gib':shared,'host_available_gib':row['host_available_memory_gib'],
            'recorded_at':row['time']}


def wait_for_capacity(study,name):
    """Wait before starting an actor, so waiting consumes no episode budget."""
    group=cgroup_path();limit=require_memory_limit(group)
    local=Path(os.environ['VLM_TAMP_CAPACITY_STATE']);last_publish=0
    while True:
        status={**capacity_status(sample(group),limit),'run':name}
        write(local,status)
        if not status['waiting'] or time.monotonic()-last_publish>=60:
            write(Path(study)/'capacity-wait.json',status)
            from report_luna_pipeline_validation import render
            render(study,force=True);last_publish=time.monotonic()
        if not status['waiting']:return
        time.sleep(5)


def require_memory_limit(group):
    limit=(group/'memory.max').read_text().strip()
    swap=(group/'memory.swap.max').read_text().strip()
    whole_job=(group/'memory.oom.group').read_text().strip()
    if limit=='max' or int(limit)>MAX_JOB_MEMORY_BYTES or swap!='0' or whole_job!='1':
        raise RuntimeError('Launch in a systemd service with MemoryMax=32G, MemorySwapMax=0 and OOMPolicy=kill; experiments must not run with unbounded host memory.')
    return int(limit)


class ProcessSampler:
    """One daemon may inspect processes; the health loop never waits for it."""
    def __init__(self,group):
        self.group=group;self.queue=queue.Queue(maxsize=1);self.stopped=threading.Event();self.latest={}
        self.thread=threading.Thread(target=self._work,daemon=True)
        self.thread.start()

    def _work(self):
        while not self.stopped.is_set():
            try:value=process_details(self.group)
            except OSError as error:value={'time':time.time(),'error':str(error)}
            try:self.queue.get_nowait()
            except queue.Empty:pass
            self.queue.put_nowait(value)
            self.stopped.wait(DETAIL_SECONDS)

    def read(self):
        try:self.latest=self.queue.get_nowait()
        except queue.Empty:pass
        return self.latest

    def close(self):
        self.stopped.set()  # Never join a potentially blocked /proc reader.


def pressure_stop(samples):
    """Stop on recurring observed server strain, not on a task/planner failure."""
    tail=samples[-3:]
    if len(tail)<3:return None
    if all(s['io_full_avg10']>=10 for s in tail):return 'sustained_host_io_pressure'
    if all(max(s['memory_full_avg10'],s.get('job_memory_full_avg10',0))>=5 for s in tail):
        shortage=any(s.get('host_available_memory_gib',0)<8
                     or (s.get('shared_headroom_including_clean_inactive_file_gib') is not None
                         and s['shared_headroom_including_clean_inactive_file_gib']<2)
                     for s in tail)
        hit_limit=any(s.get('job_memory_max_event_delta',0)>0 for s in tail)
        if shortage or hit_limit:return 'sustained_memory_pressure_with_shortage_or_job_limit'
    return None


def qualification_passed(stats):
    return stats.get('samples',0)>=3 and stats.get('maximum_sample_gap_seconds',float('inf'))<=10


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


def run(study,stages,*,first_run=None,log_tag=None):
    group=cgroup_path();limit=require_memory_limit(group)
    local=Path(tempfile.mkdtemp(prefix='vlm-luna-resource-monitor-'));details=ProcessSampler(group)
    state={'pid':os.getpid(),'started_at':time.time(),'status':'running','local_directory':str(local),
           'sample_interval_seconds':SAMPLE_SECONDS,'detail_interval_seconds':DETAIL_SECONDS,
           'memory_limit_bytes':limit,'stages':stages,'first_run':first_run,'log_tag':log_tag,
           'stop_policy':'Three consecutive full I/O pressure samples >=10%; or memory pressure >=5% with host/shared-pool shortage or new job-limit hits. Kernel memory/whole-group OOM limits remain mandatory. No automatic retries.'}
    write(study/'resource-monitor.json',{'latest_sample':str(local/'latest.json'),'samples':str(local/'samples.jsonl'),
                                       'supervisor':str(local/'supervisor.json'),'policy':state['stop_policy']})
    write(study/'supervisor-state.json',state);write(local/'supervisor.json',state)
    history=[];child=None;interrupted=None;last_max_events=0
    def stop(signum,frame):
        nonlocal interrupted
        interrupted='supervisor_signal_'+str(signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        for stage,limit in stages:
            state.update(stage=stage,limit=limit,stage_started_at=time.time());write(local/'supervisor.json',state)
            health={'samples':0,'maximum_sample_gap_seconds':0.,'peak_job_memory_gib':0.,
                    'maximum_io_full_avg10':0.,'maximum_memory_full_avg10':0.};last_active_sample=None
            label=stage+('-first' if limit else '-rest' if stage=='baseline' else '')
            if log_tag:label+='-'+log_tag
            state['driver_log']=str(study/(label+'-driver.log'))
            command=[sys.executable,str(ROOT/'scripts/run_luna_pipeline_validation.py'),'--study',str(study),'--stage',stage]
            if limit:command+=['--limit',str(limit)]
            if stage=='baseline' and limit==1 and first_run:command+=['--baseline-run',first_run]
            with (study/(label+'-driver.log')).open('x') as stream:
                capacity_file=local/'capacity.json';write(capacity_file,{'waiting':True})
                environment={**os.environ,'VLM_TAMP_CAPACITY_STATE':str(capacity_file)}
                child=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state['driver_pid']=child.pid;write(local/'supervisor.json',state)
                while child.poll() is None:
                    row=sample(group);extra=details.read()
                    row['job_memory_max_event_delta']=max(0,row['job_memory_max_events']-last_max_events)
                    last_max_events=row['job_memory_max_events']
                    row.update(processes=extra.get('processes',[]),process_io_sum=extra.get('process_io_sum',{}),
                               nfs_counters=extra.get('nfs_counters',{}),
                               nfs_sample_time=extra.get('time'),
                               process_details_age_seconds=row['time']-extra['time'] if extra.get('time') else None,
                               counter_scope='Asynchronous process diagnostics may be stale. rchar/wchar include IPC; sums can fall when processes exit.')
                    history.append(row);history=history[-3:]
                    write(local/'latest.json',row)
                    with (local/'samples.jsonl').open('a') as samples:samples.write(json.dumps(row)+'\n')
                    admission=json.loads(capacity_file.read_text())
                    if not admission.get('waiting'):
                        health['samples']+=1
                        if last_active_sample is not None:
                            health['maximum_sample_gap_seconds']=max(health['maximum_sample_gap_seconds'],row['time']-last_active_sample)
                        last_active_sample=row['time']
                        health['peak_job_memory_gib']=max(health['peak_job_memory_gib'],row['job_memory_peak_gib'])
                        health['maximum_io_full_avg10']=max(health['maximum_io_full_avg10'],row['io_full_avg10'])
                        health['maximum_memory_full_avg10']=max(health['maximum_memory_full_avg10'],row['memory_full_avg10'])
                    interrupted=interrupted or (pressure_stop(history) if not admission.get('waiting') else None)
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
            if stage=='baseline' and limit==1:
                health.update(passed=qualification_passed(health),normal_driver_completion=True,
                              resource_interruption=False,qualification_run=first_run,
                              task_success_required=False,memory_limit_bytes=state['memory_limit_bytes'])
                write(study/'resource-qualification.json',health)
                if not health['passed']:
                    state.update(status='qualification_review_required',finished_at=time.time())
                    write(local/'supervisor.json',state);write(study/'supervisor-state.json',state)
                    return 1
        state.update(status='memory_scope_review' if stages[-1][0]=='practice' else 'complete',finished_at=time.time())
        write(local/'supervisor.json',state);write(study/'supervisor-state.json',state)
        return 0
    finally:
        details.close()
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True,type=Path)
    p.add_argument('--stage',choices=['first','remaining','transfer','qualify-and-continue'],default='first')
    p.add_argument('--first-run',help='Declared baseline run to use for the first resource qualification')
    p.add_argument('--log-tag',help='Distinct orchestration log name for an investigated continuation')
    a=p.parse_args();study=a.study.resolve()
    if not study.is_relative_to(ROOT/'results'):raise ValueError('Use a study under results/')
    study.mkdir(parents=True,exist_ok=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='3' or os.environ.get('MUJOCO_EGL_DEVICE_ID')!='3':raise ValueError('GPU 3 only')
    stages={'first':[('baseline',1)],'remaining':[('baseline',None),('practice',None)],
            'transfer':[('transfer',None)],'qualify-and-continue':[('baseline',1),('baseline',None),('practice',None)]}[a.stage]
    if a.first_run and a.stage not in {'first','qualify-and-continue'}:raise ValueError('--first-run requires an initial qualification stage')
    if a.log_tag and (not a.log_tag.replace('-','').replace('_','').isalnum()):raise ValueError('Use a simple log tag')
    raise SystemExit(run(study,stages,first_run=a.first_run,log_tag=a.log_tag))


if __name__=='__main__':main()
