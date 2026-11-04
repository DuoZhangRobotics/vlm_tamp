"""Paired direct-tool Luna pilot across declared task families, under the existing watchdog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from run_restriction_study import source_hashes, validate_terminal
from run_varied_contact_practice import actor_command, audit_memory_input
from select_recorded_goal_references import model_worker
from report_visual_memory import render
from supervise_luna_dashboard_study import wait_for_capacity
from vlm_tamp.coupled_storage import write_json

ROOT=Path(__file__).resolve().parents[1]
PLANNER_REFERENCE_COMMIT='e2c9860'
NUMERICAL_FILES=(
    'src/vlm_tamp/contact_skills.py','src/vlm_tamp/contact_skill_service.py',
    'src/vlm_tamp/graspgen_selection.py','src/vlm_tamp/hinged_contact_tools.py',
    'src/vlm_tamp/surface_contact_tools.py','src/vlm_tamp/timed_execution.py',
    'src/vlm_tamp/mcvamp_contact_numerical.py','src/vlm_tamp/native/mcvamp_contact_numerical.cc',
    'src/vlm_tamp/rekep_libero/task_ik_curobo.py','src/vlm_tamp/rekep_libero/curobo_v2_worker.py',
    'src/vlm_tamp/rekep_libero/curobo_scene_points.py','configs/articulated_libero.json',
    'src/vlm_tamp/profiled_sim_worker.py','src/vlm_tamp/profiled_sim_client.py',
    'src/vlm_tamp/rekep_libero/sim_worker_v3.py','src/vlm_tamp/rekep_libero/sim_worker_v7.py',
    'src/vlm_tamp/current_robot_scene.py','src/vlm_tamp/collision_scene_points.py',
    'src/vlm_tamp/vamp_sphere_model.py','src/vlm_tamp/coupled_scene.py')


def plan_runs(config: dict) -> list[dict]:
    """Fix cases and alternate order before any model or physical outcome is observed."""
    result=[]
    for index,case in enumerate(config['cases']):
        conditions=config['conditions'] if index%2==0 else list(reversed(config['conditions']))
        for condition in conditions:
            result.append({**case,**condition,'condition':condition['name'],
                'run':f"{index+1:02d}-{case['task']}-s{case['state_index']:03d}-{condition['name']}"})
    return result


def main() -> None:
    """Keep every attempted condition and reject infrastructure contamination explicitly."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/study_visual_memory.json')
    args=parser.parse_args();study=args.study.resolve();study.mkdir(exist_ok=True)
    (study/'trials').mkdir()
    config=json.loads(args.config.read_text());frozen=source_hashes()
    numerical={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in NUMERICAL_FILES}
    for name,digest in numerical.items():
        committed=subprocess.check_output(['git','show',PLANNER_REFERENCE_COMMIT+':'+name],cwd=ROOT)
        if hashlib.sha256(committed).hexdigest()!=digest:
            raise RuntimeError('Numerical backend changed outside this interface study: '+name)
    manifest={'config':config,'runs':plan_runs(config),'completed_runs':[],
        'active_run':None,'complete':False,'started_at':time.time(),'source_sha256':frozen,
        'planner_reference_commit':PLANNER_REFERENCE_COMMIT,
        'numerical_sha256':numerical,'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'scope':'One-seed paired pilot of the combined visual and memory changes; not a statistical generalization claim'}
    write_json(study/'visual-memory-manifest.json',manifest);render(study)
    wait_for_capacity(study,'visual-memory-model-load')
    with model_worker(study) as endpoint:
        for entry in manifest['runs']:
            if source_hashes()!=frozen:raise RuntimeError('Frozen study sources changed')
            wait_for_capacity(study,entry['run'])
            out=study/'trials'/entry['run']
            command=actor_command(out,['--task',entry['task'],'--state',str(entry['state_index']),
                '--attempts','1','--agent-seconds',str(config['episode_seconds']),
                '--evaluation-case','--clean-start','--pointing-socket',str(endpoint),
                '--structured-agent','--memory-mode',entry['memory_mode'],
                '--memory-slots',str(config['memory_slots']),
                *(['--visual-interface'] if entry['visual_interface'] else []),
                '--prompt-file',str(ROOT/'prompts/visual_memory_episode.txt')],
                control_steps=config['control_steps'],episode_seconds=config['episode_seconds'])
            manifest['active_run']=entry['run'];write_json(study/'visual-memory-manifest.json',manifest);render(study)
            with (study/(entry['run']+'.log')).open('x') as log:
                process=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
                write_json(study/'active-process.json',{'pid':process.pid,'run':entry['run'],'status':'running'})
                try:
                    while process.poll() is None:time.sleep(2)
                finally:
                    if process.poll() is None:process.terminate();process.wait(timeout=60)
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,command)
            if source_hashes()!=frozen:raise RuntimeError('Sources changed during the episode')
            audit_memory_input(out,None)
            evaluation=json.loads((out/'evaluation-private.json').read_text())
            agent=json.loads((out/'agent-exit.json').read_text())
            attempt=validate_terminal(process.returncode,agent,evaluation,(out/'operator-interruption.json').exists())
            write_json(out/'source-integrity.json',{'sources_unchanged_during_run':True,
                'checked_files':len(frozen),'numerical_backend_unchanged':True})
            manifest['completed_runs'].append({**entry,'attempt':attempt,
                'actor_wall_seconds':agent['wall_seconds']})
            manifest['active_run']=None
            write_json(study/'visual-memory-manifest.json',manifest);render(study)
    manifest.update(complete=True,finished_at=time.time())
    write_json(study/'visual-memory-manifest.json',manifest);render(study,with_resources=True)


if __name__=='__main__':main()
