"""Start a fresh actor at an exactly replayed physical command boundary.

Only current public observations are published. The prefix is a declared test
initialization, not a suggested action recipe and not a simulator state edit.
"""
import argparse
from pathlib import Path
import sys
import numpy as np

import serve_subscription_robot as backend
from replay_subscription_episode import source_commands,source_reference_mode
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.profiled_sim_client import execute_batch


def install_prefix(source,checkpoint):
    source=Path(source).resolve();protocol=read_json(source/'protocol.json')
    commands=source_commands(source,1);prefix=[];steps=60
    for command in commands:
        if steps==checkpoint:break
        if steps+len(command['references'])>checkpoint:
            raise ValueError('Checkpoint must end a complete original issued trajectory')
        prefix.append(command);steps+=len(command['references'])
    if steps!=checkpoint:raise ValueError('No original command boundary at the requested checkpoint')
    references=[]
    for path in (source/'agent-workspace/robot_data/attempt01').glob('obs*/robot.json'):
        row=read_json(path)
        if row['control_steps']==checkpoint:references.append(row)
    if not references:raise ValueError('A recorded public robot-state checkpoint is required')
    expected=np.asarray(references[0]['joint_positions']);original=backend.RobotService.reset

    def reset(service):
        case=protocol['case']
        if service.state_index!=case['state_index'] or service.task!=case.get('task',service.task):
            raise ValueError('Replay source task/state must match the declared case')
        capture=service.capture
        # The original reset still captures its private calibration inputs.
        # Publish only the final current observation to this fresh actor.
        service.capture=lambda **kwargs:{}
        try:original(service)
        finally:service.capture=capture
        if service.sim is None:raise ValueError('The episode ended before prefix replay')
        if hasattr(service.sim,'set_joint_reference_mode'):
            service.state=service.sim.set_joint_reference_mode(source_reference_mode(source))
        rows=[]
        for command in prefix:
            service.calls+=1;label=f'initial-prefix-{command["event"]:03d}'
            refs=command['references']
            service.state,result=execute_batch(service.sim,service.state,refs,gripper=command['gripper'],
                event_sink=service.events.append,stop_on_success=service.stop_on_success)
            np.save(service.episode/f'{service.calls:03d}-requested-{label}.npy',refs)
            np.save(service.episode/f'{service.calls:03d}-issued-{label}.npy',refs[:result['issued_commands']])
            service.executions.append({'label':label,**{k:v for k,v in result.items() if k!='events'}})
            rows.append({'source_event':command['event'],'issued_controls':result['issued_commands']})
            if not result['all_commands_issued']:raise ValueError('Prefix ended before its declared checkpoint')
        error=float(np.max(abs(np.asarray(service.state['metadata']['joint_positions'])-expected)))
        write_json(service.output/'initial-prefix.json',{'checkpoint_control_step':checkpoint,
            'source':str(source),'source_attempt':1,'command_replay':rows,'maximum_joint_error_rad':error,
            'simulator_state_assignment':False,'actor_received_historical_regions_or_paths':False})
        write_json(service.episode/'executions.json',service.executions)
        if error>1e-6:raise ValueError('Physical replay did not reproduce the recorded robot checkpoint')
        if hasattr(service.sim,'set_joint_reference_mode'):
            service.state=service.sim.set_joint_reference_mode(getattr(service,'reference_mode','slew_limited'))
        service.calls+=1
        return {'instruction':service.task_spec['instruction'],'attempt':service.attempt,
            'initialization':'recorded physical prefix; choose your actions from current observations',**service.capture()}
    backend.RobotService.reset=reset


if __name__=='__main__':
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--prefix-source',required=True)
    parser.add_argument('--prefix-controls',type=int,default=459)
    args,rest=parser.parse_known_args();install_prefix(args.prefix_source,args.prefix_controls)
    sys.argv=[sys.argv[0],*rest];backend.main()
