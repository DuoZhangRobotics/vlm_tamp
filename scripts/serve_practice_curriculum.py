"""Run a declared private training curriculum in one persistent actor session.

The actor receives only each reset's task instruction and current observations.
No state indices, successful regions, trajectories or private outcomes are hints.
"""
import argparse
import json
from pathlib import Path
import sys
import serve_subscription_robot as backend
from vlm_tamp.subscription_tasks import TASKS,task_spec


def checked_cases(path):
    cases=json.loads(Path(path).read_text())
    if not isinstance(cases,list) or not cases:raise ValueError('Use a nonempty curriculum list')
    for row in cases:
        if set(row)!={'task','state_index'} or row['task'] not in TASKS:
            raise ValueError('Each curriculum case needs a supported task and state_index')
        if type(row['state_index']) is not int or not 0<=row['state_index']<50:
            raise ValueError('Official initial-state indices must be integers in 0..49')
    return cases


def install_curriculum(cases):
    original=backend.RobotService.reset
    def reset(service):
        if not service.learning_practice:raise ValueError('Curriculum reset is only available in declared practice')
        status=service.episode_status()
        if service.attempt>=service.limit or (service.sim is not None and not status['episode_done']):
            return original(service)
        # Finish an old wall-expired episode before changing its private labels.
        if service.sim is not None:service.finish_attempt(status['termination_reason'] or 'curriculum_transition')
        case=cases[service.attempt%len(cases)]
        service.task=case['task'];service.task_spec=task_spec(case['task']);service.state_index=case['state_index']
        return original(service)
    backend.RobotService.reset=reset


if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--curriculum-file',required=True)
    args,remaining=p.parse_known_args();install_curriculum(checked_cases(args.curriculum_file))
    sys.argv=[sys.argv[0],*remaining];backend.main()
