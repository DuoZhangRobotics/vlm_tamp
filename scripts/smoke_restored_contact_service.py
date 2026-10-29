"""One declared tool-interface smoke; no agent trial or recovery policy.

Uses recorded RGB regions on the practice state, then requests one grasp and
one five-degree probe. Normal planning, controller, physics, and success checks
remain active. An unsuccessful call ends this interface check and is reported.
"""
import argparse
import json
from pathlib import Path
import time
import traceback

from serve_subscription_robot import RobotService
from replay_subscription_episode import compress_observations
from vlm_tamp.coupled_storage import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / 'public'
    workspace.mkdir()
    regions = json.loads(args.regions.read_text())
    service = RobotService(output, workspace, state_index=1, attempts=1, stop_on_success=True,
                           fit_diagnostics=True, surface_contact_tools=True)
    result = {'scope': 'scripted development interface smoke, excluded from policy success rates',
              'state_index': 1, 'agent_model_calls': 0, 'events': [], 'error': None}

    def call(operation, arguments=None):
        before = time.monotonic()
        answer = service.dispatch(operation, arguments or {})
        row = {'sequence': service.calls, 'operation': operation, 'arguments': arguments or {},
               'wall_seconds': time.monotonic() - before, 'result': answer}
        result['events'].append(row)
        write_json(output / 'interface-events.json', result['events'])
        print(json.dumps({'operation': operation, 'seconds': row['wall_seconds']}), flush=True)
        return answer

    try:
        call('reset')
        call('ground', {'regions': regions})
        grasp = call('plan_grasp', {'probe_degrees': 3, 'max_candidates': 24, 'seed': 0})
        call('execute', {'plan_id': grasp['plan_id'], 'gripper': -1, 'speed': .35, 'hold': 6})
        if not service.episode_success:
            call('gripper', {'close': True, 'steps': 40})
        if not service.episode_success:
            call('start_tracking')
            arc = call('plan_arc', {'degrees': 5, 'grasp_constraint': 'point'})
            probe = call('execute', {'plan_id': arc['plan_id'], 'gripper': 0, 'speed': .35, 'hold': 6})
            update = probe.get('scene_pose_update')
            if not service.episode_success:
                assert update and update['source'] in {'rgbd_rigid_motion', 'hand_motion_prediction'}
                assert update['assumes_contact_retention'] == (update['source'] == 'hand_motion_prediction')
        result['interface_completed'] = True
    except Exception as error:
        result.update(interface_completed=False, error=f'{type(error).__name__}: {error}')
        (output / 'error.txt').write_text(traceback.format_exc())
    finally:
        service.finish_attempt('declared_interface_smoke_complete')
        result['lossless_sensor_compression'] = compress_observations(output)
        write_json(output / 'summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('events', 'lossless_sensor_compression')}), flush=True)
    if result['error']:
        raise RuntimeError(result['error'])


if __name__ == '__main__':
    main()
