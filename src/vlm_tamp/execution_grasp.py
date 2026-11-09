"""Complete a planned grasp with physical closure after measured approach arrival."""

from __future__ import annotations

from typing import Any

from .execution_limits import remaining_controls
from .feedback_measured import AFTER_CLOSURE


def require_new_grasp(service: Any, regrasp: bool = False) -> None:
    """Keep a closing/closed hand intact unless the caller explicitly discards that attempt."""
    closure = getattr(service, 'gripper_closure', None)
    if closure is not None and not regrasp:
        state = 'Fingers settled.' if closure['settled'] else 'Closing is still in progress; continue closing.'
        error = ValueError(state+' The previous closing attempt is preserved. '+
                           (AFTER_CLOSURE+' ' if closure['settled'] else '')+
                           'To release and retry, open explicitly or use regrasp=true on plan_grasp/grasp_object.')
        error.planning_diagnostic = {'status': 'explicit_regrasp_required',
                                    'physical_motion_issued': False,
                                    'interpretation': str(error), 'gripper_completion': closure}
        raise error


def finish_grasp(service: Any, approach: dict, close_steps: int) -> dict:
    """Close at the reached approach pose and retain separate evidence for both stages."""
    stages = []

    def record(name: str, reply: dict) -> None:
        """Keep distinct observation artifacts and control ranges for each physical stage."""
        stages.append({'stage': name, 'artifact_id': service.calls,
                       'control_steps': service.state['control_steps'], **reply})

    record('grasp-approach', approach)
    arrival = approach['execution']['arrival']
    reached = bool(arrival['goal_reached'])
    status = service.episode_status()
    last = approach
    closed_controls = 0
    completion = None
    if approach.get('episode_done') or status.get('episode_done'):
        reason = approach.get('termination_reason') or status['termination_reason']
    elif not reached:
        reason = 'approach_not_reached'
    else:
        available = min(close_steps, remaining_controls(service.state))
        if available:
            service.calls += 1
            last = service.gripper(close=True, steps=available)
            record('gripper-close', last)
            closed_controls = last['execution']['issued_commands']
            completion = last['execution']['gripper_completion']
            reason = ('closure_completed' if completion['settled'] and closed_controls >= close_steps else
                      'control_budget_exhausted' if available < close_steps else
                      last['execution']['termination_reason'])
        else:
            reason = 'control_budget_exhausted'
    summary = {
        'automatic_closure': True, 'approach_reached': reached,
        'requested_close_controls': close_steps, 'executed_close_controls': closed_controls,
        'closure_issued': closed_controls > 0,
        'closure_completed': bool(completion and completion['settled'] and closed_controls >= close_steps),
        'closing_interval_completed': closed_controls >= close_steps,
        'mechanical_completion': completion,
        'stop_reason': reason,
        'lift_executed': False,
    }
    if summary['closure_completed'] and not last.get('episode_done') and not last.get('task_success'):
        summary['next_action'] = AFTER_CLOSURE
    executions = [stage['execution'] for stage in stages]
    execution = {
        'planned_commands': approach['execution']['planned_commands'] + close_steps,
        'issued_commands': sum(item['issued_commands'] for item in executions),
        'all_commands_issued': reached and summary['closure_completed'],
        'termination_reason': reason,
        'arrival': arrival, 'arrival_scope': 'grasp_approach_before_closure',
        'terminated_on_simulator_success': any(
            item.get('terminated_on_simulator_success', False) for item in executions),
    }
    for key in ('wall_seconds', 'worker_seconds'):
        if all(key in item for item in executions):
            execution[key] = sum(item[key] for item in executions)
    comparisons = [{**stage['visual_comparison'], 'stage': stage['stage'],
                    'end_control_step': stage['control_steps']}
                   for stage in stages if 'visual_comparison' in stage]
    return {**last, 'execution': execution, 'stages': stages, 'grasp_execution': summary,
            'stop_reason': reason, 'phase_visual_comparisons': comparisons}
