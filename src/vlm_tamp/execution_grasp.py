"""Complete a planned grasp with physical closure after measured approach arrival."""

from __future__ import annotations

from typing import Any

from .execution_limits import remaining_controls


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
            reason = ('closure_completed' if closed_controls == close_steps else
                      last.get('termination_reason') or
                      ('control_budget_exhausted' if available < close_steps else
                       last['execution'].get('termination_reason', 'closure_incomplete')))
        else:
            reason = 'control_budget_exhausted'
    summary = {
        'automatic_closure': True, 'approach_reached': reached,
        'requested_close_controls': close_steps, 'executed_close_controls': closed_controls,
        'closure_issued': closed_controls > 0, 'closure_completed': closed_controls == close_steps,
        'stop_reason': reason, 'object_retention': 'unverified',
    }
    executions = [stage['execution'] for stage in stages]
    execution = {
        'planned_commands': approach['execution']['planned_commands'] + close_steps,
        'issued_commands': sum(item['issued_commands'] for item in executions),
        'all_commands_issued': reached and closed_controls == close_steps,
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
