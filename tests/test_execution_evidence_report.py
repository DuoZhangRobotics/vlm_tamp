"""Dashboard evidence must not turn a tool name or accepted request into manipulation."""

from report_visual_memory import executed_behavior


def event(number: int, operation: str, result: dict, arguments: dict | None = None) -> dict:
    """Construct a recorded tool event without a simulator."""
    return {'sequence': number, 'request': {'operation': operation, 'arguments': arguments or {}},
            'response': {'ok': True, 'result': result}}


def test_rejected_or_blocked_push_does_not_count_as_executed_stroke() -> None:
    """A solved command or an approach can end without a manipulation stroke."""
    events = [event(1, 'push', {'stop_reason': 'contact_ik_failed'}),
              event(2, 'push', {'stages': [{'stage': 'contact-approach', 'control_steps': 70,
                  'execution': {'issued_commands': 10, 'termination_reason': 'reference_not_reached',
                                'arrival': {'goal': {'reached': False, 'position_error_m': .04}}}}]}),
              event(3, 'push', {'preview': True}, {'preview': True})]
    actions = [{'step': step, 'control': [0] * 8} for step in range(61, 71)]
    result = executed_behavior(events, actions, 60)
    assert result['push_requests_for_motion'] == 2
    assert result['push_requests_without_robot_motion'] == 1
    assert result['push_requests_that_reached_a_stroke'] == 0
    assert result['push_previews'] == 1
    assert result['motion_stages'][0]['goal_error_mm'] == 40
    assert result['controls_commanding_close'] == 0


def test_gripper_motor_commands_are_counted_even_without_a_named_close_tool() -> None:
    """An execute command can close fingers; a gap reading is separate evidence."""
    events = [event(1, 'execute', {'execution': {'issued_commands': 2},
        'observation': {'control_steps': 62}, 'grasp_evidence': {'pad_gap_mm': 3.6}}),
              event(2, 'push_stroke', {'stages': [{'stage': 'push-stroke-1', 'control_steps': 65,
                  'execution': {'issued_commands': 3, 'termination_reason': 'trajectory_complete'}}]})]
    actions = [{'step': step, 'control': [0] * 7 + [1 if step <= 62 else 0]}
               for step in range(61, 66)]
    result = executed_behavior(events, actions, 60)
    assert result['explicit_close_requests_executed'] == 0
    assert result['controls_commanding_close'] == 2
    assert result['direct_hand_stroke_stages'] == 1
    assert result['push_requests_that_reached_a_stroke'] == 0
    assert result['jaw_gap_samples'] == [{'control': 62, 'gap_mm': 3.6}]


def test_automatic_grasp_closure_is_visible_separately_from_manual_requests() -> None:
    """One execute request contains two stages without counting either twice."""
    events = [event(5, 'execute', {'grasp_execution': {'automatic_closure': True},
        'execution': {'issued_commands': 5}, 'observation': {'control_steps': 65},
        'stages': [{'stage': 'grasp-approach', 'control_steps': 63,
                    'execution': {'issued_commands': 3}},
                   {'stage': 'gripper-close', 'control_steps': 65,
                    'execution': {'issued_commands': 2}}]})]
    actions = [{'step': step, 'control': [0] * 7 + [-1 if step <= 63 else 1]}
               for step in range(61, 66)]
    result = executed_behavior(events, actions, 60)
    assert result['automatic_grasp_closures_executed'] == 1
    assert result['explicit_close_requests_executed'] == 0
    assert result['controls_commanding_close'] == 2
    assert sum(stage['issued_controls'] for stage in result['motion_stages']) == 5
