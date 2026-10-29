"""Per-episode control budget shared by service, planners and simulator RPCs."""
DEFAULT_CONTROL_STEPS = 600
LEARNING_CONTROL_STEPS = 1200
CONTROL_HZ = 20.


def checked_control_steps(value):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError('maximum_control_steps must be a positive integer')
    return value


def control_limit(state=None, config=None):
    for source in (state, config):
        if isinstance(source, dict) and 'maximum_control_steps' in source:
            return checked_control_steps(source['maximum_control_steps'])
    return DEFAULT_CONTROL_STEPS


def remaining_controls(state, config=None):
    return max(0, control_limit(state, config)-int(state.get('control_steps', 0)))
