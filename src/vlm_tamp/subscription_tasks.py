"""Public task selection for the existing upright-door subscription interface.

These are task instructions and official dataset identifiers, never geometry,
target poses, simulator joint values or recovery policies.
"""
DEFAULT_TASK = 'open_microwave'
TASKS = {
    'open_microwave': {'suite': 'libero_90', 'task_name': 'KITCHEN_SCENE7_open_the_microwave',
                       'instruction': 'open the microwave', 'motion': 'open'},
    'close_microwave': {'suite': 'libero_90', 'task_name': 'KITCHEN_SCENE6_close_the_microwave',
                        'instruction': 'close the microwave', 'motion': 'close'},
}


def task_spec(name=DEFAULT_TASK):
    if name not in TASKS:
        raise ValueError('Unsupported subscription task; choose ' + ', '.join(TASKS))
    return dict(TASKS[name])
