"""Official task instructions and identifiers for the robot-tool interface.

These are task instructions and official dataset identifiers, never geometry,
target poses, simulator joint values or recovery policies.
"""
DEFAULT_TASK = 'open_microwave'
TASKS = {
    'open_microwave': {'suite': 'libero_90', 'task_name': 'KITCHEN_SCENE7_open_the_microwave',
                       'instruction': 'open the microwave', 'motion': 'open'},
    'close_microwave': {'suite': 'libero_90', 'task_name': 'KITCHEN_SCENE6_close_the_microwave',
                        'instruction': 'close the microwave', 'motion': 'close'},
    'close_top_drawer': {'suite': 'libero_90',
                         'task_name': 'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet',
                         'instruction': 'close the top drawer of the cabinet', 'motion': 'close'},
    'push_plate': {'suite': 'libero_goal', 'task_name': 'push_the_plate_to_the_front_of_the_stove',
                   'instruction': 'push the plate to the front of the stove', 'motion': None},
    'place_bowl': {'suite': 'libero_spatial',
                   'task_name': 'pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate',
                   'instruction': 'pick up the black bowl from table center and place it on the plate',
                   'motion': None},
}


def task_spec(name=DEFAULT_TASK):
    if name not in TASKS:
        raise ValueError('Unsupported subscription task; choose ' + ', '.join(TASKS))
    return dict(TASKS[name])
