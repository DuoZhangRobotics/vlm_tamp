"""Locate planning failures without importing simulator evaluation signals."""
from collections import Counter
from pathlib import Path
from .coupled_storage import read_json

def failure_loci(folder):
    loci = Counter()
    mapping = {'observed_robot_collision_at_start':'start_collision',
        'no_robot_valid_ik':'goal_ik', 'no_terminal_ik':'goal_ik',
        'requested_posture_unavailable':'posture_unavailable',
        'terminal_projection':'goal_constraint',
        'contact_pose_collision':'goal_collision','open_grasp_scene_collision':'goal_collision',
        'closure_scene_collision':'goal_collision','no_handle_surface_inside_finger_aperture':'contact_geometry'}
    def visit(value):
        if isinstance(value,list):
            for row in value: visit(row)
        elif isinstance(value,dict):
            reason=value.get('reason'); status=value.get('status')
            if reason in mapping:loci[mapping[reason]]+=1
            if status=='invalid_endpoint':
                for name,check in value.get('endpoints',{}).items():
                    if check.get('accepted') is False:loci[f'{name}_collision']+=1
            elif status=='endpoint_constraint_mismatch':loci['endpoint_constraint']+=1
            elif value.get('exact_solution') is False:loci['path_search']+=1
            for key in ('attempts','rejected_candidates','planning','native_planning'):
                if key in value:visit(value[key])
    for name in ('contact-mode-admission.json','grasp-admission.json','planning.json','retreat-planning.json'):
        path=Path(folder)/name
        if path.exists():visit(read_json(path))
    return dict(loci)
