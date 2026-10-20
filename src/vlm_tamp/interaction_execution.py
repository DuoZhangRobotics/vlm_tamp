"""Explicit capability adapters from shared measured fits to existing planners."""
import copy
import numpy as np
from scipy.spatial import cKDTree

from .interaction_contract import backend_for, InteractionError
from .interaction_geometry import robot_policy
from .mechanism_surface import SurfaceScene as BaseSurfaceScene


def execution_backend(grounded):
    backend=backend_for(grounded.selection)
    if backend!='hinge':return backend
    model=grounded.evidence.get('contact_model',{})
    # The mature partial-pose tracker has explicit upright assumptions. Other
    # orientations use the existing general 3-D full-pose mechanism planner.
    hinge=np.asarray(grounded.spec['joint_axis_world']);bar=np.asarray(model.get('axis_world',[0,0,0]))
    if ('fingers_on_local_moving_surface' in grounded.selection['contact_roles']
            and abs(hinge[2])>np.cos(np.deg2rad(3)) and abs(bar[2])>np.cos(np.deg2rad(3))):
        return 'upright_bar_hinge'
    return 'mechanism'


def hinge_inputs(grounded):
    if execution_backend(grounded)!='upright_bar_hinge':
        raise InteractionError('Upright partial-pose executor cannot accept this fitted mechanism')
    spec=copy.deepcopy(grounded.spec);model=grounded.evidence['contact_model'];selection=grounded.selection
    spec['measured_joint_axis_world']=spec['joint_axis_world']
    spec['joint_axis_world']=[0.,0.,float(np.sign(spec['joint_axis_world'][2]))]
    normal=np.asarray(spec['panel_normal_world']);normal[2]=0;normal/=np.linalg.norm(normal)
    spec['panel_normal_world']=normal.tolist();spec['front_plane']['normal_world']=normal.tolist()
    spec['observed_handle_radius_m']=model['radius_m']
    evidence={**grounded.evidence,'cylinder_radius_m':model['radius_m'],
        'executor_capability':'upright hinge and bar within 3 degrees; projected onto declared upright model',
        'upright_projection_applied':True}
    choices={'camera':selection['camera'],'moving_panel_box_xyxy':selection['moving']['box_xyxy'],
        'contact_box_xyxy':selection['contact']['box_xyxy'],'grasp_pixel_xy':selection['contact']['pixel_xy'],
        'hinge_pixels_xy':selection['mechanism']['axis_pixels_xy'],'motion':selection['mechanism']['motion'],
        'initial_image_direction_xy':selection['mechanism']['initial_image_direction_xy']}
    goal=float(spec['nominal_target_progress'])
    goal_evidence={'source':'VLM-selected goal relation fitted from current depth','simulator_scene_state_used':False}
    if spec['motion']=='close':goal_evidence['selected']={'normal_world':spec['terminal_normal_world']}
    anchor=np.asarray(model['origin_world_m'])
    return spec,grounded.cloud,grounded.panel_mask,[anchor],evidence,choices,goal,goal_evidence


class InteractionSurfaceScene(BaseSurfaceScene):
    """Compile contact roles with robot-scaled bounds and sensor uncertainty."""
    def __init__(self,cameras,metadata,spec,cloud,panel_mask,config,**kwargs):
        super().__init__(cameras,metadata,spec,cloud,panel_mask,config,**kwargs)
        policy=robot_policy(metadata);radius=policy['hand_clearance_m']/2;band=policy['contact_association_m']
        local_enabled='fingers_on_local_moving_surface' in spec['selected_contact_roles']
        tree=cKDTree(self.contact_reference)
        same_face=(np.abs((self.static_points-self.panel_origin)@self.panel_normal)<band)&(tree.query(self.static_points)[0]<radius)
        # Ownership is distinct from permission: selected moving-face points
        # must move with the mechanism even when fingers may not touch them.
        added=self.static_points[same_face];self.static_points=self.static_points[~same_face]
        self.moving_reference=np.vstack([self.moving_reference,added])
        distance=tree.query(self.moving_reference)[0]
        self.selected_contact_mask=distance<band
        if local_enabled:
            self.selected_contact_mask|=(distance<radius)&(np.abs((self.moving_reference-self.panel_origin)@self.panel_normal)<band)
        self.interaction_contact_record={'selected_roles':spec['selected_contact_roles'],'local_face_enabled':local_enabled,
            'radius_from_robot_m':radius,'band_from_depth_resolution_m':band,'reassigned_observed_points':len(added),
            'task_label_used':False,'arm_contact_exemption':False}

    def record(self):
        result=super().record();result['shared_contact_compilation']=self.interaction_contact_record;return result
