"""One retained push contact and one preview, tied to actual simulator state."""
import copy
import json
from types import SimpleNamespace
import numpy as np

from .rekep_libero.scene_state_v7 import observed_points


def clear_motion_context(service):
    service.active_push_contact=None
    service.push_preview_cache=None


def observed_contact_cloud(service):
    key=(service.attempt,service.latest_observation['observation_id'],service.state['control_steps'])
    cached=getattr(service,'contact_observation_cache',None)
    if cached is not None and cached['key']==key:return cached['value'],True
    value=observed_points(service.cameras,service.state['metadata'],spacing=.003)
    service.contact_observation_cache={'key':key,'value':value}
    return value,False


def preview_key(service,arguments):
    metadata=service.state['metadata']
    return json.dumps({'attempt':service.attempt,'observation_id':service.latest_observation['observation_id'],
        'control_step':service.state['control_steps'],'geometry_epoch':service.epoch,
        'joints':list(metadata['joint_positions']),'gripper_qpos':list(metadata.get('gripper_qpos',[])),
        'request':arguments},sort_keys=True,allow_nan=False)


def active_contact(service):
    context=getattr(service,'active_push_contact',None)
    if context is None or context['control_step']!=service.state['control_steps']:
        service.active_push_contact=None
        raise ValueError('No retained push contact at the current state. Use push with a current target region to acquire contact first.')
    return context


class UnassociatedContactTracker:
    """A current-hand stroke does not pretend to have an observed object model."""
    def __init__(self):self.tracker=SimpleNamespace(references={})
    def observe(self,cameras,step):
        return {'accepted':False,'pairs':0,'reason':'no_object_target_associated',
            'simulator_object_state_used':False}


def current_hand_reference(service,direction):
    point=np.asarray(service.state['metadata']['eef_pose_world_xyzw'][:3],float)
    empty=np.empty((0,3))
    # This is robot-motion bookkeeping, with no observed or invented obstacles.
    scene=SimpleNamespace(static_points=empty,moving_reference=empty,contact_reference=empty,
        inferred_panel_reference=empty,config={},origin=point,axis=np.array([0.,0.,1.]),
        panel_origin=point,panel_normal=np.array([0.,0.,1.]),skill_kind='push',skill_contact_surface='fingertips')
    camera='agentview' if 'agentview' in service.cameras else next(iter(service.cameras))
    return {'contact_id':f'hand-{service.attempt}-{service.calls}','control_step':service.state['control_steps'],
        'scene':scene,'evidence':{'point_world_m':point.tolist(),'direction_world':list(direction),
            'joint_associated':False,'contact_surface':'fingertips','object_target_associated':False,
            'source':'current robot hand reference; physical target/contact not assumed'},
        'contact_local':np.zeros(3),'contact_world':point,'contact_name':'current_hand_reference',
        'contact_surface':'fingertips','camera':camera,'direction_world':np.asarray(direction),
        'contact_speed_m_s':.03,'stroke_length_m':None,'stroke_plan_seconds':5.,'ineffective_strokes':0,
        'tracker':UnassociatedContactTracker(),'object_pose_current':False,'scene_control_step':None,
        'object_target_associated':False}


def contact_summary(context):
    if context is None:return {'available':False}
    return {'available':True,'contact_id':context['contact_id'],'control_step':context['control_step'],
        'robot_contact_surface':context['contact_name'],'contact_surface':context['contact_surface'],
        'reference_direction_world':context['direction_world'].tolist(),
        'direction_semantics':'Previous requested direction transported by measured part rotation when available; not a task-goal recommendation.',
        'task_direction_verified':False,
        'contact_speed_m_s':context['contact_speed_m_s'],'last_stroke_length_m':context['stroke_length_m'],
        'default_continuation':'one requested stroke','ineffective_strokes':context.get('ineffective_strokes',0),
        'object_pose_current':context.get('object_pose_current',True),
        'object_pose_control_step':context.get('scene_control_step',context['control_step']),
        'object_target_associated':context.get('object_target_associated',True),
        'execution_policy':'physical push feedback; object tracking does not veto direction-only strokes',
        'physical_contact_verified':False,
        'source':('retained contact geometry; measured part pose is updated by RGB-D feedback'
            if context.get('object_target_associated',True) else 'current hand reference; no object model or contact assumption')}


def resumed_engagement(context):
    empty=np.empty((0,7))
    return {'path':empty,'references':empty,'free_references':empty,'insertion_references':empty,
        'contact_local':context['contact_local'].copy(),'contact_world':context['contact_world'].copy(),
        'rotation':np.eye(3),'insertion_direction':context['direction_world'].copy(),
        'contact_name':context['contact_name'],'reuse_existing_contact':context.get('object_target_associated',True),
        'reuse_current_hand_pose':True,
        'attempts':[],'candidate_index':None,'timing_seconds':{},'total_seconds':0.,
        'predicted_contact_only':True,'native_contact_policy':{}}


def copied_preview(cached):
    scene=copy.copy(cached['scene']);engagement=dict(cached['engagement'])
    engagement['cached_search_timing_seconds']=engagement['timing_seconds']
    engagement['timing_seconds']={};engagement['total_seconds']=0.
    return scene,copy.deepcopy(cached['evidence']),engagement
