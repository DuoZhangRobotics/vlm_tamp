"""One retained push contact and one preview, tied to actual simulator state."""
import copy
import json
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


def contact_summary(context):
    if context is None:return {'available':False}
    return {'available':True,'contact_id':context['contact_id'],'control_step':context['control_step'],
        'robot_contact_surface':context['contact_name'],'contact_surface':context['contact_surface'],
        'suggested_direction_world':context['direction_world'].tolist(),
        'contact_speed_m_s':context['contact_speed_m_s'],'last_stroke_length_m':context['stroke_length_m'],
        'default_continuation':'one requested stroke','ineffective_strokes':context.get('ineffective_strokes',0),
        'physical_contact_verified':False,
        'source':'retained contact geometry; measured part pose is updated by RGB-D feedback'}


def resumed_engagement(context):
    empty=np.empty((0,7))
    return {'path':empty,'references':empty,'free_references':empty,'insertion_references':empty,
        'contact_local':context['contact_local'].copy(),'contact_world':context['contact_world'].copy(),
        'rotation':np.eye(3),'insertion_direction':context['direction_world'].copy(),
        'contact_name':context['contact_name'],'reuse_existing_contact':True,
        'attempts':[],'candidate_index':None,'timing_seconds':{},'total_seconds':0.,
        'predicted_contact_only':True,'native_contact_policy':{}}


def copied_preview(cached):
    scene=copy.copy(cached['scene']);engagement=dict(cached['engagement'])
    engagement['cached_search_timing_seconds']=engagement['timing_seconds']
    engagement['timing_seconds']={};engagement['total_seconds']=0.
    return scene,copy.deepcopy(cached['evidence']),engagement
