"""Revise a fixed-joint hypothesis from two separately observed intervals."""
import copy
from pathlib import Path
import numpy as np

from .part_motion import estimate_and_verify,verify_upright_axis
from .coupled_storage import write_json


def verify_feedback_joint(records,old_origin):
    if len(records)<2:return {'status':'waiting_for_second_interval','motion_supported':False}
    first,second=records[-2:]
    if first['current_observation_id']!=second['reference_observation_id']:
        return {'status':'nonconsecutive_observations','motion_supported':False}
    a,b=np.asarray(first['first']),np.asarray(first['second']);c,d=np.asarray(second['first']),np.asarray(second['second'])
    world=np.full((3,len(a)+len(c),3),np.nan)
    world[0,:len(a)]=a;world[1,:len(a)]=b;world[1,len(a):]=c;world[2,len(a):]=d
    valid=np.isfinite(world).all(2)
    result=estimate_and_verify(world,valid,[0,1],[1,2])
    previous=verify_upright_axis(world,valid,1,2,old_origin)
    result['previous_model_verification']=previous
    result['observation_ids']=[first['reference_observation_id'],first['current_observation_id'],second['current_observation_id']]
    result['new_model_predicts_better']=bool(result.get('motion_supported') and
        result['verification']['axis_prediction_rms_m']<previous.get('axis_prediction_rms_m',float('inf')))
    result['source']='RGB-D correspondences; fit and later verification intervals have separate observed feature pairs'
    result['native_object_state_used']=False
    return result


def revise_from_feedback(service,records):
    if service.spec is None:return {'status':'no_joint_model','applied':False}
    result=verify_feedback_joint(records,service.spec['joint_origin_world_m'])
    result['applied']=False
    if result.get('new_model_predicts_better'):
        old=copy.deepcopy(service.spec);fit=result['estimate']
        axis=np.asarray(fit['axis_world_unsigned'],float);axis/=np.linalg.norm(axis)
        if axis@np.asarray(old['joint_axis_world'])<0:axis=-axis
        service.spec=copy.deepcopy(old)
        service.spec.update(joint_origin_world_m=list(fit['joint_origin_world_m']),joint_axis_world=axis.tolist(),
            hinge_evidence={'status':'motion_consistency_checked','source':result['source'],
                'motion_supported':True,'observation_ids':result['observation_ids'],
                'calibrated_accuracy_guarantee':False})
        service.spec['joint_estimation_evidence']={'source':'sequence RGB-D feedback','motion_supported':True,
            'estimated_axis_world_unsigned':fit['axis_world_unsigned'],'observation_ids':result['observation_ids']}
        service._update_joint_parameters();service.epoch+=1;service.tracker=None
        result.update(applied=True,previous_origin_world_m=old['joint_origin_world_m'],
            updated_origin_world_m=service.spec['joint_origin_world_m'],geometry_epoch=service.epoch,
            fixed_joint_revised_separately_from_part_pose=True)
        result['geometry']=service.export_geometry(copy.deepcopy(result))
    if len(records)>=2:
        folder=service.public/f'attempt{service.attempt:02d}'/f'joint-feedback-{service.calls:03d}'
        folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'pairs.npz',first_a=records[-2]['first'],first_b=records[-2]['second'],
            second_a=records[-1]['first'],second_b=records[-1]['second'])
        result['evidence_file']=str((folder/'verification.json').relative_to(service.workspace))
        write_json(folder/'verification.json',result)
    return result
