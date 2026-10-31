from types import SimpleNamespace
import numpy as np
import pytest

from vlm_tamp.contact_skills import Budget,SkillStop,hook_feature
from vlm_tamp.robot_contracts import validate_operation,RequestValidationError
from vlm_tamp.compact_robot_interface import compact_response


@pytest.mark.parametrize('operation',['push','hook_pull'])
@pytest.mark.parametrize('preview',[False,True])
def test_missing_motion_is_an_explicit_input_error_not_a_silent_normal_push(operation,preview):
    args={'observation_id':1,'camera':'test','region_xyxy':[1,1,20,20],'preview':preview}
    with pytest.raises(RequestValidationError,match='Supply direction_world') as error:validate_operation(operation,args)
    assert error.value.request_diagnostic['reason']=='motion_direction_required'
    validate_operation(operation,{**args,'direction_world':[1,0,0]})
    validate_operation(operation,{**args,'joint_degrees':-15,'joint_model_epoch':1})


def test_motion_direction_and_image_coordinates_have_separate_causes():
    args={'observation_id':1,'camera':'test','region_xyxy':[823,497,842,569],'direction_world':[1,0,0]}
    with pytest.raises(RequestValidationError,match='512×512') as error:
        validate_operation('push',args,{'test':{'width':512,'height':512}})
    assert error.value.request_diagnostic['reason']=='pixel_out_of_bounds'
    validate_operation('push',{**args,'region_xyxy':[500,500,512,512]}, {'test':{'width':512,'height':512}})


def test_opposite_direction_can_reuse_current_push_posture_without_admission(monkeypatch):
    import vlm_tamp.contact_skills as module
    volume={'robot_geom_name':'gripper0_finger1_pad_collision',
        'vertices_world':np.array([[x,y,z] for x in [.018,.022] for y in [-.002,.002] for z in [1.02,1.03]])}
    meta={'joint_positions':[0.]*7,'eef_pose_world_xyzw':[0,0,1,0,0,0,1],
        'robot_collision_volumes_world':[volume]}
    point=np.array([.018,0,1.025])
    scene=SimpleNamespace(contact_reference=point[None],skill_collision_policy='veto')
    evidence={'point_world_m':point.tolist(),'direction_world':[1,0,0],
        'normal_toward_camera_world':[1,0,0],'contact_surface':'fingertips','joint_associated':False}
    def forbidden(*a,**k):raise AssertionError('Existing push posture must not be re-admitted or replanned')
    monkeypatch.setattr(module,'skill_native_scene',forbidden)
    result=module._plan_contact_at_point(SimpleNamespace(solve_many=forbidden),scene,{},meta,evidence,point[None],'push',Budget(5))
    assert result['reuse_existing_contact'] and len(result['references'])==0
    assert not evidence['surface_alignment_prediction']['execution_veto_applied']
    assert evidence['surface_alignment_prediction']['force_alignment']==pytest.approx(-1.)


def test_stubby_retaining_feature_is_not_rejected_only_for_aspect_ratio():
    points=np.array([[x,y,z] for x in np.linspace(-.028,.028,7)
        for y in np.linspace(-.016,.016,7) for z in np.linspace(-.0095,.0095,7)])
    _,feature=hook_feature(points,[1,0,0])
    assert feature['aspect_ratio']<1.8 and not feature['aspect_ratio_is_admission_gate']
    assert not feature['failed_checks'] and not feature['catch_verified']
    flat=points.copy();flat[:,2]=0
    with pytest.raises(SkillStop) as error:hook_feature(flat,[1,0,0])
    assert any(c['quantity']=='thickness_m' for c in error.value.evidence['feature']['failed_checks'])


def test_compact_motion_measurement_does_not_claim_task_progress():
    result=compact_response({'ok':True,'result':{'skill':'push','observed_progress':.02,
        'measured_object_motion_along_request':.02,'progress_units':'metres',
        'motion_measurement_scope':'along request, not task goal','task_direction_verified':False}})['result']
    assert 'observed_progress' not in result
    assert result['measured_object_motion_along_request']==.02 and not result['task_direction_verified']


def test_acquisition_uses_physical_insertion_policy_but_checks_free_approach(monkeypatch):
    import vlm_tamp.contact_skills as module
    checks=[]
    class Context:
        def __init__(self,allow):self.allow=allow;self.record={}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def validate(self,q):
            checks.append((self.allow,float(q[0])))
            return {'accepted':not self.allow or q[0]==0.}
        def plan(self,start,goal,**kwargs):return np.vstack([start,goal]),{'exact_solution':True}
    monkeypatch.setattr(module,'skill_native_scene',lambda *a,**k:Context(k.get('allow_contact',True)))
    monkeypatch.setattr(module,'current_contact_candidate',lambda *a:None)
    target=np.eye(4);target[0,3]=.1;standoff=target.copy();standoff[0,3]=.065
    contact={'name':'pad','point':np.zeros(3)}
    monkeypatch.setattr(module,'contact_pose_menu',lambda *a:[(contact,target,standoff,np.array([1.,0,0]))])
    class Chain:
        def forward(self,q):return np.r_[np.asarray(q)[:3],[0.,0,0,1.]]
    ik=SimpleNamespace(chain=Chain(),self_valid=lambda q:True,
        solve_many=lambda pose,seed,**k:[{'joints':np.r_[pose[:3],np.zeros(4)]}])
    meta={'joint_positions':[0.]*7,'eef_pose_world_xyzw':[0.,0,0,0,0,0,1.]}
    evidence={'point_world_m':[.1,0,0],'direction_world':[1.,0,0],
        'normal_toward_camera_world':[-1.,0,0],'contact_surface':'fingertips'}
    scene=SimpleNamespace(skill_collision_policy='veto',skill_physical_contact_execution=True)
    result=module._plan_contact_at_point(ik,scene,{},meta,evidence,None,'push',Budget(5))
    assert result['contact_insertion_policy']=='physical_execution_feedback'
    assert not any(allow and x!=0 for allow,x in checks)
    assert any(not allow and x!=0 for allow,x in checks)
