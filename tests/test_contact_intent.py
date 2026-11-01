from types import SimpleNamespace
import numpy as np
import pytest

from vlm_tamp.contact_skills import Budget,SkillStop,hook_feature
from vlm_tamp.robot_contracts import validate_operation,RequestValidationError
from vlm_tamp.compact_robot_interface import compact_response


@pytest.mark.parametrize('operation',['push','hook_pull'])
@pytest.mark.parametrize('preview',[False,True])
def test_missing_motion_is_an_explicit_input_error_not_a_silent_normal_push(operation,preview):
    args={'observation_id':1,'camera':'test','contact_pixel_xy':[5,5],'contact_name':'gripper0_finger1_pad_collision','region_xyxy':[1,1,20,20],'preview':preview}
    with pytest.raises(RequestValidationError,match='Supply direction_world') as error:validate_operation(operation,args)
    assert error.value.request_diagnostic['reason']=='motion_direction_required'
    validate_operation(operation,{**args,'direction_world':[1,0,0]})
    validate_operation(operation,{**args,'joint_degrees':-15,'joint_model_epoch':1})


def test_motion_direction_and_image_coordinates_have_separate_causes():
    args={'observation_id':1,'camera':'test','contact_pixel_xy':[5,5],'contact_name':'gripper0_finger1_pad_collision','region_xyxy':[823,497,842,569],'direction_world':[1,0,0]}
    with pytest.raises(RequestValidationError,match='512×512') as error:
        validate_operation('push',args,{'test':{'width':512,'height':512}})
    assert error.value.request_diagnostic['reason']=='pixel_out_of_bounds'
    validate_operation('push',{**args,'region_xyxy':[500,500,512,512]}, {'test':{'width':512,'height':512}})




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
