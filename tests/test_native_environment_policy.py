"""Actual MC-VAMP contact-pair checks using retained robot-only calibration."""
from pathlib import Path

import numpy as np
import pytest

from vlm_tamp.coupled_storage import read_json
from vlm_tamp.robot_contracts import validate_operation,RequestValidationError


def test_runtime_defaults_cannot_silently_clear_the_observed_world():
    from vlm_tamp.hinged_contact_tools import tool_configuration
    root=Path(__file__).resolve().parents[1]
    config=tool_configuration(read_json(root/'configs/articulated_libero.json'))
    assert config['environment_collision_policy']=='veto'
    for operation,args in [('plan_grasp',{}),('plan_arc',{'degrees':10}),
                           ('push',{'observation_id':1,'camera':'frontview','region_xyxy':[1,1,20,20]})]:
        validate_operation(operation,{**args,'collision_policy':'veto'})
        with pytest.raises(RequestValidationError):
            validate_operation(operation,{**args,'collision_policy':'advisory'})


@pytest.fixture
def calibrated():
    pytest.importorskip('vamp')
    folder=Path(__file__).resolve().parents[1]/'results/small-graspgen-20261029-planner-path/open_microwave-state037/attempt01'
    if not folder.exists():pytest.skip('Retained robot calibration for native integration test unavailable')
    return read_json(folder/'opened/metadata.json'),read_json(folder/'robot-chain.json')


@pytest.mark.parametrize('part,allowed',[
    ('gripper0_finger1_pad_collision',True),
    ('gripper0_hand_collision',False),
    ('robot0_link4_collision',False),
])
@pytest.mark.parametrize('surface',['target','unrelated'])
def test_native_arm_palm_and_unrelated_surfaces_block_but_target_pad_contact_is_allowed(calibrated,part,allowed,surface):
    from vlm_tamp.coupled_scene import Scene
    from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache
    meta,record=calibrated
    volume=next(v for v in meta['robot_collision_volumes_world'] if v['robot_geom_name']==part)
    point=np.mean(volume['vertices_world'],axis=0)[None]
    far=point+3
    spec={'joint_origin_world_m':point[0].tolist(),'joint_axis_world':[0,0,1],
        'front_plane':{'point_world_m':point[0].tolist()},'panel_normal_world':[1,0,0]}
    scene=(Scene.from_points(np.empty((0,3)),point,point,spec) if surface=='target'
        else Scene.from_points(point,far,far,spec))
    with NativeScene(scene,record,meta,meta['joint_positions'],.05,articulated=False) as context:
        result=context.validate(meta['joint_positions'])
        assert result['accepted']==(allowed and surface=='target'),result
        assert context.record['environment_collision_policy']=='veto'
        assert not context.record['existing_palm_contact']['enabled']
        assert context.record['existing_palm_contact']['maximum_allowed_depth_m']==0.
        assert context.record['proximal_fingers_checked']
        assert context.record['static_points']+context.record['selected_contact_points']>0
    clear_native_cache()
