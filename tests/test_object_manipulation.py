"""Persistent object references must survive scene selection and remain observation-grounded."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vlm_tamp.object_manipulation import ObjectModels
from vlm_tamp.robot_contracts import validate_operation


@pytest.fixture
def registry(tmp_path: Path) -> tuple:
    """Provide two selected regions and a deterministic sensor-only tracking backend."""
    y,x=np.indices((40,40))
    points=np.stack([x*.005,y*.005,np.ones((40,40))],axis=-1)
    points[12:28,12:28,2]+=.04
    camera=np.eye(4);camera[:3,3]=[.1,.1,2.]
    view={'world_points':points,'depth':np.ones((40,40)), 'rgb':np.zeros((40,40,3),np.uint8),
          'known_robot_depth_prior':np.zeros((40,40)), 'camera_to_world':camera}
    service=SimpleNamespace(public=tmp_path/'robot_data',attempt=1,calls=1,
        state={'control_steps':60},latest_observation={'observation_id':1},scene=object(),
        observation_records={1:{'control_steps':60},2:{'control_steps':90},3:{'control_steps':102}})
    (service.public/'attempt01').mkdir(parents=True)
    engine=SimpleNamespace(parts={},tracks={},_view=lambda oid, name:view)
    service._perception_tools=engine
    service.tracking_supported=True
    def part_tool(operation: str, arguments: dict) -> dict:
        """Return measured correspondences, without a hand-motion prediction."""
        if operation=='segment_parts':
            mask=np.zeros((40,40),bool)
            if arguments['points_xy'][0][0]<30:mask[12:28,12:28]=True
            else:mask[8:24,30:39]=True
            identifier=f'part{service.calls}'
            engine.parts[identifier]={'mask':mask,'observation_id':1,'camera':'agentview'}
            return {'parts':[{'part_id':identifier,'sam_mask_score':.9,
                              'overlay':identifier+'.png','mask_file':identifier+'.npz'}], 'timing_seconds':{}}
        if operation=='track_video_motion':
            xx,yy=np.meshgrid(np.linspace(0,.08,5),np.linspace(0,.08,4))
            original=np.c_[xx.ravel(),yy.ravel(),np.ones(20)]
            if not service.tracking_supported:original=original[:4]
            count=len(arguments['observation_ids'])
            world=np.stack([original+t*np.array([.03,-.02,.01]) for t in np.linspace(0,1,count)])
            engine.tracks['tracks']={'world':world,'valid':np.ones(world.shape[:2],bool)}
            return {'track_id':'tracks','images':[]}
        raise AssertionError('Unexpected perception operation: '+operation)
    service._part_tool=part_tool
    return ObjectModels(service),service


def test_source_reference_uses_declared_support_and_survives_destination_selection(registry: tuple) -> None:
    """The visible top is not silently used as the object's bottom; selection keeps separate identities."""
    models,service=registry
    scene=service.scene
    source=models.select('agentview',[5,5,30,30],[20,20],'supported_body')['object_model']
    assert source['shape']['low_z_m']==pytest.approx(1.04)
    assert source['reference_point_world_m'][2]==pytest.approx(1.)
    service.calls+=1
    destination=models.select('agentview',[30,5,39,26],[34,17],'surface_center')['object_model']
    assert source['object_id']!=destination['object_id']
    assert models.summary(source['object_id'])==source
    assert service.scene is scene
    assert len(models.models)==2


def test_observed_motion_updates_only_the_selected_object(registry: tuple) -> None:
    """A destination selection cannot overwrite the carried object's reference or pose."""
    models,service=registry
    source=models.select('agentview',[5,5,30,30],[20,20],'supported_body')['object_model']
    service.calls+=1
    destination=models.select('agentview',[30,5,39,26],[34,17],'surface_center')['object_model']
    service.state['control_steps']=90;service.latest_observation={'observation_id':2};service.calls+=1
    measured=models.measure(source['object_id'],[1,2])['object_model']
    np.testing.assert_allclose(measured['reference_point_world_m'],
                               np.array(source['reference_point_world_m'])+[.03,-.02,.01],atol=1e-10)
    assert measured['pose_current']
    assert not models.summary(destination['object_id'])['pose_current']
    assert models.summary(destination['object_id'])['reference_point_world_m']==destination['reference_point_world_m']
    assert measured['attachment_verified'] is False


def test_failed_measurement_never_becomes_identity_or_current_pose(registry: tuple) -> None:
    """Insufficient evidence remains unresolved instead of substituting a previous transform."""
    models,service=registry
    source=models.select('agentview',[5,5,30,30],[20,20],'supported_body')['object_model']
    service.state['control_steps']=90;service.latest_observation={'observation_id':2}
    service.tracking_supported=False;service.calls+=1
    reply=models.measure(source['object_id'],[1,2])
    assert not reply['object_motion']['accepted']
    assert reply['object_model']['pose_status']=='unresolved'
    assert reply['object_model']['pose_control_step']==60
    assert not reply['object_model']['pose_current']
    with pytest.raises(ValueError,match='unresolved'):
        models.current(source['object_id'])


def test_grasp_composes_existing_tools_and_measures_postclosure_pose(registry: tuple) -> None:
    """Acquisition and measurement need no model-side coordinate calculation."""
    models,service=registry
    source=models.select('agentview',[5,5,30,30],[20,20],'supported_body')['object_model']
    calls=[]
    service.select_surface_contact=lambda *args:calls.append(('selection',args))
    service.plan_grasp=lambda **kwargs:calls.append(('planning',kwargs)) or {'plan_id':'p'}
    def execute(plan_id: str) -> dict:
        """Provide a reached approach and physical closure, not retained-object certification."""
        calls.append(('execution',plan_id))
        service.state['control_steps']=102;service.latest_observation={'observation_id':3}
        return {'grasp_execution':{'closure_completed':True},'episode_done':False,
                'stages':[{'observation':{'observation_id':2}},{'observation':{'observation_id':3}}]}
    service.execute_plan=execute
    result=models.grasp(source['object_id'],close_steps=12,seed=0)
    assert [row[0] for row in calls]==['selection','planning','execution']
    assert len(calls[0][1])==3  # Grasp input remains the caller's region, not a substituted mask.
    assert result['object_model']['pose_current']
    assert result['object_pose_update']['object_motion']['accepted']


def test_model_reply_is_a_snapshot_not_a_mutable_registry_alias(registry: tuple) -> None:
    """Later bookkeeping must not rewrite previously returned geometric evidence."""
    models,_=registry
    result=models.select('agentview',[5,5,30,30],[20,20],'supported_body')['object_model']
    models.models[result['object_id']]['shape']['center_xy_m'][0]=99.
    assert result['shape']['center_xy_m'][0]!=99.


def test_object_tools_are_declared_in_actual_callable_schemas() -> None:
    """The stable-ID workflow is callable rather than documentation-only."""
    from vlm_tamp.actor_structured import OPERATIONS
    requests={
        'select_object':{'camera':'agentview','region_xyxy':[1,1,20,20],'contact_pixel_xy':[10,10],
                         'reference_type':'supported_body'},
        'grasp_object':{'object_id':'object001'},
        'place_object':{'object_id':'object001','destination_id':'object002','lift_distance_m':.12,
                        'attachment_model':'rigid'},
        'object_status':{'object_id':'object001'}}
    for name,args in requests.items():
        validate_operation(name,args)
        assert name in OPERATIONS
