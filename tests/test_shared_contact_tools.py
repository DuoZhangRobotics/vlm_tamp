"""Both policy callers must use the same observed-contact implementation."""
from pathlib import Path

import numpy as np
import pytest

from vlm_tamp import hinged_contact_tools as shared
from vlm_tamp.coupled_scene import Scene
from vlm_tamp.coupled_storage import read_json

ROOT=Path(__file__).resolve().parents[1]


def test_both_callers_share_geometry_ik_and_planning_implementations():
    import serve_subscription_robot as agent
    import reliable_microwave_final as scripted
    from vlm_tamp.arc_execution import contact_arc
    from vlm_tamp.numerical_arc_execution import contact_arc as numerical_arc
    for name in ['ground_handle','select_grasp','contact_plan','NativeScene','TaskIK']:
        assert getattr(agent,name) is getattr(scripted,name) is getattr(shared,name)
    assert contact_arc is numerical_arc


def test_shared_tool_configuration_preserves_caller_motion_requests():
    old_agent={'maximum_grasp_search_seconds':180.,'motion_policy':'full','grasp_constraint':'rigid'}
    old_script={'maximum_grasp_search_seconds':600.,'motion_policy':'short'}
    a=shared.tool_configuration(old_agent);b=shared.tool_configuration(old_script)
    assert a['maximum_grasp_search_seconds']==180.
    assert b['maximum_grasp_search_seconds']==600.
    ca=shared.tool_contract(a);cb=shared.tool_contract(b)
    ca['settings'].pop('maximum_grasp_search_seconds');cb['settings'].pop('maximum_grasp_search_seconds')
    assert ca==cb
    assert a['motion_policy']=='full' and a['grasp_constraint']=='rigid'
    assert b['motion_policy']=='short'
    assert old_agent['maximum_grasp_search_seconds']==180.


def test_grounding_uses_measured_finite_pad_support(monkeypatch):
    from types import SimpleNamespace
    cloud=np.array([[0.,0.,.1],[0.,0.,.2]])
    original=[np.array([0.,0.,.19])]
    supported=[np.array([0.,0.,.16])]
    mechanism=SimpleNamespace(spec={});evidence={'cylinder_radius_m':.01}
    monkeypatch.setattr(shared,'fit_observed_handle',
        lambda *args,**kwargs:(mechanism,cloud,np.ones((2,2),bool),original,evidence))
    received=[]
    def menu(axis,points,metadata):
        received.append((axis.copy(),points.copy(),metadata))
        return supported,{'retained':1,'simulator_object_geometry_used':False}
    monkeypatch.setattr(shared,'handle_anchor_menu',menu)
    result=shared.ground_handle({}, {}, {}, {'robot':'calibrated'})
    np.testing.assert_array_equal(result[3],supported)
    np.testing.assert_array_equal(received[0][1],cloud)
    assert result[4]['pad_support']['retained']==1
    assert 'pad_support' not in evidence


@pytest.fixture
def calibrated_scene():
    folder=ROOT/'results/prepared-final-all-tasks-20261020/open_microwave-state001'
    if not (folder/'opened/metadata.json').exists():
        pytest.skip('Local calibrated RGB-D regression fixture unavailable')
    metadata=read_json(folder/'opened/metadata.json')
    record=read_json(folder/'robot-chain.json')
    q=np.asarray(metadata['joint_positions'])
    pad=next(v for v in metadata['robot_collision_volumes_world']
             if v['robot_geom_name'].endswith('finger1_pad_collision'))
    point=np.asarray(pad['vertices_world']).mean(0)
    spec={'joint_origin_world_m':(point+[-.25,0,0]).tolist(),
        'joint_axis_world':[0,0,1],'panel_normal_world':[0,1,0],
        'front_plane':{'point_world_m':(point+[0,-.1,0]).tolist()}}
    return record,metadata,q,point,spec


@pytest.mark.parametrize('mode',['point','rigid','jaw_axis'])
def test_shared_native_modes_keep_calibrated_arm_and_contact_scope(calibrated_scene,mode):
    record,metadata,q,point,spec=calibrated_scene
    allowed=Scene.from_points(np.empty((0,3)),[point],[point],spec)
    stationary=Scene.from_points([point],[point],[point],spec)
    with shared.NativeScene(allowed,record,metadata,q,.4,articulated=False,
                            grasp_constraint=mode) as context:
        assert context.record['configured_convex_arm_links']>=6
        assert context.record['grasp_constraint']==mode
        assert context.validate(q)['accepted']
    with shared.NativeScene(stationary,record,metadata,q,.4,articulated=False,
                            grasp_constraint=mode) as context:
        result=context.validate(q)
        assert not result['accepted']
        assert result['last_rejected_surface']==0
    shared.clear_native_cache()
