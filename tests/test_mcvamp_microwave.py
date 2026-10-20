from pathlib import Path

import numpy as np
import pytest

from vlm_tamp.coupled_scene import Scene
from vlm_tamp.coupled_storage import read_json
from vlm_tamp.mcvamp_microwave import NativeScene,LIBRARY,clear_native_cache,cleanup_worker_dumps
from vlm_tamp.rekep_libero.geometry import pose_matrix,transform_points
from vlm_tamp.coupled_manifold import CoupledHandlePoint

ROOT=Path(__file__).resolve().parents[1]
EXAMPLE=ROOT/'results/coupled-planning-resumed-20261015/final-development-open/open_microwave-state001'


@pytest.fixture
def configured_robot():
    if not LIBRARY.exists() or not EXAMPLE.exists():pytest.skip('Local native build and calibrated Panda fixture required')
    record=read_json(EXAMPLE/'robot-chain.json')
    metadata=read_json(EXAMPLE/'plan-00/inputs.json')['metadata']
    q=np.asarray(metadata['joint_positions'])
    pad=next(v for v in metadata['robot_collision_volumes_world'] if v['robot_geom_name'].endswith('finger1_pad_collision'))
    point=np.asarray(pad['vertices_world']).mean(0)
    spec={'joint_origin_world_m':(point+np.array([-.25,0,0])).tolist(),
        'joint_axis_world':[0,0,1],'panel_normal_world':[0,1,0],
        'front_plane':{'point_world_m':(point+np.array([0,-.1,0])).tolist()}}
    return record,metadata,q,point,spec


def test_only_selected_finger_contact_is_exempted(configured_robot):
    record,metadata,q,point,spec=configured_robot
    scene=Scene.from_points(np.empty((0,3)),[point],[point],spec)
    with NativeScene(scene,record,metadata,q,.4,articulated=False,allow_contact=True) as context:
        assert context.validate(q)['accepted']
    with NativeScene(scene,record,metadata,q,.4,articulated=False,allow_contact=False) as context:
        check=context.validate(q)
        assert not check['accepted']
        assert check['last_rejected_surface']==2


def test_stationary_contact_is_not_exempted(configured_robot):
    record,metadata,q,point,spec=configured_robot
    scene=Scene.from_points([point],[point],[point],spec)
    with NativeScene(scene,record,metadata,q,.4,articulated=False,allow_contact=True) as context:
        check=context.validate(q)
        assert not check['accepted']
        assert check['last_rejected_surface']==0


def test_unrelated_moving_geometry_is_not_exempted(configured_robot):
    record,metadata,q,point,spec=configured_robot
    selected=point+np.array([.15,0,0])
    scene=Scene.from_points(np.empty((0,3)),[point],[selected],spec)
    with NativeScene(scene,record,metadata,q,.4,articulated=False,allow_contact=True) as context:
        assert not context.validate(q)['accepted']


def test_closed_state11_palm_sphere_overhang_does_not_block():
    folder=ROOT/'results/mcvamp-microwave-frozen-20261015/open_microwave-state011'
    reference=folder/'plan-00/horizon-0/scene-reference.json'
    if not reference.exists():pytest.skip('Retained state11 RGB-D fixture required')
    inputs=read_json(reference.parent/'planning-input.json')
    arrays=np.load(read_json(reference)['arrays_path'])
    scene=Scene.from_points(arrays['static_points'],arrays['moving_reference'],arrays['contact_reference'],inputs['task_spec'])
    metadata=inputs['metadata'];q=inputs['reference_joints'];record=read_json(folder/'robot-chain.json')
    with NativeScene(scene,record,metadata,q,inputs['goal_angle_rad']) as context:
        assert context.validate(q)['accepted']
        assert context.record['gripper_convex_point_padding_m']==0.
    # Removing sphere overhang must preserve collisions inside the actual palm.
    palm=next(v for v in metadata['robot_collision_volumes_world'] if v['robot_geom_name']=='gripper0_hand_collision')
    center=np.asarray(palm['vertices_world']).mean(0)
    scene=Scene.from_points([center],arrays['moving_reference'],arrays['contact_reference'],inputs['task_spec'])
    with NativeScene(scene,record,metadata,q,inputs['goal_angle_rad']) as context:
        result=context.validate(q)
        assert not result['accepted'] and result['last_rejected_surface']==0


def test_tracked_handle_reference_does_not_invent_observed_palm_collision():
    folder=ROOT/'results/state11-rigid-pose-recovery-20261015/open_microwave-state011'
    reference=folder/'plan-01/horizon-0/scene-reference.json'
    if not reference.exists():pytest.skip('Retained measured-pose failure fixture required')
    inputs=read_json(reference.parent/'planning-input.json')
    arrays=np.load(read_json(reference)['arrays_path'])
    metadata=inputs['metadata'];q=inputs['reference_joints'];record=read_json(folder/'robot-chain.json')
    scene=Scene.from_points(arrays['static_points'],arrays['moving_reference'],arrays['contact_reference'],inputs['task_spec'])
    with NativeScene(scene,record,metadata,q,inputs['goal_angle_rad'],aligned=False) as context:
        assert context.validate(q)['accepted']
        assert not context.record['transported_handle_reference_added_as_obstacle']
    # The same point must still block the palm when it is an actual observed
    # moving endpoint, rather than just a transported tracking reference.
    palm=next(v for v in metadata['robot_collision_volumes_world'] if v['robot_geom_name']=='gripper0_hand_collision')
    planes=np.asarray(palm['planes_world']);points=arrays['contact_reference']
    values=np.max(points@planes[:,:3].T+planes[:,3],axis=1)
    witness=points[np.argmin(values)];assert values.min()<0
    scene=Scene.from_points(arrays['static_points'],np.vstack([arrays['moving_reference'],witness]),
        arrays['contact_reference'],inputs['task_spec'])
    with NativeScene(scene,record,metadata,q,inputs['goal_angle_rad'],aligned=False) as context:
        result=context.validate(q)
        assert not result['accepted'] and result['last_rejected_surface']==2
        assert result['last_rejected_part']==1000


def test_moving_palm_volume_uses_hinge_frame(configured_robot):
    record,metadata,q,point,spec=configured_robot
    far=point+np.array([1.,0,0])
    empty=Scene.from_points(np.empty((0,3)),[far],[far],spec)
    with NativeScene(empty,record,metadata,q,.4,aligned=False) as context:
        model=CoupledHandlePoint(context.chain,spec['joint_origin_world_m'],spec['joint_axis_world'],
            pose_matrix(context.chain.forward(q)),pad_local=context.pad_local,align_axis=False,
            jaw_axis_local=None,jaw_tilt_limit=np.pi/3,orientation_cone=np.pi/2,
            progress_bounds=(-.03,.43),joint_margin=.025,position_tolerance=.001,orientation_tolerance=.01)
        projected=model.project(np.r_[q,.08],fixed_progress=.08,max_iterations=150)
        assert projected['accepted']
        target=projected['state'][:7]
        palm=next(v for v in metadata['robot_collision_volumes_world'] if v['robot_geom_name']=='gripper0_hand_collision')
        first_pose=pose_matrix(context.chain.forward(q));last_pose=pose_matrix(context.chain.forward(target))
        local=transform_points([np.asarray(palm['vertices_world']).mean(0)],np.linalg.inv(first_pose))[0]
        moved=transform_points([local],last_pose)[0]
        from scipy.spatial.transform import Rotation
        origin=np.asarray(spec['joint_origin_world_m'])
        reference=origin+Rotation.from_rotvec(-.08*np.asarray(spec['joint_axis_world'])).apply(moved-origin)
    scene=Scene.from_points(np.empty((0,3)),[reference],[far],spec)
    with NativeScene(scene,record,metadata,q,.4,aligned=False) as context:
        result=context.validate(target)
        assert not result['accepted'] and result['last_rejected_surface']==1


def test_cache_separates_static_and_permitted_contact_partitions(configured_robot):
    record,metadata,q,point,spec=configured_robot
    far=point+np.array([1.,0,0])
    clear_native_cache()
    permitted=Scene.from_points(np.empty((0,3)),[point,far],[point,far],spec)
    stationary=Scene.from_points([point],[far],[far],spec)
    try:
        with NativeScene(permitted,record,metadata,q,.4,articulated=False) as context:
            assert context.validate(q)['accepted']
        with NativeScene(stationary,record,metadata,q,.4,articulated=False) as context:
            assert not context.validate(q)['accepted']
    finally:
        clear_native_cache()


def test_crash_cleanup_only_removes_this_worker_and_executable(tmp_path):
    executable=tmp_path/'project-env/bin/python3.10'
    tag=str(executable.resolve()).replace('/','_')
    owned=tmp_path/f'wsl-crash-123-4242-{tag}-11.dmp'
    different_pid=tmp_path/f'wsl-crash-123-4243-{tag}-11.dmp'
    other_program=tmp_path/'wsl-crash-123-4242-_other_program-11.dmp'
    for path in [owned,different_pid,other_program]:path.write_bytes(b'test dump')
    result=cleanup_worker_dumps(4242,0.,folder=tmp_path,executable=executable)
    assert len(result)==1 and result[0]['removed']
    assert not owned.exists()
    assert different_pid.exists() and other_program.exists()


def test_projected_grip_stays_on_perceived_hinge_circle(configured_robot):
    record,metadata,q,point,spec=configured_robot
    scene=Scene.from_points(np.empty((0,3)),[point],[point],spec)
    with NativeScene(scene,record,metadata,q,.4,aligned=True) as context:
        result,info=context.project(q+np.array([.001,0,0,0,0,0,0]))
        assert info['accepted']
        tcp=pose_matrix(context.chain.forward(result))
        current=transform_points([context.pad_local],tcp)[0]-context.origin
        start=context.anchor-context.origin
        assert abs(current[2]-start[2])<1e-5
        assert abs(np.linalg.norm(current[:2])-np.linalg.norm(start[:2]))<1e-5
        assert abs((tcp[:3,:3]@np.array([1,0,0]))@context.axis)<1e-4


def test_native_plan_handles_unaligned_numpy_inputs(configured_robot):
    record,metadata,q,point,spec=configured_robot
    scene=Scene.from_points(np.empty((0,3)),[point],[point],spec)
    with NativeScene(scene,record,metadata,q,.4) as context:
        for offset in range(4):
            raw=np.zeros(16,np.float32);raw[offset:offset+7]=q
            goal,projection=context.project(q+np.array([.001,0,0,0,0,0,0]))
            assert projection['accepted']
            path,result=context.plan(raw[offset:offset+7],goal,seconds=1.,iterations=50)
            assert result['exact_solution'],result
            assert len(path)>=2


def test_point_constraint_preserves_free_gripper_twist(configured_robot):
    record,metadata,q,point,spec=configured_robot
    scene=Scene.from_points(np.empty((0,3)),[point],[point],spec)
    checked=0
    with NativeScene(scene,record,metadata,q,.4) as context:
        model=CoupledHandlePoint(context.chain,context.origin,context.axis,pose_matrix(context.chain.forward(q)),
            pad_local=context.pad_local,jaw_axis_local=[1,0,0],orientation_cone=np.pi/2,
            progress_bounds=(-.03,.43),joint_margin=.025)
        rng=np.random.default_rng(4)
        for seed in rng.uniform(context.chain.limits[:,0]+.05,context.chain.limits[:,1]-.05,(12,7)):
            result=model.project(np.r_[seed,.2],fixed_progress=.2,max_iterations=100)
            if not result['accepted'] or result['weighted_residual_norm']>1e-7:continue
            candidate=result['state'][:7]
            projected,info=context.project(candidate)
            assert info['accepted']
            np.testing.assert_allclose(projected,candidate,atol=1e-4)
            assert abs(context.progress(projected)-.2)<1e-4
            checked+=1
        assert checked>=2
