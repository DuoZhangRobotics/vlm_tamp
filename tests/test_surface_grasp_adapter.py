"""Surface-selected grasping reaches the unchanged generic GraspGen path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import serve_subscription_robot as service_module
from vlm_tamp.rekep_libero.geometry import pose_matrix


def test_surface_grasp_uses_the_selected_cloud_without_fabricating_a_joint(tmp_path: Path,monkeypatch) -> None:
    service=service_module.RobotService.__new__(service_module.RobotService)
    service.contact_scene_kind='surface';service.spec=None;service.scene=SimpleNamespace();service.anchors=[]
    service.config={'grasp_selection_method':'graspgen_direct','graspgen_samples':8,'graspgen_topk':8}
    service.state={'control_steps':60,'maximum_control_steps':600,'metadata':{
        'joint_positions':[0.]*7,'gripper_qpos':[.04,-.04],'robot_collision_volumes_world':[]}}
    service.ik=SimpleNamespace(events=[],chain=object());service.record={};service.cloud=np.ones((20,3))
    service.folder=lambda name:tmp_path/name
    service._query_scene=lambda policy:SimpleNamespace()
    service.sim=SimpleNamespace(fk=lambda *a,**kw:{'robot_collision_volumes_world':[]})
    service.resources=SimpleNamespace(grasps=lambda cloud,*a,**kw:np.asarray(cloud))
    monkeypatch.setattr(service_module,'RobotVolumes',lambda *a:object())
    monkeypatch.setattr(service_module,'require_grasp_budget',lambda *a:{})
    import vlm_tamp.current_robot_scene as current_scene
    monkeypatch.setattr(current_scene,'points_outside_current_robot',lambda p,m:(p,{}))
    def select(generated,anchors,state,ik,record,opened,closed,spec,goal,folder,config):
        np.testing.assert_array_equal(generated,service.cloud)
        assert spec is None and closed is None and goal==0.
        assert config['maximum_grasp_poses']==8
        return {'source':'graspgen','pose_world_xyzw':[0,0,0,0,0,0,1]}
    monkeypatch.setattr(service_module,'select_grasp',select)
    result=service._select_contact(0.,8,0)
    assert result['source']=='graspgen' and service.spec is None
    assert result['approach_aperture_qpos']==[.04,-.04]


def test_official_registry_contains_only_task_identifiers_and_instructions() -> None:
    import ast
    from vlm_tamp.subscription_tasks import TASKS
    source=Path(__file__).resolve().parents[1]/'third_party/LIBERO/libero/libero/benchmark/libero_suite_task_map.py'
    mapping=ast.literal_eval(ast.parse(source.read_text()).body[0].value)
    for spec in TASKS.values():
        assert set(spec)=={'suite','task_name','instruction','motion'}
        assert spec['task_name'] in mapping[spec['suite']]


def test_visual_pose_axis_schema_does_not_require_a_task_or_known_direction() -> None:
    from vlm_tamp.robot_contracts import validate_operation
    from vlm_tamp.visual_scene import pose_from_axes
    args={'position_world_m':[.1,.2,1.],'approach_world':[0,0,-1.],'jaw_world':[1.,0,0],
          'aperture':'open'}
    validate_operation('preview_hand',args)
    pose=pose_matrix(pose_from_axes(args['position_world_m'],args['approach_world'],args['jaw_world']))
    np.testing.assert_allclose(pose[:3,2],[0,0,-1])
