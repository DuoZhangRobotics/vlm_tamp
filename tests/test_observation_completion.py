import hashlib
import json
import numpy as np
import pytest
from vlm_tamp.observation_completion import plane_fit,compare_planes,CompletionEvidence


def surface(angle=0,noise=.0005):
    rng=np.random.default_rng(4)
    xy=rng.uniform(-.12,.12,(1600,2));points=np.c_[xy,rng.normal(0,noise,len(xy))]
    a=np.radians(angle);rotation=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    return points@rotation.T+np.array([.1,.2,1])


def test_measures_actual_plane_separation_and_rejects_wrong_goal_range():
    a,_=plane_fit(surface());b,_=plane_fit(surface(62))
    result=compare_planes(a,b)
    assert result['supported']
    assert abs(result['values']['plane_angle_deg']-62)<.3
    low,high=result['intervals']['plane_angle_deg']
    assert 60<low<high<64


def test_narrow_edge_has_insufficient_plane_support():
    points=surface();points[:,1]=.2+np.linspace(0,.001,len(points))
    fit,_=plane_fit(points)
    assert not fit['supported']


def test_current_relative_geometry_does_not_require_same_material_identity(tmp_path):
    c=CompletionEvidence(tmp_path)
    a,_=plane_fit(surface());b,_=plane_fit(surface(62))
    c.measurements['a']={**a,'kind':'surface','control_steps':90,'observation_ids':[4]}
    c.measurements['b']={**b,'kind':'surface','control_steps':90,'observation_ids':[4]}
    relation=c.geometry_relation('a','b',90)
    assert relation['supported'] and abs(relation['values']['plane_angle_deg']-62)<.3
    assert not relation['semantic_identity_verified'] and not relation['object_motion_claimed']
    with pytest.raises(ValueError,match='current physical state'):c.geometry_relation('a','b',91)


def test_partial_progress_and_stale_measurement_do_not_certify_completion(tmp_path):
    c=CompletionEvidence(tmp_path)
    c.set_contract('Requested orientation', [{'id':'orientation','quantity':'plane_angle_deg',
        'minimum':80,'maximum':90,'rationale':'Declared desired orientation'}],60)
    c.measurements['a']={'kind':'comparison','supported':True,'control_steps':100,
        'values':{'plane_angle_deg':62},'intervals':{'plane_angle_deg':[61,63]}}
    result=c.check({'orientation':'a'},100,'The same material surface is visible in two views.')
    assert not result['supported']
    assert not c.finish_claim('completed',result['check_id'],'It appears open',100)['finish_allowed']
    assert c.finish_claim('uncertain',None,'Occluded and no supported progress check',100)['finish_allowed']
    c.measurements['b']={'kind':'comparison','supported':True,'control_steps':100,
        'values':{'plane_angle_deg':85},'intervals':{'plane_angle_deg':[83,87]}}
    passed=c.check({'orientation':'b'},100,'Current measurements support the goal range.')
    assert c.finish_claim('completed',passed['check_id'],'Measured goal met',100)['finish_allowed']
    assert not c.finish_claim('completed',passed['check_id'],'Old measurement',101)['finish_allowed']
    assert not c.check({'orientation':'b'},101,'Old measurement')['supported']
    with pytest.raises(ValueError,match='fixed'):c.set_contract('Easier',[],100)


def test_declaring_goal_late_is_not_allowed(tmp_path):
    with pytest.raises(ValueError,match='before'):
        CompletionEvidence(tmp_path).set_contract('Late',[],61)


def test_surface_uses_real_current_sensor_arrays_and_detects_edits(tmp_path):
    h=w=80;y,x=np.mgrid[:h,:w];xyz=np.stack([x/200,y/200,np.ones((h,w))],-1)
    p=tmp_path/'sensors.npz';np.savez_compressed(p,cam_rgb=np.zeros((h,w,3),np.uint8),
        cam_depth=np.ones((h,w)),cam_world_points=xyz)
    record={'sensors':p,'sensor_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'control_steps':60}
    c=CompletionEvidence(tmp_path/'measurements')
    result=c.surface(record,1,'cam',[[5,5],[74,5],[74,74],[5,74]])
    assert result['supported'] and result['observation_ids']==[1]
    assert result['normal_world']==[0,0,1]
    assert 'evaluation' not in json.dumps(result)
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):c.surface(record,1,'cam',[[5,5],[74,5],[74,74]])


def test_measurement_module_has_no_robot_or_task_dependencies():
    import inspect
    import vlm_tamp.observation_completion as module
    text=inspect.getsource(module).lower()
    assert all(name not in text for name in ['microwave','libero','mujoco','plan_arc','execute_batch'])


def test_cleanup_preserves_completion_measurement_inputs(tmp_path):
    from run_subscription_dino_study import retained_inputs
    root=tmp_path/'agent-workspace/robot_data/attempt01'
    for n in [1,3,5,9]:
        d=root/f'obs{n:03d}';d.mkdir(parents=True);(d/'sensors.npz').write_bytes(b'test array evidence')
    event={'request':{'operation':'measure_surface','arguments':{'observation_id':5}},'response':{'ok':True,'result':{}}}
    (tmp_path/'tool-events.jsonl').write_text(json.dumps(event)+'\n')
    result=retained_inputs(tmp_path)
    assert (root/'obs005/sensors.npz').exists()
    assert not (root/'obs003/sensors.npz').exists()
    assert len(result['removed'])==1


def test_completion_protocol_never_intercepts_execute(tmp_path):
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/serve_subscription_robot.py'
    spec=importlib.util.spec_from_file_location('completion_service_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    robot=module.RobotService.__new__(module.RobotService)
    robot.calls=0;robot.dino_tools=False;robot.completion_checks=True
    robot.completion=CompletionEvidence(tmp_path);robot.state={'control_steps':120}
    executions=[];finishes=[]
    robot.execute_plan=lambda **args:executions.append(args) or {'all_commands_issued':True}
    robot.finish_attempt=lambda reason:finishes.append(reason)
    assert robot.dispatch('execute',{'plan_id':'fresh'})['all_commands_issued']
    assert executions==[{'plan_id':'fresh'}]
    rejected=robot.dispatch('finish',{'claim':'completed','reason':'No evidence'})
    assert not rejected['finished'] and not finishes
    assert robot.dispatch('finish',{'claim':'uncertain','reason':'Could not verify'})['finished']
    assert finishes==['agent_finish']


def test_advisory_completion_records_claim_without_certifying_it(tmp_path):
    c=CompletionEvidence(tmp_path)
    strict=c.finish_claim('completed',None,'Visual evidence only',120)
    assert not strict['finish_allowed']
    advisory=c.finish_claim('completed',None,'Visual evidence only',120,advisory=True)
    assert advisory['finish_allowed']
    assert advisory['completion_policy']=='advisory'
    assert not advisory['current_measurement_supported']
    assert not advisory['native_success_used']
    c.checks['old']={'supported':True,'control_steps':100}
    stale=c.finish_claim('completed','old','Fresh visual evidence; old measurement',120,advisory=True)
    assert not stale['current_measurement_supported']


def test_practice_finish_does_not_require_a_measurement_or_interrupt_motion(tmp_path):
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'scripts/serve_subscription_robot.py'
    spec=importlib.util.spec_from_file_location('advisory_service_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    robot=module.RobotService.__new__(module.RobotService)
    robot.calls=0;robot.dino_tools=False;robot.completion_checks=True;robot.learning_practice=True
    robot.completion=CompletionEvidence(tmp_path);robot.state={'control_steps':120}
    executions=[];finishes=[]
    robot.execute_plan=lambda **args:executions.append(args) or {'all_commands_issued':True}
    robot.finish_attempt=lambda reason:finishes.append(reason)
    assert robot.dispatch('execute',{'plan_id':'fresh'})['all_commands_issued']
    result=robot.dispatch('finish',{'claim':'completed','reason':'Current views support my interpretation'})
    assert result['finished'] and finishes==['agent_finish']
    assert result['completion_claim']['completion_policy']=='advisory'
    assert not result['completion_claim']['current_measurement_supported']
    assert executions==[{'plan_id':'fresh'}]


def test_material_chain_composes_rotation_and_does_not_connect_unrelated_patches(tmp_path):
    c=CompletionEvidence(tmp_path)
    angle=np.radians(20);r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    edge={'rotation':r.tolist(),'translation_m':[0,0,0],'orientation_stability_radius_deg':1.0}
    c.association_edges=[('initial','middle',edge),('middle','final',edge)]
    result=c.association_path('initial','final')
    assert abs(result['rigid_rotation_deg']-40)<1e-8
    assert result['orientation_stability_radius_deg']==2
    assert result['measurement_path']==['initial','middle','final']
    assert c.association_path('initial','unrelated') is None
    assert abs(c.association_path('final','initial')['rigid_rotation_deg']-40)<1e-8
