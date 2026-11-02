"""Inspect the actor's original coordinates without inventing later clicks."""
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from luna_selection_history import selection_history, project


def setup(tmp_path):
    study=tmp_path/'study';run=study/'trials/case';run.mkdir(parents=True)
    return study,run


def capture(run,oid,step,camera='front',color='white'):
    folder=run/f'agent-workspace/robot_data/attempt01/obs{oid:03d}';folder.mkdir(parents=True,exist_ok=True)
    Image.new('RGB',(100,80),color).save(folder/(camera+'.png'))
    (folder/'robot.json').write_text(json.dumps({'control_steps':step}))
    calibration={'intrinsic':[[100,0,50],[0,100,40],[0,0,1]],
        'camera_to_world':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
    path=folder/'calibration.json';calibrations=json.loads(path.read_text()) if path.exists() else {}
    calibrations[camera]=calibration;path.write_text(json.dumps(calibrations))
    return folder,calibration


def event(n,op,args,result,ok=True):
    return {'sequence':n,'request':{'operation':op,'arguments':args},'response':{'ok':ok,'result':result}}


def test_push_draws_exact_input_pixel_on_before_image_not_result_capture(tmp_path):
    study,run=setup(tmp_path);before,_=capture(run,1,60);capture(run,4,120,color='black')
    original=hashlib.sha256((before/'front.png').read_bytes()).hexdigest()
    events=[event(1,'reset',{}, {'attempt':1,'observation_id':1}),
        event(4,'push',{'observation_id':1,'camera':'front','contact_pixel_xy':[13,27],
            'direction_world':[1,0,0],'region_xyxy':[2,3,30,40]},
            {'contact':{'point_world_m':[0,0,1]},'stop_reason':'approach_not_reached',
             'observation':{'observation_id':4}})]
    row=selection_history(study,run,events)[0]
    assert row['contact_pixel_xy']==[13,27] and row['pixel_source']=='Luna request'
    assert row['image'].endswith('obs001/front.png') and row['control_step']==60
    assert row['direction_arrow']['start']==[50.,40.]
    assert row['direction_arrow']['end']==[55.,40.]
    assert hashlib.sha256((before/'front.png').read_bytes()).hexdigest()==original


def test_continuation_never_presents_a_carried_pixel_as_a_new_annotation(tmp_path):
    study,run=setup(tmp_path);capture(run,1,60);capture(run,3,100)
    events=[event(1,'reset',{}, {'observation_id':1}),event(3,'observe',{}, {'observation_id':3}),
        event(4,'continue_push',{'direction_world':[1,0,0],'distance_m':.1},
            {'contact':{'camera':'front','contact_pixel_xy':[20,20],'point_world_m':[0,0,1]},
             'observation':{'observation_id':5}})]
    row=selection_history(study,run,events)[0]
    assert row['contact_pixel_xy'] is None
    assert row['image'].endswith('obs003/front.png') and row['control_step']==100
    assert row['projected_world_point_xy']==[50.,40.]
    assert 'No new pixel' in row['notes'][0]


def test_ground_without_observation_id_uses_current_capture_and_links_requested_model(tmp_path):
    study,run=setup(tmp_path);capture(run,1,60);capture(run,1,60,'side')
    regions={'camera':'front','contact_pixel_xy':[20,30],'hinge_pixels_xy':[[10,12],[10,70]],
        'moving_panel_box_xyxy':[2,3,90,75],'initial_image_direction_xy':[0,1]}
    model={'joint_origin_world_m':[0,0,1],'joint_axis_world':[0,1,0]}
    events=[event(1,'reset',{}, {'observation_id':1}),
        event(2,'ground',{'regions':regions},{'model':model,'geometry_epoch':3}),
        event(3,'push',{'observation_id':1,'camera':'side','contact_pixel_xy':[25,30],
            'joint_degrees':10,'joint_model_epoch':3},{'contact':{'point_world_m':[.1,0,1],'direction_world':[0,1,0]}})]
    ground,push=selection_history(study,run,events)
    assert ground['hinge_pixels_xy']==regions['hinge_pixels_xy']
    assert ground['observation_id']==1
    assert push['model_reference']['event']==2 and push['model_reference']['camera']=='front'
    assert push['hinge_pixels_xy'] is None  # Never transplant source-view pixels.
    assert push['model_axis_line'] is not None
    assert push['direction_source'].startswith('tool-computed')


def test_missing_original_image_is_not_replaced_by_a_later_capture(tmp_path):
    study,run=setup(tmp_path);capture(run,2,100)
    events=[event(1,'reset',{}, {'observation_id':2}),event(2,'push',
        {'observation_id':1,'camera':'front','contact_pixel_xy':[10,20]}, {})]
    row=selection_history(study,run,events)[0]
    assert row['image'] is None and row['contact_pixel_xy']==[10,20]
    assert row['observation_id']==1


def test_mask_provenance_and_optional_evaluator_line_remain_separate(tmp_path):
    study,run=setup(tmp_path);capture(run,1,60)
    overlay=run/'agent-workspace/robot_data/part.png';Image.new('RGB',(100,80),'green').save(overlay)
    parts=[{'part_id':'part1','camera':'front','observation_id':1,'overlay':'robot_data/part.png',
        'mask_box_xyxy':[1,2,30,40],'caller_label':'door'}]
    (run/'selection-evaluation-reference.json').write_text(json.dumps({'states':{'60':{
        'joint_origin_world_m':[0,0,1],'joint_axis_world':[0,1,0]}}}))
    events=[event(1,'reset',{}, {'observation_id':1}),
        event(2,'segment_parts',{'observation_id':1,'camera':'front','box_xyxy':[0,0,40,60]},{'parts':parts}),
        event(3,'push',{'observation_id':1,'camera':'front','contact_pixel_xy':[5,6],
            'moving_part_id':'part1'},{})]
    segmentation,push=selection_history(study,run,events)
    assert segmentation['regions'][0]['source']=='Luna request'
    assert push['regions'][0]['source']=='SAM output'
    assert push['masks'][0]['part_id']=='part1' and push['masks'][0]['same_capture']
    assert push['contact_pixel_xy']==[5,6] and push['evaluator_axis_line']
    assert 'never supplied to Luna' in push['evaluator_reference_scope']


def test_projection_accounts_for_camera_rotation_and_rejects_behind_camera(tmp_path):
    _,run=setup(tmp_path);_,camera=capture(run,1,60)
    camera['camera_to_world']=[[0,-1,0,1],[1,0,0,2],[0,0,1,0],[0,0,0,1]]
    assert project([1,2.1,1],camera)==pytest.approx([60,40])
    assert project([1,2,-1],camera) is None


def test_sam_prompt_clicks_are_not_contact_pixels(tmp_path):
    study,run=setup(tmp_path);capture(run,1,60)
    rows=selection_history(study,run,[event(1,'segment_parts',{'observation_id':1,'camera':'front',
        'points_xy':[[0,0],[25,30]],'point_labels':[1,0]}, {})])
    assert rows[0]['contact_pixel_xy'] is None
    assert rows[0]['segmentation_points']==[{'xy':[0,0],'include':1},{'xy':[25,30],'include':0}]
