"""Shared admission and diagnostics protect every caller before numerical work."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest

from vlm_tamp.robot_contracts import (GROUND_HELP, RequestValidationError, operation_schemas,
                                     validate_ground_regions, validate_operation)


def regions():
    return {'camera':'frontview','contact_box_xyxy':[10,20,30,80],
            'moving_panel_box_xyxy':[5,10,90,95],'hinge_pixels_xy':[[8,15],[8,90]],
            'initial_image_direction_xy':[0,1]}


SIZES={'frontview':{'width':100,'height':100}}


@pytest.mark.parametrize('field,value',[
    ('initial_image_direction_xy',[0,0]),
    ('initial_image_direction_xy',[float('nan'),1]),
    ('initial_image_direction_xy',[False,1]),
    ('initial_image_direction_xy',[1]),
    ('initial_image_direction_xy','down'),
    ('contact_box_xyxy',[10,20,10,80]),
    ('contact_box_xyxy',[10,20,float('inf'),80]),
    ('contact_box_xyxy',[10,20,130,80]),
    ('moving_panel_box_xyxy',[90,10,5,95]),
    ('hinge_pixels_xy',[[8,-1],[8,90]]),
    ('hinge_pixels_xy',[[8,15],[8,100]]),
    ('hinge_pixels_xy',[[8,15],[8,15]]),
    ('camera','missing_camera'),
])
def test_invalid_grounding_is_a_request_error_not_search_failure(field,value):
    request=regions();request[field]=value
    with pytest.raises(RequestValidationError) as exc:
        validate_operation('ground',{'regions':request},SIZES)
    diagnostic=exc.value.request_diagnostic
    assert diagnostic['status']=='invalid_request'
    assert diagnostic['physical_motion_issued'] is False
    assert diagnostic['expensive_work_started'] is False


def test_missing_direction_and_flat_argument_envelope_are_not_defaulted():
    request=regions();del request['initial_image_direction_xy']
    with pytest.raises(RequestValidationError):validate_operation('ground',{'regions':request},SIZES)
    with pytest.raises(RequestValidationError):validate_operation('ground',regions(),SIZES)
    with pytest.raises(RequestValidationError):validate_operation('ground(regions)',{'regions':regions()},SIZES)


def test_explicit_unknown_and_nonunit_directions_preserve_input():
    for direction in [None,[0,.25],[-5,1],[0,1e-300],[1e300,1e300]]:
        request=regions();request['initial_image_direction_xy']=direction;before=copy.deepcopy(request)
        assert validate_ground_regions(request,image_sizes=SIZES) is request
        assert request==before


def test_rectangular_camera_uses_its_own_width_and_height():
    request=regions();request['hinge_pixels_xy']=[[8,15],[8,120]]
    with pytest.raises(RequestValidationError,match='200x100'):
        validate_ground_regions(request,image_sizes={'frontview':{'width':200,'height':100}})


def test_service_rejects_before_geometry_or_state_changes(monkeypatch):
    import serve_subscription_robot as service
    robot=service.RobotService.__new__(service.RobotService)
    robot.sim=object();robot.cameras={'frontview':{'rgb':np.zeros((100,100,3),np.uint8)}}
    robot.epoch=7;robot.spec={'untouched':True}
    monkeypatch.setattr(service,'ground_handle',lambda *a,**k:pytest.fail('Geometry must not start'))
    request=regions();request['initial_image_direction_xy']=[0,0]
    with pytest.raises(RequestValidationError):robot.ground(request)
    assert robot.epoch==7 and robot.spec=={'untouched':True}


def test_direct_shared_geometry_call_also_rejects_before_plane_fit(monkeypatch):
    from vlm_tamp import handle_contact
    cameras={'frontview':{'rgb':np.zeros((100,100,3),np.uint8)}}
    monkeypatch.setattr(handle_contact,'upright_plane',lambda *a,**k:pytest.fail('Plane fitting must not start'))
    request=regions();request['initial_image_direction_xy']=[0,0]
    with pytest.raises(RequestValidationError):handle_contact.ground_handle(cameras,request,{}, {})
    request['initial_image_direction_xy']=None
    with pytest.raises(RequestValidationError,match='unresolved'):
        handle_contact.ground_handle(cameras,request,{}, None)


def test_projection_tie_is_unresolved_and_direction_scale_is_irrelevant():
    from vlm_tamp.handle_contact import image_motion_axis
    camera={'intrinsic':np.eye(3),'camera_to_world':np.eye(4)}
    anchor=np.array([1.,0.,5.]);pivot=np.array([0.,0.,5.])
    axis,scores=image_motion_axis(anchor,pivot,camera,[1,0])
    assert axis is None and scores[0]==pytest.approx(scores[1])
    assert image_motion_axis(anchor,pivot,camera,None)==(None,[])
    for hint in [[0,1],[0,20],[0,1e-300],[0,1e300]]:
        np.testing.assert_array_equal(image_motion_axis(anchor,pivot,camera,hint)[0],[0,0,1])
    np.testing.assert_array_equal(image_motion_axis(anchor,pivot,camera,[0,-1])[0],[0,0,-1])


def test_rpc_and_compact_view_preserve_precise_rejection(tmp_path):
    from serve_subscription_robot import handle_request
    from vlm_tamp.compact_robot_interface import compact_response
    work=tmp_path/'work';work.mkdir();replies=work/'replies';replies.mkdir()
    out=tmp_path/'private';out.mkdir();request_path=work/('a'*32+'.json')
    request=regions();request['initial_image_direction_xy']=[0,0]
    request_path.write_text(json.dumps({'operation':'ground','arguments':{'regions':request}}))
    class Service:
        workspace=work
        output=out
        calls=0
        def dispatch(self,operation,arguments):
            self.calls+=1
            validate_operation(operation,arguments,SIZES)
            pytest.fail('Invalid request must not be admitted')
    handle_request(Service(),request_path,replies)
    reply=json.loads((replies/request_path.name).read_text())
    display=compact_response(reply)
    assert display['request_diagnostic']==reply['request_diagnostic']
    assert display['request_diagnostic']['field'].endswith('initial_image_direction_xy')
    assert 'null' in display['error'] and 'planning_diagnostic' not in display


def test_all_registered_schemas_are_valid_and_basic_operations_are_strict():
    for item in operation_schemas().values():
        jsonschema.Draft202012Validator.check_schema(item['function']['parameters'])
        assert item['function']['parameters']['additionalProperties'] is False
    with pytest.raises(RequestValidationError):validate_operation('execute',{'plan_id':'p1','speed':float('nan')})
    with pytest.raises(RequestValidationError):validate_operation('gripper',{'close':'false'})
    with pytest.raises(RequestValidationError):validate_operation('observe',{'unexpected':1})


def test_schema_help_is_local_and_does_not_expose_all_contracts(tmp_path,capsys):
    from vlm_tamp.compact_robot_interface import install,main
    install(tmp_path,'- observe(): current sensors.\n'+GROUND_HELP,attempts=1,initial_attempts=1,practice_mode=False)
    main(lambda *a,**k:pytest.fail('Schema help must not make an RPC'),tmp_path,['schema','ground'])
    schema=json.loads(capsys.readouterr().out)
    assert schema['function']['name']=='ground'
    index=(tmp_path/'tools.txt').read_text()
    assert '"properties"' not in index
    assert 'schema OPERATION' in index


def test_preview_dimensions_and_direction_source_survive_compaction():
    from vlm_tamp.compact_robot_interface import compact_response
    raw={'ok':True,'result':{'images':{'frontview':'a.png','agentview':'b.png','side':'c.png'},
        'image_sizes':{'frontview':{'width':512,'height':512},'agentview':{'width':640,'height':480},'side':{'width':600,'height':500}},
        'image_coordinate_system':'original_pixels_top_left_x_right_y_down',
        'fit_diagnostics':{'motion_direction_check':{'source':'observed depth','image_hint_status':'unresolved',
            'image_direction_xy':None,'selected_axis':[0,0,-1]}}}}
    result=compact_response(raw)['result']
    assert set(result['image_sizes'])=={'frontview','agentview'}
    assert result['fit_diagnostics']['motion_direction_check']['image_hint_status']=='unresolved'
    assert result['image_coordinate_system'].startswith('original_pixels')
