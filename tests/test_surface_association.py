import hashlib
import json
import numpy as np
from vlm_tamp.surface_association import rigid_correspondences,associate_surfaces
from vlm_tamp.observation_completion import plane_fit


def test_rigid_fit_rejects_nonrigid_geometry_despite_image_similarity():
    rng=np.random.default_rng(2);a=rng.uniform(-.2,.2,(30,3));b=rng.uniform(-.2,.2,(30,3))
    assert not rigid_correspondences(a,b)['supported']
    assert rigid_correspondences(a,a+[.1,.2,.3])['supported']


def test_collinear_feature_matches_do_not_certify_a_plane():
    a=np.c_[np.linspace(0,1,20),np.zeros((20,2))]
    assert not rigid_correspondences(a,a+[.1,0,0])['supported']


def test_in_plane_rotation_is_visible_to_correspondences_even_when_normal_is_unchanged():
    rng=np.random.default_rng(8);a=np.c_[rng.uniform(-.2,.2,(30,2)),np.ones(30)]
    angle=np.radians(40);r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    result=rigid_correspondences(a,a@r.T)
    assert result['supported'] and abs(result['rigid_rotation_deg']-40)<1e-6
    assert abs(result['rotation'][2][2]-1)<1e-8


def test_automatic_association_supports_same_material_and_rejects_unrelated_texture(tmp_path):
    rng=np.random.default_rng(4);h=w=160;y,x=np.mgrid[:h,:w]
    texture=rng.integers(0,256,(h,w,3),dtype=np.uint8)
    points=np.stack([x/300,y/300,np.ones((h,w))],-1)
    angle=np.radians(35);r=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    def frame(name,rgb,xyz):
        p=tmp_path/(name+'.npz');np.savez_compressed(p,cam_rgb=rgb,cam_depth=np.ones((h,w)),cam_world_points=xyz)
        record={'sensors':p,'sensor_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        fit,_=plane_fit(xyz.reshape(-1,3));fit.update(camera='cam',polygon_xy=[[5,5],[154,5],[154,154],[5,154]])
        return record,fit
    first=frame('first',texture,points);second=frame('second',texture,points@r.T+[.1,0,.2])
    matched,a,b=associate_surfaces(*first,*second)
    json.dumps({'ok':True,'result':matched})
    assert matched['supported'] and matched['inliers']>=6
    assert matched['transformed_normal_disagreement_deg']<.1
    other=frame('other',rng.integers(0,256,(h,w,3),dtype=np.uint8),points@r.T)
    assert not associate_surfaces(*first,*other)[0]['supported']


def test_complete_public_measurement_response_serializes_and_checks_goal(tmp_path):
    from vlm_tamp.observation_completion import CompletionEvidence
    rng=np.random.default_rng(19);h=w=160;y,x=np.mgrid[:h,:w]
    rgb=rng.integers(0,256,(h,w,3),dtype=np.uint8)
    xyz=np.stack([x/300,y/300,np.ones((h,w))],-1)
    angle=np.radians(35);r=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    protocol=CompletionEvidence(tmp_path/'completion')
    protocol.set_contract('Observed rigid orientation', [{'id':'angle','quantity':'rigid_rotation_deg','minimum':30,'maximum':40,'rationale':'Synthetic known motion'}],60)
    ids=[]
    for n,points in enumerate([xyz,xyz@r.T]):
        path=tmp_path/f'frame{n}.npz';np.savez_compressed(path,cam_rgb=rgb,cam_depth=np.ones((h,w)),cam_world_points=points)
        record={'sensors':path,'sensor_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'control_steps':60+40*n}
        result=protocol.surface(record,n,'cam',[[5,5],[154,5],[154,154],[5,154]])
        json.dumps({'ok':True,'result':result});ids.append(result['measurement_id'])
    result=protocol.compare(*ids);json.dumps({'ok':True,'result':result})
    assert result['quantity_supported']['rigid_rotation_deg']
    assert abs(result['values']['rigid_rotation_deg']-35)<1e-5
    check=protocol.check({'angle':result['measurement_id']},100,'Matched the same synthetic texture and metric points.')
    assert check['supported']
    assert protocol.finish_claim('completed',check['check_id'],'Measured condition met',100)['finish_allowed']
