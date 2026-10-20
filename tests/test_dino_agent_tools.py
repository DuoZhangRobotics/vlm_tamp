import numpy as np
import pytest
from vlm_tamp.dino_agent_tools import propose_landmarks,match_landmarks


def fixture():
    y,x=np.indices((48,64));depth=np.ones((48,64));world=np.stack([x/100,y/100,depth],axis=-1)
    rng=np.random.default_rng(0);features=rng.normal(size=(48,64,12)).astype(np.float32)
    return features,world,depth


def test_landmarks_preserve_pixels_and_observed_metric_positions():
    f,w,d=fixture();robot=np.zeros_like(d);robot[:,:24]=d[:,:24]
    w[20:24,30:34]=np.nan
    result=propose_landmarks(f,w,d,box=[10,8,55,40],count=8,robot_depth=robot)
    pixels=[]
    for row in result['candidates']:
        x,y=row['pixel_xy'];pixels.append((x,y))
        assert 24<=x<=55 and 8<=y<=40
        assert np.isfinite(w[y,x]).all()
        assert np.allclose(row['world_m'],w[y,x])
    assert len(set(pixels))==len(pixels)>0


def test_identity_correspondences_have_exact_matches():
    f,w,d=fixture();pixels=[[30,22],[40,36]]
    result=match_landmarks(f,f,w,w,d,d,pixels)
    for expected,row in zip(pixels,result['matches']):
        assert row['candidates'][0]['pixel_xy']==expected
        assert row['reciprocal_pixel_error_px']==0
        assert row['candidates'][0]['observed_displacement_m']==0
        assert not row['ambiguous']


def test_repeated_appearance_is_reported_ambiguous():
    f,w,d=fixture();f[:]=1
    result=match_landmarks(f,f,w,w,d,d,[[30,22]])
    assert result['matches'][0]['ambiguous']
    assert result['matches'][0]['nonlocal_similarity_margin']==0


def test_invalid_depth_and_reference_pixels_do_not_invent_matches():
    f,w,d=fixture();d[22,30]=np.nan
    result=match_landmarks(f,f,w,w,d,d,[[30,22]])
    assert result['matches'][0]['candidates']==[]
    with pytest.raises(ValueError):match_landmarks(f,f,w,w,d,d,[[999,22]])
    with pytest.raises(ValueError):propose_landmarks(f,w,d,box=[30,20,5,5])
