import gc
from pathlib import Path

import numpy as np
import pytest

from vlm_tamp.native_scene_cache import NativeSceneCache,MAX_CACHED_SCENES


class Library:
    def __init__(self):self.released=[]
    def mcvamp_destroy(self,handle):self.released.append(handle)


def test_scene_history_has_a_bound_and_eviction_releases_native_ownership():
    cache=NativeSceneCache();library=Library()
    for i in range(12):
        _,reused=cache.acquire(str(i),library,lambda i=i:i+1)
        assert not reused and len(cache)<=MAX_CACHED_SCENES
    assert len(library.released)==12-MAX_CACHED_SCENES
    cache.clear();assert sorted(library.released)==list(range(1,13))
    cache.clear();assert len(library.released)==12


def test_reuse_keeps_recent_geometry_and_does_not_rebuild():
    cache=NativeSceneCache();library=Library()
    cache.acquire('a',library,lambda:1);cache.acquire('b',library,lambda:2)
    _,reused=cache.acquire('a',library,lambda:pytest.fail('unexpected rebuild'))
    assert reused
    cache.acquire('c',library,lambda:3)
    assert list(cache)==['a','c'] and library.released==[2]
    cache.clear()


def test_live_native_clone_survives_cache_eviction_and_clear():
    pytest.importorskip('vamp')
    from vlm_tamp.coupled_storage import read_json
    from vlm_tamp.coupled_scene import Scene
    from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache,_TREE_CACHE
    root=Path(__file__).resolve().parents[1]
    folder=root/'results/small-graspgen-20261029-planner-path/open_microwave-state037/attempt01'
    if not folder.exists():pytest.skip('Recorded robot calibration unavailable')
    meta=read_json(folder/'opened/metadata.json');record=read_json(folder/'robot-chain.json')
    palm=next(v for v in meta['robot_collision_volumes_world'] if v['robot_geom_name']=='gripper0_hand_collision')
    point=np.mean(palm['vertices_world'],axis=0)[None];far=point+3
    spec={'joint_origin_world_m':[0,0,0],'joint_axis_world':[0,0,1],
          'front_plane':{'point_world_m':[0,0,0]},'panel_normal_world':[1,0,0]}
    def scene(i):return Scene.from_points(point+np.array([float(i),0,0]),far,far,spec)
    clear_native_cache()
    with NativeScene(scene(0),record,meta,meta['joint_positions'],.05,articulated=False) as live:
        before=live.validate(meta['joint_positions'])
        assert not before['accepted']
        for i in range(1,8):
            with NativeScene(scene(i),record,meta,meta['joint_positions'],.05,articulated=False):pass
            assert len(_TREE_CACHE)<=MAX_CACHED_SCENES
        clear_native_cache()
        assert live.validate(meta['joint_positions'])['accepted']==before['accepted']
    assert not live._release.alive
    gc.collect();clear_native_cache()
