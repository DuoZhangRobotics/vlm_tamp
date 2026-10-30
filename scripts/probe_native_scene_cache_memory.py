"""Offline index-lifetime check: saved points, no model, simulation or motion."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from vlm_tamp.coupled_storage import read_json
from vlm_tamp.coupled_scene import Scene,RobotVolumes
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache,_TREE_CACHE
import vlm_tamp.native_scene_cache as policy

ROOT=Path(__file__).resolve().parents[1]


def rss_bytes():
    line=next(x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:'))
    return int(line.split()[1])*1024


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-entries',type=int,default=2,choices=[2,1000],help='1000 emulates the old effectively unbounded policy for a short capped probe')
    p.add_argument('--iterations',type=int,default=8)
    p.add_argument('--max-rss-gib',type=float,default=3.)
    p.add_argument('--point-stride',type=int,default=4)
    a=p.parse_args()
    if not 1<=a.iterations<=12 or a.point_stride<1:raise ValueError('Use a small offline probe')
    folder=ROOT/'results/open-hand-grasp-admission-20261030/open_microwave-state037'
    meta=read_json(folder/'attempt01/001-observation/metadata.json')
    record=read_json(folder/'attempt01/robot-chain.json');chain=RobotChain(record)
    spec=read_json(folder/'attempt01/002-grounding/model.json')
    robot=RobotVolumes(chain,meta['robot_collision_volumes_world'],meta['joint_positions'])
    with np.load(folder/'workspace/robot_data/attempt01/geometry002/points.npz') as data:
        points={k:data[k][::a.point_stride].copy() for k in ['static','moving','contact']}
    policy.MAX_CACHED_SCENES=a.cache_entries
    clear_native_cache()
    print(json.dumps({'event':'start','no_simulation_or_robot_motion':True,'cache_entries':a.cache_entries,
        'point_stride':a.point_stride,'point_counts':{k:len(v) for k,v in points.items()},'rss_bytes':rss_bytes()}),flush=True)
    for i in range(a.iterations):
        shift=np.array([0.,0.,i*.0002])
        scene=Scene.from_points(points['static'],points['moving']+shift,points['contact']+shift,spec,robot=robot)
        start=time.perf_counter()
        with NativeScene(scene,record,meta,meta['joint_positions'],.05,articulated=False) as native:
            accepted=native.validate(meta['joint_positions'])['accepted']
        rss=rss_bytes()
        print(json.dumps({'event':'query','index':i,'cached_scenes':len(_TREE_CACHE),'rss_bytes':rss,
            'accepted':accepted,'seconds':time.perf_counter()-start}),flush=True)
        if rss>a.max_rss_gib*1024**3:
            print(json.dumps({'event':'probe_memory_stop','rss_bytes':rss}),flush=True);break
    clear_native_cache()
    print(json.dumps({'event':'cleared','cached_scenes':len(_TREE_CACHE),'rss_bytes':rss_bytes()}),flush=True)


if __name__=='__main__':main()
