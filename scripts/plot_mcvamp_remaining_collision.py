"""Scientific diagnostic of a pad/static-cloud intersection after the palm fix."""
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from vlm_tamp.coupled_storage import read_json
from vlm_tamp.coupled_scene import _transform_volume
from vlm_tamp.rekep_libero.geometry import pose_matrix
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain

ROOT=Path(__file__).resolve().parents[1]


def main():
    audit=ROOT/'results/mcvamp-convex-fix-audit-20261015'
    source=ROOT/'results/mcvamp-convex-frozen-grasp-state11-v2-20261015/open_microwave-state011'
    folder=source/'plan-00/horizon-1'
    inputs=read_json(folder/'planning-input.json');a=np.load(read_json(folder/'scene-reference.json')['arrays_path'])
    evidence=read_json(audit/'remaining-finger-collision.json');q=np.load(audit/'ee-continuation-probe.npy')[-1]
    chain=RobotChain(read_json(source/'robot-chain.json'))
    volume=next(v for v in inputs['metadata']['robot_collision_volumes_world'] if v['robot_geom_name']==evidence['configured_volume'])
    transform=pose_matrix(chain.forward(q))@np.linalg.inv(pose_matrix(chain.forward(inputs['reference_joints'])))
    vertices=np.asarray(_transform_volume(volume,transform)['vertices_world'])
    witnesses=np.asarray(evidence['witness_points_world']);center=witnesses.mean(0)
    spec=inputs['task_spec'];origin=np.asarray(spec['joint_origin_world_m'])
    handle=origin+Rotation.from_rotvec(np.radians(3.5)*np.asarray(spec['joint_axis_world'])).apply(a['contact_reference']-origin)
    points=a['static_points'];points=points[np.linalg.norm(points-center,axis=1)<.035]
    handle=handle[np.linalg.norm(handle-center,axis=1)<.035]
    fig,axes=plt.subplots(1,2,figsize=(10,4.6),layout='constrained')
    for axis,dims,title in zip(axes,[(0,1),(1,2)],['Top view','Side view']):
        p=(vertices-center)[:,dims]*1000;hull=ConvexHull(p)
        axis.fill(*p[hull.vertices].T,color='#4087cb',alpha=.24,label='Configured finger pad')
        axis.scatter(*((points-center)[:,dims]*1000).T,s=9,color='#89929b',alpha=.5,label='Static-tagged observations')
        axis.scatter(*((handle-center)[:,dims]*1000).T,s=7,color='#269b70',alpha=.7,label='Predicted moving handle')
        axis.scatter(*((witnesses-center)[:,dims]*1000).T,s=55,color='#d34343',marker='x',linewidths=2,label='Blocked static points')
        axis.set(xlabel=f'{"xyz"[dims[0]]} offset (mm)',ylabel=f'{"xyz"[dims[1]]} offset (mm)',
                 xlim=(-32,32),ylim=(-32,32),title=title,aspect='equal')
        axis.grid(alpha=.2)
    axes[1].legend(fontsize=8,loc='lower left')
    fig.suptitle('First blocked pull posture: predicted +3.5°; no physical motion\n'
                 'Three static-tagged points enter the pad by 0.27–0.54 mm; surface ownership remains unverified',fontsize=11)
    output=ROOT/'results/mcvamp-corrections-report';output.mkdir(exist_ok=True)
    fig.savefig(output/'remaining-finger-collision.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':main()
