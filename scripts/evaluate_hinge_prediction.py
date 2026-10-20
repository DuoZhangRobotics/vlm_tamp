"""Evaluation-only hinge error; never imported or read by the policy runner."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.hinge_uncertainty import rotate_cloud,hinge_sensitivity

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-run',required=True)
    args=parser.parse_args();folder=ROOT/args.policy_run
    policy_hash=hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest()
    reference=read_json(folder/'scene-reference.json');arrays=np.load(reference['arrays_path'])
    spec=reference['spec'];origin=np.asarray(spec['joint_origin_world_m']);axis=np.asarray(spec['joint_axis_world'])
    source=Path(read_json(folder/'summary.json')['source'])
    steps=read_json(source/'initial/evaluation_contact_diagnostics.json')['steps']
    # This asset identity is confined to this post-hoc evaluator.
    truth=next(j for j in steps[-1]['evaluation_only_articulation_state'] if j['joint_name']=='microwave_1_microjoint')
    actual_origin=np.asarray(truth['anchor_world_m']);actual_axis=np.asarray(truth['axis_world'])
    if actual_axis@axis<0:actual_axis=-actual_axis
    delta=origin-actual_origin;sideways=delta-actual_axis*(delta@actual_axis)
    cloud=arrays['moving_reference'];rows=[]
    for degrees in [0.,15.,45.,90.]:
        nominal=rotate_cloud(cloud,origin,axis,np.radians(degrees))
        actual=rotate_cloud(cloud,actual_origin,actual_axis,np.radians(degrees))
        errors=np.linalg.norm(nominal-actual,axis=1)
        rows.append({'progress_deg':degrees,'median_point_error_m':float(np.median(errors)),
                     'p95_point_error_m':float(np.quantile(errors,.95)),'max_point_error_m':float(errors.max())})
    output=folder/'hinge-evaluation';output.mkdir(exist_ok=True)
    write_json(output/'sensitivity.json',hinge_sensitivity(cloud,origin,axis))
    report={'evaluation_only':True,'simulator_hinge_used_by_policy':False,
        'policy_run_summary_sha256_before_evaluation':policy_hash,
        'estimated_origin_world_m':origin,'true_origin_world_m':actual_origin,
        'estimated_axis':axis,'true_axis_with_matching_motion_sign':actual_axis,
        'perpendicular_axis_offset_m':float(np.linalg.norm(sideways)),
        'axis_direction_error_deg':float(np.degrees(np.arccos(np.clip(axis@actual_axis,-1,1)))),
        'rows':rows,'same_observed_partial_cloud_used_for_both_predictions':True,
        'unobserved_geometry_not_reconstructed':True,'no_planning_or_execution_performed':True,
        'policy_summary_unchanged':hashlib.sha256((folder/'summary.json').read_bytes()).hexdigest()==policy_hash}
    write_json(output/'summary.json',report)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.2),layout='constrained')
    for ax,degrees in zip(axes,[0.,45.,90.]):
        nominal=rotate_cloud(cloud,origin,axis,np.radians(degrees))
        actual=rotate_cloud(cloud,actual_origin,actual_axis,np.radians(degrees))
        ax.scatter(*actual[::4,:2].T,s=2,alpha=.5,c='#5965b2',label='True hinge: evaluation only')
        ax.scatter(*nominal[::4,:2].T,s=2,alpha=.5,c='#d08029',label='Estimated hinge: policy model')
        ax.scatter(*origin[:2],marker='x',s=60,c='#d08029');ax.scatter(*actual_origin[:2],marker='x',s=60,c='#5965b2')
        error=next(r['median_point_error_m'] for r in rows if r['progress_deg']==degrees)
        ax.set(title=f'{degrees:g}°: median displacement {error*1000:.1f} mm',xlabel='World x (m)',ylabel='World y (m)',aspect='equal')
        ax.set_xlim(-.14,.29);ax.set_ylim(-.22,.13)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7,loc='best')
    fig.suptitle('Predicted observed door cloud: hinge error grows with opening angle\n'
                 'Ground-truth hinge is used only for this evaluation; no true hinge data enters the planner',fontsize=10)
    fig.savefig(output/'prediction-error.png',dpi=150,bbox_inches='tight',pad_inches=.15);plt.close(fig)
    print({'perpendicular_hinge_error_mm':report['perpendicular_axis_offset_m']*1000,
           'predicted_point_error_at_90_mm':rows[-1]['median_point_error_m']*1000})


if __name__=='__main__':main()
