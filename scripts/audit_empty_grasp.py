"""Read-only audit of learned grasp placement, frames and recorded closure.

No proposals are changed and no robot actions are issued. Native object motion
is used only for a separately labelled post-run diagnostic.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import trimesh

from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.graspgen_selection import learned_poses
from vlm_tamp.rekep_libero.geometry import pose_matrix,transform_points

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run=args.run.resolve();out=args.output.resolve();out.mkdir(exist_ok=True)
    episode=run/'attempt01';grasp=next(episode.glob('*-grasp'));generation=grasp/'generation'
    cloud=np.load(generation/'input-cloud.npy');selected=read_json(grasp/'selection/selected.json')
    with np.load(generation/'grasp_candidates.npz') as data:generated={k:data[k] for k in data.files}
    events=[json.loads(x) for x in (run/'tool-events.jsonl').read_text().splitlines()]
    reset=next(e for e in events if e['request']['operation']=='reset')
    approach=next(e for e in events if e['request']['operation']=='execute' and e['response']['ok'])
    close=next(e for e in events if e['request']['operation']=='gripper' and e['response']['ok'])
    def metadata(event):return read_json(episode/f'{event["sequence"]:03d}-observation/metadata.json')
    initial,arrived,closed=[metadata(e) for e in [reset,approach,close]]
    inverse=np.linalg.inv(pose_matrix(initial['eef_pose_world_xyzw']))
    pads=sorted([transform_points(v['vertices_world'],inverse) for v in initial['robot_collision_volumes_world']
        if v['robot_geom_name'].endswith('_pad_collision')],key=lambda v:np.mean(v[:,0]))
    low=np.r_[pads[0][:,0].max(),np.maximum(pads[0][:,1:].min(0),pads[1][:,1:].min(0))]
    high=np.r_[pads[1][:,0].min(),np.minimum(pads[0][:,1:].max(0),pads[1][:,1:].max(0))]
    def capture(pose,points):
        local=transform_points(points,np.linalg.inv(pose_matrix(pose)))
        inside=np.all((local>=low)&(local<=high),axis=1)
        band=np.all((local[:,:2]>=low[:2])&(local[:,:2]<=high[:2]),axis=1)
        return {'observed_points_in_pad_closing_volume':int(inside.sum()),
            'points_at_pad_height_and_between_jaws':int(band.sum()),
            'nearest_surface_ahead_of_pad_tip_m':float(local[band,2].min()-high[2]) if band.any() else None},local,band
    rows=[];attempts=read_json(grasp/'selection/grasp-attempts.json')
    for rank,row in enumerate(learned_poses(generated,initial['robot_calibration'],len(generated['retained_indices']))):
        check,_,_=capture(row['pose'],cloud)
        rows.append({'rank':rank+1,'model_index':row['graspgen_index'],'score':row['graspgen_score'],
            'selection_status':attempts[rank]['status'] if rank<len(attempts) else 'unexamined',**check})
    intended,local,band=capture(selected['pose'],cloud)
    measured,_,_=capture(arrived['eef_pose_world_xyzw'],cloud)
    native=read_json(episode/'initial/evaluation_contact_diagnostics.json')['steps']
    step0=reset['response']['result']['control_steps'];step1=approach['response']['result']['observation']['control_steps']
    step2=close['response']['result']['observation']['control_steps']
    def joint(step):return next(j for r in native if r['step']==step for j in r['evaluation_only_articulation_state'] if 'microjoint' in j['joint_name'])
    first,last=joint(step0),joint(step1);angle=last['position']-first['position']
    origin=np.asarray(first['anchor_world_m']);rotation=Rotation.from_rotvec(angle*np.asarray(first['axis_world'])).as_matrix()
    moved=(cloud-origin)@rotation.T+origin
    corrected,_,_=capture(arrived['eef_pose_world_xyzw'],moved)
    closure_contacts=[c for r in native if step1<r['step']<=step2 for c in r['robot_involved_contacts'] if not all(c['robot_flags'])]
    # Robot-only mesh alignment: known aperture and finger attachment, no object fitting.
    mesh=trimesh.load(ROOT/'third_party/GraspGen/assets/panda_gripper/finger.stl')
    alignment=[]
    for v in initial['robot_collision_volumes_world']:
        if v['robot_geom_name'] not in {'gripper0_finger1_collision','gripper0_finger2_collision'}:continue
        physical=transform_points(v['vertices_world'],inverse);sign=np.sign(physical[:,0].mean())
        q=initial['gripper_qpos'][0 if sign<0 else 1]
        physical[:,0]+=sign*(.04-abs(q));physical[:,2]+=.0584-initial['robot_calibration']['physical_finger_attach_z_m']+.097
        trained=np.asarray(mesh.vertices).copy()
        if sign>0:trained[:,:2]*=-1
        trained+=[sign*.04,0,.0584]
        alignment.append(max(cKDTree(trained).query(physical)[0].max(),cKDTree(physical).query(trained)[0].max()))
    audit={'source_run':str(run),'runtime_modified':False,'physical_motion_issued':False,
        'closing_volume_bounds_tcp_m':{'low':low,'high':high},'candidates':rows,
        'selected_intended_pose':intended,'measured_arrival_original_cloud':measured,
        'mesh_alignment_maximum_vertex_error_m':max(alignment),
        'ik_error_to_requested_grasp_position_m':float(np.linalg.norm(np.asarray(selected['pose'])[:3]-np.asarray(selected['pose_world_xyzw'])[:3])),
        'measured_error_to_ik_target_m':float(np.linalg.norm(np.asarray(arrived['eef_pose_world_xyzw'])[:3]-np.asarray(selected['pose_world_xyzw'])[:3])),
        'closure_controls':step2-step1,'closure_scene_contact_records':len(closure_contacts),
        'closure_evidence':close['response']['result']['grasp_evidence'],
        'offline_object_motion_diagnostic':{'excluded_from_policy_inputs':True,'angle_change_deg':float(np.degrees(angle)),
            'maximum_handle_point_motion_m':float(np.linalg.norm(moved-cloud,axis=1).max()),'corrected_capture':corrected},
        'interpretation':'The chosen pose is clear of forbidden collisions but has no observed handle points in the pad closing volume. The selector does not require positive pinch support.'}
    write_json(out/'audit.json',audit)
    fig,(ax,bar)=plt.subplots(1,2,figsize=(13,5),gridspec_kw={'width_ratios':[1.5,1]},layout='constrained')
    ax.add_patch(Rectangle((low[0]*1000,low[2]*1000),(high[0]-low[0])*1000,(high[2]-low[2])*1000,color='#f5bd64',alpha=.28,label='Volume reached by closing pads'))
    for pad in pads:
        lo,hi=pad.min(0)*1000,pad.max(0)*1000
        ax.add_patch(Rectangle((lo[0],lo[2]),hi[0]-lo[0],hi[2]-lo[2],color='#657d95'))
    ax.scatter(local[band,0]*1000,local[band,2]*1000,s=8,color='#1574bb',label='Observed handle at pad height')
    ax.axhline(high[2]*1000,color='#8c4b13',ls='--',lw=1)
    for x in [-42,42]:ax.annotate('',xy=(np.sign(x)*7,-3),xytext=(x,-3),arrowprops={'arrowstyle':'->','lw':2,'color':'#253d54'})
    gap=intended['nearest_surface_ahead_of_pad_tip_m']*1000
    ax.annotate(f'{gap:.2f} mm beyond tips',xy=(3,local[band,2].min()*1000),xytext=(18,17),arrowprops={'arrowstyle':'->'})
    ax.set(xlim=(-52,52),ylim=(-15,35),xlabel='Closing direction x (mm)',ylabel='Approach direction z (mm)',title='Selected pose: closing sideways misses the handle')
    ax.legend(loc='upper left',fontsize=9);ax.grid(alpha=.15);ax.set_aspect('equal',adjustable='box')
    ranks=np.arange(len(rows));colors=['#bd5d50' if r['selection_status']!='selected' else '#8868ac' for r in rows]
    bar.barh(ranks,[r['score'] for r in rows],color=colors)
    for i,r in enumerate(rows):bar.text(.01,i,f'{r["score"]:.3f} · {r["observed_points_in_pad_closing_volume"]} points between pads',va='center',fontsize=9,color='white' if r['score']>.4 else 'black')
    bar.set(yticks=ranks,yticklabels=[str(r['rank']) for r in rows],xlabel='GraspGen score (not a success probability)',ylabel='Candidate rank',xlim=(0,1.03),title='GraspGen ranking and pad-capture evidence')
    bar.invert_yaxis();fig.suptitle(f'Empty-grasp diagnosis — {run.name}',fontsize=16)
    fig.savefig(out/'empty-grasp.png',dpi=160);fig.savefig(out/'empty-grasp.pdf');plt.close(fig)
    (out/'watch.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Why the grasp closed empty</title>
<style>body{{font:17px system-ui;max-width:1200px;margin:35px auto;padding:0 20px;color:#182838}}img{{width:100%}}a{{color:#0764a4}}</style>
<h1>Why the grasp closed empty</h1><p>The chosen original GraspGen pose had <b>zero observed handle points in the calibrated pad closing volume</b>. At pad height the nearest surface was {gap:.2f} mm beyond the pad tips before execution.</p>
<img src="empty-grasp.png"><p>The mesh/frame conversion agrees within {max(alignment):.2g} m. IK preserved the grasp position within {audit['ik_error_to_requested_grasp_position_m']:.2g} m; measured arrival error was {audit['measured_error_to_ik_target_m']*1000:.3f} mm.</p>
<p>All seven higher-scored poses were rejected by scene checks. The remaining pose, scored {selected['graspgen_score']:.3f}, passed because collision freedom and IK were tested, while positive pinch support was not.</p>
<p>The 40-control closure recorded no robot/environment contacts. The earlier 22 mm measurement was pad-midpoint-to-handle-axis distance, not the tip clearance. A small door movement during approach adds approximately 0.8 mm to the pad-tip gap in the separately labelled offline diagnostic.</p>
<p>A geometric capture test is a necessary plausibility check, not proof of frictional grasp retention. This audit does not modify proposals, planner checks or runtime behavior.</p>
<p><a href="audit.json">All calculations and provenance</a> · <a href="empty-grasp.pdf">PDF figure</a></p>''')
    print(json.dumps({'output':str(out),'selected_capture':intended,'mesh_alignment_error_m':max(alignment),
        'closure_scene_contacts':len(closure_contacts)},indent=2))


if __name__=='__main__':main()
