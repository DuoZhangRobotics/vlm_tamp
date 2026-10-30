"""Reproduce saved endpoint rejections and show their exact collision geometry.

Uses recorded RGB-D and calibrated robot FK only. No simulator is started,
no robot action is issued, and planner/contact settings are not changed.
"""
import argparse
import copy
import json
import html
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull,HalfspaceIntersection,cKDTree
import vamp

from vlm_tamp.coupled_scene import Scene,RobotVolumes
from vlm_tamp.coupled_storage import read_json,write_json
from vlm_tamp.hinged_contact_tools import tool_configuration
from vlm_tamp.mcvamp_contact_numerical import NativeScene,clear_native_cache
from vlm_tamp.rekep_libero.geometry import pose_matrix,transform_points
from vlm_tamp.rekep_libero.robot_chain_v4 import RobotChain
from vlm_tamp.rekep_libero.scene_state_v7 import bounded_points,voxel_points
from try_rekep_bowl_v3 import read_observation_v3

ROOT=Path(__file__).resolve().parents[1]


def aperture_metadata(initial,qpos):
    """Known prismatic finger kinematics; no hypothetical scene modification."""
    meta=copy.deepcopy(initial);axis=pose_matrix(meta['eef_pose_world_xyzw'])[:3,0]
    for volume in meta['robot_collision_volumes_world']:
        name=volume['robot_geom_name']
        finger=0 if 'finger1' in name else 1 if 'finger2' in name else None
        if finger is None:continue
        shift=axis*(initial['gripper_qpos'][finger]-qpos[finger])
        volume['vertices_world']=np.asarray(volume['vertices_world'])+shift
        planes=np.asarray(volume['planes_world']).copy();planes[:,3]-=planes[:,:3]@shift
        volume['planes_world']=planes
    meta['gripper_qpos']=list(qpos)
    return meta


def mesh(volume):
    vertices=np.asarray(volume['vertices_world']);triangles=ConvexHull(vertices).simplices
    return {'name':volume['robot_geom_name'],'vertices':vertices,'triangles':triangles}


def hits_for_check(context,arrays,metadata,q,check,volumes):
    if check['accepted']:return np.empty((0,3)),np.empty(0),None
    code=int(check['last_rejected_part']);surface=int(check['last_rejected_surface'])
    native=arrays[['stationary','moving','contact'][surface]]
    world=transform_points(native,context.world_robot)
    if code>=2000:
        volume=next(v for v in volumes if v['robot_geom_name']==check['blocking_robot_part'])
        ids,distance=bounded_points(world,cKDTree(world),volume,0.)
        return world[ids],-distance,mesh(volume)
    # These are the exact float32 planes/piece boundaries installed in MC-VAMP.
    eef=np.asarray(vamp.panda.eefk(np.asarray(q,np.float32)),np.float32)[0]
    bounds=arrays['gripper_bounds'];planes=arrays['gripper_planes'];offsets=arrays['gripper_plane_offsets']
    points=[];depths=[];pieces=[]
    for i,b in enumerate(bounds):
        if 1000+int(b[5])!=code or (surface==2 and b[4]):continue
        center=eef[:3,:3]@b[:3]+eef[:3,3]
        ids=np.asarray(cKDTree(native).query_ball_point(center,float(b[3])+1e-6),int)
        local=(native[ids]-eef[:3,3])@eef[:3,:3]
        equations=planes[offsets[i]:offsets[i+1]]
        distance=np.max(local@equations[:,:3].T+equations[:,3],axis=1)
        inside=distance<=0
        if not inside.any():continue
        points.extend(world[ids[inside]]);depths.extend(-distance[inside])
        vertices=HalfspaceIntersection(equations.astype(float),b[:3].astype(float)).intersections
        vertices=transform_points(vertices,context.world_robot@eef)
        pieces.append(mesh({'vertices_world':vertices,'robot_geom_name':check['blocking_robot_part']}))
    if not pieces:raise ValueError('Native rejection has no recovered intersecting observation points')
    vertices=np.vstack([p['vertices'] for p in pieces])
    return np.asarray(points),np.asarray(depths),mesh({'vertices_world':vertices,'robot_geom_name':check['blocking_robot_part']})


def project(points,camera):
    p=transform_points(points,np.linalg.inv(pose_matrix(camera['camera_to_world'])))
    xy=p@np.asarray(camera['intrinsic']).T
    return xy[:,:2]/p[:,2,None],p[:,2]


def draw_rgb(ax,camera,robot,hits,*,zoom=False):
    from matplotlib.collections import LineCollection
    ax.imshow(camera['rgb'])
    for part in robot:
        vertices,triangles=np.asarray(part['vertices']),np.asarray(part['triangles'])
        xy,z=project(vertices,camera)
        edges=np.unique(np.sort(np.vstack([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1),axis=0)
        edges=edges[(z[edges]>0).all(1)]
        ax.add_collection(LineCollection(xy[edges],colors='#14a5c4',linewidths=.45,alpha=.25))
    xy,z=project(hits,camera)
    ax.scatter(xy[:,0],xy[:,1],s=15 if zoom else 8,c='#e72c35',edgecolors='white',linewidths=.3)
    h,w=camera['rgb'].shape[:2]
    if zoom and len(hits):
        lo,hi=np.quantile(xy,[0,1],axis=0);center=(lo+hi)/2;radius=max(28.,float((hi-lo).max())*.8)
        ax.set(xlim=(max(0,center[0]-radius),min(w,center[0]+radius)),
            ylim=(min(h,center[1]+radius),max(0,center[1]-radius)))
    else:ax.set(xlim=(0,w),ylim=(h,0))
    ax.axis('off')


def figures(out,frames,cameras,scene_cloud):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # Show one actual rejection per pose; open/closed variants remain interactive.
    for rank in range(1,9):
        options=[f for f in frames if f['rank']==rank]
        f=next((f for f in options if not f['accepted']),options[0])
        if not len(f['hits']):continue
        points=np.asarray(f['hits']);center=points.mean(0)
        fig=plt.figure(figsize=(13,7),layout='constrained')
        for n,name in enumerate(['frontview','arm_wide_front','arm_wide_side']):
            ax=fig.add_subplot(2,3,n+1);draw_rgb(ax,cameras[name],f['robot'],points)
            ax.set_title(name.replace('_',' ')+' — proposed robot in cyan',fontsize=10)
        for n,name in enumerate(['frontview','arm_wide_front']):
            ax=fig.add_subplot(2,3,n+4);draw_rgb(ax,cameras[name],f['robot'],points,zoom=True)
            ax.set_title('Zoom: red points inside rejected volume',fontsize=10)
        ax=fig.add_subplot(236,projection='3d')
        local=scene_cloud[np.linalg.norm(scene_cloud-center,axis=1)<.08]
        ax.scatter(*local[::max(1,len(local)//5000)].T,s=1,c='#a5afb8',alpha=.35)
        part=f['blocking_mesh'];vertices=np.asarray(part['vertices']);triangles=np.asarray(part['triangles'])
        ax.add_collection3d(Poly3DCollection(vertices[triangles],facecolor='#dc734d',edgecolor='#6a605b',lw=.15,alpha=.2))
        ax.scatter(*points.T,s=13,c='#e72c35',depthshade=False)
        radius=max(.02,min(.08,float(np.ptp(points,axis=0).max())*.8+.01))
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius),
            xlabel='x (m)',ylabel='y (m)',zlabel='height (m)',title='3D overlap neighborhood',box_aspect=(1,1,1))
        ax.view_init(25,55)
        fig.suptitle(f'Candidate {rank}, {f["aperture"]} fingers: {f["part_label"]}\n{len(points)} observed points inside geometry; maximum inward face distance {f["max_depth_mm"]:.3f} mm',fontsize=13)
        fig.savefig(out/f'candidate-{rank}.png',dpi=150);plt.close(fig)


def interactive(out,frames,clouds):
    import plotly.graph_objects as go
    from plotly.offline import get_plotlyjs
    (out/'plotly.min.js').write_text(get_plotlyjs())
    for f in frames:
        fig=go.Figure()
        for role,color,label in [('stationary','#929fab','Other observed surfaces'),('moving','#59b4b4','Moving panel'),('contact','#dcb549','Selected handle')]:
            points=np.asarray(clouds[role]);points=points[::max(1,int(np.ceil(len(points)/12000)))]
            fig.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='markers',
                marker=dict(size=1.7,color=color,opacity=.4),name=label,hoverinfo='name'))
        for part in f['robot']:
            v=np.asarray(part['vertices']);t=np.asarray(part['triangles']);name=part['name']
            color='#2eab76' if name.endswith('_pad_collision') else '#278aaa'
            fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=t[:,0],j=t[:,1],k=t[:,2],
                color=color,opacity=.32,flatshading=True,name=name,showlegend=False,hoverinfo='name'))
        if f['blocking_mesh']:
            v=np.asarray(f['blocking_mesh']['vertices']);t=np.asarray(f['blocking_mesh']['triangles'])
            fig.add_trace(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=t[:,0],j=t[:,1],k=t[:,2],
                color='#ed7753',opacity=.6,flatshading=True,name='Rejected collision volume',showlegend=True))
        points=np.asarray(f['hits'])
        if len(points):
            fig.add_trace(go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='markers',
                marker=dict(size=5,color='#e52d3a'),name='Points inside rejected volume',customdata=f['depths_mm'],
                hovertemplate='Inward face distance: %{customdata:.3f} mm<extra>Observed surface point</extra>'))
        whole={'scene.xaxis.range':[-.85,.5],'scene.yaxis.range':[-.55,.45],'scene.zaxis.range':[.75,1.65],
            'scene.camera.eye':{'x':1.3,'y':1.8,'z':1.0}}
        center=points.mean(0) if len(points) else np.asarray(f['robot'][-1]['vertices']).mean(0)
        radius=max(.035,min(.1,float(np.ptp(points,axis=0).max())*.9+.015)) if len(points) else .09
        close={f'scene.{a}axis.range':[float(center[i]-radius),float(center[i]+radius)] for i,a in enumerate('xyz')}
        close['scene.camera.eye']={'x':1.3,'y':1.8,'z':1.0}
        fig.update_layout(height=580,margin=dict(l=0,r=0,t=45,b=0),
            scene=dict(aspectmode='data',xaxis_title='x (m)',yaxis_title='y (m)',zaxis_title='height (m)',
                xaxis=dict(range=whole['scene.xaxis.range']),yaxis=dict(range=whole['scene.yaxis.range']),
                zaxis=dict(range=whole['scene.zaxis.range']),camera=dict(eye=whole['scene.camera.eye'])),
            legend=dict(orientation='h',y=-.04),paper_bgcolor='white',
            updatemenus=[dict(type='buttons',direction='right',x=0,y=1.08,
                buttons=[dict(label='Whole scene',method='relayout',args=[whole]),
                         dict(label='Zoom contact',method='relayout',args=[close])])])
        fig.write_html(out/f'candidate-{f["rank"]}-{f["aperture"]}.html',include_plotlyjs=False,
            full_html=True,config={'displaylogo':False,'scrollZoom':True})
        path=out/f'candidate-{f["rank"]}-{f["aperture"]}.html'
        page=path.read_text().replace('<head>','<head><script src="plotly.min.js"></script>');path.write_text(page)
    summaries=[{k:v for k,v in f.items() if k not in {'robot','blocking_mesh','hits','depths_mm'}}|
        {'point_count':len(f['hits'])} for f in frames]
    first=next(r for r in summaries if r['rank']==1 and r['aperture']=='closed')
    second=next(r for r in summaries if r['rank']==2 and r['aperture']=='open')
    comparison=(f'<b>Candidate 1, predicted holding width:</b> {first["point_count"]} points, '
        f'up to {first["max_depth_mm"]:.3f} mm inside the reported {html.escape(first["part_label"])} volume. '
        f'<b>Candidate 2, open hand:</b> {second["point_count"]} points, '
        f'up to {second["max_depth_mm"]:.3f} mm inside the reported {html.escape(second["part_label"])} volume. '
        'The pictures distinguish local finger/target contact from larger arm intersections.')
    table=''.join(f'<tr><td>{f["rank"]}</td><td>{f["aperture"]}</td><td>{html.escape(f["part_label"] if not f["accepted"] else "Accepted")}</td><td>{f["point_count"]}</td><td>{f["max_depth_mm"]:.3f}</td></tr>' for f in summaries)
    page='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>See the predicted grasp collisions</title><style>body{font:17px system-ui;max-width:1350px;margin:28px auto;padding:0 18px;color:#182b3b;background:#f3f5f8}section{background:white;padding:20px;border-radius:12px;margin:16px 0}h1{font-size:29px}select{font:inherit;padding:7px}iframe{border:0;width:100%;height:650px}img{width:100%;display:block}td,th{padding:7px;text-align:left;border-bottom:1px solid #ddd}table{width:100%;border-collapse:collapse;font-size:14px}a{color:#075daf}.key{padding:14px;background:#fff0d4}#summary{font-weight:600;margin:14px 0}</style></head><body>
<h1>“Scene collisions” means contact with the surrounding environment</h1>
<p>These are reconstructed planner predictions, not physical trials of the rejected poses. Each displayed accept/reject result matches the saved MC-VAMP result.</p>
<p class="key">__COMPARISON_NOTE__</p>
<section><label>Grasp <select id="rank">'''+''.join(f'<option value="{i}">{i}</option>' for i in range(1,9))+'''</select></label>
<label>Hand <select id="aperture"><option value="open">Open</option><option value="closed" selected>Predicted holding width</option></select></label>
<div id="summary"></div><p>Drag to rotate; scroll to zoom; use “Zoom contact” for the red points. Cyan: proposed robot. Orange: rejected robot volume. Red: observed points inside it. Green: finger pads. Gray/teal/yellow: observed environment and target.</p>
<iframe id="viewer" title="Rotatable collision geometry"></iframe></section>
<section><h2>Camera overlays</h2><p>The RGB background shows the original scene and initial robot. The cyan wireframe is the hypothetical proposed posture. Red dots are the blocking observed points. The sheet uses the first rejected hand aperture for that candidate; the 3D hand selector above allows open/holding comparison.</p><a id="photoLink"><img id="photo" alt="Predicted robot and collision points over recorded RGB views"></a><p id="photoNote"></p></section>
<section><h2>All reproduced endpoint checks</h2><p>We show the closest saved IK branch for each grasp; the original search examined additional branches. Red highlights recover the first reported blocking pair, not every possible collision. “Inward face distance” measures points inside the configured convex collision volume, including any contact-permission clipping face. It is not a certificate of physical object penetration.</p><table><tr><th>Candidate</th><th>Hand aperture</th><th>First blocked robot part</th><th>Observed points</th><th>Max inward face distance (mm)</th></tr>'''+table+'''</table><p><a href="geometry.json">Exact geometry and provenance</a></p></section>
<script>const rows='''+json.dumps(summaries)+''';const rank=document.getElementById('rank'),aperture=document.getElementById('aperture');
function update(){const n=Number(rank.value),a=aperture.value,r=rows.find(x=>x.rank===n&&x.aperture===a);document.getElementById('viewer').src=`candidate-${n}-${a}.html`;document.getElementById('summary').textContent=r.accepted?`Candidate ${n}: accepted at this hand aperture; this does not prove a grasp.`:`Candidate ${n}: ${r.part_label}; ${r.point_count} points; maximum inward face distance ${r.max_depth_mm.toFixed(3)} mm.`;const photo=document.getElementById('photo');photo.style.display=n===8?'none':'block';if(n!==8){photo.src=`candidate-${n}.png`;document.getElementById('photoLink').href=photo.src;}document.getElementById('photoNote').textContent=n===8?'Candidate 8 was executed and closed empty; its failure is grasp placement rather than a scene collision.':`Static sheet: candidate ${n}, ${n===1?'predicted holding width':'open hand'}.`;}
rank.onchange=()=>{aperture.value=Number(rank.value)===1?'closed':'open';update();};aperture.onchange=update;update();</script></body></html>'''
    (out/'watch.html').write_text(page.replace('__COMPARISON_NOTE__',comparison))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();out=args.output.resolve();out.mkdir(exist_ok=True)
    ep=run/'attempt01';grasp=next(ep.glob('*-grasp'));initial=read_json(ep/'001-observation/metadata.json')
    record=read_json(ep/'robot-chain.json');chain=RobotChain(record);spec=read_json(ep/'002-grounding/model.json')
    with np.load(run/'workspace/robot_data/attempt01/geometry002/points.npz') as d:raw={k:d[k].copy() for k in d.files}
    cameras,_=read_observation_v3(ep/'opened',read_json(ep/'opened/metadata.json'))
    config=tool_configuration(read_json(ROOT/'configs/articulated_libero.json'))
    saved=read_json(grasp/'selection/approach-initial-scene.json');attempts=read_json(grasp/'selection/grasp-attempts.json')
    frames=[];clouds=None
    for aperture in ['open','closed']:
        width=spec['observed_handle_radius_m']+.001
        meta=copy.deepcopy(initial) if aperture=='open' else aperture_metadata(initial,[width,-width])
        robot=RobotVolumes(chain,meta['robot_collision_volumes_world'],meta['joint_positions'])
        scene=Scene.from_points(raw['static'],raw['moving'],raw['contact'],spec,config=config,robot=robot)
        scene.skill_contact_surface='fingertips'
        folder=out/('native-'+aperture)
        with NativeScene(scene,record,meta,meta['joint_positions'],np.radians(3),articulated=False,allow_contact=True,collision_policy='veto',output=folder) as context:
            for key in ['static_points','moving_noncontact_points','selected_contact_points']:
                if context.record[key]!=saved['record' if aperture=='open' else 'closed_contact_record'][key]:
                    raise ValueError('Observed native cloud counts do not reproduce')
            with np.load(folder/'native-geometry.npz') as d:arrays={k:d[k] for k in d.files}
            if clouds is None:clouds={key:transform_points(arrays[key],context.world_robot) for key in ['stationary','moving','contact']}
            for row in attempts:
                item=row['ik_endpoint_checks'][0];q=np.asarray(item['joints']);old=item[aperture+'_aperture'];new=context.validate(q)
                for key in ['accepted','blocking_robot_part','blocking_surface']:
                    if old[key]!=new[key]:raise ValueError(f'Candidate {row["pose_index"]+1}: native {key} differs: {old[key]} vs {new[key]}')
                volumes=robot.forward(np.asarray(q,np.float32))
                points,depths,blocking=hits_for_check(context,arrays,meta,q,new,volumes)
                label=str(new['blocking_robot_part']).replace('gripper0_hand_collision','palm').replace('gripper0_finger1_collision','finger 1 body').replace('gripper0_finger2_collision','finger 2 body').replace('robot0_link','arm link ').replace('_collision','')
                frame={'rank':row['pose_index']+1,'score':row['graspgen_score'],'aperture':aperture,
                    'ik_branch':0,'tested_ik_branches':len(row['ik_endpoint_checks']),'accepted':new['accepted'],
                    'part_label':label,'surface':new['blocking_surface'],'native_rejection_reproduced':True,
                    'hits':points,'depths_mm':depths*1000,'max_depth_mm':float(depths.max()*1000) if len(depths) else 0.,
                    'robot':[mesh(v) for v in volumes],'blocking_mesh':blocking}
                frames.append(frame)
                print(json.dumps({k:v for k,v in frame.items() if k not in {'hits','depths_mm','robot','blocking_mesh'}}),flush=True)
    clear_native_cache()
    write_json(out/'geometry.json',{'frames':frames,'clouds':clouds,'source':str(run),
        'runtime_modified':False,'physical_motion_issued':False,'scope':'Saved closest IK branch for each grasp, in open and predicted-closed apertures; no physical collision trial',
        'depth_definition':'Minimum inward distance to a convex collision-volume face, including a contact-permission clipping face; not an object penetration certificate.'})
    figures(out,frames,cameras,np.vstack(list(clouds.values())))
    interactive(out,frames,clouds)


if __name__=='__main__':main()
