"""Read-only previews of native collision rejections using measured surfaces.

Capture is cheap and opt-in. Geometry reconstruction and rendering occur after
the tool finishes, outside planner search budgets. No simulator scene state is
queried and no collision permission or planning result is changed.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull,cKDTree
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
from .coupled_storage import write_json
from .rekep_libero.geometry import transform_points
from .rekep_libero.mechanism_geometry import project

_ACTIVE=ContextVar('native_collision_evidence',default=None)

def active_capture():return _ACTIVE.get()

class Capture:
    def __init__(self,maximum=3):self.maximum=maximum;self.samples=[];self.keys=set();self.rejections=0
    def record(self,context,q,result):
        if result['accepted']:return
        self.rejections+=1
        part=int(result.get('last_rejected_part',-1));surface=int(result.get('last_rejected_surface',-1))
        if not (1<=part<37 or part>=1000) or surface not in (0,1,2):return
        kind='start' if np.max(np.abs(np.asarray(q)-context.reference_q))<1e-6 else 'candidate'
        key=(part,surface,kind)
        if not np.isfinite(q).all():return
        if key in self.keys:
            self.samples=[s for s in self.samples if (int(s[2]['last_rejected_part']),int(s[2]['last_rejected_surface']),s[3])!=key]
        elif len(self.samples)>=self.maximum:
            old=self.samples.pop(0);self.keys.discard((int(old[2]['last_rejected_part']),int(old[2]['last_rejected_surface']),old[3]))
        self.keys.add(key);self.samples.append((context,np.asarray(q,float).copy(),dict(result),kind))

@contextmanager
def capture_rejections():
    result=Capture();token=_ACTIVE.set(result)
    try:yield result
    finally:_ACTIVE.reset(token)

def reconstruct(context,q,result):
    import vamp
    geometry=context._diagnostic_geometry
    world_robot=context.world_robot
    native=np.asarray([[s.x,s.y,s.z,s.r] for s in vamp.panda.fk(q.tolist())],float)
    spheres=np.c_[transform_points(native[:,:3],world_robot),native[:,3]]
    eef=world_robot@np.asarray(vamp.panda.eefk(q.tolist()),float)[0]
    motion=np.eye(4);progress=0.
    if geometry['articulated']:
        progress=context.progress(q)
        if context.mechanism_kind==2:motion[:3,3]=progress*context.axis
        else:
            motion[:3,:3]=Rotation.from_rotvec(progress*context.axis).as_matrix()
            motion[:3,3]=context.origin-motion[:3,:3]@context.origin
    surfaces=[transform_points(geometry[k],world_robot) for k in ('stationary','moving','contact')]
    surfaces[1]=transform_points(surfaces[1],motion);surfaces[2]=transform_points(surfaces[2],motion)
    surface=int(result['last_rejected_surface']);part=int(result['last_rejected_part']);points=surfaces[surface]
    shapes={name:transform_points(vertices,eef) for name,vertices in geometry['gripper_vertices'].items()}
    witness=None;verified=False;gap=None;name=None;center=None
    if len(points) and 1<=part<37:
        center=spheres[part,:3];distance,index=cKDTree(points).query(center)
        gap=float(distance-spheres[part,3]-geometry['point_radius'])
        witness=points[index];verified=gap<=2e-5;name=f'Panda arm sphere {part}'
    elif len(points) and part>=1000:
        ordinal=next((i for i,row in enumerate(geometry['parts']) if row['sphere_range'][0]+1000==part),None)
        if ordinal is not None:
            row=geometry['parts'][ordinal];name=row['robot_geom_name'];center=shapes[name].mean(0)
            low,high=geometry['offsets'][ordinal:ordinal+2]
            planes=geometry['planes'][low:high]
            local=transform_points(points,np.linalg.inv(eef))
            residual=np.max(local@planes[:,:3].T+planes[:,3],axis=1)-geometry['gripper_padding']
            inside=np.flatnonzero(residual<=2e-5)
            if len(inside):
                index=inside[np.argmin(np.linalg.norm(points[inside]-center,axis=1))]
                witness=points[index];gap=float(residual[index]);verified=True
    record={'joints':q.tolist(),'native_accepted':False,'reported_part':part,'reported_surface':surface,
        'robot_part':name or 'unresolved','surface_role':('observed stationary','predicted moving','selected contact')[surface],
        'witness_verified':bool(verified),'witness_world_m':witness.tolist() if verified else None,
        'robot_center_world_m':center.tolist() if center is not None else None,
        'sphere_clearance_or_max_halfspace_residual_m':gap,'point_radius_m':geometry['point_radius'],
        'predicted_progress':progress,'simulator_object_truth_used':False,
        'interpretation':'Reconstructed geometric overlap at the native-reported part; not a measured physical collision or exact CAPT traversal order.'}
    return record,spheres,shapes,surfaces

def draw_pose(camera,spheres,shapes,evidence):
    canvas=Image.fromarray(np.asarray(camera['rgb'],np.uint8)).convert('RGBA')
    layer=Image.new('RGBA',canvas.size);draw=ImageDraw.Draw(layer)
    def lines(points,color,width=1):
        pixels=project(np.asarray(points),camera)
        if not np.isfinite(pixels).all() or np.max(np.abs(pixels))>10000:return
        draw.line([tuple(p) for p in pixels],fill=color,width=width)
    angles=np.linspace(0,2*np.pi,25)
    for index,s in enumerate(spheres[1:37],1):
        color=(245,30,60,230) if index==evidence['reported_part'] else (170,65,220,140)
        for a,b in ((0,1),(1,2),(0,2)):
            xyz=np.repeat(s[None,:3],len(angles),axis=0)
            xyz[:,a]+=s[3]*np.cos(angles);xyz[:,b]+=s[3]*np.sin(angles)
            lines(xyz,color,2 if index==evidence['reported_part'] else 1)
    for name,vertices in shapes.items():
        color=(245,30,60,230) if name==evidence['robot_part'] else (170,65,220,170)
        for face in ConvexHull(vertices).simplices:lines(vertices[np.r_[face,face[0]]],color)
    if evidence['witness_verified']:
        uv=project([evidence['witness_world_m']],camera)[0]
        if np.isfinite(uv).all():
            x,y=uv;draw.ellipse((x-7,y-7,x+7,y+7),outline=(255,0,0,255),width=3)
    return Image.alpha_composite(canvas,layer).convert('RGB')

def save_capture(capture,views,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);records=[]
    for number,(context,q,result,kind) in enumerate(capture.samples):
        record,spheres,shapes,surfaces=reconstruct(context,q,result)
        record.update(sample=number,query_kind=kind)
        sub=folder/f'sample-{number:02d}';sub.mkdir(exist_ok=True)
        write_json(sub/'evidence.json',record)
        np.savez_compressed(sub/'geometry.npz',spheres_world=spheres,
            stationary=surfaces[0],moving=surfaces[1],selected_contact=surfaces[2],
            **{f'gripper_{i}':v for i,v in enumerate(shapes.values())})
        observed_views=getattr(context.scene,'diagnostic_views',views)
        for camera in ('agentview','frontview'):
            draw_pose(observed_views[camera],spheres,shapes,record).save(sub/f'{camera}.png')
        records.append(record)
    summary={'native_validation_rejections_observed':capture.rejections,'representative_samples':records,
        'execution_checks_added':False,'geometry_used_for_planning_changed':False,
        'diagnostic_rendering_outside_search_budget':True}
    write_json(folder/'summary.json',summary)
    return summary
