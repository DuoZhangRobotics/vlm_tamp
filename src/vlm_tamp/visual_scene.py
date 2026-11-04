"""Observed-cloud views and known-robot pose previews, without motion or scene repair."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

from .coupled_storage import write_json
from .rekep_libero.geometry import matrix_pose, pose_matrix, transform_points

CAMERAS = ('agentview', 'wrist')
PROJECTIONS = {'xy': (0, 1, 2, 1), 'yz': (1, 2, 0, 1), 'xz': (0, 2, 1, -1)}
SIZE = 640
RECT = (48, 48, 591, 591)


def cloud_samples(cameras: dict[str, dict]) -> dict[str, np.ndarray]:
    """Retain original-pixel provenance while thinning only the display by two."""
    points, colors, sources = [], [], []
    for index, name in enumerate(CAMERAS):
        if name not in cameras:
            continue
        view = cameras[name]
        h, w = view['rgb'].shape[:2]
        yy, xx = np.mgrid[0:h:2, 0:w:2]
        xyz = np.asarray(view['world_points'])[yy, xx]
        valid = np.isfinite(xyz).all(axis=-1) & (np.asarray(view['depth'])[yy, xx] > 0)
        points.append(xyz[valid])
        colors.append(np.asarray(view['rgb'])[yy, xx][valid])
        sources.append(np.c_[np.full(valid.sum(), index), xx[valid], yy[valid]])
    if not points or not any(len(p) for p in points):
        raise ValueError('No measured points in the two native camera views')
    return {'points': np.vstack(points).astype(np.float32),
            'colors': np.vstack(colors).astype(np.uint8), 'sources': np.vstack(sources).astype(np.int32)}


def projection_spec(points: np.ndarray, plane: str, center: list[float] | None,
                    span: float | None) -> dict[str, Any]:
    """Define a metric screen map; never change the underlying points."""
    horizontal, vertical, depth, sign = PROJECTIONS[plane]
    low, high = points.min(axis=0), points.max(axis=0)
    origin = (low+high)/2 if center is None else np.asarray(center, float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('Use a finite XYZ viewing center')
    width = float(max(high[horizontal]-low[horizontal], high[vertical]-low[vertical])) if span is None else float(span)
    if not np.isfinite(width) or width <= 0:
        raise ValueError('The viewing span must be positive')
    return {'plane': plane, 'horizontal_axis': horizontal, 'vertical_axis': vertical,
            'depth_axis': depth, 'observer_axis_sign': sign, 'center_world_m': origin.tolist(),
            'span_m': width, 'data_rectangle_xyxy': list(RECT), 'image_size_xy': [SIZE, SIZE],
            'meters_per_pixel': width/(RECT[2]-RECT[0])}


def project_world(points: np.ndarray, spec: dict) -> np.ndarray:
    """Map world points to projection pixels, including the labelled border."""
    center = np.asarray(spec['center_world_m'])
    scale = spec['meters_per_pixel']
    middle = (RECT[0]+RECT[2])/2
    return np.c_[middle+(points[:, spec['horizontal_axis']]-center[spec['horizontal_axis']])/scale,
                 middle-(points[:, spec['vertical_axis']]-center[spec['vertical_axis']])/scale]


def pixel_coordinates(pixel: list[int], spec: dict) -> dict[int, float]:
    """Return the two world coordinates explicitly chosen by a projection click."""
    if len(pixel) != 2 or any(type(v) is not int for v in pixel):
        raise ValueError('Use two integer image coordinates')
    x, y = pixel
    if not RECT[0] <= x <= RECT[2] or not RECT[1] <= y <= RECT[3]:
        raise ValueError('Select a point inside the plotted data rectangle')
    center, scale = spec['center_world_m'], spec['meters_per_pixel']
    middle = (RECT[0]+RECT[2])/2
    return {spec['horizontal_axis']: center[spec['horizontal_axis']]+(x-middle)*scale,
            spec['vertical_axis']: center[spec['vertical_axis']]-(y-middle)*scale}


def render_projection(cloud: dict[str, np.ndarray], spec: dict) -> tuple[Image.Image, np.ndarray]:
    """Rasterize measured samples with a depth buffer and a matching index image."""
    points, colors = cloud['points'], cloud['colors']
    pixels = np.rint(project_world(points, spec)).astype(int)
    pixel_rows, point_rows = [], []
    for dx, dy in ((-1,-1), (-1,0), (-1,1), (0,-1), (0,0), (0,1), (1,-1), (1,0), (1,1)):
        xy = pixels+np.array([dx, dy])
        valid = ((xy[:,0] >= RECT[0]) & (xy[:,0] <= RECT[2]) &
                 (xy[:,1] >= RECT[1]) & (xy[:,1] <= RECT[3]))
        pixel_rows.append(xy[valid,1]*SIZE+xy[valid,0])
        point_rows.append(np.flatnonzero(valid))
    linear, indices = np.concatenate(pixel_rows), np.concatenate(point_rows)
    depth = spec['observer_axis_sign']*points[indices, spec['depth_axis']]
    order = np.lexsort((-depth, linear))
    first = np.unique(linear[order], return_index=True)[1]
    selected = order[first]
    index_map = np.full(SIZE*SIZE, -1, np.int32)
    index_map[linear[selected]] = indices[selected]
    canvas = np.full((SIZE*SIZE, 3), (27, 35, 45), np.uint8)
    canvas[linear[selected]] = colors[indices[selected]]
    picture = Image.fromarray(canvas.reshape(SIZE,SIZE,3));draw = ImageDraw.Draw(picture)
    axis = 'XYZ'
    h, v, d = (spec[k] for k in ('horizontal_axis','vertical_axis','depth_axis'))
    draw.text((8,8), f"Observed cloud | right +{axis[h]}, up +{axis[v]} | view from {'+' if spec['observer_axis_sign']>0 else '-'}{axis[d]}", fill='white')
    draw.text((8,25), 'Two-camera observations only; holes are unknown. Marks are pixel coordinates.', fill='white')
    for fraction in np.linspace(0,1,5):
        x = RECT[0]+fraction*(RECT[2]-RECT[0]);y = RECT[3]-fraction*(RECT[3]-RECT[1])
        values = pixel_coordinates([int(x),int(y)],spec)
        draw.text((x-14,RECT[3]+8),f'{values[h]:.2f}',fill='white')
        draw.text((2,y-5),f'{values[v]:.2f}',fill='white')
    draw.text((8,SIZE-19),f"Axes in metres | display stride 2 | sample splats 3 px | span {spec['span_m']:.3f} m",fill='white')
    return picture, index_map.reshape(SIZE,SIZE)


def scene_views(service: Any, center_world_m: list[float] | None = None,
                span_m: float | None = None) -> dict[str, Any]:
    """Publish immutable, markable views of the current observed point cloud."""
    cloud = cloud_samples(service.cameras)
    folder = service.public/f'attempt{service.attempt:02d}'/f'views{service.calls:03d}'
    folder.mkdir()
    np.savez_compressed(folder/'samples.npz', **cloud)
    views = []
    for plane in PROJECTIONS:
        spec = projection_spec(cloud['points'],plane,center_world_m,span_m)
        picture, index_map = render_projection(cloud,spec)
        view_id = f'view{service.calls:03d}-{plane}'
        picture.save(folder/f'{plane}.png')
        np.savez_compressed(folder/f'{plane}-index.npz',index=index_map)
        record = {**spec, 'view_id':view_id, 'observation_id':service.latest_observation['observation_id'],
                  'control_steps':service.state['control_steps'],
                  'image':str((folder/f'{plane}.png').relative_to(service.workspace)),
                  'source_cameras':[name for name in CAMERAS if name in service.cameras]}
        write_json(folder/f'{plane}.json',record)
        if not hasattr(service,'visual_views'):
            service.visual_views = {}
        service.visual_views[view_id] = (folder,record)
        views.append(record)
    return {'views':views, 'images':[view['image'] for view in views],
            'new_sensor_capture':False, 'robot_motion_issued':False,
            'geometry_source':'measured RGB-D only; no hidden surfaces, native contacts or object poses',
            'display_points':len(cloud['points']), 'planner_scene_changed':False}


def mark_scene_point(service: Any, view_id: str, pixel_xy: list[int], mode: str = 'surface',
                     depth_view_id: str | None = None, depth_pixel_xy: list[int] | None = None) -> dict[str, Any]:
    """Resolve a rendered sample or an explicitly chosen free-space coordinate."""
    folder, first = service.visual_views[view_id]
    coords = pixel_coordinates(pixel_xy,first)
    result = {'mode':mode, 'observation_id':first['observation_id'],
              'source_control_step':first['control_steps'], 'physical_contact_verified':False,
              'current_observation':first['observation_id']==service.latest_observation['observation_id']}
    if mode == 'surface':
        with np.load(folder/f"{first['plane']}-index.npz") as data:
            index = int(data['index'][pixel_xy[1],pixel_xy[0]])
        if index < 0:
            raise ValueError('No observed sample at that pixel; choose a coloured sample explicitly')
        with np.load(folder/'samples.npz') as data:
            point = data['points'][index]
            camera,x,y = data['sources'][index]
        result.update(source_camera=CAMERAS[camera],source_pixel_xy=[int(x),int(y)],
                      point_source='exact original RGB-D sample represented by the selected rendered splat')
    elif mode == 'free_space':
        if depth_view_id is None or depth_pixel_xy is None:
            raise ValueError('Free-space marking needs a second projection and click')
        second_folder, second = service.visual_views[depth_view_id]
        if first['observation_id'] != second['observation_id']:
            raise ValueError('Use two projections of the same observation')
        second_coords = pixel_coordinates(depth_pixel_xy,second)
        missing = set(range(3))-set(coords)
        if not missing <= set(second_coords):
            raise ValueError('The second projection must show the remaining world axis')
        axis = missing.pop()
        coords[axis] = second_coords[axis]
        point = np.array([coords[i] for i in range(3)])
        result.update(point_source='first click fixes two coordinates; second supplies only the missing axis',
                      surface_support_claimed=False,second_coordinate_axis='XYZ'[axis])
    else:
        raise ValueError('Choose surface or free_space explicitly')
    image = Image.open(folder/f"{first['plane']}.png").convert('RGB');draw = ImageDraw.Draw(image)
    x,y = project_world(point[None],first)[0]
    draw.ellipse((x-5,y-5,x+5,y+5),outline='#ffdd00',width=2)
    target = folder/f'mark{service.calls:03d}.png';image.save(target)
    images=[str(target.relative_to(service.workspace))]
    if mode=='free_space':
        depth_image=Image.open(second_folder/f"{second['plane']}.png").convert('RGB')
        depth_draw=ImageDraw.Draw(depth_image);sx,sy=project_world(point[None],second)[0]
        depth_draw.ellipse((sx-5,sy-5,sx+5,sy+5),outline='#ffdd00',width=2)
        depth_target=folder/f'mark{service.calls:03d}-depth.png';depth_image.save(depth_target)
        images.append(str(depth_target.relative_to(service.workspace)))
    if mode=='surface':
        source=folder.parent/f"obs{first['observation_id']:03d}"/f"{result['source_camera']}.png"
        native=Image.open(source).convert('RGB');draw=ImageDraw.Draw(native)
        sx,sy=result['source_pixel_xy'];draw.ellipse((sx-5,sy-5,sx+5,sy+5),outline='#ffdd00',width=2)
        marked=folder/f'mark{service.calls:03d}-source.png';native.save(marked)
        images.append(str(marked.relative_to(service.workspace)))
    result.update(point_world_m=point.tolist(),view_id=view_id,pixel_xy=pixel_xy,
                  images=images,robot_motion_issued=False)
    return result


def pose_from_axes(position: list[float], approach: list[float], jaw: list[float]) -> np.ndarray:
    """Set TCP +Z to approach and +X to the jaw component orthogonal to +Z."""
    p,z,x = [np.asarray(v,float) for v in (position,approach,jaw)]
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in (p,z,x)) or np.linalg.norm(z)==0:
        raise ValueError('Use finite XYZ position and nonzero orientation directions')
    z = z/np.linalg.norm(z);x = x-z*(x@z)
    if np.linalg.norm(x)<1e-8:
        raise ValueError('Jaw and approach directions must not be parallel')
    x = x/np.linalg.norm(x)
    transform = np.eye(4);transform[:3,:3] = np.column_stack([x,np.cross(z,x),z]);transform[:3,3] = p
    return matrix_pose(transform)


def preview_hand(service: Any, pose_world_xyzw: list[float] | None = None,
                 position_world_m: list[float] | None = None,
                 approach_world: list[float] | None = None, jaw_world: list[float] | None = None,
                 aperture: str = 'current') -> dict[str, Any]:
    """Overlay known hand geometry at an explicit current or calibrated-open aperture."""
    began=time.perf_counter()
    if pose_world_xyzw is None:
        if any(value is None for value in (position_world_m,approach_world,jaw_world)):
            raise ValueError('Supply a full pose, or position plus approach and jaw directions')
        pose_world_xyzw = pose_from_axes(position_world_m,approach_world,jaw_world).tolist()
    elif any(value is not None for value in (position_world_m,approach_world,jaw_world)):
        raise ValueError('Choose the full-pose interface or the two-direction interface')
    target = pose_matrix(pose_world_xyzw)
    metadata = service.state['metadata']
    if aperture=='open':
        qpos=service.open_gripper_reference_qpos
        robot=service.sim.fk(metadata['joint_positions'],include_volumes=True,gripper_qpos=qpos)
        metadata={**metadata,'gripper_qpos':list(qpos),
                  'robot_collision_volumes_world':robot['robot_collision_volumes_world']}
    elif aperture!='current':raise ValueError('Choose current or calibrated-open aperture')
    transform = target@np.linalg.inv(pose_matrix(metadata['eef_pose_world_xyzw']))
    parts = [(v['robot_geom_name'],transform_points(v['vertices_world'],transform))
             for v in metadata['robot_collision_volumes_world'] if v['robot_geom_name'].startswith('gripper0_')]
    folder = service.public/f'attempt{service.attempt:02d}'/f'hand{service.calls:03d}';folder.mkdir()
    images=[]
    for name in CAMERAS:
        if name not in service.cameras:
            continue
        view=service.cameras[name];picture=Image.fromarray(view['rgb']).convert('RGBA')
        overlay=Image.new('RGBA',picture.size);draw=ImageDraw.Draw(overlay)
        inverse=np.linalg.inv(np.asarray(view['camera_to_world']));intrinsic=np.asarray(view['intrinsic'])
        for _,points in parts:
            optical=transform_points(points,inverse)
            if not np.all(optical[:,2]>0):continue
            pixels=optical@intrinsic.T;pixels=pixels[:,:2]/pixels[:,2,None]
            if len(pixels)<3 or np.linalg.matrix_rank(pixels-pixels.mean(0))<2:continue
            hull=ConvexHull(pixels)
            draw.polygon([tuple(pixels[i]) for i in hull.vertices],fill=(255,153,0,35),outline=(255,153,0,255),width=2)
        picture=Image.alpha_composite(picture,overlay).convert('RGB');draw=ImageDraw.Draw(picture)
        origin=transform_points([target[:3,3]],inverse)[0]
        if origin[2]>0:
            uv=intrinsic@origin;gx,gy=uv[:2]/uv[2]
            draw.ellipse((gx-4,gy-4,gx+4,gy+4),outline='#ff55df',width=2)
        draw.rectangle((0,0,picture.width,37),fill='#172635')
        draw.text((6,4),'REQUESTED HAND POSE (orange) - NOT EXECUTED',fill='white')
        draw.text((6,20),f'{aperture} aperture; no IK or collision verdict.',fill='white')
        path=folder/f'{name}.png';picture.save(path);images.append(str(path.relative_to(service.workspace)))
    result={'pose_world_xyzw':pose_world_xyzw,'images':images,
            'gripper_aperture_qpos':metadata['gripper_qpos'],
            'aperture_source':aperture,
            'approach_world':target[:3,2].tolist(),'jaw_world':target[:3,0].tolist(),
            'orientation_convention':'TCP +Z approach; +X jaw projected orthogonal to approach',
            'robot_motion_issued':False,'collision_checked':False,'ik_solved':False,
            'geometry_source':'known robot geometry at the explicit aperture; two native camera images only'}
    result['render_seconds']=time.perf_counter()-began
    write_json(folder/'preview.json',result)
    return result


def observation_pair(workspace: Path, observation: dict) -> Path:
    """Pack the two original-resolution RGB views with explicit camera labels."""
    folder=workspace/'robot_data'/f"attempt{observation.get('attempt',1):02d}"/f"obs{observation['observation_id']:03d}"
    output=folder/'camera-pair.png'
    if output.exists():return output
    pictures=[(name,Image.open(folder/f'{name}.png').convert('RGB')) for name in CAMERAS if (folder/f'{name}.png').exists()]
    if not pictures:raise ValueError('Observation has no native RGB views')
    canvas=Image.new('RGB',(sum(p.width for _,p in pictures),max(p.height for _,p in pictures)+36),'#172635')
    draw=ImageDraw.Draw(canvas);x=0
    state=json.loads((folder/'robot.json').read_text())
    for name,picture in pictures:
        draw.text((x+5,4),f"{name} | obs {observation['observation_id']} | control {state['control_steps']}",fill='white')
        draw.text((x+5,19),f'Original image coordinates: 0..{picture.width-1}, 0..{picture.height-1}',fill='white')
        canvas.paste(picture,(x,36));x+=picture.width
    canvas.save(output)
    return output
