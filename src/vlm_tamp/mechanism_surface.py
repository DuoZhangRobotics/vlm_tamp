"""Measured drawer/control surfaces without a fabricated solid payload hull."""
from pathlib import Path
import copy
import json
import shutil

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

from .cohort_resources import CohortResources,prepare_contact
from .coupled_storage import write_json,read_json
from .rekep_libero.articulated_grounding import parse_choices
from .rekep_libero.mechanism_scene import PanelRegion
from .rekep_libero.scene_state_v7 import observed_points,voxel_points
from .rekep_libero.geometry import transform_points

ROOT=Path(__file__).resolve().parents[2]
BASELINE=ROOT/'results/all-prior-tasks-frozen-v3-20261015'


def normalize_sheet_choices(raw):
    """Repair only a coherent whole-sheet camera offset; never clamp pixels."""
    choices=copy.deepcopy(raw)
    try:return parse_choices(json.dumps(choices)),None
    except ValueError:pass
    camera=choices.get('camera')
    if camera not in {'agentview','frontview'}:raise ValueError('Unknown camera')
    yoff=24 if camera=='agentview' else 560
    candidates=[]
    for xoff in [0,512]:
        out=copy.deepcopy(raw)
        for key in ['contact_box_xyxy','moving_panel_box_xyxy']:
            out[key]=(np.asarray(raw[key],float)-[xoff,yoff,xoff,yoff]).tolist()
        out['grasp_pixel_xy']=(np.asarray(raw['grasp_pixel_xy'],float)-[xoff,yoff]).tolist()
        if raw.get('hinge_pixels_xy') is not None:
            out['hinge_pixels_xy']=(np.asarray(raw['hinge_pixels_xy'],float)-[xoff,yoff]).tolist()
        try:candidates.append((parse_choices(json.dumps(out)),[xoff,yoff]))
        except ValueError:pass
    if len(candidates)!=1:raise ValueError('Invalid image coordinates have no unique camera-sheet interpretation')
    out,offset=candidates[0]
    return out,{'kind':'unambiguous_sheet_to_camera_translation','offset_xy':offset,
                'original_choices':raw,'manual_role_or_pixel_selection':False}


class MechanismResources(CohortResources):
    def grasps(self,cloud,output):
        # Each experiment owns its proposal cache. Parallel experiments cannot
        # delete or partially read one another's temporary candidate files.
        shared=self.cache
        private=self.output/'temporary-proposal-cache';private.mkdir(exist_ok=True)
        self.cache=private
        try:return super().grasps(cloud,output)
        finally:self.cache=shared

    def cleanup_proposals(self):
        private=self.output/'temporary-proposal-cache'
        if private.exists():shutil.rmtree(private)

    def candidates(self,label,index):
        return [BASELINE/f'{label}-state{index:03d}'/'grounding',*super().candidates(label,index)]

    def grounding(self,label,index,observation,metadata,output):
        old=BASELINE/f'{label}-state{index:03d}'/'grounding'
        if (old/'response.txt').exists() and not (old/'grounding.json').exists():
            raw=json.loads((old/'response.txt').read_text());choices,repair=normalize_sheet_choices(raw)
            if repair:
                observation,output=Path(observation),Path(output)
                for name in ['agentview','frontview']:
                    if not np.array_equal(np.asarray(Image.open(observation/f'{name}.png')),
                            np.asarray(Image.open(old/name/'rgb.png'))):
                        raise ValueError('Coordinate-repair cache differs from current RGB')
                shutil.copytree(old,output)
                from .cohort_execution import read_cohort_observation
                cameras,_=read_cohort_observation(observation/'obs.npz',metadata)
                with np.load(old/choices['camera']/'keypoints.npz') as a:bundle={k:a[k].copy() for k in a.files}
                bundle.update(cameras=cameras,meta={**metadata,'main_camera':choices['camera']},
                    main_camera=choices['camera'])
                prepare_contact(self.ensure_perception(),bundle,choices,output)
                write_json(output/'coordinate-repair.json',repair)
                write_json(output/'input-provenance.json',{'cached_source':str(old),'new_api_calls':0,
                    'both_rgb_views_byte_identical':True,'coordinate_repair':repair['kind']})
                return choices
        return super().grounding(label,index,observation,metadata,output)


class SurfaceScene:
    """Current depth endpoints with task contact labels; no solid completion.

    A reconstructed convex drawer/control envelope is not checked against its
    own cabinet/base. Robot arm/palm versus the measured appliance is retained.
    The simulator resolves mechanism/base contact and all execution contact.
    """
    def __init__(self,cameras,metadata,spec,cloud,panel_mask,config,*,previous=None,transform=None):
        self.spec=copy.deepcopy(spec);self.config=config;self.robot=None
        if config.get('collision_evidence_views',False):self.diagnostic_views=cameras
        self.origin=np.asarray(spec['joint_origin_world_m'],float)
        self.axis=np.asarray(spec['joint_axis_world'],float)
        self.panel_origin=np.asarray(spec['front_plane']['point_world_m'],float)
        self.panel_normal=np.asarray(spec['panel_normal_world'],float)
        self.panel_band=float(config.get('panel_contact_band_m',.004))
        self.inferred_panel_reference=np.empty((0,3))
        points,self.self_filter=observed_points(cameras,metadata,spacing=.003,
            robot_depth_tolerance=config['robot_self_depth_tolerance_m'])
        self.contact_reference=np.asarray(cloud,float)
        if previous is None:
            region=PanelRegion(spec,cloud,panel_mask,cameras[spec['camera']])
            moving=region.mask(points)
        else:
            transform=np.eye(4) if transform is None else np.asarray(transform)
            predicted=transform_points(previous.moving_reference,transform)
            self.contact_reference=transform_points(previous.contact_reference,transform)
            self.panel_origin=transform_points([previous.panel_origin],transform)[0]
            self.panel_normal=transform[:3,:3]@previous.panel_normal
            moving=(cKDTree(predicted).query(points)[0]<.009)
        proximity=cKDTree(self.contact_reference).query(points)[0]
        moving|=proximity<.010
        self.static_points=points[~moving];self.moving_reference=points[moving]
        if len(self.moving_reference)<30:raise ValueError('Measured mechanism region has too few points')
        distance=cKDTree(self.contact_reference).query(self.moving_reference)[0]
        plane=np.abs((self.moving_reference-self.panel_origin)@self.panel_normal)
        self.selected_contact_mask=(distance<.028)|((distance<.050)&(plane<self.panel_band))

    def record(self):
        return {'schema_version':'observed-mechanism-surface-1','joint_family':self.spec['joint_family'],
            'stationary_points':len(self.static_points),'moving_points':len(self.moving_reference),
            'finger_contact_points':int(self.selected_contact_mask.sum()),'self_depth_filter':self.self_filter,
            'solid_payload_hull_used':False,'mechanism_environment_collision_checked':False,
            'mechanical_interface_contact_resolved_by':'simulator physics',
            'robot_environment_collision_checked_during_planning':True,
            'allowed_contact':'functional fingers with RGB-D selected control/handle and bounded local face',
            'palm_arm_contact_exempt':False,'unknown_space':'unobserved space is unmodeled',
            'simulator_object_truth_used':False,'execution_veto':False}


def refine_control_axis(cameras,choices,spec,cloud):
    """Estimate knob center from its visible base footprint, not a contact pixel."""
    camera=cameras[spec['camera']];a,b,c,d=np.asarray(choices['moving_panel_box_xyxy'],int)
    points=np.asarray(camera['world_points'])[b:d+1,a:c+1].reshape(-1,3)
    points=points[np.isfinite(points).all(1)]
    normal=np.asarray(spec['panel_normal_world']);normal/=np.linalg.norm(normal)
    origin=np.asarray(spec['front_plane']['point_world_m'])
    side=np.cross(normal,[1,0,0] if abs(normal[0])<.8 else [0,1,0]);side/=np.linalg.norm(side)
    basis=np.column_stack([side,np.cross(normal,side)])
    heights=(points-origin)@normal
    # Use measured control surfaces above the supporting face, within the RGB box.
    selected=points[(heights>=-.003)&(heights<=.09)]
    if len(selected)<30:raise ValueError('Control box lacks usable depth')
    xy=(selected-origin)@basis
    low,high=np.quantile(xy,[.01,.99],axis=0)
    center_xy=(low+high)/2
    center=origin+basis@center_xy
    new=copy.deepcopy(spec);new['joint_origin_world_m']=center.tolist()
    # A contact cloud can include the round base. Prefer observed protruding
    # control material so that grasping and direction tracking concern the lever.
    cloud=np.asarray(cloud,float);height=(cloud-origin)@normal
    high_cloud=cloud[height>=max(.008,float(np.quantile(height,.35)))]
    refined=high_cloud if len(high_cloud)>=40 else cloud
    new['source_anchor_world_m']=np.median(refined,axis=0).tolist()
    new['control_axis_refinement']={'source':'RGB box and measured control footprint',
        'previous_contact_pixel_origin_world_m':spec['joint_origin_world_m'],
        'center_world_m':center.tolist(),'footprint_extent_m':(high-low).tolist(),
        'input_contact_points':len(cloud),'protruding_contact_points':len(refined),
        'simulator_object_truth_used':False}
    return new,refined


def observed_control_rotation(reference,current,origin,axis,*,bound=np.deg2rad(18)):
    """Depth-template angular registration, with ambiguity explicitly reported."""
    reference=np.asarray(reference,float);current=np.asarray(current,float)
    if len(reference)<30 or len(current)<30:return {'valid':False,'reason':'insufficient_depth'}
    origin=np.asarray(origin);axis=np.asarray(axis);tree=cKDTree(current)
    grid=np.linspace(-bound,bound,145);cost=[];coverage=[]
    for value in grid:
        predicted=Rotation.from_rotvec(value*axis).apply(reference-origin)+origin
        dist=tree.query(predicted)[0]
        cost.append(float(np.mean(np.minimum(dist,.015)**2)));coverage.append(float(np.mean(dist<.006)))
    best=int(np.argmin(cost));angle=float(grid[best]);far=np.abs(grid-angle)>np.deg2rad(4)
    separation=float(np.min(np.asarray(cost)[far])-cost[best]) if far.any() else 0.
    return {'valid':bool(coverage[best]>=.35 and separation>2e-7 and best not in {0,len(grid)-1}),
        'angle_rad':angle,'angle_deg':float(np.rad2deg(angle)),'coverage_6mm':coverage[best],
        'cost_separation_m2':separation,'boundary_solution':best in {0,len(grid)-1},
        'source':'depth template registered about observed control axis','simulator_state_used':False}
