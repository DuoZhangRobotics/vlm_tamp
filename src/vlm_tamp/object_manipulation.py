"""Persistent sensor-selected objects and prepared placement through existing numerical tools."""

from __future__ import annotations

import time
import copy
from typing import Any

import numpy as np

from .contact_skills import fit_object_motion
from .coupled_storage import write_json
from .observation_completion import plane_fit
from .part_motion import pair_points
from .part_perception import association, region_mask, usable_pixels
from .planner_object_transfer import plan_object_transfer
from .rekep_libero.geometry import pose_matrix, transform_points
from .rekep_libero.task_geometry_v4 import surface_summary
from .timed_execution import schedule
from .execution_limits import remaining_controls


class ObjectModels:
    """Keep object identity, reference geometry and observed pose separate from the active scene."""

    def __init__(self, service: Any) -> None:
        """Use the episode's existing perception tools and observation registry."""
        self.service = service
        self.models: dict[str, dict] = {}

    def summary(self, object_id: str) -> dict:
        """Return compact provenance and pose evidence, without duplicating point clouds."""
        if object_id not in self.models:
            selected = {key: model.get('label') for key, model in self.models.items()}
            raise ValueError(f'Unknown selected object {object_id!r}. Previously selected IDs/labels: '
                             f'{selected}. Call object_status without an ID to list their currentness.')
        model = self.models[object_id]
        return {**copy.deepcopy({key: value for key, value in model.items() if key != 'cloud'}),
                'pose_current': model['pose_control_step'] == self.service.state['control_steps']
                                and model['pose_status'] == 'observed'}

    def selected_objects(self) -> list[dict]:
        """List only caller-created references; never discover native simulator objects."""
        return [{key: value for key, value in self.summary(object_id).items()
                 if key in {'object_id', 'label', 'reference_type', 'camera', 'pose_control_step',
                            'pose_status', 'pose_current'}} for object_id in self.models]

    def save(self, object_id: str) -> dict:
        """Persist the latest model independently of source and destination scene selection."""
        value = self.summary(object_id)
        destination = self.service.public/f'attempt{self.service.attempt:02d}'/'objects'
        destination.mkdir(exist_ok=True)
        write_json(destination/(object_id+'.json'), value)
        return value

    def select(self, camera: str, region_xyxy: list, contact_pixel_xy: list,
               reference_type: str, label: str | None = None) -> dict:
        """Segment a caller-selected object and construct its declared observed reference."""
        if reference_type not in {'supported_body', 'surface_center', 'surface_point'}:
            raise ValueError('Choose supported_body, surface_center or surface_point')
        service = self.service
        observation = service.latest_observation['observation_id']
        segmentation = service._part_tool('segment_parts', {
            'observation_id': observation, 'camera': camera, 'box_xyxy': region_xyxy,
            'points_xy': [contact_pixel_xy], 'point_labels': [1], 'label': label})
        selected = max(segmentation['parts'], key=lambda part: part['sam_mask_score'])
        engine = service._perception_tools
        view = engine._view(observation, camera)
        part = engine.parts[selected['part_id']]
        cloud = np.asarray(view['world_points'][part['mask']])
        shape = surface_summary(cloud)
        if reference_type == 'supported_body':
            region, _ = region_mask(view, region_xyxy)
            support = view['world_points'][region & ~part['mask'] & usable_pixels(view)]
            plane, _ = plane_fit(support)
            if not plane['supported']:
                raise ValueError('Object region does not resolve the declared planar resting surface')
            center = np.array([*shape['center_xy_m'], shape['median_z_m']])
            normal = np.asarray(plane['normal_world'])
            point = center-normal*((center-np.asarray(plane['centroid_world_m']))@normal)
            assumption = 'Selected rigid body initially rests on the measured planar surface in its region.'
        else:
            plane, _ = plane_fit(cloud)
            if not plane['supported']:
                raise ValueError('Selected destination does not resolve the declared planar support')
            normal = np.asarray(plane['normal_world'])
            if normal@(pose_matrix(view['camera_to_world'])[:3, 3]-plane['centroid_world_m']) < 0:
                normal = -normal
            if reference_type == 'surface_center':
                point = np.array([*shape['center_xy_m'], shape['median_z_m']])
            else:
                from .surface_contact_tools import selected_pixel_point
                point = selected_pixel_point(view, contact_pixel_xy, region_xyxy, part['mask'])
            assumption = 'Selected visible material surface is the intended destination.'
        identifier = f'object{service.calls:03d}'
        model = {
            'object_id': identifier, 'label': label, 'part_id': selected['part_id'],
            'camera': camera, 'region_xyxy': list(region_xyxy),
            'contact_pixel_xy': list(contact_pixel_xy), 'reference_type': reference_type,
            'reference_observation_id': observation, 'reference_control_step': service.state['control_steps'],
            'pose_observation_id': observation, 'pose_control_step': service.state['control_steps'],
            'cloud': cloud, 'shape': shape, 'reference_point_initial_world_m': point.tolist(),
            'axis_initial_world': normal.tolist(), 'transform_initial_to_current_world': np.eye(4).tolist(),
            'reference_point_world_m': point.tolist(), 'axis_world': normal.tolist(),
            'pose_status': 'observed', 'pose_measurement': None, 'attachment_verified': False,
            'reference_assumption': assumption, 'semantic_identity_verified': False,
            'mask_selection': 'highest MobileSAM proposal score from caller box/click; no alternate mask retry',
            'mask_overlay': selected['overlay'], 'mask_file': selected['mask_file'],
            'support_fit': {key: plane[key] for key in (
                'normal_world', 'centroid_world_m', 'rms_m', 'inlier_fraction', 'inlier_points')},
            'grasp_input': 'unchanged caller-selected region; mask used for object geometry and tracking',
            'simulator_object_geometry_used': False,
        }
        self.models[identifier] = model
        return {'object_model': self.save(identifier), 'images': [selected['overlay']],
                'robot_motion_executed': False, 'active_planning_scene_changed': False,
                'timing_seconds': segmentation['timing_seconds']}

    def measure(self, object_id: str, observation_ids: list[int]) -> dict:
        """Update rigid pose from the existing video tracker and measured RGB-D correspondences."""
        service = self.service
        model = self.models[object_id]
        association_id = None
        if model['camera'] != 'agentview':
            associated = service._part_tool('associate_views', {
                'part_id': model['part_id'], 'cameras': ['agentview']})
            association_id = associated['association_id']
            service.calls += 1
        tracked = service._part_tool('track_video_motion', {
            'part_id': model['part_id'], 'observation_ids': observation_ids, 'max_rgb_frames': 64,
            **({'association_id': association_id} if association_id is not None else {})})
        record = service._perception_tools.tracks[tracked['track_id']]
        source = observation_ids.index(model['reference_observation_id'])
        first, last = pair_points(record['world'], record['valid'], source, len(observation_ids)-1)
        fit = fit_object_motion(first, last)
        model['pose_measurement'] = fit
        model['last_requested_observation_id'] = observation_ids[-1]
        if fit['accepted']:
            transform = np.asarray(fit['transform'])
            model.update(
                transform_initial_to_current_world=transform.tolist(),
                reference_point_world_m=transform_points([model['reference_point_initial_world_m']], transform)[0].tolist(),
                axis_world=(transform[:3, :3]@model['axis_initial_world']).tolist(),
                pose_observation_id=observation_ids[-1],
                pose_control_step=service.observation_records[observation_ids[-1]]['control_steps'],
                pose_status='observed')
        else:
            model['pose_status'] = 'unresolved'
        return {'object_model': self.save(object_id), 'tracking': tracked,
                'object_motion': fit, 'robot_motion_executed': False}

    def current(self, object_id: str) -> dict:
        """Obtain current observed geometry; never relabel an old pose as a fresh measurement."""
        model = self.models[object_id]
        current = self.service.latest_observation['observation_id']
        if model['pose_control_step'] != self.service.state['control_steps']:
            self.service.calls += 1
            self.measure(object_id, sorted(set([model['reference_observation_id'],
                                               model['pose_observation_id'], current])))
        if model['pose_status'] != 'observed':
            raise ValueError('Object motion is unresolved; inspect the recorded tracking or select current geometry')
        return model

    def grasp(self, object_id: str, **arguments: Any) -> dict:
        """Execute the existing grasp and measure the selected object's post-closure pose."""
        service = self.service
        model = self.models[object_id]
        if model['reference_control_step'] != service.state['control_steps']:
            raise ValueError('Select a current object region before requesting a new grasp')
        service.select_surface_contact(model['camera'], model['contact_pixel_xy'], model['region_xyxy'])
        service.calls += 1
        planned = service.plan_grasp(**arguments)
        service.calls += 1
        result = service.execute_plan(planned['plan_id'])
        result['planning'] = planned
        result['object_id'] = object_id
        if result['grasp_execution']['closure_completed'] and not result.get('episode_done'):
            observations = [model['reference_observation_id'],
                            *[stage['observation']['observation_id'] for stage in result['stages']]]
            service.calls += 1
            result['object_pose_update'] = self.measure(object_id, observations)
        result['object_model'] = self.save(object_id)
        return result

    def place(self, object_id: str, destination_id: str, lift_distance_m: float,
              attachment_model: str, *, speed: float = .35, hold: int = 6,
              release_steps: int = 20, budget_seconds: float = 60.) -> dict:
        """Prepare a complete lifted transfer from observed object frames, then execute and release."""
        if attachment_model != 'rigid':
            raise ValueError('Placement requires the caller-declared rigid attachment model')
        if not np.isfinite(lift_distance_m) or lift_distance_m <= 0:
            raise ValueError('Choose a positive physical lift distance')
        if not 0 < speed <= 1 or type(hold) is not int or hold < 0 or type(release_steps) is not int or release_steps < 1:
            raise ValueError('Invalid execution speed, hold, or release duration')
        service = self.service
        source = self.current(object_id)
        destination = self.current(destination_id)
        planning_models = [self.summary(object_id), self.summary(destination_id)]
        engine = service._perception_tools
        camera = destination['camera']
        view = service.cameras[camera]
        points = transform_points(destination['cloud'], destination['transform_initial_to_current_world'])
        original_part = engine.parts[destination['part_id']]
        if original_part['observation_id'] == service.latest_observation['observation_id']:
            mask = original_part['mask']
        else:
            mask, _ = association(points, view)
        from .part_perception import box_of
        selected = np.asarray(destination['reference_point_world_m'])
        optical = transform_points([selected], np.linalg.inv(pose_matrix(view['camera_to_world'])))[0]
        projected = np.asarray(view['intrinsic'])@optical
        pixel = np.rint(projected[:2]/projected[2]).astype(int).tolist()
        current_part = f'{destination_id}-current-{service.calls:03d}'
        engine.parts[current_part] = {'observation_id': service.latest_observation['observation_id'],
                                      'camera': camera, 'mask': mask}
        service.select_surface_contact(camera, pixel, box_of(mask), contact_part_id=current_part)
        query = service._query_scene()
        point = np.asarray(source['reference_point_world_m'])
        axis = np.asarray(source['axis_world'])
        normal = np.asarray(destination['axis_world'])
        normal /= np.linalg.norm(normal)
        hand = pose_matrix(service.state['metadata']['eef_pose_world_xyzw'])
        local_point = transform_points([point], np.linalg.inv(hand))[0]
        local_axis = hand[:3, :3].T@axis
        lifted, over = point+lift_distance_m*normal, selected+lift_distance_m*normal
        floor = lifted if lifted@normal < over@normal else over
        targets = [('lift', lifted, point), ('transfer', over, floor), ('lower', selected, selected)]
        metadata = dict(service.state['metadata'])
        folder = service.folder('object-transfer');folder.mkdir()
        began = time.perf_counter()
        def metadata_at(joints: np.ndarray) -> dict:
            """Predict robot geometry for an unexecuted branch without clearing future scene points."""
            geometry = service.sim.fk(joints, include_volumes=True,
                                      gripper_qpos=service.state['metadata']['gripper_qpos'])
            return {**metadata, 'joint_positions': joints.tolist(),
                    'eef_pose_world_xyzw': service.ik.chain.forward(joints).tolist(),
                    'robot_collision_volumes_world': geometry['robot_collision_volumes_world']}
        combined, reports, search = plan_object_transfer(
            service.ik, query, service.record, metadata, metadata_at, local_point, local_axis,
            axis, normal, targets, folder, speed=speed, hold=hold, release_steps=release_steps,
            available_controls=remaining_controls(service.state), budget_seconds=budget_seconds)
        controls = len(schedule(combined, maximum_joint_speed=speed, endpoint_hold_steps=hold))
        planning_seconds = time.perf_counter()-began
        if controls+release_steps > remaining_controls(service.state):
            raise ValueError(f'Prepared transfer and release need {controls+release_steps} controls; '
                             f'{remaining_controls(service.state)} remain')
        plan = service.keep_plan(combined, {'status': 'solved', 'stages': reports}, 'free', extra={
            'arrival': {'maximum_wait_steps': 20, 'contact_local_m': local_point.tolist()},
            'required_gripper_command': 0, 'object_point_goal': reports[-1]['result']['object_point_goal'],
            'approach_speed': speed, 'approach_hold_steps': hold})
        service.calls += 1
        moved = service.execute_plan(plan['plan_id'])
        stages = [{'stage': 'object-transfer', 'control_steps': service.state['control_steps'], **moved}]
        result = {**moved, 'stages': stages, 'object_id': object_id, 'destination_id': destination_id,
                  'planning_object_models': planning_models,
                  'prepared_transfer': {'stages': reports, 'nominal_controls': controls,
                                        'planning_seconds': planning_seconds,
                                        'lift_distance_m': lift_distance_m,
                                        'sequence_search': {key: search[key] for key in (
                                            'retained_prefix_limit','selected_branches','ik_queries','path_queries',
                                            'budget_rejected_prefixes','complete_controls_including_release',
                                            'global_optimality_claimed')}},
                  'release_issued': False, 'attachment_verified': False}
        status = service.episode_status()
        if moved.get('episode_done') or status['episode_done']:
            result.update(task_success=status['task_success'],
                          task_success_available=status['task_success_available'],
                          episode_done=True, stop_reason=status['termination_reason'])
            return result
        if not moved['execution']['arrival']['goal_reached']:
            result['stop_reason'] = 'placement_reference_not_reached'
            return result
        service.calls += 1
        available = min(release_steps, remaining_controls(service.state))
        released = service.gripper(close=False, steps=available)
        stages.append({'stage': 'object-release', 'control_steps': service.state['control_steps'], **released})
        result.update({key: value for key, value in released.items() if key != 'execution'})
        result['execution'] = {'issued_commands': sum(stage['execution']['issued_commands'] for stage in stages),
                               'termination_reason': released['execution'].get('termination_reason')}
        issued_release = released['execution']['issued_commands']
        release_complete = issued_release == release_steps
        result.update(release_issued=issued_release > 0, release_complete=release_complete,
                      stop_reason='object_transfer_released' if release_complete else
                      'control_budget_exhausted' if remaining_controls(service.state) == 0 else
                      released['execution']['termination_reason'])
        source['pose_status'] = 'requires_observation_after_release'
        self.save(object_id)
        return result


def models_for(service: Any) -> ObjectModels:
    """Use one registry per physical episode without retaining data across resets."""
    if getattr(service, '_object_models', None) is None:
        service._object_models = ObjectModels(service)
    return service._object_models
