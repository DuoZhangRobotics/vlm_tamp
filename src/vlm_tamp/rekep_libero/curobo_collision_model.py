"""Calibrated arm/palm spheres for GPU proposal guidance, never final admission.

Inscribed spheres approximate measured/configured robot convex volumes. Fingers
and exact occupied surfaces remain checked by the native contact-aware checker.
"""
import re

import numpy as np

from .curobo_model import axis_frame


def inscribed_spheres(volume, maximum=48):
    vertices = np.asarray(volume['vertices_world'], float)
    planes = np.asarray(volume['planes_world'], float).copy()
    planes /= np.linalg.norm(planes[:, :3], axis=1)[:, None]
    low, high = vertices.min(0), vertices.max(0)
    spacing = .012 if 'hand' in volume['robot_geom_name'] else .025
    axes = [np.arange(a+spacing/2, b, spacing) for a, b in zip(low, high)]
    if any(not len(a) for a in axes):
        candidates = vertices.mean(0)[None]
    else:
        candidates = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        candidates = np.vstack([candidates, vertices.mean(0)])
    radii = np.empty(len(candidates))
    for i in range(0, len(candidates), 256):
        radii[i:i+256] = -(candidates[i:i+256] @ planes[:, :3].T+planes[:, 3]).max(1)
    chosen = []
    for index in np.argsort(-radii):
        radius = min(float(radii[index]), .045)
        if radius < .003:
            continue
        point = candidates[index]
        if any(np.linalg.norm(point-old[:3]) < .65*old[3] for old in chosen):
            continue
        chosen.append(np.r_[point, radius])
        if len(chosen) >= maximum:
            break
    if not chosen:
        raise ValueError('No internal sphere for '+volume['robot_geom_name'])
    return np.asarray(chosen)


def calibrated_spheres(chain, robot, world_robot=None):
    spheres, groups = {}, []
    for volume in robot.forward(chain.reference_joints):
        name = volume['robot_geom_name']
        if name == 'robot0_link0_collision' or 'finger' in name:
            continue
        match = re.fullmatch(r'robot0_link([1-7])_collision', name)
        if match:
            index = int(match.group(1))
            link = f'link{index}'
            frame = np.eye(4)
            frame[:3, :3] = axis_frame(chain.axes[index-1])
            frame[:3, 3] = chain.anchors[index-1]
        elif name == 'gripper0_hand_collision':
            index, link, frame = 7, 'tcp', chain.reference_tcp
        else:
            raise ValueError('Unsupported calibrated robot volume: '+name)
        values = inscribed_spheres(volume)
        local = (values[:, :3]-frame[:3, 3]) @ frame[:3, :3]
        spheres[link] = [{'center': c.tolist(), 'radius': float(r)}
                         for c, r in zip(local, values[:, 3])]
        groups.append({'link': link, 'joint_count': index,
                       'world_at_reference': values.tolist(), 'source_robot_volume': name})
    if world_robot is not None:
        # Match the arm guidance used by PalmClearanceObjective. A much tighter
        # inscribed arm proxy can produce endpoints in narrow components that
        # the complete path cannot reach. This still is only proposal geometry.
        import vamp
        def native(q):
            return np.asarray([[s.x, s.y, s.z, s.r] for s in vamp.panda.fk(np.asarray(q).tolist())[1:37]])
        reference = native(chain.reference_joints)
        downstream = np.zeros(len(reference), dtype=int)
        for j in range(7):
            for step in (.19, -.27):
                q = chain.reference_joints.copy();q[j] += step
                moved = np.linalg.norm(native(q)[:, :3]-reference[:, :3], axis=1) > 2e-6
                downstream[moved] = j+1
        world = reference.copy()
        world[:, :3] = world[:, :3] @ world_robot[:3, :3].T+world_robot[:3, 3]
        arm_spheres, arm_groups = {}, []
        for index in range(1, 8):
            ids = np.flatnonzero(downstream == index)
            if not len(ids):
                continue
            link = f'link{index}'
            local = (world[ids, :3]-chain.anchors[index-1]) @ axis_frame(chain.axes[index-1])
            arm_spheres[link] = [{'center': c.tolist(), 'radius': float(r)}
                                for c, r in zip(local, world[ids, 3])]
            arm_groups.append({'link': link, 'joint_count': index,
                'world_at_reference': world[ids].tolist(), 'vamp_indices': ids.tolist(),
                'world_robot': world_robot.tolist(), 'source_robot_volume': 'legacy_arm_clearance_spheres'})
        spheres = {**arm_spheres, 'tcp': spheres['tcp']}
        groups = arm_groups+[g for g in groups if g['link'] == 'tcp']
    return spheres, {'groups': groups, 'sphere_count': sum(map(len, spheres.values())),
        'scope': 'Calibrated arm guidance and inscribed palm proxy; native exact geometry and finger contact checks remain required.',
        'simulator_object_geometry_used': False}


def expected_spheres(robot, groups, joints):
    prefixes = robot._prefixes(joints)
    result = []
    for group in groups:
        if 'vamp_indices' in group:
            import vamp
            values = np.asarray([[s.x, s.y, s.z, s.r] for s in vamp.panda.fk(np.asarray(joints).tolist())[1:37]])[group['vamp_indices']]
            world = np.asarray(group['world_robot'])
            result.extend(np.c_[values[:, :3] @ world[:3, :3].T+world[:3, 3], values[:, 3]].tolist())
            continue
        values = np.asarray(group['world_at_reference'])
        transform = prefixes[group['joint_count']]
        result.extend(np.c_[values[:, :3] @ transform[:3, :3].T+transform[:3, 3], values[:, 3]].tolist())
    return np.asarray(result)
