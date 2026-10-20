"""Dense Panda tool surfaces from configured current robot convex geometry.

The collision hull is the same configured robot prior used by the simulator.
No scene mesh or object identity is queried. Samples cover every hull triangle;
the requested lattice spacing is never weakened to reduce the point count.
"""

from copy import deepcopy
import re

import numpy as np
from scipy.spatial import ConvexHull

from .geometry import pose_matrix, transform_points


def _triangle_lattice(vertices, spacing):
    """Sample a deterministic barycentric lattice with edges at most spacing."""
    vertices = np.asarray(vertices, float)
    edges = vertices[[1, 2, 0]] - vertices
    maximum_edge = float(np.linalg.norm(edges, axis=1).max())
    subdivisions = max(1, int(np.ceil(maximum_edge / spacing)))
    first = np.concatenate([np.full(subdivisions - i + 1, i, dtype=float)
                            for i in range(subdivisions + 1)])
    second = np.concatenate([np.arange(subdivisions - i + 1, dtype=float)
                             for i in range(subdivisions + 1)])
    a, b = first / subdivisions, second / subdivisions
    points = ((1 - a - b)[:, None] * vertices[0] + a[:, None] * vertices[1]
              + b[:, None] * vertices[2])
    return points, maximum_edge / subdivisions


def _deduplicate(points):
    # Numeric equality only: this is not a voxel subsampler that drops nearby
    # samples and silently widens the requested surface coverage.
    _, selected = np.unique(np.round(points, decimals=10), axis=0, return_index=True)
    return np.asarray(points)[np.sort(selected)]


def densify_tool_calibration(calibration, currentrobotvolumes, currenttcp, spacing=.005):
    """Return a copied calibration and exact-source surface sampling metadata.

    Only names with the known configured gripper prefix are admitted. Entire
    finger meshes remain ordinary collision geometry; intended-contact labels
    apply solely to the inward local-x face of configured pad collision boxes.
    """
    spacing = float(spacing)
    if not np.isfinite(spacing) or spacing <= 0 or spacing > .005:
        raise ValueError('Physical tool lattice spacing must be positive and at most 5mm')
    closing = np.asarray(calibration.get('tcp_closing_axis', [1, 0, 0]), float)
    if not np.allclose(closing, [1, 0, 0], atol=1e-9):
        raise ValueError('Configured Panda contact faces require TCP local-x closing')
    tcp = pose_matrix(currenttcp)
    world_to_tcp = np.linalg.inv(tcp)
    prefix = calibration.get('gripper_naming_prefix')
    blocks, finger_blocks, contact_blocks, name_blocks, rows = [], [], [], [], []
    for volume in currentrobotvolumes:
        name = str(volume.get('robot_geom_name', ''))
        is_tool = name.startswith(prefix) if prefix else bool(re.match(r'^gripper[0-9]+_', name))
        if not is_tool:
            continue
        if volume.get('source') != 'configured robot collision geometry' or volume.get('type') != 'convex':
            raise ValueError('Dense tool surfaces require exact configured robot convex geometry')
        vertices = np.asarray(volume['vertices_world'], float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError('Configured tool hull vertices must be finite Nx3 points')
        local_vertices = transform_points(vertices, world_to_tcp)
        hull = ConvexHull(local_vertices)
        sampled, max_lattice_edge = [], 0.
        for simplex in hull.simplices:
            points, edge = _triangle_lattice(local_vertices[simplex], spacing)
            sampled.append(points)
            max_lattice_edge = max(max_lattice_edge, edge)
        local_points = _deduplicate(np.vstack(sampled))
        fingers = np.full(len(local_points), 'finger' in name, bool)
        contacts = np.zeros(len(local_points), bool)
        if name.endswith('_pad_collision'):
            # The two fingers lie on opposite sides of local x. This labels
            # their inward physical box face using exact mesh coordinates.
            inner_x = float(np.min(np.abs(local_vertices[:, 0])))
            contacts = np.abs(np.abs(local_points[:, 0]) - inner_x) <= 1e-7
            if not fingers.all() or not contacts.any():
                raise ValueError('Configured Panda pad must have an inward finger face')
        blocks.append(local_points)
        finger_blocks.append(fingers)
        contact_blocks.append(contacts)
        name_blocks.extend([name] * len(local_points))
        rows.append({'robot_geom_name': name, 'input_hull_vertices': len(vertices),
                     'hull_triangles': len(hull.simplices), 'surface_samples': len(local_points),
                     'maximum_lattice_edge_m': max_lattice_edge,
                     'finger_samples': int(fingers.sum()), 'contact_face_samples': int(contacts.sum()),
                     'source': volume['source']})
    if not blocks:
        raise ValueError('No configured Panda gripper collision hulls were supplied')
    all_points = np.vstack(blocks)
    all_fingers, all_contacts = np.concatenate(finger_blocks), np.concatenate(contact_blocks)
    keys, inverse = np.unique(np.round(all_points, decimals=10), axis=0, return_inverse=True)
    _, first = np.unique(inverse, return_index=True)
    points = all_points[first]
    # Coincident samples can belong to multiple configured bodies. A contact
    # exemption is allowed only if every contributing body permits contact.
    fingers = np.ones(len(keys), bool)
    contacts = np.ones(len(keys), bool)
    np.logical_and.at(fingers, inverse, all_fingers)
    np.logical_and.at(contacts, inverse, all_contacts)
    names = np.asarray(name_blocks, dtype=str)[first]
    result = deepcopy(calibration)
    result.update(tool_collision_points_tcp=points.tolist(),
                  tool_collision_finger_flags=fingers.tolist(),
                  tool_collision_contact_face_flags=contacts.tolist(),
                  tool_collision_geom_names=names.tolist(),
                  tool_geometry_configuration='every configured gripper convex hull triangle; current physical opening',
                  tool_surface_sample_spacing_m=spacing)
    metadata = {'version': 'dense-tool-surface-3.0', 'requested_spacing_m': spacing,
                'maximum_lattice_edge_m': max(row['maximum_lattice_edge_m'] for row in rows),
                'source_geometry': 'configured current robot convex collision hulls only',
                'frame': 'controller TCP', 'input_calibration_samples': len(calibration['tool_collision_points_tcp']),
                'surface_samples': len(points), 'finger_samples': int(fingers.sum()),
                'contact_face_samples': int(contacts.sum()), 'per_geometry': rows,
                'deduplication': 'numeric equality 1e-10m; contact allowed only if all coincident bodies permit it',
                'contact_policy': 'inward faces of configured Panda pad collision boxes only',
                'coverage_limit': 'point sampled hull surfaces; no continuous swept-volume guarantee'}
    return result, metadata
