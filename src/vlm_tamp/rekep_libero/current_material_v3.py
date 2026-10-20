"""Exact current configured robot visual material, with concavities retained.

Closed manifold components with consistent winding become a known-material
prior. CAD self-intersection neighborhoods stay ambiguous and are excluded.
Open components are ignored; no hole filling or convexification occurs.
"""

import json
from pathlib import Path
import re

import numpy as np


MATERIAL_SOURCE = 'configured current robot visual meshes only'


class CurrentRobotMaterialPrior:
    def __init__(self, meshes, *, source=MATERIAL_SOURCE, current_pose_only=True,
                 boundary_tolerance=1e-6, artifact=None):
        if source != MATERIAL_SOURCE or current_pose_only is not True:
            raise ValueError('Material prior requires configured CURRENT robot visual meshes only')
        tolerance = float(boundary_tolerance)
        if not np.isfinite(tolerance) or not 0 <= tolerance <= 1e-6:
            raise ValueError('Exact robot material boundary tolerance must be at most 1 micrometer')
        import open3d as o3d
        import trimesh

        self.o3d, self.tolerance, self.artifact = o3d, tolerance, artifact
        self.components, self.rows = [], []
        for item in meshes:
            name = str(item.get('robot_geom_name', ''))
            if not re.match(r'^(robot|gripper)[0-9]+_', name):
                raise ValueError('Only configured robot-named visual meshes may enter material prior')
            vertices, faces = np.asarray(item['vertices_world'], float), np.asarray(item['faces'], int)
            if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3
                    or not len(vertices) or not len(faces) or not np.isfinite(vertices).all()
                    or faces.min() < 0 or faces.max() >= len(vertices)):
                raise ValueError('Robot material needs finite vertices and valid triangle indices')
            # Meshes often repeat identical vertices for texture/normal seams.
            # Weld only exact coordinate duplicates, preserving every surface.
            unique, inverse = np.unique(vertices, axis=0, return_inverse=True)
            welded_faces = inverse[faces]
            mesh = trimesh.Trimesh(vertices=unique, faces=welded_faces, process=False)
            pieces = mesh.split(only_watertight=False, repair=False)
            row = {'robot_geom_name': name, 'input_vertices': len(vertices), 'input_triangles': len(faces),
                   'exact_welded_vertices': len(unique), 'components': len(pieces),
                   'accepted_watertight_components': 0, 'accepted_closed_components': 0,
                   'cad_seam_components': [], 'ignored_components': []}
            for component_index, piece in enumerate(pieces):
                if not piece.is_watertight:
                    row['ignored_components'].append({'component_index': component_index,
                        'triangles': len(piece.faces), 'reason': 'open_or_nonmanifold_component'})
                    continue
                legacy = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(np.asarray(piece.vertices)),
                    o3d.utility.Vector3iVector(np.asarray(piece.faces, np.int32)))
                if (not legacy.is_edge_manifold(allow_boundary_edges=False)
                        or not legacy.is_vertex_manifold() or not piece.is_winding_consistent):
                    row['ignored_components'].append({'component_index': component_index,
                        'triangles': len(piece.faces), 'reason': 'not_closed_manifold_with_consistent_winding'})
                    continue
                pairs = np.asarray(legacy.get_self_intersecting_triangles(), dtype=int).reshape(-1, 2)
                ambiguity = np.empty((0, 2, 3), float)
                if len(pairs):
                    pair_points = np.asarray(piece.vertices)[np.asarray(piece.faces)[pairs]].reshape(len(pairs), 6, 3)
                    ambiguity = np.stack([pair_points.min(axis=1) - tolerance,
                                          pair_points.max(axis=1) + tolerance], axis=1)
                    row['cad_seam_components'].append({'component_index': component_index,
                        'self_intersection_pairs': len(pairs), 'ambiguity_aabbs_world': ambiguity.tolist(),
                        'policy': 'deny component membership inside every intersecting triangle-pair AABB'})
                tensor = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
                scene = o3d.t.geometry.RaycastingScene()
                scene.add_triangles(tensor)
                self.components.append({'scene': scene, 'low': np.asarray(piece.vertices).min(axis=0),
                    'high': np.asarray(piece.vertices).max(axis=0), 'robot_geom_name': name,
                    'component_index': component_index, 'vertices': len(piece.vertices), 'triangles': len(piece.faces),
                    'ambiguity_regions': ambiguity, 'geometrically_watertight': len(pairs) == 0})
                row['accepted_closed_components'] += 1
                row['accepted_watertight_components'] += int(len(pairs) == 0)
            self.rows.append(row)

    @classmethod
    def from_artifact(cls, artifact):
        if artifact.get('source') != MATERIAL_SOURCE or artifact.get('current_pose_only') is not True:
            raise ValueError('Mesh artifact must contain only configured current robot visual material')
        manifest = json.loads(Path(artifact['manifest_path']).read_text())
        if manifest.get('source', MATERIAL_SOURCE) != MATERIAL_SOURCE:
            raise ValueError('Scene meshes cannot enter a robot material manifest')
        meshes = []
        with np.load(artifact['npz_path'], allow_pickle=False) as arrays:
            for record in manifest['meshes']:
                local = np.asarray(arrays[record['vertices_key']], float)
                rotation = np.asarray(arrays[record['rotation_key']], float)
                position = np.asarray(arrays[record['position_key']], float)
                if rotation.shape != (3, 3) or position.shape != (3,) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
                    raise ValueError('Robot material world transform must be rigid')
                meshes.append({'robot_geom_name': record['robot_geom_name'],
                               'vertices_world': local @ rotation.T + position,
                               'faces': arrays[record['faces_key']]})
        return cls(meshes, artifact=dict(artifact))

    def contains(self, points):
        points = np.asarray(points, float).reshape(-1, 3)
        inside = np.zeros(len(points), bool)
        valid = np.isfinite(points).all(axis=1)
        for component in self.components:
            near = (valid & ~inside & (points >= component['low'] - self.tolerance).all(axis=1)
                    & (points <= component['high'] + self.tolerance).all(axis=1))
            # CAD files may have a few crossing triangles at color/seam joins.
            # Never claim material in any such bounding region. Another
            # independently closed component can still prove robot material.
            for low, high in component['ambiguity_regions']:
                ambiguous = (points >= low).all(axis=1) & (points <= high).all(axis=1)
                near[ambiguous] = False
            indices = np.flatnonzero(near)
            for offset in range(0, len(indices), 20000):
                chunk = indices[offset:offset + 20000]
                tensor = self.o3d.core.Tensor(np.asarray(points[chunk], np.float32))
                signed = component['scene'].compute_signed_distance(tensor, nsamples=5).numpy()
                inside[chunk] = signed <= self.tolerance
        return inside

    def metadata(self):
        return {'version': 'exact-current-robot-material-3.0', 'source': MATERIAL_SOURCE,
                'current_pose_only': True, 'mesh_count': len(self.rows),
                'accepted_topologically_closed_components': len(self.components),
                'watertight_components': sum(item['geometrically_watertight'] for item in self.components),
                'cad_seam_components': sum(not item['geometrically_watertight'] for item in self.components),
                'cad_self_intersection_pairs': sum(len(item['ambiguity_regions']) for item in self.components),
                'accepted_triangles': sum(item['triangles'] for item in self.components),
                'ignored_components': sum(len(item['ignored_components']) for item in self.rows),
                'boundary_numeric_tolerance_m': self.tolerance,
                'boundary_numeric_basis': 'Open3D float32 world-coordinate query precision',
                'per_mesh': self.rows, 'convexification': False, 'hole_filling': False,
                'query': 'union of closed consistent component odd-ray interiors away from explicit CAD seam ambiguities',
                'field_application': 'exact point exclusion only; no coarse visual-material voxel carving',
                'unknown_outside_current_material': 'unchanged'}
