"""Dense tool geometry tests without learned inference or simulator access."""

import numpy as np
import pytest
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

from vlm_tamp.rekep_libero.dense_tool_v3 import densify_tool_calibration
from vlm_tamp.rekep_libero.geometry import transform_points
from vlm_tamp.rekep_libero.grasp_v3 import AnalyticRobotSDF


def box(name, low, high, transform=None):
    vertices = np.asarray([[x, y, z] for x in [low[0], high[0]]
                           for y in [low[1], high[1]] for z in [low[2], high[2]]])
    if transform is not None:
        vertices = transform_points(vertices, transform)
    return {'robot_geom_name': name, 'type': 'convex', 'source': 'configured robot collision geometry',
            'vertices_world': vertices.tolist(), 'planes_world': ConvexHull(vertices).equations.tolist()}


def calibration():
    base = np.eye(4)
    base[2, 3] = .097
    return {'tool_collision_points_tcp': [[0, 0, 0]], 'tool_collision_finger_flags': [False],
            'tcp_closing_axis': [1, 0, 0], 'graspgen_base_to_tcp': base.tolist()}


def test_dense_surface_covers_face_interiors_and_preserves_tcp_frame():
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('xyz', [.2, .4, -.6]).as_matrix()
    transform[:3, 3] = [1, 2, 3]
    volume = box('gripper0_hand_collision', [-.05, -.03, -.10], [.05, .03, -.07], transform)
    original = calibration()
    dense, metadata = densify_tool_calibration(original, [volume], transform)
    points = np.asarray(dense['tool_collision_points_tcp'])
    assert len(points) > 8 and metadata['maximum_lattice_edge_m'] <= .005 + 1e-12
    assert np.allclose(points.min(axis=0), [-.05, -.03, -.10])
    assert np.allclose(points.max(axis=0), [.05, .03, -.07])
    face_grid = np.asarray([[x, y, -.07] for x in np.linspace(-.05, .05, 21)
                            for y in np.linspace(-.03, .03, 13)])
    assert cKDTree(points).query(face_grid)[0].max() <= .005
    assert original['tool_collision_points_tcp'] == [[0, 0, 0]]
    assert dense['graspgen_base_to_tcp'][2][3] == .097
    assert not np.asarray(dense['tool_collision_contact_face_flags']).any()


def test_only_inward_pad_faces_allow_contact_and_all_meshes_remain_present():
    volumes = [box('gripper0_hand_collision', [-.02, -.02, -.07], [.02, .02, -.05]),
               box('gripper0_finger1_collision', [-.055, -.01, -.04], [-.048, .01, -.01]),
               box('gripper0_finger1_pad_collision', [-.048, -.008, -.012], [-.040, .008, .004]),
               box('gripper0_finger2_pad_collision', [.040, -.008, -.012], [.048, .008, .004]),
               box('robot0_link6_collision', [-.1, -.1, -.2], [.1, .1, -.1])]
    dense, metadata = densify_tool_calibration(calibration(), volumes, np.eye(4))
    points = np.asarray(dense['tool_collision_points_tcp'])
    names = np.asarray(dense['tool_collision_geom_names'])
    faces = np.asarray(dense['tool_collision_contact_face_flags'])
    assert len(metadata['per_geometry']) == 4
    assert faces.any() and np.allclose(np.abs(points[faces, 0]), .040)
    assert all(name.endswith('_pad_collision') for name in names[faces])
    assert not faces[names == 'gripper0_finger1_collision'].any()
    assert not faces[names == 'gripper0_hand_collision'].any()
    assert not any(name.startswith('robot0') for name in names)


def test_dense_current_geometry_is_exact_robot_prior_but_nearby_unknown_stays_blocked():
    volumes = [box('gripper0_hand_collision', [.003, .003, .003], [.0149, .0149, .0149])]
    dense, _ = densify_tool_calibration(calibration(), volumes, np.eye(4), .003)
    shape = (11, 11, 11)
    field = AnalyticRobotSDF(np.zeros(3), np.full(3, .02), np.zeros(shape, bool), np.zeros(shape, bool),
                             np.full(shape, .003), np.full(3, .002))
    field.exclude_known_robot_volumes(volumes)
    points = np.asarray(dense['tool_collision_points_tcp'])
    assert field.states(points)['free'].all()
    assert (field.collision_values(points)[0] <= 0).all()
    assert field.states([[.0159, .01, .01]])['unknown'][0]
    assert field.collision_values([[.0159, .01, .01]])[0][0] > 0


def test_scene_geometry_and_overly_sparse_requested_spacing_rejected():
    volume = box('gripper0_hand_collision', [0, 0, 0], [.02, .02, .02])
    volume['source'] = 'simulator environment geometry'
    with pytest.raises(ValueError, match='exact configured robot'):
        densify_tool_calibration(calibration(), [volume], np.eye(4))
    with pytest.raises(ValueError, match='at most 5mm'):
        densify_tool_calibration(calibration(), [], np.eye(4), .01)
    with pytest.raises(ValueError, match='No configured'):
        densify_tool_calibration(calibration(), [], np.eye(4))


def test_coincident_non_contact_mesh_disables_pad_contact_exemption():
    volumes = [box('gripper0_finger1_pad_collision', [.04, -.008, -.012], [.048, .008, .004]),
               box('gripper0_hand_collision', [.04, -.008, -.012], [.048, .008, .004])]
    dense, _ = densify_tool_calibration(calibration(), volumes, np.eye(4))
    assert not np.asarray(dense['tool_collision_contact_face_flags']).any()
