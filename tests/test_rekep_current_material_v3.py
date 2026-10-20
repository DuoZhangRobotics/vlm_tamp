"""Exact known material keeps concave holes and open components unknown."""

import json

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

pytest.importorskip('open3d')
trimesh = pytest.importorskip('trimesh')

from vlm_tamp.rekep_libero.current_material_v3 import MATERIAL_SOURCE, CurrentRobotMaterialPrior
from vlm_tamp.rekep_libero.grasp_v3 import AnalyticRobotSDF


def l_mesh():
    base = np.asarray([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float) * .01
    vertices = np.r_[np.c_[base, np.zeros(6)], np.c_[base, np.full(6, .01)]]
    tops = np.asarray([[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]])
    faces = [tops[:, ::-1], tops + 6]
    side = []
    for index in range(6):
        nxt = (index + 1) % 6
        side.extend([[index, nxt, nxt + 6], [index, nxt + 6, index + 6]])
    faces.append(np.asarray(side))
    return {'robot_geom_name': 'robot0_test_visual', 'vertices_world': vertices, 'faces': np.vstack(faces)}


def field():
    shape = (8, 8, 5)
    return AnalyticRobotSDF(np.asarray([-.005, -.005, -.005]), np.asarray([.03, .03, .015]),
        np.zeros(shape, bool), np.zeros(shape, bool), np.full(shape, .005), np.full(3, .005))


def test_concave_material_retains_unknown_inside_convex_hull_hole():
    mesh = l_mesh()
    material = CurrentRobotMaterialPrior([mesh])
    hole = [.014, .014, .005]
    hull = ConvexHull(mesh['vertices_world'])
    assert np.all(hull.equations[:, :3] @ hole + hull.equations[:, 3] <= 1e-10)
    assert material.contains([[.005, .015, .005], [.015, .005, .005], hole]).tolist() == [True, True, False]
    observed = field()
    observed.exclude_current_visual_material(material)
    assert observed.states([[.005, .015, .005]])['free'][0]
    assert observed.states([hole])['unknown'][0]
    assert observed.collision_values([hole])[0][0] > 0
    assert observed.collision_values([[.005, .015, .005]])[0][0] < 0


def test_one_mm_outside_current_material_stays_unknown():
    material = CurrentRobotMaterialPrior([l_mesh()])
    observed = field()
    observed.exclude_current_visual_material(material)
    outside = [[-.001, .005, .005], [.011, .019, .005]]
    assert observed.states(outside)['unknown'].all()
    assert (observed.collision_values(outside)[0] > 0).all()


def test_open_component_never_filled_or_used_as_known_material():
    mesh = l_mesh()
    mesh['faces'] = mesh['faces'][:-1]  # A single triangle hole must not be repaired.
    material = CurrentRobotMaterialPrior([mesh])
    assert material.metadata()['watertight_components'] == 0
    assert material.metadata()['ignored_components'] > 0
    assert not material.contains([[.005, .005, .005]])[0]
    observed = field()
    assert observed.exclude_current_visual_material(material) == 0
    assert observed.states([[.005, .005, .005]])['unknown'][0]


def test_union_of_overlapping_watertight_components_does_not_use_parity_cancellation():
    first, second = l_mesh(), l_mesh()
    second['robot_geom_name'] = 'robot0_other_visual'
    second['vertices_world'] = second['vertices_world'] + [.001, 0, 0]
    material = CurrentRobotMaterialPrior([first, second])
    assert material.metadata()['watertight_components'] == 2
    assert material.contains([[.005, .005, .005]])[0]


def test_current_artifact_world_transform_and_save_load_replay(tmp_path):
    mesh = l_mesh()
    rotation = Rotation.from_euler('z', .3).as_matrix()
    position = np.asarray([-.2, .1, 1.])
    keys = {'vertices_key': 'geom_000_vertices_local', 'faces_key': 'geom_000_faces',
            'rotation_key': 'geom_000_rotation_world', 'position_key': 'geom_000_position_world'}
    archive, manifest = tmp_path / 'robot.npz', tmp_path / 'robot.json'
    np.savez_compressed(archive, **{keys['vertices_key']: mesh['vertices_world'], keys['faces_key']: mesh['faces'],
                                    keys['rotation_key']: rotation, keys['position_key']: position})
    manifest.write_text(json.dumps({'source': MATERIAL_SOURCE, 'meshes': [{**keys,
        'robot_geom_name': mesh['robot_geom_name']}] }))
    artifact = {'source': MATERIAL_SOURCE, 'current_pose_only': True,
                'npz_path': str(archive), 'manifest_path': str(manifest)}
    material = CurrentRobotMaterialPrior.from_artifact(artifact)
    inside = np.asarray([.005, .015, .005]) @ rotation.T + position
    hole = np.asarray([.014, .014, .005]) @ rotation.T + position
    assert material.contains([inside, hole]).tolist() == [True, False]
    shape = (8, 8, 5)
    observed = AnalyticRobotSDF(position + [-.005, -.005, -.005], position + [.03, .03, .015],
        np.zeros(shape, bool), np.zeros(shape, bool), np.full(shape, .005), np.full(3, .005))
    observed.exclude_current_visual_material(material)
    output = tmp_path / 'field.npz'
    observed.save(output)
    saved = AnalyticRobotSDF.load(output)
    assert saved.states([inside, hole])['free'].tolist() == [True, False]


def test_scene_and_future_meshes_are_rejected():
    with pytest.raises(ValueError, match='CURRENT robot'):
        CurrentRobotMaterialPrior([l_mesh()], current_pose_only=False)
    with pytest.raises(ValueError, match='CURRENT robot'):
        CurrentRobotMaterialPrior([l_mesh()], source='simulator environment mesh')
    mesh = l_mesh()
    mesh['robot_geom_name'] = 'wooden_cabinet_visual'
    with pytest.raises(ValueError, match='robot-named'):
        CurrentRobotMaterialPrior([mesh])


def cad_seam_mesh():
    box = trimesh.creation.box(extents=[.02, .02, .02])
    box.apply_translation([.01, .01, .01])
    vertices, faces = box.vertices, box.faces
    for _ in range(3):
        vertices, faces = trimesh.remesh.subdivide(vertices, faces)
    vertices = vertices.copy()
    index = np.argmin(np.linalg.norm(vertices - [.0175, .02, .0175], axis=1))
    vertices[index] = [.0215, .018, .0175]
    return {'robot_geom_name': 'robot0_cad_visual', 'vertices_world': vertices, 'faces': faces}


def test_localized_cad_seam_denies_ambiguity_but_admits_proven_interior_away():
    material = CurrentRobotMaterialPrior([cad_seam_mesh()])
    metadata = material.metadata()
    assert metadata['cad_seam_components'] == 1
    assert metadata['accepted_topologically_closed_components'] == 1
    assert metadata['watertight_components'] == 0  # Do not call self-intersecting CAD geometrically watertight.
    assert metadata['cad_self_intersection_pairs'] > 0
    seam = material.components[0]['ambiguity_regions'][0].mean(axis=0)
    assert material.contains([[.005, .005, .005], seam, [-.001, .005, .005]]).tolist() == [True, False, False]
    observed = field()
    observed.exclude_current_visual_material(material)
    assert observed.states([seam])['unknown'][0]
    assert observed.states([[.005, .005, .005]])['free'][0]


def test_independent_closed_material_can_prove_a_cad_ambiguity_region():
    first = CurrentRobotMaterialPrior([cad_seam_mesh()])
    seam = first.components[0]['ambiguity_regions'][0].mean(axis=0)
    other = trimesh.creation.box(extents=[.002] * 3)
    other.apply_translation(seam)
    material = CurrentRobotMaterialPrior([cad_seam_mesh(), {'robot_geom_name': 'robot0_independent_visual',
        'vertices_world': other.vertices, 'faces': other.faces}])
    assert material.contains([seam])[0]
