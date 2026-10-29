"""Private evaluator: compare replays and attribute stale points to native geometry.

Native articulation and asset geometry are read only after the replay/planner
probes have finished. This report is never an actor or planner input.
"""
import argparse
import hashlib
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from vlm_tamp.coupled_storage import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'third_party/LIBERO/libero/libero'


def articulation(row):
    return next(j for j in row['evaluation_only_articulation_state'] if j['joint_name'] == 'microwave_1_microjoint')


def pairs(row):
    return {tuple(sorted(c['geom_names'])) for c in row['robot_involved_contacts']}


def main(output):
    xml = ASSETS / 'assets/articulated_objects/microwave.xml'
    bddl = ASSETS / 'bddl_files/libero_90/KITCHEN_SCENE7_open_the_microwave.bddl'
    handle = ET.parse(xml).getroot().find(".//body[@name='microdoorroot']/geom[@class='micro_collision'][@type='capsule']")
    offset = np.fromstring(handle.attrib['pos'], sep=' ')
    radius, length = np.fromstring(handle.attrib['size'], sep=' ')
    region = bddl.read_text().split('(microwave_init_region', 1)[1].split('(plate_init_region', 1)[0]
    yaw_range = re.search(r':yaw_rotation\s*\(\s*\(([^)]+)\)', region)[1]
    yaws = np.fromstring(yaw_range, sep=' ')
    assert len(yaws) == 2 and yaws[0] == yaws[1], 'Evaluator requires fixed fixture yaw'
    yaw = float(yaws[0])
    comparisons = []
    for root in sorted(output.glob('state*-prefix')):
        protocol = read_json(root / 'protocol.json')
        summary = read_json(root / 'summary.json')
        source = Path(protocol['source'])
        native = read_json(root / 'attempt01/initial/evaluation_contact_diagnostics.json')['steps']
        original = read_json(source / 'attempt01/initial/evaluation_contact_diagnostics.json')['steps'][:len(native)]
        opening = [-np.degrees(articulation(row)['position']) for row in native]
        delta = np.asarray(opening) - np.asarray([-np.degrees(articulation(row)['position']) for row in original])
        timeline = []
        first_static = None
        for snapshot in sorted((root / 'snapshots').glob('event*')):
            obs = read_json(snapshot / 'observation.json')
            reference = read_json(snapshot / 'scene-reference.json')
            a = articulation(native[obs['control_steps'] - 1])
            rotation = Rotation.from_rotvec([0, 0, yaw + a['position']]).as_matrix()
            centre = np.asarray(a['anchor_world_m']) + rotation @ offset
            with np.load(reference['arrays_path']) as arrays:
                static = arrays['static_points']
                if first_static is None:
                    first_static = static.copy()
                unchanged = np.array_equal(static, first_static)
                point_counts = {key: len(arrays[key]) for key in arrays.files}
            timeline.append({'event': snapshot.name, 'control_step': obs['control_steps'],
                             'native_opening_degrees': float(-np.degrees(a['position'])),
                             'native_handle_center_world_m': centre,
                             'model_handle_center_xy_error_m': float(np.linalg.norm(centre[:2] - np.asarray(obs['spec']['source_anchor_world_m'])[:2])),
                             'stationary_points_identical_to_grounding': unchanged, 'point_counts': point_counts})
        hashes_match = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h
                           for p, h in protocol['planner_source_hashes'].items())
        comparisons.append({'case': root.name, 'replay': summary['replay'],
                            'native_opening_maximum_difference_degrees': float(np.abs(delta).max()),
                            'native_contact_pair_set_differences': sum(pairs(a) != pairs(b) for a, b in zip(native, original)),
                            'planner_source_hashes_unchanged': hashes_match,
                            'final_opening_degrees': opening[-1], 'timeline': timeline, 'probes': summary['probes']})
        analysis_path = root / 'probes/event018/point-analysis.json'
        if analysis_path.exists():
            points = np.asarray(read_json(analysis_path)['world_points'])
            rows = []
            for step in [60, 247, 259, 275]:
                a = articulation(native[step - 1])
                rotation = Rotation.from_rotvec([0, 0, yaw + a['position']]).as_matrix()
                centre = np.asarray(a['anchor_world_m']) + rotation @ offset
                local = (points - centre) @ rotation
                nearest = local.copy()
                nearest[:, :2] = 0
                nearest[:, 2] = np.clip(local[:, 2], -length, length)
                distances = np.linalg.norm(local - nearest, axis=1) - radius
                rows.append({'control_step': step, 'native_opening_degrees': float(-np.degrees(a['position'])),
                             'handle_center_world_m': centre, 'stationary_point_signed_distances_to_native_handle_m': distances})
            write_json(root / 'evaluator-handle-provenance.json', {
                'evaluator_only': True, 'excluded_from_planner_and_actor': True,
                'geometry_sources': [str(xml), str(bddl)],
                'asset_source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [xml, bddl]},
                'fixture_yaw_radians': yaw, 'native_handle_capsule_local_position': offset,
                'native_handle_radius_m': radius, 'native_handle_half_length_m': length,
                'points_world_m': points, 'steps': rows})
    write_json(output / 'comparison.json', {'scope': 'development evaluator-only, no fresh agent trials',
                                           'cases': comparisons})
    for row in comparisons:
        print({key: row[key] for key in ('case', 'native_opening_maximum_difference_degrees',
                                         'native_contact_pair_set_differences', 'planner_source_hashes_unchanged', 'final_opening_degrees')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output.resolve())
