"""Pointing preserves camera provenance and feeds existing primitive interfaces."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vlm_tamp.compact_robot_interface import compact_response
from vlm_tamp.perception_tools import PerceptionTools
from vlm_tamp.robot_contracts import validate_operation
from vlm_tamp.surface_contact_tools import selected_pixel_point


class Pointer:
    def __init__(self, image_index: int = 2) -> None:
        self.image_index = image_index
        self.calls = []

    def point(self, paths: list[Path], prompt: str) -> dict:
        self.calls.append((paths, prompt))
        return {'status': 'parsed', 'backend': 'allenai/Molmo2-8B',
                'points': [{'image_index': self.image_index, 'point_id': 1,
                            'coordinates_1000': [250, 500], 'pixel_xy': [20.25, 30.25]}],
                'raw_text': '<points coords="2 1 250 500">part</points>',
                'worker_pid': 123, 'worker_request_sequence': len(self.calls),
                'preprocessing_seconds': .01, 'generation_seconds': .1}


class Segmenter:
    def predict(self, view, key, points, labels, box):
        self.input = (view, key, points, labels, box)
        mask = np.zeros(view['rgb'].shape[:2], bool)
        mask[25:35, 15:25] = True
        return [mask], [1.], {'mask_prediction': .01}


def engine(tmp_path: Path, pointer=None) -> PerceptionTools:
    folder = tmp_path / 'capture'
    folder.mkdir()
    for camera in ['agentview', 'wrist']:
        Image.new('RGB', (80, 60), 'gray').save(folder / (camera + '.png'))
    return PerceptionTools({7: {'sensors': folder / 'intentionally-absent.npz',
                                'cameras': ['agentview', 'wrist'], 'attempt': 1}},
                           tmp_path / 'perception', tmp_path, pointer=pointer or Pointer())


def test_point_reads_only_rgb_and_preserves_luna_query(tmp_path: Path) -> None:
    tools = engine(tmp_path)
    query = 'Point to a visible lip that I could hook without pinching.'
    result = tools.point(8, 7, 'wrist', query)
    assert result['points'][0]['contact_pixel_xy'] == [20, 30]
    assert result['points'][0]['pixel_xy'] == [20.25, 30.25]
    assert query in tools.pointer.calls[0][1]
    assert [p.name for p in tools.pointer.calls[0][0]] == ['agentview.png', 'wrist.png']
    request = json.loads((tmp_path / result['request_file']).read_text())
    assert request['query'] == query
    assert not result['planner_model_updated'] and not result['robot_motion_executed']
    compact = compact_response({'ok': True, 'result': result})
    assert compact['result']['points'] == result['points']
    assert compact['result']['images'] == result['images']


def test_wrong_view_is_reported_not_relabelled(tmp_path: Path) -> None:
    tools = engine(tmp_path, Pointer(image_index=1))
    result = tools.point(8, 7, 'wrist', 'Point to the handle.')
    point = result['points'][0]
    assert point['camera'] == 'agentview'
    assert not point['matches_requested_camera']
    assert result['camera'] == 'wrist'


def test_point_to_segmentation_and_contact_primitives(tmp_path: Path, monkeypatch) -> None:
    tools = engine(tmp_path)
    result = tools.point(8, 7, 'wrist', 'Point to a graspable handle.')
    pixel = result['points'][0]['contact_pixel_xy']
    y, x = np.indices((60, 80))
    world = np.stack([x / 100, y / 100, np.ones_like(x)], axis=-1)
    view = {'rgb': np.zeros((60, 80, 3), np.uint8), 'world_points': world,
            'depth': np.ones((60, 80)), 'known_robot_depth_prior': np.zeros((60, 80))}
    monkeypatch.setattr(tools, '_view', lambda *_: view)
    tools.segmenter = Segmenter()
    segmentation = tools.segment_parts(9, 7, 'wrist', points_xy=[pixel], point_labels=[1])
    part = tools.parts[segmentation['parts'][0]['part_id']]
    np.testing.assert_array_equal(selected_pixel_point(view, pixel, selected_mask=part['mask']),
                                  world[30, 20])
    assert tools.segmenter.input[2] == [pixel]
    for operation in ['push', 'hook_pull']:
        validate_operation(operation, {'observation_id': 7, 'camera': 'wrist',
            'contact_pixel_xy': pixel, 'contact_name': 'pad', 'direction_world': [1, 0, 0],
            'contact_part_id': segmentation['parts'][0]['part_id'], 'distance_m': .01})


def test_source_observation_identity_is_not_replaced(tmp_path: Path) -> None:
    tools = engine(tmp_path)
    with pytest.raises(ValueError, match='observation ID'):
        tools.point(8, 99, 'wrist', 'Point to the handle.')
    assert not tools.pointer.calls


def test_no_points_remains_a_visible_model_outcome(tmp_path: Path) -> None:
    class Missing(Pointer):
        def point(self, paths, prompt):
            result = super().point(paths, prompt)
            result.update(status='no_point_tag', points=[], raw_text='No visible lip.')
            return result
    tools = engine(tmp_path, Missing())
    result = tools.point(8, 7, 'wrist', 'Point to a retaining lip.')
    assert result['points'] == [] and result['raw_text'] == 'No visible lip.'
    assert result['status'] == 'no_point_tag'


def test_point_help_exports_an_actually_callable_schema(tmp_path: Path) -> None:
    from vlm_tamp.compact_robot_interface import install
    from vlm_tamp.robot_contracts import POINTING_HELP
    install(tmp_path, POINTING_HELP, attempts=1, initial_attempts=1,
            practice_mode=False, control_steps=600)
    assert (tmp_path / 'tools/point.txt').exists()
    assert 'point' in json.loads((tmp_path / 'tools/schema-index.json').read_text())['operations']
    validate_operation('point', {'observation_id': 7, 'camera': 'agentview',
                                 'query': 'Point to a graspable part.'})
