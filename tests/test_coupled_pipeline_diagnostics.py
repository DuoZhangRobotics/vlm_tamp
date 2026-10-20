import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from vlm_tamp import coupled_pipeline


def save(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, default=convert))


class TestStartGateDiagnostics(unittest.TestCase):
    def run_rejected_start(self, projection_accepted):
        pose = np.array([0., 0., 1., 0., 0., 0., 1.])
        start = {'joints': [0.]*7, 'pose_world_xyzw': pose.tolist(), 'joint_reserve_rad': .2}
        ik = SimpleNamespace(chain=SimpleNamespace(forward=Mock(return_value=pose)),
            solve_many=Mock(side_effect=[[start]]+[[]]*20), self_valid=Mock(return_value=True),
            approach=Mock(side_effect=AssertionError('Approach must remain short-circuited')))
        scene_result = {'accepted': False, 'collisions': [{'robot_geom_name': 'gripper0_finger1_collision',
            'scene_part': 'moving', 'blocked_inferred_panel_points': 8}]}
        closed = SimpleNamespace(check=Mock(return_value=scene_result))
        opened = SimpleNamespace(check=Mock(side_effect=AssertionError('No approach scene checks expected')))
        problem = SimpleNamespace(pose=Mock(return_value=np.eye(4)), transform=Mock(return_value=np.eye(4)))
        projection = {'accepted': projection_accepted, 'weighted_residual_norm': 0.,
            'reason': 'accepted' if projection_accepted else 'local_projection_failed',
            'state': np.zeros(8)}
        row = {'pose': pose, 'source': 'diagnostic fixture', 'anchor_index': 0}
        with TemporaryDirectory() as temporary, \
                patch.object(coupled_pipeline, 'grasp_poses', return_value=([row], np.zeros(3))), \
                patch('vlm_tamp.coupled_manifold.CoupledHandlePoint', return_value=problem), \
                patch.object(coupled_pipeline, 'project_ik_candidate', return_value=projection) as project, \
                patch.object(coupled_pipeline, 'partial_goal_candidates', return_value=([],[])), \
                patch.object(coupled_pipeline, 'isolated_plan') as planner:
            with self.assertRaisesRegex(ValueError, 'No tested grasp'):
                coupled_pipeline.complete_motion_selection({}, [], {'joint_positions': [0.]*7}, ik,
                    opened, closed, {'panel_normal_world': [1., 0., 0.],
                    'joint_origin_world_m': [0., 0., 0.], 'joint_axis_world': [0., 0., 1.]},
                    1., temporary, save, contact_model='jaw_contact')
            diagnostic = json.loads((Path(temporary)/'start-validation-000.json').read_text())
            search = json.loads((Path(temporary)/'grasp-search.json').read_text())
        self.assertEqual(ik.solve_many.call_count, 21)
        self.assertEqual(project.call_count, 1)
        self.assertEqual(ik.approach.call_count, 0)
        self.assertEqual(opened.check.call_count, 0)
        planner.assert_not_called()
        self.assertFalse(search['search_was_dispatched'])
        self.assertEqual(search['failure_classification'], 'start_admission_exclusion')
        self.assertEqual(len(diagnostic['goal_queries']), 20)
        self.assertEqual(diagnostic['goal_fallback'], 'milestones_searches_free_terminal_joint_posture')
        return diagnostic['starts'][0], search, ik, closed

    def test_scene_rejected_start_records_blocking_geometry_without_extra_queries(self):
        record, search, ik, closed = self.run_rejected_start(True)
        self.assertEqual(record['terminal_gate'], 'start_scene_collision')
        self.assertEqual(record['closed_scene']['collisions'][0]['robot_geom_name'],
                         'gripper0_finger1_collision')
        self.assertEqual(record['closed_scene']['collisions'][0]['blocked_inferred_panel_points'], 8)
        self.assertEqual(search['gate_counts']['start_scene_collision'], 1)
        self.assertEqual(ik.self_valid.call_count, 1)
        self.assertEqual(closed.check.call_count, 1)

    def test_failed_projection_keeps_later_checks_skipped_and_records_reason(self):
        record, search, ik, closed = self.run_rejected_start(False)
        self.assertEqual(record['terminal_gate'], 'start_projection_failed')
        self.assertEqual(record['projection']['reason'], 'local_projection_failed')
        self.assertIsNone(record['self_valid'])
        self.assertIsNone(record['closed_scene'])
        self.assertEqual(search['gate_counts']['start_projection_failed'], 1)
        ik.self_valid.assert_not_called()
        closed.check.assert_not_called()

    def test_successful_dispatch_reuses_checks_and_records_planner_result(self):
        pose = np.array([0., 0., 1., 0., 0., 0., 1.])
        start = {'joints': [0.]*7, 'pose_world_xyzw': pose.tolist(), 'joint_reserve_rad': .2}
        ik = SimpleNamespace(chain=SimpleNamespace(forward=Mock(return_value=pose)),
            solve_many=Mock(side_effect=[[start]]+[[]]*20), self_valid=Mock(return_value=True),
            approach=Mock(return_value={'accepted': True, 'joint_postures': [[0.]*7, [.01]*7]}))
        opened = SimpleNamespace(check=Mock(return_value={'accepted': True, 'collisions': []}))
        closed = SimpleNamespace(check=Mock(return_value={'accepted': True, 'collisions': []}))
        problem = SimpleNamespace(pose=Mock(return_value=np.eye(4)), transform=Mock(return_value=np.eye(4)))
        projection = {'accepted': True, 'weighted_residual_norm': 0., 'state': np.zeros(8)}
        path = np.array([[0.]*8, [.01]*7+[1.]])
        with TemporaryDirectory() as temporary, \
                patch.object(coupled_pipeline, 'grasp_poses', return_value=([
                    {'pose': pose, 'source': 'fixture', 'anchor_index': 0}], np.zeros(3))), \
                patch('vlm_tamp.coupled_manifold.CoupledHandlePoint', return_value=problem), \
                patch.object(coupled_pipeline, 'project_ik_candidate', return_value=projection), \
                patch.object(coupled_pipeline, 'partial_goal_candidates', return_value=([],[])), \
                patch.object(coupled_pipeline, 'isolated_plan', return_value=(path,
                    {'exact_solution': True})) as planner:
            chosen, _, returned = coupled_pipeline.complete_motion_selection({}, [],
                {'joint_positions': [0.]*7}, ik, opened, closed,
                {'panel_normal_world': [1., 0., 0.], 'joint_origin_world_m': [0., 0., 0.],
                 'joint_axis_world': [0., 0., 1.]}, 1., temporary, save, contact_model='jaw_contact')
            diagnostic = json.loads((Path(temporary)/'start-validation-000.json').read_text())
            search = json.loads((Path(temporary)/'grasp-search.json').read_text())
        self.assertEqual(ik.solve_many.call_count, 21)
        self.assertEqual(ik.self_valid.call_count, 1)
        self.assertEqual(closed.check.call_count, 1)
        self.assertEqual(ik.approach.call_count, 1)
        self.assertEqual(opened.check.call_count, 2)
        self.assertEqual(planner.call_count, 1)
        self.assertEqual(chosen['pose_index'], 0)
        np.testing.assert_array_equal(returned, path)
        self.assertEqual(diagnostic['starts'][0]['terminal_gate'], 'planner_exact_solution')
        self.assertEqual(search['gate_counts']['planner_calls'], 1)
        self.assertTrue(search['search_was_dispatched'])


if __name__ == '__main__':
    unittest.main()
