"""Control-protocol checks without loading a simulator or moving a robot."""

import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from vlm_tamp.coupled_worker import CoupledWorker, checked_joint_target, normalized_joint_command


class FakeRobot:
    def __init__(self, data):
        self.data = data
        self.controller = SimpleNamespace(control_dim=6)
        self.controller_config = {'type': 'OSC_POSE'}
        self.recent_actions = SimpleNamespace(dim=7)
        self.base_pos = np.zeros(3)
        self.base_ori = np.array([0., 0., 0., 1.])
        self._ref_joint_pos_indexes = np.arange(7)
        self._ref_joint_vel_indexes = np.arange(7)
        self._ref_gripper_joint_pos_indexes = np.arange(7, 9)

    @property
    def action_dim(self):
        return self.controller.control_dim + 1

    def _load_controller(self):
        robot = self
        class JointController:
            control_dim = 7
            kp = np.full(7, 50.)
            kd = np.full(7, 2 * np.sqrt(50.))
            goal_qpos = None
            def update_base_pose(self, *args):
                pass
            def reset_goal(self):
                self.goal_qpos = robot.data.qpos[:7].copy()
            def set_goal(self,action,set_qpos=None):
                self.goal_qpos=robot.data.qpos[:7]+np.asarray(action)*.05 if set_qpos is None else np.asarray(set_qpos).copy()
        self.controller = JointController()


def mock_worker():
    worker = CoupledWorker()
    data = SimpleNamespace(qpos=np.zeros(9), qvel=np.zeros(9), act=np.zeros(0),
                           ctrl=np.zeros(9), mocap_pos=np.zeros((0, 3)), mocap_quat=np.zeros((0, 4)))
    robot = FakeRobot(data)
    class CoreEnv:
        _action_dim = 7
        @property
        def action_dim(self):
            return self._action_dim
    class Env:
        env = CoreEnv()
        robots = [robot]
        sim = SimpleNamespace(data=data)
        controls = []
        def check_success(self):
            return False
        def step(self, control):
            self.controls.append(control.copy())
            robot.controller.set_goal(control[:7])
            data.qpos[:7] += getattr(self,'tracking_fraction',1.)*(robot.controller.goal_qpos-data.qpos[:7])
            return {}, 0, False, {}
    worker.env, worker.robot = Env(), robot
    worker.ik_solver = SimpleNamespace(limits=np.tile([-2., 2.], (7, 1)),
        fk=lambda q: np.array([0., 0., 1., 0., 0., 0., 1.]))
    worker._eef_pose = lambda: np.array([0., 0., 1., 0., 0., 0., 1.])
    worker._response_state = lambda: {'control_steps': worker.control_steps}
    worker._save_frame = lambda **kwargs: None
    worker._write_action_log = lambda: None
    worker._evaluation_contacts = lambda: {'evaluation_only': True}
    worker.evaluation_contact_log = []
    return worker


class CoupledWorkerTests(unittest.TestCase):
    def mocked_modules(self):
        controllers = ModuleType('robosuite.controllers')
        controllers.load_controller_config = lambda **kwargs: {'type': 'JOINT_POSITION'}
        buffers = ModuleType('robosuite.utils.buffers')
        buffers.DeltaBuffer = lambda dim: SimpleNamespace(dim=dim)
        return {'robosuite': ModuleType('robosuite'), 'robosuite.controllers': controllers,
                'robosuite.utils': ModuleType('robosuite.utils'), 'robosuite.utils.buffers': buffers}

    def enable(self, worker,mode='measured_delta'):
        with patch.dict(sys.modules, self.mocked_modules()):
            return worker.enable_joint_tracking(reference_mode=mode)

    def test_switch_resizes_bookkeeping_without_steps_or_state_assignment(self):
        worker = mock_worker()
        before = worker.env.sim.data.qpos.copy()
        result = self.enable(worker)
        np.testing.assert_array_equal(worker.env.sim.data.qpos, before)
        self.assertEqual(worker.control_steps, 0)
        self.assertEqual(worker.env.env.action_dim, 8)
        self.assertEqual(worker.robot.recent_actions.dim, 8)
        self.assertTrue(worker.joint_tracking_enabled)
        self.assertFalse(result['controller_change']['state_assignment'])

    def test_osc_commands_rejected_after_switch(self):
        worker = mock_worker()
        self.enable(worker)
        with self.assertRaisesRegex(ValueError, 'OSC action is disabled'):
            worker.action([0, 0, 1, 0, 0, 0, 1])

    def test_clipped_joint_command_is_logged_and_physically_stepped(self):
        worker = mock_worker()
        self.enable(worker)
        result = worker.joint_action(np.full(7, .08), gripper=1)
        self.assertEqual(result['executed_steps'], 1)
        self.assertEqual(len(worker.env.controls), 1)
        np.testing.assert_allclose(worker.env.controls[0], np.ones(8))
        np.testing.assert_allclose(worker.action_log[0]['controller_goal_joints'], .05)
        self.assertTrue(worker.action_log[0]['reference_clipped'])
        self.assertEqual(len(worker.evaluation_contact_log), 1)

    def test_invalid_targets_and_gripper_fail_before_any_motion(self):
        worker = mock_worker()
        self.enable(worker)
        for target in (np.full(7, 3.), np.full(7, np.nan), np.zeros(6)):
            with self.assertRaises(ValueError):
                worker.joint_action(target)
        with self.assertRaises(ValueError):
            worker.joint_action(np.zeros(7), gripper=.5)
        with self.assertRaises(ValueError):
            worker.joint_action(np.zeros(7), max_steps=1.5)
        self.assertEqual(len(worker.env.controls), 0)

    def test_episode_budget_remains_shared(self):
        worker = mock_worker()
        self.enable(worker)
        worker.control_steps = 599
        result = worker.joint_action(np.zeros(7), max_steps=5)
        self.assertEqual(result['executed_steps'], 1)
        self.assertEqual(worker.control_steps, 600)
        self.assertEqual(worker.joint_action(np.zeros(7))['executed_steps'], 0)

    def test_mode_and_target_helpers(self):
        worker = mock_worker()
        with self.assertRaisesRegex(ValueError, 'enable_joint_tracking'):
            worker.joint_action(np.zeros(7))
        np.testing.assert_array_equal(checked_joint_target(np.zeros(7), np.tile([-1., 1.], (7, 1))), np.zeros(7))
        control, goal = normalized_joint_command(np.full(7, .01), np.zeros(7), 0)
        np.testing.assert_allclose(control[:7], .2)
        np.testing.assert_allclose(goal, .01)

    def test_reference_slew_is_independent_of_measured_tracking_error(self):
        worker=mock_worker();self.enable(worker,'slew_limited');worker.env.tracking_fraction=.1
        worker.joint_action(np.ones(7),gripper=0,max_steps=5)
        goals=np.asarray([e['controller_goal_joints'] for e in worker.action_log])
        np.testing.assert_allclose(goals[:,0],[.05,.10,.15,.20,.25])
        self.assertLess(worker.env.sim.data.qpos[0],.1)
        self.assertTrue(np.allclose(np.diff(goals,axis=0),.05))
        self.assertTrue(np.all(np.asarray(worker.env.controls)[:,:7]==0))

    def test_reference_mode_switch_does_not_assign_physical_state(self):
        worker=mock_worker();self.enable(worker)
        before=worker.env.sim.data.qpos.copy();record=worker.set_joint_reference_mode('slew_limited')
        np.testing.assert_array_equal(worker.env.sim.data.qpos,before)
        self.assertFalse(record['reference_mode_change']['state_assignment'])


if __name__ == '__main__':
    unittest.main()
