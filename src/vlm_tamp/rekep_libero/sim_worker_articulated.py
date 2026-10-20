"""Official suite resets with the frozen synchronized sensor/action interface."""

import contextlib
import json
from pathlib import Path
import random
import sys
import traceback
from types import SimpleNamespace

import numpy as np

from vlm_tamp.libero import configure_libero
from .articulated_kinematics import ArticulatedPandaIK, RobotSimNames
from .sim_worker import IMAGE_SIZE
from .sim_worker_v8 import SimulatorWorkerV8
from .sim_worker_v7 import SimulatorWorkerV7


class ArticulatedWorker(SimulatorWorkerV8):
    def _evaluation_contacts(self):
        result=super()._evaluation_contacts()
        model,data=self.env.sim.model,self.env.sim.data
        prefix=self.robot.robot_model.naming_prefix
        joints=[]
        for name in model.joint_names:
            index=int(model.joint_name2id(name))
            kind=int(model.jnt_type[index])
            if not name or name.startswith(prefix) or kind not in {2,3}:
                continue
            joints.append({'joint_name':name,'type':'slide' if kind==2 else 'hinge',
                'axis_world':data.xaxis[index].tolist(),'anchor_world_m':data.xanchor[index].tolist(),
                'position':float(data.qpos[int(model.jnt_qposadr[index])]),
                'limits':model.jnt_range[index].tolist()})
        result['evaluation_only_articulation_state']=joints
        return result

    def reset(self, task_name, state_index, output_dir, seed=0, suite_name='libero_90', scene_variant='nominal'):
        from vlm_tamp.subscription_scene_variants import VARIANTS
        if scene_variant not in VARIANTS:raise ValueError('Unknown predeclared scene variant')
        if scene_variant!='nominal' and task_name!='KITCHEN_SCENE7_open_the_microwave':
            raise ValueError('These scene variants are defined only for the microwave opening task')
        if suite_name not in {'libero_goal', 'libero_90', 'libero_10'}:
            raise ValueError('Unsupported official task suite')
        if (Path(output_dir)/'obs.npz').exists():
            raise FileExistsError('Preserve existing observations')
        if self.env is not None:
            self._finish_episode()
            self.env.close()
            self.env = None
        configure_libero(gl='egl')
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        matches = [i for i in range(suite.n_tasks)
                   if Path(suite.get_task_bddl_file_path(i)).stem == task_name]
        if len(matches) != 1:
            raise ValueError('Task name must identify exactly one official task')
        index = matches[0]
        states = suite.get_task_init_states(index)
        if not 0 <= int(state_index) < len(states):
            raise ValueError('Invalid official initial state')
        random.seed(seed)
        np.random.seed(seed)
        self.task_index, self.state_index = index, int(state_index)
        self.task_name, self.suite_name = task_name, suite_name
        self.instruction = suite.get_task(index).language
        self.env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(index),
            camera_names=['agentview', 'robot0_eye_in_hand'], camera_heights=IMAGE_SIZE,
            camera_widths=IMAGE_SIZE, camera_depths=True, camera_segmentations=None,
            control_freq=20, horizon=1000, ignore_done=True)
        self.env.seed(seed)
        self.env.reset()
        self.latest_obs = self.env.set_init_state(np.asarray(states[state_index]))
        from vlm_tamp.subscription_scene_variants import apply_scene_variant
        from .sim_worker import FRONTVIEW_POLICY
        variant_record,self.frontview_policy=apply_scene_variant(self.env,scene_variant,FRONTVIEW_POLICY)
        self.control_steps, self.settling_steps = 0, 0
        self.episode_dir = Path(output_dir).resolve()
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        (self.episode_dir/'scene-variant-private.json').write_text(json.dumps(variant_record,indent=2)+'\n')
        self.action_log, self.frame_paths, self.video_path, self.evaluation_contact_log = [], [], None, []
        self.robot = self.env.robots[0]
        self.ik_solver = ArticulatedPandaIK(self.env.sim, self.robot)
        for _ in range(5):
            self.latest_obs, _, _, _ = self.env.step(np.zeros(7))
            self.settling_steps += 1
        self.calibration = self._robot_calibration()
        self._save_frame(force=True)
        return self.observe(self.episode_dir)

    def _robot_depth_priors(self, cameras):
        # Reuse the exact render/restore implementation with a corrected
        # name-to-ID view. No live model, physics, or controller is replaced.
        worker=self
        class SensorView:
            env=SimpleNamespace(sim=RobotSimNames(worker.env.sim,worker.robot))
            def __getattr__(self,name):
                return getattr(worker,name)
        return SimulatorWorkerV7._robot_depth_priors(SensorView(),cameras)


def main():
    protocol, worker = sys.stdout, ArticulatedWorker()
    with contextlib.redirect_stdout(sys.stderr):
        try:
            for line in sys.stdin:
                request = {}
                try:
                    request = json.loads(line)
                    response = {'ok': True, 'id': request.get('id'), **worker.dispatch(request)}
                except Exception as error:
                    traceback.print_exc(file=sys.stderr)
                    response = {'ok': False, 'id': request.get('id'), 'error': str(error)}
                protocol.write(json.dumps(response, allow_nan=False)+'\n')
                protocol.flush()
                if request.get('op') == 'close':
                    break
        finally:
            worker.close()


if __name__ == '__main__':
    main()
