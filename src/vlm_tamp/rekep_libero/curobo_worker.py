"""GPU IK service in the isolated cuRobo environment; no scene or API access."""

import contextlib
import copy
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np
import torch


class Worker:
    def initialize(self, robot_config, config):
        import warp as wp
        wp.config.quiet = True
        wp.config.kernel_cache_dir = os.path.join(os.environ['XDG_CACHE_HOME'], 'warp-curobo')
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError('Warp cannot initialize the CUDA device')
        from curobo.types.base import TensorDeviceType
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        self.tensor_args = TensorDeviceType()
        self.count = int(config.get('curobo_num_seeds', 64))
        self.return_count = int(config.get('curobo_return_seeds', 32))
        self.reproducible = bool(config.get('curobo_query_reproducible', False))
        self.robot_config = copy.deepcopy(robot_config)
        self.seed = int(config.get('seed', 0))
        self.solver = self.create_solver()
        self.fixed_seeds = (self.solver.generate_seed(self.count-1, batch=1).clone()
                            if self.reproducible and self.count > 1 else None)
        if self.reproducible:
            q = robot_config['kinematics']['cspace']['retract_config']
            self.solve(self.fk([q])['poses_world_xyzw'][0], q)
        return {'torch': torch.__version__, 'cuda': torch.version.cuda,
                'warp': wp.__version__, 'warp_cuda_available': wp.is_cuda_available(),
                'gpu': torch.cuda.get_device_name(), 'joint_names': self.solver.joint_names,
                'num_seeds': self.count, 'return_seeds': self.return_count,
                'query_reproducible': self.reproducible,
                'cuda_graph': not self.reproducible,
                'particle_optimizer': True,
                'optimizer_lifetime': 'one query' if self.reproducible else 'worker'}

    def create_solver(self):
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        torch.manual_seed(self.seed)
        return IKSolver(IKSolverConfig.load_from_robot_config(
            copy.deepcopy(self.robot_config), None, tensor_args=self.tensor_args, num_seeds=self.count,
            self_collision_check=False, self_collision_opt=False,
            position_threshold=.0005, rotation_threshold=.005,
            use_cuda_graph=not self.reproducible, regularization=False, seed=self.seed))

    def fk(self, joints):
        state = self.solver.kinematics.get_state(self.tensor_args.to_device(joints))
        position = state.ee_position.detach().cpu().numpy()
        quaternion = state.ee_quaternion.detach().cpu().numpy()  # wxyz
        return {'poses_world_xyzw': np.c_[position, quaternion[:, 1:], quaternion[:, :1]].tolist()}

    def solve(self, pose, seed):
        from curobo.types.math import Pose
        start = time.monotonic()
        if self.reproducible:
            # Fresh optimizer state also removes warmup and quasi-Newton
            # history effects. This correctness mode trades setup time for
            # an independently reproducible query; the legacy mode is intact.
            self.solver = self.create_solver()
        pose = np.asarray(pose, float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError('Expected one finite xyz/xyzw TCP pose')
        goal = Pose(self.tensor_args.to_device(pose[:3][None]),
                    self.tensor_args.to_device(pose[[6, 3, 4, 5]][None]))
        initial = self.tensor_args.to_device(np.asarray(seed, float).reshape(1, 1, 7))
        retract = initial.reshape(1, 7)
        audit = {}
        if self.reproducible:
            if self.count > 1:
                initial = torch.cat((initial, self.fixed_seeds.clone()), dim=1)
            seed_bytes = initial.detach().cpu().numpy().tobytes()
            audit = {'seed_pool_sha256': hashlib.sha256(seed_bytes).hexdigest(),
                     'query_sha256': hashlib.sha256(pose.tobytes()+seed_bytes).hexdigest(),
                     'explicit_seed_count': self.count}
        result = self.solver.solve_single(goal, seed_config=initial,
            retract_config=retract, return_seeds=self.return_count)
        torch.cuda.synchronize()
        return {'solutions': result.solution.detach().cpu().reshape(-1, 7).tolist(),
                'success': result.success.detach().cpu().reshape(-1).tolist(),
                'elapsed_seconds': time.monotonic()-start,
                'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(), **audit}


def main():
    protocol, worker = sys.stdout, Worker()
    with contextlib.redirect_stdout(sys.stderr):
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                op = request['op']
                if op == 'initialize':
                    result = worker.initialize(request['robot_config'], request['config'])
                elif op == 'fk':
                    result = worker.fk(request['joints'])
                elif op == 'solve':
                    result = worker.solve(request['pose'], request['seed'])
                elif op == 'close':
                    result = {}
                else:
                    raise ValueError(f'Unknown cuRobo operation: {op}')
                response = {'ok': True, 'id': request.get('id'), **result}
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                response = {'ok': False, 'id': request.get('id'), 'error': str(error)}
            protocol.write(json.dumps(response, allow_nan=False)+'\n')
            protocol.flush()
            if request.get('op') == 'close':
                break


if __name__ == '__main__':
    main()
