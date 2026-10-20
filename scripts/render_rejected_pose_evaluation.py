"""Solid pose renders and simulator geometry diagnostics, evaluation-only.

This owns a fresh private simulator. Assigning joints is for rendering isolated
hypothetical configurations, never for producing executed grasp/task results.
No contact forces, grasp retention or success rates are inferred from renders.
"""

import contextlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src')]
from vlm_tamp.rekep_libero.sim_worker_v8 import SimulatorWorkerV8


def main():
    output = ROOT/'results/rejected-grasp-visualization-v11'
    poses = json.loads((output/'pose-render-input.json').read_text())
    worker = SimulatorWorkerV8()
    records = []
    try:
        for item in poses:
            state = worker.reset(0, item['state'], output/f'evaluation-render-{item["case"]}', seed=0)
            worker.action(state['metadata']['eef_pose_world_xyzw'], gripper=-1, max_steps=60)
            sim = worker.env.sim
            sim.data.qpos[worker.robot._ref_joint_pos_indexes] = item['joints']
            sim.data.qpos[worker.robot._ref_gripper_joint_pos_indexes] = item['gripper_qpos']
            sim.data.qvel[:] = 0
            drawer_joint = None
            if item['drawer_displacement_m']:
                matches = [name for name in sim.model.joint_names if 'cabinet' in name and 'middle' in name]
                if len(matches)!=1:
                    raise ValueError('Evaluation renderer cannot identify the middle drawer joint uniquely')
                drawer_joint = matches[0]
                index = sim.model.get_joint_qpos_addr(drawer_joint)
                joint_id = sim.model.joint_name2id(drawer_joint)
                alignment = float(np.asarray(sim.data.xaxis[joint_id]) @ np.asarray(item['pull_axis_world']))
                if abs(alignment) < .95:
                    raise ValueError('Evaluation drawer axis differs from the observed pull direction')
                sim.data.qpos[index] += item['drawer_displacement_m'] / alignment
            sim.forward()
            model = sim.model._model
            prefixes = (worker.robot.robot_model.naming_prefix, worker.robot.gripper.naming_prefix)
            flags = [{'name':name,'contype':int(model.geom_contype[i]),'conaffinity':int(model.geom_conaffinity[i]),
                      'render_group':int(model.geom_group[i])}
                     for i,name in enumerate(sim.model.geom_names) if name and name.startswith(prefixes) and model.geom_group[i]==0]
            contacts = []
            for i in range(int(sim.data.ncon)):
                contact = sim.data.contact[i]
                names = [sim.model.geom_id2name(int(index)) or '' for index in [contact.geom1,contact.geom2]]
                robot = [name.startswith(prefixes) for name in names]
                if sum(robot)!=1:
                    continue
                contacts.append({'geom_names_evaluation_only':names,'distance_m':float(contact.dist),
                                 'point_world_m':np.asarray(contact.pos).tolist()})
            rgb, _, _ = worker._frontview()
            Image.fromarray(rgb).save(output/f'{item["case"]}-solid.png')
            record = {'case':item['case'],'render_only':True,'hypothetical_configuration_executed':False,
                      'robot_collision_masks':flags,'instantaneous_simulator_geometry_contacts':contacts,
                      'scene_identities_and_shapes_used_for_evaluation_render_only':True,
                      'inputs_to_policy_or_grasp_selection':False,'contact_forces_or_success_not_measured':True,
                      'evaluation_drawer_joint':drawer_joint,'observed_drawer_displacement_m':item['drawer_displacement_m']}
            records.append(record)
    finally:
        worker.close()
    (output/'simulator-pose-render-audit.json').write_text(json.dumps({'scope':'evaluation-only static hypothetical pose rendering',
         'no_rejected_trajectory_execution':True,'cases':records},indent=2)+'\n')
    print(json.dumps([{'case':r['case'],'instantaneous_robot_scene_contacts':len(r['instantaneous_simulator_geometry_contacts']),
                      'deepest_simulator_contact_m':min([c['distance_m'] for c in r['instantaneous_simulator_geometry_contacts']],default=None)}
                     for r in records]),flush=True)


if __name__=='__main__':
    main()
