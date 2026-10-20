"""Short-lived OMPL worker; input is a trusted, locally generated experiment bundle."""

import argparse
import json
from pathlib import Path
import pickle
import sys
import traceback
import copy
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.runtime/mc-vamp-deps'),str(ROOT/'.runtime/ompl-deps')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    folder=Path(args.output);folder.mkdir(parents=True,exist_ok=False)
    try:
        # These bundles are produced by our parent runner, never by a model.
        with Path(args.input).open('rb') as stream:
            bundle=pickle.load(stream)
        from vlm_tamp.coupled_manifold import ompl_plan
        import vamp
        environment=vamp.Environment()
        scene=bundle['scene']
        def valid(x):
            return bool(vamp.panda.validate(np.asarray(x[:7]).tolist(),environment,True)
                        and scene.check(x[:7],float(x[7]))['accepted'])
        if bundle.get('strategy')=='warmstart':
            from vlm_tamp.coupled_warmstart import retarget_path
            proposal_limit=float(bundle.get('warm_maximum_joint_correction_rad',.04))
            path,result=retarget_path(bundle['problem'],bundle['start'],bundle['warm_path'],valid,
                maximum_joint_correction_rad=proposal_limit)
            result['maximum_hypothetical_reference_correction_rad']=proposal_limit
            result['measured_start_binding_checked_separately']=True
            result['goal_angle_rad']=float(bundle['goal'][7])
            result['full_trajectory_built_before_execution']=True
            if len(path) and abs(float(path[-1,7]-bundle['goal'][7]))>1e-8:
                result.update(exact_solution=False,reason='Warm start misses requested terminal angle')
                path=np.empty((0,8))
        elif bundle.get('strategy')=='milestones':
            problem=bundle['problem'];current=np.asarray(bundle['start']).copy()
            goal=float(bundle['goal'][7]);segments=[];pieces=[];begin=time.monotonic()
            initial_seed=int(bundle.get('seed',1));rng=np.random.default_rng(initial_seed)
            from ompl import util as ou
            ou.RNG.setSeed(initial_seed)
            exact=True
            milestone_diagnostics=[]
            target_goal=np.asarray(bundle['goal'],float)
            real_goal=(problem.satisfied(target_goal)
                and problem.errors(target_goal)['weighted_residual_norm']<=1e-5 and valid(target_goal))
            for theta in np.linspace(current[7],goal,max(2,int(np.ceil(abs(goal-current[7])/.12))+1))[1:]:
                candidates=[]
                diagnostics=[]
                if real_goal and abs(theta-goal)<1e-10:candidates.append(target_goal.copy())
                for trial in range(12):
                    initial=current.copy();initial[7]=theta
                    if real_goal and trial==0:
                        weight=float((theta-current[7])/(goal-current[7]))
                        initial[:7]+=weight*(target_goal[:7]-current[:7])
                    elif trial>1:initial[:7]+=rng.normal(0.,.18,7)
                    projected=problem.project(initial,fixed_progress=float(theta))
                    state=projected['state']
                    self_clear=bool(vamp.panda.validate(state[:7].tolist(),environment,True))
                    scene_result=scene.check(state[:7],state[7]) if projected['accepted'] and self_clear else None
                    if trial<4:
                        diagnostics.append({'projection_accepted':projected['accepted'],
                            'weighted_residual':projected['weighted_residual_norm'],
                            'self_collision_free':self_clear,'scene_check':scene_result})
                    if (projected['accepted'] and projected['weighted_residual_norm']<=1e-5
                            and self_clear and scene_result['accepted']):
                        candidates.append(state)
                milestone_diagnostics.append({'theta':float(theta),'valid_projected_candidates':len(candidates),
                    'examples':diagnostics})
                candidates.sort(key=lambda x:np.linalg.norm(x[:7]-current[:7]))
                if real_goal and abs(theta-goal)<1e-10:
                    candidates.sort(key=lambda x:np.linalg.norm(x-target_goal)>1e-9)
                accepted=False
                for target in candidates[:4]:
                    remaining=bundle.get('seconds',60.)-(time.monotonic()-begin)
                    if remaining<=0:break
                    local=copy.deepcopy(problem)
                    local.bounds[7]=[max(problem.bounds[7,0],current[7]-.04),min(problem.bounds[7,1],theta+.04)]
                    samples=max(8,int(np.ceil(np.max(np.abs(target[:7]-current[:7]))/.015)))
                    segment,record=ompl_plan(local,current,target,valid,seconds=min(3.,remaining),
                        interpolation_count=samples,delta=.02,seed=None)
                    segments.append(record)
                    if record['exact_solution'] and len(segment)>1:
                        pieces.append(segment if not pieces else segment[1:]);current=segment[-1];accepted=True;break
                if not accepted:
                    exact=False;break
            path=np.vstack(pieces) if pieces else np.empty((0,8))
            result={'planner':'OMPL Atlas/RRTConnect through offline contact-progress milestones',
                'exact_solution':exact and len(path)>1,'segments':segments,'solve_seconds':time.monotonic()-begin,
                'goal_angle_rad':goal,'final_planned_angle_rad':float(current[7]),'states':len(path),
                'full_trajectory_built_before_execution':True,'initial_rng_seed':initial_seed,
                'real_valid_goal_guidance_used':bool(real_goal),
                'constraint_codimension':problem.codimension,'ambient_dimension':8,
                'milestone_diagnostics':milestone_diagnostics,
                'maximum_position_error_m':max((problem.errors(x)['position_error_m'] for x in path),default=0.)}
            if not result['exact_solution']:path=np.empty((0,8))
        else:
            path,result=ompl_plan(bundle['problem'],bundle['start'],bundle['goal'],valid,
                seconds=bundle.get('seconds',12.),interpolation_count=120,delta=.035)
        np.save(folder/'path.npy',path)
        result['scene_scope']=scene.record()
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    except Exception as error:
        (folder/'result.json').write_text(json.dumps({'exact_solution':False,'error':str(error),
            'traceback':traceback.format_exc()},indent=2)+'\n')
        raise


if __name__=='__main__':main()
