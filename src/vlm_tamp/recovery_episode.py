"""Generic recovery tools on the same continuous physical episode and checker."""
import numpy as np
from .strategy_episode import StrategyTools
from .strategy_feedback import planner_feedback
from .recovery_feedback import failure_loci
from .reliable_mechanisms import segment
from .mcvamp_mechanism import NativeScene
from .rekep_libero.geometry import pose_matrix
from .reliability_geometry import SurfaceScene
from .coupled_storage import read_json,write_json

class RecoveryTools(StrategyTools):
    step_m = .025
    posture_rank = None

    def motion(self,extent,folder):
        if self.current_choice is None:return {'status':'no_active_contact','manipulation_executed':False}
        start=self.progress.observed_progress;completed=0;last={};last_plan={};amounts=[]
        for _ in range(1 if extent=='probe' else 4):
            remaining=float(self.spec['target_progress'])-self.progress.progress
            tolerance=float(self.config['scene_voxel_spacing_m'])
            if remaining<=tolerance:self.goal_reached=True;break
            if 600-int(self.state['control_steps'])<=12:break
            seating=self.spec['motion']=='close' and remaining<=.020
            amount=min(self.step_m,max(remaining,tolerance))
            q=np.asarray(self.state['metadata']['joint_positions']);self.stage='motion_planning'
            output=self.folder/f'plan-{self.plan_number:02d}';self.plan_number+=1
            path,info=segment(self.scene,self.record,self.state['metadata'],self.ik,q,amount,output,
                              seconds=8.,posture_rank=self.posture_rank)
            last_plan={**planner_feedback(output),'failure_loci':failure_loci(output)}
            if path is None:
                return {'status':'planning_failed','phase':'manipulation','reason':info['reason'],
                    'manipulation_executed':bool(completed),'completed_segments':completed,
                    'requested_step_mm':self.step_m*1000,'attempted_step_m':amount,
                    'executed_plan_distances_m':amounts,'requested_posture_rank':self.posture_rank,
                    'object_progress_delta_m':self.progress.observed_progress-start if last.get('tracking_accepted') else None,
                    **last,**last_plan}
            before=np.asarray(self.state['metadata']['eef_pose_world_xyzw'][:3]);self.stage='execution'
            self.issue(path,f'manipulate-{self.plan_number:02d}',1);completed+=1;amounts.append(amount)
            last=self.observe(before_hand=before)
            # Preserve the previous bounded seating rule only when this segment
            # actually covers its requested remaining distance.
            if seating and amount>=remaining:self.goal_reached=True;break
        if float(self.spec['target_progress'])-self.progress.progress<=float(self.config['scene_voxel_spacing_m']):
            self.goal_reached=True
        return {'status':'motion_completed','completed_segments':completed,'manipulation_executed':bool(completed),
            'object_progress_delta_m':self.progress.observed_progress-start if last.get('tracking_accepted') else None,
            'estimated_goal_reached':self.goal_reached,'requested_step_mm':self.step_m*1000,
            'executed_plan_distances_m':amounts,'requested_posture_rank':self.posture_rank,**last,**last_plan}

    def try_contact(self,decision,folder):
        result=super().try_contact(decision,folder)
        # A requested rank selects the contact posture, not every continuation
        # or approach IK query. Motion uses the ordinary search after contact.
        result.setdefault('failure_loci',failure_loci(folder))
        if (folder/'chosen-grasp.json').exists():
            selected=read_json(folder/'chosen-grasp.json')
            self.current_choice.update(actual_ik_rank=selected.get('ik_rank'),
                actual_tilt_deg=selected.get('approach_tilt_deg'))
            result['selected_strategy']=dict(self.current_choice)
        result.update(requested_step_mm=decision['step_mm'],requested_posture_rank=decision['posture_rank'])
        return result

    def release_reposition(self,decision,folder):
        before=np.asarray(self.state['metadata']['eef_pose_world_xyzw'][:3])
        q=np.asarray(self.state['metadata']['joint_positions'])
        context=self.sim.context;self.sim.context=None;self.stage='release'
        try:self.issue([q,q],'release-reposition-open',-1,hold=8)
        finally:self.sim.context=context
        observation=self.observe(before_hand=before)
        self.current_choice=None;self.sim.context=None;self.sim.visual_offset=np.zeros(3)
        self.generated=self.resources.grasps(self.cloud,folder/'grasps')
        q=np.asarray(self.state['metadata']['joint_positions'])
        target=np.asarray(self.state['metadata']['eef_pose_world_xyzw'],float).copy()
        target[:3]-=(decision['retreat_mm']/1000)*pose_matrix(target)[:3,2]
        self.stage='retreat_planning';attempts=[];path=None
        with NativeScene(self.scene,self.record,self.state['metadata'],q,.03,articulated=False) as free:
            initial=free.validate(q)
            if not initial['accepted']:
                info={'reason':'observed_robot_collision_at_start','initial':initial,'attempts':[]}
            else:
                solutions=self.ik.solve_many(target,q,maximum=8)
                rank=decision['posture_rank']
                if rank is not None:solutions=solutions[rank:rank+1]
                for solution in solutions:
                    candidate,planned=free.plan(q,np.asarray(solution['joints']),constrained=False,seconds=5.,iterations=4000)
                    attempts.append(planned)
                    if planned['exact_solution']:path=candidate;break
                info={'reason':None if path is not None else ('no_terminal_ik' if not solutions and rank is None else
                    'requested_posture_unavailable' if not solutions else 'no_complete_free_retreat'),
                    'target_pose_world_xyzw':target.tolist(),'attempts':attempts}
        write_json(folder/'retreat-planning.json',info)
        if path is not None:
            before=np.asarray(self.state['metadata']['eef_pose_world_xyzw'][:3]);self.stage='retreat_execution'
            try:self.issue(path,'release-reposition-retreat',-1)
            except ValueError as error:
                if 'budget' not in str(error).lower():raise
                return {'status':'tool_budget_exhausted','phase':'retreat',
                    'reason':'Released, but the complete retreat does not fit in the remaining control budget',
                    'release_executed':True,'retreat_executed':False,'manipulation_executed':False,
                    'requested_retreat_mm':decision['retreat_mm'],'failure_loci':{'control_budget':1},**observation}
            observation=self.observe(before_hand=before)
            self.scene=SurfaceScene(self.views,self.state['metadata'],self.spec,self.cloud,self.panel,
                                    self.config,previous=self.scene,transform=np.eye(4))
            self.geometry=self.geometry_record()
        return {'status':'retreat_completed' if path is not None else 'planning_failed',
            'phase':'retreat','reason':info['reason'],'release_executed':True,'retreat_executed':path is not None,
            'manipulation_executed':False,'requested_retreat_mm':decision['retreat_mm'],
            'failure_loci':failure_loci(folder),'requested_posture_rank':decision['posture_rank'],**observation}

    def dispatch(self,decision,folder):
        self.step_m=(decision['step_mm'] or 25)/1000
        self.posture_rank=decision['posture_rank'] if decision['operation']=='continue_motion' else None
        if decision['operation']=='release_reposition':return self.release_reposition(decision,folder)
        return super().dispatch(decision,folder)
