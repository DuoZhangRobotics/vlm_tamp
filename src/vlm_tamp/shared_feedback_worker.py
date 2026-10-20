"""Physical feedback execution in the isolated LIBERO process."""
import contextlib
import json
import sys
import time
import traceback
import numpy as np
from .profiled_sim_worker import ProfiledWorker
from .coupled_worker import checked_joint_target
from .rekep_libero.robot_chain_v4 import RobotChain
from .rekep_libero.sim_worker import MAX_CONTROL_STEPS
from .shared_feedback import FeedbackConfig
from .shared_feedback_directional import DirectionalFeedback


class FeedbackWorker(ProfiledWorker):
    def feedback_trajectory(self, references, gripper, context):
        refs=np.asarray(references,float)
        if refs.ndim!=2 or refs.shape[1]!=7 or not np.isfinite(refs).all():
            raise ValueError('Finite seven-joint references required')
        if len(refs)>MAX_CONTROL_STEPS-self.control_steps:
            raise ValueError('Entire trajectory must fit before dispatch')
        for q in refs: checked_joint_target(q,self.ik_solver.limits)
        controller=DirectionalFeedback(RobotChain(self.robot_chain()),refs,self._eef_pose(),
            variant=context['variant'],normal=context['normal_world'],phase=context['phase'],
            config=FeedbackConfig(**context.get('config',{})),tangent=context.get('motion_tangent_world'),
            mode=context.get('mode','pinch'),visual_offset=context.get('visual_offset_world_m'))
        self.in_batch=True;events=[];began=time.perf_counter();correction_seconds=0.
        try:
            for index, nominal in enumerate(refs):
                tick=time.perf_counter()
                # Avoid applying a contact accommodation rule during free travel.
                if context['phase']=='approach' and index<len(refs)-40:
                    command=nominal;detail={'offset_world_m':[0.,0.,0.],'free_approach':True}
                else:
                    command,detail=controller.command(index,nominal,self._eef_pose())
                correction_seconds+=time.perf_counter()-tick
                state=super().joint_action(command,gripper=int(gripper),max_steps=1)
                actual=np.asarray(state['metadata']['joint_positions'])
                events.append({'control_step':state['control_steps'],'reference_index':index,
                    'nominal_reference_joints':nominal.tolist(),'reference_joints':command.tolist(),
                    'measured_joints':actual.tolist(),'joint_reference_error_rad':float(np.max(abs(actual-command))),
                    'command_issued':True,'execution_veto_checks':0,**detail})
        finally:
            self.in_batch=False;self._write_action_log()
        return {**self._response_state(),'trajectory':{'all_commands_issued':True,
            'planned_commands':len(refs),'issued_commands':len(events),'events':events,
            'worker_seconds':time.perf_counter()-began,'correction_compute_seconds':correction_seconds,
            'execution_veto_checks':0,'maximum_joint_reference_error_rad':max((e['joint_reference_error_rad'] for e in events),default=0.),
            'controller':context['variant'],'nominal_samples_skipped':0,
            'policy_inputs':['nominal robot trajectory','robot proprioception','observed contact frame','bounded RGB-D tracking correction'],
            'native_contact_or_success_used_for_control':False}}

    def dispatch(self, request):
        if request['op']=='feedback_trajectory':
            return self.feedback_trajectory(**{k:v for k,v in request.items() if k not in {'id','op'}})
        return super().dispatch(request)


def main():
    output=sys.stdout;worker=FeedbackWorker(fast=True)
    with contextlib.redirect_stdout(sys.stderr):
        try:
            for line in sys.stdin:
                request={};worker.stats={};before=time.perf_counter()
                try:
                    request=json.loads(line);response={'ok':True,'id':request.get('id'),**worker.dispatch(request)}
                except Exception as error:
                    traceback.print_exc(file=sys.stderr)
                    response={'ok':False,'id':request.get('id'),'error':str(error),'error_type':type(error).__name__}
                response['worker_profile']={'server_seconds':time.perf_counter()-before,'stages':worker.stats}
                output.write(json.dumps(response,allow_nan=False)+'\n');output.flush()
                if request.get('op')=='close':break
        finally:worker.close()

if __name__=='__main__':main()
