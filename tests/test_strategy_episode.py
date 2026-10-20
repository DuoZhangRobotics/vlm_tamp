from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

from vlm_tamp import strategy_episode as module


def test_instruction_survives_robot_only_metadata_after_control_enable(tmp_path,monkeypatch):
    captured={}
    class Session:
        def __init__(self,task,index,folder,config,resources):
            self.stage='ready';self.ik=None;self.sim=SimpleNamespace(observe=Mock(),close=Mock())
            self.events=[];self.phases=[];self.history=[];self.goal_reached=False;self.latest_pose=None
            self.current_choice=None;self.geometry={};self.cloud=np.zeros((20,3));self.views=self.previous_views={}
            self.state={'control_steps':60,'metadata':{'eef_pose_world_xyzw':[0,0,0,0,0,0,1],
                'gripper_qpos':[.04,-.04],'evaluation_task_success':True}}
            self.spec={'instruction':'Open the selected part','joint_family':'prismatic','motion':'open',
                'joint_axis_world':[0,1,0],'panel_normal_world':[0,1,0],
                'source_anchor_world_m':[0,0,0],'target_progress':.15,'progress_units':'meters'}
            self.progress=SimpleNamespace(records=[],observed_progress=0.,progress=0.)
        def prepare(self):pass
        def dispatch(self,decision,folder):return {'status':'agent_finished','manipulation_executed':False}
    monkeypatch.setattr(module,'StrategyTools',Session)
    monkeypatch.setattr(module,'render_feedback',lambda *a,**kw:None)
    monkeypatch.setattr(module,'compact_episode',lambda folder:{'bytes_reclaimed_this_pass':0})
    def choose(packet,*args):
        captured.update(packet)
        return {'version':'strategy-v1','operation':'finish','contact_mode':None,'approach_tilt_deg':None,
            'motion_extent':None,'evidence':'No further action requested.','expected_observation':'The scene remains unchanged.'}
    policy=SimpleNamespace(choose=choose,calls=0);resources=SimpleNamespace(new_api_calls=0,cleanup_proposals=Mock())
    result=module.run_episode({'label':'case','name':'native task','suite':'test'},0,tmp_path/'case',{},resources,policy)
    assert captured['instruction']=='Open the selected part'
    assert 'evaluation_task_success' not in captured['observed_state']
    assert result['decisions']==1 and 'failure_stage' not in result
