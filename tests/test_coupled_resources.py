import hashlib,json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock,patch
import numpy as np
import pytest
from PIL import Image
from vlm_tamp import coupled_resources as module
from vlm_tamp import run_api_budget as budget


def test_fresh_task_keeps_current_camera_and_drops_old_scene_geometry():
    previous={'camera':'frontview','joint_family':'revolute','motion':'open',
        'target_progress':1.57,'simulator_scene_state_used':False,
        'joint_origin_world_m':[999.,999.,999.], 'source_anchor_world_m':[999.,0.,0.],
        'constraint':{'type':'point_on_circle'}}
    current={'camera':'agentview','joint_family':'revolute','motion':'open'}
    result=module.fresh_task_template(previous,current)
    assert result['camera']=='agentview' and result['target_progress']==1.57
    assert 'joint_origin_world_m' not in result and 'source_anchor_world_m' not in result
    result['constraint']['type']='modified'
    assert previous['constraint']['type']=='point_on_circle' and previous['camera']=='frontview'


def test_fresh_task_rejects_oracle_provenance_instead_of_relabelling_it():
    with pytest.raises(ValueError,match='non-oracle'):
        module.fresh_task_template({'simulator_scene_state_used':True}, {})


def fixture(root):
    obs=root/'obs';obs.mkdir()
    for name in ['agentview','frontview']:Image.new('RGB',(3,3)).save(obs/(name+'.png'))
    ledger=root/'ledger.json';ledger.write_text(json.dumps({'budget_usd':2.,
        'cumulative_conservative_spend_usd':.2,'remaining_usd':1.8,'requests':{}}))
    return obs,ledger


def test_validated_rgbd_cache_does_not_call_api():
    with TemporaryDirectory() as temp:
        root=Path(temp);obs,ledger=fixture(root)
        source=root/'results/articulated-physics-pilot/open_microwave-state001/grounding'
        source.mkdir(parents=True);(source/'grounding.json').write_text(json.dumps({'provided_inventory':False,'simulator_scene_state_used':False}))
        def cached(source,observation,metadata,out):
            out.mkdir();return {},{'motion':'open'},None,None
        with patch.object(module,'ROOT',root),patch.object(module,'LEDGER',ledger), \
             patch.object(module,'_cached',side_effect=cached),patch.object(module,'_fresh') as paid:
            service=module.CoupledResources(root/'cache');out=root/'grounding'
            assert service.grounding('open_microwave',1,obs,{},out)=={'motion':'open'}
            paid.assert_not_called()
            assert module.read(out/'input-provenance.json')['new_api_calls']==0


def test_failed_paid_request_records_own_reservation_and_is_not_retried():
    with TemporaryDirectory() as temp:
        root=Path(temp);obs,ledger=fixture(root)
        def failed(perception,observation,metadata,out,scene,*,ledger):
            data=ledger.snapshot();data['requests'][scene]={'status':'uncertain','reservation_usd':.005,
                'request_sha256':'test-request'}
            ledger._save(data);raise RuntimeError('uncertain response')
        with patch.object(module,'ROOT',root),patch.object(module,'LEDGER',ledger), \
             patch.object(budget,'RUN_ROOT',root/'api-runs'), \
             patch.object(module,'_fresh',side_effect=failed) as paid:
            service=module.CoupledResources(root/'cache',api_output=root/'run');service.perception=object();out=root/'grounding'
            with pytest.raises(RuntimeError,match='uncertain'):
                service.grounding('open_microwave',11,obs,{},out)
            assert paid.call_count==1
            record=module.read(out/'input-provenance.json')
            assert record['new_api_calls']==1 and record['automatic_api_retries']==0
            assert record['run_spend_usd']==.005 and record['shared_remaining_usd']==1.995


def test_project_history_above_two_dollars_does_not_block_a_new_run():
    with TemporaryDirectory() as temp:
        root=Path(temp);obs,ledger=fixture(root)
        data=json.loads(ledger.read_text());data['cumulative_conservative_spend_usd']=2.01
        ledger.write_text(json.dumps(data))
        with patch.object(module,'ROOT',root),patch.object(module,'LEDGER',ledger), \
             patch.object(budget,'RUN_ROOT',root/'api-runs'), \
             patch.object(module,'_fresh',return_value=({}, {'motion':'open'},None,None)) as paid:
            service=module.CoupledResources(root/'cache',api_output=root/'run');service.perception=object()
            assert service.grounding('open_microwave',11,obs,{},root/'out')=={'motion':'open'}
            paid.assert_called_once()
            assert paid.call_args.kwargs['ledger'].snapshot()['remaining_run_budget_usd']==2.
            record=module.read(root/'out/input-provenance.json')
            assert record['shared_cumulative_spend_usd']==2.01 and record['run_spend_usd']==0.


def test_grasp_cache_requires_exact_point_cloud_and_checkpoint_pins():
    with TemporaryDirectory() as temp:
        root=Path(temp);service=module.CoupledResources(root/'cache');service.pins={'model':'pinned'}
        xyz=np.arange(90,dtype=np.float32).reshape(30,3)
        key=hashlib.sha256(xyz.tobytes()+json.dumps(service.pins,sort_keys=True).encode()).hexdigest()
        cache=service.cache/('graspgen-'+key);cache.mkdir()
        module.save(cache/'input-provenance.json',{'cache_key':key,'checkpoint_pins':service.pins})
        np.save(cache/'input-cloud.npy',xyz+1.)
        with pytest.raises(ValueError,match='point cloud differs'):service.grasps(xyz,root/'out')
        np.save(cache/'input-cloud.npy',xyz)
        np.savez(cache/'grasp_candidates.npz',poses_world_base=np.eye(4)[None],scores=[.9],retained_indices=[0])
        output=service.grasps(xyz,root/'out')
        assert output['scores'][0]==.9
        assert module.read(root/'out/input-provenance.json')['reused']


def test_grasp_sampling_seed_is_part_of_cache_identity(tmp_path):
    service=module.CoupledResources(tmp_path/'cache');service.pins={'model':'pinned'}
    calls=[]
    class Generator:
        def generate(self,xyz,output,seed=0):
            calls.append(seed);output.mkdir()
            result={'poses_world_base':np.eye(4)[None],'scores':np.array([.9]),'retained_indices':np.array([0])}
            np.savez(output/'grasp_candidates.npz',**result)
            return result
    service.graspgen=Generator();xyz=np.zeros((30,3),np.float32)
    service.grasps(xyz,tmp_path/'zero')
    service.grasp_seed=1
    service.grasps(xyz,tmp_path/'one')
    service.grasps(xyz,tmp_path/'one_again')
    assert calls==[0,1]
    assert len(list(service.cache.glob('graspgen-*')))==2
    assert module.read(tmp_path/'one_again/input-provenance.json')['grasp_seed']==1
    service.grasp_seed=True
    with pytest.raises(ValueError,match='integer'):service.grasps(xyz,tmp_path/'invalid')
