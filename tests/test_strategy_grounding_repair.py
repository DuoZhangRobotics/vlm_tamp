import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest

from vlm_tamp import strategy_grounding_repair as module
from vlm_tamp.rekep_libero.articulated_grounding import PROMPT


def test_feedback_contains_machine_error_without_task_specific_pixel_patch():
    text=module.correction_prompt('Close the selected part','{"contact_box_xyxy":[1,600,2,700]}','Invalid RGB box: contact_box_xyxy')
    added=text[len(PROMPT.format(instruction='Close the selected part')):]
    assert '600' in added and 'Invalid RGB box' in added
    assert '560' not in added and 'subtract' not in added.lower() and 'bottom drawer' not in added.lower()


def test_one_completed_invalid_response_is_revised_and_original_is_preserved(tmp_path,monkeypatch):
    output=tmp_path/'grounding';output.mkdir();obs=tmp_path/'obs';obs.mkdir()
    for name in ['agentview','frontview']:
        Image.new('RGB',(8,8)).save(obs/f'{name}.png')
        (output/name).mkdir();np.savez(output/name/'keypoints.npz',points_world=np.zeros((1,3)))
    Image.new('RGB',(16,16)).save(output/'role-choice-sheet.png')
    original={'status':'completed','output':[]}
    (output/'response.json').write_text(json.dumps(original));before=(output/'response.json').read_bytes()
    captured={};calls=[]
    class Ledger:
        run_id='shared-run'
        def snapshot(self):return {'api_attempts':len(calls),'run_conservative_spend_usd':.01*len(calls)}
        def perform(self,args,key,**kwargs):captured.update(args);calls.append(key);return {'status':'completed','output':[]}
    class Base:
        def grounding(self,*a,**k):raise ValueError('Invalid RGB box: contact_box_xyxy')
        def api_ledger(self):return Ledger()
        def ensure_perception(self):return object()
    class Service(module.GroundingRepairMixin,Base):pass
    service=Service();service.new_api_calls=0;service.allow_paid=True;service.cache=tmp_path/'cache';service.cache.mkdir()
    monkeypatch.setattr(module,'response_text',lambda raw:'{"camera":"frontview"}')
    monkeypatch.setattr(module,'parse_choices',lambda text:json.loads(text))
    monkeypatch.setattr(module,'_read_observation',lambda *a:({'frontview':{}},{}))
    monkeypatch.setattr(module,'prepare_contact',lambda *a:None)
    choices=service.grounding('task',0,obs,{'instruction':'Close the selected part'},output)
    assert choices['camera']=='frontview' and len(calls)==1 and service.new_api_calls==1
    assert captured['max_output_tokens']==32000 and captured['reasoning']=={'effort':'high'}
    assert (output/'response.json').read_bytes()==before
    assert (output/'validation-repair/verification.json').exists()


def test_incomplete_response_does_not_trigger_network_retry(tmp_path):
    output=tmp_path/'grounding';output.mkdir();(output/'response.json').write_text('{"status":"incomplete"}')
    (output/'role-choice-sheet.png').write_bytes(b'image')
    class Base:
        def grounding(self,*a,**k):raise ValueError('Response incomplete')
        def api_ledger(self):raise AssertionError('No new request allowed')
    class Service(module.GroundingRepairMixin,Base):pass
    with pytest.raises(ValueError,match='incomplete'):Service().grounding('task',0,tmp_path,{},output)


def test_new_runner_paid_opt_in_also_covers_initial_grounding(tmp_path,monkeypatch):
    from unittest.mock import Mock
    from vlm_tamp import cohort_resources as cohort,cli
    from vlm_tamp.rekep_libero.api import ApiError
    obs=tmp_path/'obs';obs.mkdir()
    for name in ['agentview','frontview']:Image.new('RGB',(8,8)).save(obs/f'{name}.png')
    class Perception:
        def prepare(self,observation,metadata,output):
            output.mkdir(parents=True)
            for name in ['rgb','numbered']:Image.new('RGB',(8,8)).save(output/f'{name}.png')
            return {}
    service=cohort.CohortResources(tmp_path/'experiment');service.cache=tmp_path/'cache';service.cache.mkdir()
    service.perception=Perception();service.candidates=lambda *a:[];service.allow_paid=False
    ledger=SimpleNamespace(run_id='test',perform=Mock(side_effect=ApiError('Paid inference disabled')))
    service.api_ledger=lambda:ledger
    monkeypatch.setattr(cohort,'ROOT',tmp_path)
    monkeypatch.setattr(cohort,'api_snapshot',lambda *a:{'requests':{},'run_conservative_spend_usd':0.,
        'remaining_run_budget_usd':2.,'cumulative_conservative_spend_usd':0.})
    credentials=Mock();monkeypatch.setattr(cli,'load_credentials',credentials)
    with pytest.raises(ApiError,match='disabled'):
        service.grounding('new_task',17,obs,{'instruction':'Close the selected part'},tmp_path/'grounding')
    assert ledger.perform.call_args.kwargs['allow_paid'] is False
    credentials.assert_not_called()
