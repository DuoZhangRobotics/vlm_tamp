"""Exercise dollar accounting with fake responses; never contact the provider."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from types import SimpleNamespace

import pytest

from vlm_tamp import run_api_budget as module
from vlm_tamp.rekep_libero import api


class FakeClient:
    def __init__(self, fail=False):
        self.responses=self;self.calls=0;self.fail=fail

    def create(self,**arguments):
        self.calls+=1
        if self.fail:raise RuntimeError('connection interrupted')
        return SimpleNamespace(model_dump=lambda **_: {'status':'completed','output':[]})


@pytest.fixture
def accounting(tmp_path,monkeypatch):
    root=tmp_path/'api-runs';history=tmp_path/'legacy/ledger.json';history.parent.mkdir()
    history.write_text(json.dumps({'budget_usd':2.,'cumulative_conservative_spend_usd':3.,'requests':{}}))
    monkeypatch.setattr(module,'RUN_ROOT',root);monkeypatch.setattr(module,'LEGACY_LEDGER',history)
    monkeypatch.delenv('VLM_TAMP_API_RUN_ID',raising=False)
    # Large synthetic reservations exercise limits with only one fake response.
    monkeypatch.setattr(api,'reservation_usd',lambda arguments:1.25)
    return root,history


def arguments(tmp_path,text):
    image=tmp_path/'rgb.png';image.write_bytes(b'fake numbered RGB')
    return api.request_arguments(image,text,12000)


def test_independent_runs_each_get_two_dollars_and_history_is_informational(tmp_path,accounting):
    _,history=accounting;before=history.read_bytes();client=FakeClient()
    first=module.RunLedger('first');second=module.RunLedger('second')
    first.perform(arguments(tmp_path,'first scene'),'scene',allow_paid=True,client=client)
    second.perform(arguments(tmp_path,'second scene'),'scene',allow_paid=True,client=client)
    assert first.snapshot()['run_conservative_spend_usd']==1.25
    assert second.snapshot()['remaining_run_budget_usd']==.75
    assert module.snapshot(second)['cumulative_conservative_spend_usd']==5.5
    assert module.snapshot(second)['project_total_is_informational']
    with pytest.raises(api.BudgetError):
        first.perform(arguments(tmp_path,'third scene'),'third',allow_paid=True,client=client)
    assert client.calls==2 and history.read_bytes()==before


def test_same_run_workers_and_resume_share_balance(tmp_path,accounting):
    first=module.ledger_for_output(tmp_path/'cohort-a','one-experiment')
    second=module.ledger_for_output(tmp_path/'cohort-b','one-experiment')
    assert first.path==second.path
    first.perform(arguments(tmp_path,'first'),'scene',allow_paid=True,client=FakeClient())
    restarted=module.ledger_for_output(tmp_path/'cohort-a')
    assert restarted.run_id=='one-experiment' and restarted.snapshot()['remaining_run_budget_usd']==.75
    with pytest.raises(api.BudgetError,match='resumed experiment'):
        module.ledger_for_output(tmp_path/'cohort-a','new-budget')
    with pytest.raises(api.BudgetError):
        second.perform(arguments(tmp_path,'second'),'second',allow_paid=True,client=FakeClient())


def test_concurrent_workers_cannot_oversubscribe_one_budget(tmp_path,accounting):
    args=[arguments(tmp_path,'scene-a'),arguments(tmp_path,'scene-b')];clients=[FakeClient(),FakeClient()]
    def issue(i):
        try:
            module.RunLedger('parallel').perform(args[i],f'scene-{i}',allow_paid=True,client=clients[i])
            return 'admitted'
        except api.BudgetError:return 'blocked'
    with ThreadPoolExecutor(max_workers=2) as workers:
        results=list(workers.map(issue,range(2)))
    assert sorted(results)==['admitted','blocked']
    assert sum(c.calls for c in clients)==1
    assert module.RunLedger('parallel').snapshot()['run_conservative_spend_usd']==1.25


def test_uncertain_request_keeps_reservation_and_cannot_retry_in_another_run(tmp_path,accounting):
    client=FakeClient(fail=True);args=arguments(tmp_path,'uncertain')
    first=module.RunLedger('first')
    with pytest.raises(api.UncertainRequest):first.perform(args,'scene',allow_paid=True,client=client)
    for ledger in [module.RunLedger('first'),module.RunLedger('second')]:
        with pytest.raises(api.UncertainRequest):ledger.perform(args,'scene',allow_paid=True,client=client)
    assert client.calls==1 and first.snapshot()['run_conservative_spend_usd']==1.25
    assert module.RunLedger('second').snapshot()['run_conservative_spend_usd']==0


def test_exact_cache_reused_without_recharging_new_run(tmp_path,accounting):
    client=FakeClient();args=arguments(tmp_path,'shared image and instruction')
    first=module.RunLedger('first');raw=first.perform(args,'scene',allow_paid=True,client=client)
    assert first.perform(args,'alias',client=client)==raw
    second=module.RunLedger('second')
    assert second.perform(args,'scene',client=client)==raw
    assert second.perform(args,'scene',client=client)==raw
    snapshot=second.snapshot()
    assert client.calls==1 and snapshot['api_attempts']==0 and snapshot['run_conservative_spend_usd']==0
    assert snapshot['cache_hits']['scene']['charged_usd']==0
    with pytest.raises(api.ApiError,match='different inputs'):
        second.perform(arguments(tmp_path,'changed'),'scene',allow_paid=True,client=client)
    assert client.calls==1


def test_legacy_cached_response_reused_read_only(tmp_path,accounting):
    _,history=accounting;args=arguments(tmp_path,'legacy cached scene')
    digest=hashlib.sha256(api._canonical(args).encode()).hexdigest()
    raw={'status':'completed','output':[]}
    (history.parent/f'response-{digest}.json').write_text(json.dumps(raw))
    saved=json.loads(history.read_text());saved['requests']['old']={'request_sha256':digest,'status':'completed',
        'reservation_usd':.01,'cost_usd':.001}
    history.write_text(json.dumps(saved));before=history.read_bytes();client=FakeClient()
    ledger=module.RunLedger('new')
    assert ledger.perform(args,'now',client=client)==raw
    assert client.calls==0 and ledger.snapshot()['remaining_run_budget_usd']==2
    assert history.read_bytes()==before


def test_scope_and_environment_group_cohorts_without_affecting_next_run(tmp_path,accounting,monkeypatch):
    with module.api_run_scope('shared'):
        assert module.ledger_for_output(tmp_path/'a').run_id=='shared'
        assert module.ledger_for_output(tmp_path/'b').run_id=='shared'
        assert module.snapshot()['run_id']=='shared'
    assert module.current_run_id() is None
    assert module.ledger_for_output(tmp_path/'c').run_id!='shared'
    monkeypatch.setenv('VLM_TAMP_API_RUN_ID','workers')
    assert module.ledger_for_output(tmp_path/'d').run_id=='workers'
    assert module.ledger_for_output(tmp_path/'e').run_id=='workers'


def test_run_budget_and_identity_cannot_be_changed_on_disk(tmp_path,accounting):
    ledger=module.RunLedger('fixed');saved=ledger.snapshot();saved['budget_usd']=3
    ledger.path.write_text(json.dumps(saved))
    client=FakeClient()
    with pytest.raises(api.BudgetError):ledger.perform(arguments(tmp_path,'scene'),'scene',allow_paid=True,client=client)
    assert client.calls==0


@pytest.mark.parametrize('run_id',['../other','/tmp/other','',True,'bad space'])
def test_invalid_run_id_never_creates_ledger(run_id,accounting):
    with pytest.raises(api.BudgetError):module.RunLedger(run_id)


def test_invalid_request_is_rejected_before_cache_lookup(tmp_path,accounting):
    args=arguments(tmp_path,'scene');ledger=module.RunLedger('one');client=FakeClient()
    args['reasoning']={'effort':'low'}
    with pytest.raises(api.ApiError):ledger.perform(args,'scene',allow_paid=True,client=client)
    assert client.calls==0
