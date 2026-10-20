import json
from types import SimpleNamespace

import pytest

from vlm_tamp.baseline_api import BoundedChat


def caller(tmp_path, limit=0.10, failure=False):
    api = BoundedChat.__new__(BoundedChat)
    api.output = tmp_path
    api.limit, api.max_calls, api.cost, api.calls, api.unknown_reserve = limit, 1, 0.0, 0, 0.0
    api.encoder = SimpleNamespace(encode=lambda text: list(text))
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if failure:
            raise RuntimeError("Transport failed")
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
                               choices=[SimpleNamespace(message=SimpleNamespace(content="actual reply"))],
                               model_dump=lambda **kwargs: {"usage": {"prompt_tokens": 100, "completion_tokens": 10}})

    api.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return api, calls


def request():
    return {"model": "gpt-4o-2024-11-20", "messages": [{"role": "user", "content": "test"}]}


def test_component_budget_rejects_before_a_call(tmp_path):
    api, calls = caller(tmp_path, limit=0.0001)
    with pytest.raises(RuntimeError):
        api.create(**request())
    assert not calls and api.calls == 0


def test_accounting_cap_and_stream_adapter_preserve_the_actual_reply(tmp_path):
    api, calls = caller(tmp_path)
    chunks = list(api.create(**request(), stream=True, max_tokens=2048))
    assert chunks[0].choices[0].delta.content == "actual reply"
    assert calls[0]["max_tokens"] == 1536 and calls[0]["store"] is False
    assert abs(api.cost - 0.00035) < 1e-12
    with pytest.raises(RuntimeError):
        api.create(**request())
    assert len(calls) == 1


def test_unknown_transport_cost_is_reserved_and_wrong_model_is_rejected(tmp_path):
    api, calls = caller(tmp_path, failure=True)
    with pytest.raises(ValueError):
        api.create(model="unpriced", messages=[])
    assert not calls
    with pytest.raises(RuntimeError):
        api.create(**request())
    saved = json.loads((tmp_path / "ledger.json").read_text())
    assert saved["api_attempts"] == 1 and saved["unknown_reserved_usd"] > 0
