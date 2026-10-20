"""Small audited API budget for native external component smoke tests."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from .libero import ROOT


class BoundedChat:
    def __init__(self, output, limit=0.10, max_calls=8):
        sys.path.insert(0, str(ROOT / ".runtime/external-api-deps"))
        import tiktoken
        from openai import OpenAI
        self.encoder = tiktoken.get_encoding("o200k_base")
        self.client = OpenAI(base_url="https://api.openai.com/v1", timeout=120, max_retries=0)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.limit, self.max_calls, self.cost, self.calls = limit, max_calls, 0.0, 0
        self.unknown_reserve = 0.0
        ledger = self.output / "ledger.json"
        if ledger.exists():
            saved = json.loads(ledger.read_text())
            self.cost, self.calls = saved["estimated_usage_cost_usd"], saved["api_attempts"]
            self.unknown_reserve = saved["unknown_reserved_usd"]
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        kwargs = dict(kwargs)
        if kwargs.get("model") != "gpt-4o-2024-11-20":
            raise ValueError("Component API pricing is frozen for the GPT-4o snapshot")
        kwargs.pop("extra_body", None)  # Local Qwen repetition_penalty is not an OpenAI parameter.
        wants_stream = kwargs.pop("stream", False)
        kwargs.update(stream=False, store=False, max_tokens=min(kwargs.get("max_tokens", 1536), 1536))
        images, text_tokens = 0, 100
        for message in kwargs["messages"]:
            content = message["content"]
            for item in content if isinstance(content, list) else [{"type": "text", "text": content}]:
                if item["type"] == "image_url":
                    images += 1
                elif item["type"] == "text":
                    text_tokens += len(self.encoder.encode(item["text"]))
        text_tokens += len(self.encoder.encode(json.dumps(kwargs.get("response_format", {}))))
        reservation = ((text_tokens + images * 2048) * 2.5 + kwargs["max_tokens"] * 10) / 1e6
        if self.calls >= self.max_calls or self.cost + self.unknown_reserve + reservation > self.limit:
            raise RuntimeError("External component API budget exhausted")
        index = self.calls
        self.calls += 1
        (self.output / f"request-{index:02d}.json").write_text(json.dumps(kwargs) + "\n")
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception:
            self.unknown_reserve += reservation
            self.save_ledger()
            raise
        raw = response.model_dump(mode="json")
        (self.output / f"response-{index:02d}.json").write_text(json.dumps(raw, indent=2) + "\n")
        if response.usage:
            self.cost += (response.usage.prompt_tokens * 2.5 + response.usage.completion_tokens * 10) / 1e6
        else:
            self.unknown_reserve += reservation
        self.save_ledger()
        if wants_stream:
            text = response.choices[0].message.content
            return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])])
        return response

    def save_ledger(self):
        (self.output / "ledger.json").write_text(json.dumps({
            "api_attempts": self.calls, "estimated_usage_cost_usd": self.cost,
            "unknown_reserved_usd": self.unknown_reserve, "budget_usd": self.limit,
            "model": "gpt-4o-2024-11-20", "input_per_million": 2.5, "output_per_million": 10,
            "scope": "Ported component smoke tests only; not full paper reproduction",
        }, indent=2) + "\n")
