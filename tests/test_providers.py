import json

import httpx
from openai import OpenAI

from vlm_tamp.dataset import build_request
from vlm_tamp.providers import GeminiProvider, OpenAIProvider


def test_official_openai_sdk_image_and_strict_schema(dataset):
    root, record = dataset
    captured = []

    def handle(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "object": "response",
                "created_at": 0,
                "model": "test-model",
                "status": "completed",
                "output": [
                    {
                        "id": "m1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": '{"entities":[],"goals":[]}', "annotations": []}
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 101,
                    "output_tokens": 11,
                    "total_tokens": 112,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )

    client = OpenAI(
        api_key="unit-test-only",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handle)),
    )
    provider = OpenAIProvider(
        {
            "model": "test-model",
            "timeout_seconds": 10,
            "max_output_tokens": 1000,
            "temperature": 0,
            "reasoning_effort": "none",
            "image_detail": "high",
        },
        client=client,
    )
    response = provider.infer(build_request(record["model_inputs"], root, "rgb", "prompt"))
    assert response.input_tokens == 101 and response.output_tokens == 11
    body = captured[0]
    assert body["text"]["format"]["strict"] is True
    assert body["store"] is False
    assert body["temperature"] == 0 and body["reasoning"]["effort"] == "none"
    assert body["text"]["format"]["schema"]["properties"]["schema_version"]["const"] == "1.2"
    assert body["input"][0]["content"][1]["image_url"].startswith("data:image/png;base64,")
    assert "DO_NOT_SEND_123" not in json.dumps(body)
    provider.config.update(
        temperature=None, reasoning_effort="medium", service_tier="default", max_output_tokens=25000
    )
    provider.infer(build_request(record["model_inputs"], root, "rgb", "prompt"))
    reasoning_body = captured[1]
    assert "temperature" not in reasoning_body
    assert reasoning_body["reasoning"]["effort"] == "medium"
    assert reasoning_body["service_tier"] == "default"
    assert reasoning_body["max_output_tokens"] == 25000


def test_gemini_adapter_uses_official_types_and_counts_thinking(dataset):
    from types import SimpleNamespace
    from google.genai import types
    from vlm_tamp.grounding import GroundingPrediction

    root, record = dataset
    captured = []
    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model", parts=[types.Part.from_text(text='{"entities":[],"goals":[]}')]
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=20, candidates_token_count=10, thoughts_token_count=5, total_token_count=35
        ),
    )

    def generate_content(**kwargs):
        captured.append(kwargs)
        return response

    fake = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    provider = GeminiProvider(
        {"model": "gemini-test", "timeout_seconds": 10, "max_output_tokens": 1000}, client=fake
    )
    result = provider.infer(build_request(record["model_inputs"], root, "rgb", "prompt"))
    assert result.input_tokens == 20 and result.output_tokens == 15
    assert captured[0]["config"].response_schema is GroundingPrediction
    assert captured[0]["config"].response_mime_type == "application/json"
    assert captured[0]["contents"][1].inline_data.mime_type == "image/png"
