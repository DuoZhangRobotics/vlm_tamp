import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vlm_tamp.rgb_choices import png
from vlm_tamp.rgb_object_choices import build_choice_request
from vlm_tamp.rgb_object_trial import ChoiceOpenAIProvider


@pytest.fixture
def object_request(dataset, monkeypatch):
    from PIL import Image

    root, _ = dataset
    pixels = (root / "rgb.png").read_bytes()
    regions = [
        {"choice": 1, "point": [0.2, 0.4], "pixel": [4, 8], "box": [0, 0, 10, 20], "area": 200},
        {"choice": 2, "point": [0.8, 0.4], "pixel": [16, 8], "box": [10, 0, 20, 20], "area": 200},
    ]
    visual = png(Image.open(root / "rgb.png").convert("RGB"))
    monkeypatch.setattr("vlm_tamp.rgb_object_choices.object_bundle", lambda image: (regions, visual, visual))
    return build_choice_request("put the bowl on the plate", pixels, "test"), regions


def test_public_spatial_hints_are_image_only_and_cached_mappings_are_isolated(object_request):
    request, cached = object_request
    assert set(request.payload) == {"instruction", "visual_choices"}
    assert all(
        set(choice) == {"number", "image_point", "image_box"} for choice in request.payload["visual_choices"]
    )
    assert request.payload["visual_choices"][0]["image_box"] == [0.0, 0.0, 0.5, 1.0]
    request.regions[0]["point"][0] = 0.9
    assert cached[0]["point"] == [0.2, 0.4]


def test_provider_rejects_oracle_metadata_and_changed_image_points(object_request):
    request, _ = object_request
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text="{}", usage=None, model_dump=lambda mode: {})

    config = {
        "model": "fake",
        "image_detail": "high",
        "reasoning_effort": "medium",
        "max_output_tokens": 8000,
    }
    provider = ChoiceOpenAIProvider(config, client=SimpleNamespace(responses=SimpleNamespace(create=create)))
    bad_payload = copy.deepcopy(request.payload)
    bad_payload["visual_choices"][0]["identifier"] = "bowl_1"
    with pytest.raises(ValueError):
        provider.infer(replace(request, payload=bad_payload))
    request.regions[0]["point"][0] = 0.9
    with pytest.raises(ValueError):
        provider.infer(request)
    assert not calls
