import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from vlm_tamp.simulator_probe import contact_penetrations, contained_vertices, observation_only
from vlm_tamp.verification import axis_check, limits_check, verifier_feedback, verify_proposal


def facts_fixture():
    fact = {
        "entity_id": "cabinet_top_region",
        "parent_id": "cabinet",
        "joint_id": "slide",
        "body": "drawer_body",
        "joint_type": "prismatic",
        "axis_world": [0, 1, 0],
        "axis_local": [1, 0, 0],
        "axis_point_world": [0, 0, 1],
        "limits": [-0.16, 0.01],
        "coordinate_units": "m",
        "initial_q": 0,
        "initial_open": False,
        "open_toggled_in_sweep": True,
        "translation_axis_residual_m": 0.0,
        "rotation_residual": 0.0,
        "sweep": [{"q": -0.16, "open": True}, {"q": 0.01, "open": False}],
    }
    return {
        "example_id": "example",
        "moving_object": "bowl",
        "target_region": "cabinet_top_region",
        "joints": {"cabinet_top_region": fact},
        "collision_penetration_tolerance_m": 0.0005,
        "placement_probes": {
            "open": {
                "tested": 125,
                "nonpenetrating_in_count": 20,
                "fully_contained_collision_box_count": 0,
                "joint_q": -0.16,
            },
            "closed": {
                "tested": 125,
                "nonpenetrating_in_count": 1,
                "fully_contained_collision_box_count": 0,
                "joint_q": 0.01,
            },
        },
    }


def proposal_fixture():
    return {
        "schema_version": "1.2",
        "goal_logic": "and",
        "entities": [
            {"id": "b", "category": "bowl", "description": "bowl", "reference_id": "bowl"},
            {"id": "c", "category": "cabinet", "description": "cabinet", "reference_id": "cabinet"},
            {
                "id": "d",
                "category": "drawer",
                "description": "top drawer",
                "kind": "region",
                "parent": "c",
                "reference_id": "cabinet_top_region",
            },
        ],
        "goals": [
            {
                "name": "In",
                "arguments": ["b", "d"],
                "intent": "task_outcome",
                "evidence_text": "put the bowl inside",
            }
        ],
        "constraints": [
            {
                "id": "slide",
                "category": "articulation",
                "type": "follow_joint",
                "entities": ["d"],
                "articulated_part": "d",
                "joint_type": "prismatic",
                "scope": "path",
                "requirement": "conditional",
                "evidence": "planning_hypothesis",
                "verified": False,
                "parameters": [
                    {"name": "axis", "value": None, "status": "unresolved"},
                    {"name": "joint_limits", "value": None, "status": "unresolved"},
                ],
            }
        ],
    }


def test_verified_kinematics_does_not_claim_full_containment_or_execution():
    facts = facts_fixture()
    proposal = proposal_fixture()
    report = verify_proposal(proposal, facts)
    checks = {c["kind"]: c for c in report["claims"]}
    assert checks["joint_type"]["status"] == "verified"
    assert checks["mechanism_sweep"]["status"] == "verified"
    assert checks["sampled_placement"]["status"] == "verified"
    assert checks["whole_object_site_proxy"]["status"] == "unresolved"
    assert checks["access_requirement"]["status"] == "unresolved"
    assert checks["execution"]["status"] == "unresolved"
    assert checks["claimed_axis"]["status"] == "unresolved"
    assert not report["ready_for_robot_execution"]
    assert all(c["verified"] is False for c in proposal["constraints"])


def test_wrong_joint_mechanism_axis_and_limits_are_contradicted():
    facts = facts_fixture()
    proposal = proposal_fixture()
    proposal["constraints"][0]["joint_type"] = "revolute"
    proposal["constraints"][0]["parameters"] = [
        {"name": "axis", "value": [1, 0, 0], "status": "known", "frame": "world", "units": "unit_vector"},
        {
            "name": "joint_limits",
            "value": [0, 1],
            "status": "known",
            "frame": "joint_coordinate",
            "units": "m",
        },
    ]
    report = verify_proposal(proposal, facts)
    assert sum(c["status"] == "contradicted" for c in report["claims"]) == 3
    assert facts["joints"]["cabinet_top_region"]["joint_type"] == "prismatic"


def test_axis_sign_frames_and_units_are_not_silently_invented():
    fact = facts_fixture()["joints"]["cabinet_top_region"]
    assert axis_check({"value": [0, -1, 0], "status": "known", "frame": "world"}, fact)[0] == "verified"
    assert axis_check({"value": [0, 0, 0], "status": "known", "frame": "world"}, fact)[0] == "contradicted"
    assert axis_check({"value": [1, 0, 0], "status": "known", "frame": "camera"}, fact)[0] == "unresolved"
    assert (
        limits_check(
            {"value": [-0.16, 0.01], "status": "known", "frame": "joint_coordinate", "units": "rad"}, fact
        )[0]
        == "unresolved"
    )
    assert (
        limits_check(
            {"value": [-0.16, 0.01], "status": "known", "frame": "joint_coordinate", "units": "m"}, fact
        )[0]
        == "verified"
    )


def test_probe_filters_contact_pairs_and_respects_rotated_boxes():
    contacts = [
        SimpleNamespace(geom1=1, geom2=2, dist=-1),
        SimpleNamespace(geom1=4, geom2=5, dist=-0.4),
        SimpleNamespace(geom1=1, geom2=8, dist=-0.0002),
    ]
    assert contact_penetrations(contacts, {1, 2}) == 0.0002
    assert contained_vertices(np.array([[0, 0, 0], [0.5, 0.5, 0.5]]), np.zeros(3), np.eye(3), [1, 1, 1])
    assert not contained_vertices(np.array([[2, 0, 0]]), np.zeros(3), np.eye(3), [1, 1, 1])
    assert contained_vertices(None, np.zeros(3), np.eye(3), [1, 1, 1]) is None


def test_answer_compartment_is_removed_and_feedback_excludes_benchmark_answers():
    record = {
        "example_id": "x",
        "model_inputs": {},
        "metadata": {},
        "reference": {"goal_predicates": "SECRET_ANSWER"},
    }
    assert set(observation_only(record)) == {"example_id", "model_inputs", "metadata"}
    facts = facts_fixture()
    facts["reference"] = {"goal_predicates": "SECRET_ANSWER"}
    proposal = proposal_fixture()
    original = copy.deepcopy(proposal)
    report = verify_proposal(proposal, facts)
    feedback = verifier_feedback(proposal, report, facts)
    assert "SECRET_ANSWER" not in json.dumps(feedback)
    assert proposal == original and not report["benchmark_goal_answers_used"]


@pytest.mark.integration
def test_real_independent_drawer_probes_restore_scene(tmp_path):
    import os

    if os.environ.get("RUN_LIBERO_INTEGRATION") != "1":
        pytest.skip("Enable real LIBERO integration checks explicitly")
    from vlm_tamp.libero import collect
    from vlm_tamp.simulator_probe import collect_physical_facts

    records = collect(tmp_path / "dataset", tasks=[3], states=[0], gl="egl")
    facts = collect_physical_facts(
        tmp_path / "dataset",
        records[0]["example_id"],
        tmp_path / "facts",
        "akita_black_bowl_1",
        "wooden_cabinet_1_top_region",
        grid_size=3,
    )
    assert len(facts["joints"]) == 3
    assert facts["original_scene_restored"]
    assert facts["benchmark_goal_answers_used"] is False
    assert all(j["translation_axis_residual_m"] < 1e-7 for j in facts["joints"].values())
    assert facts["placement_probes"]["open"]["tested"] == 27
    assert facts["placement_probes"]["open"]["official_in_count"] == 27


def test_trial_paid_guard_and_scoped_feedback_with_local_fake(dataset, config, tmp_path):
    from vlm_tamp.verifier_trial import run_verifier_trial
    from vlm_tamp.grounding import strict_output_schema
    from vlm_tamp.providers import ProviderResponse
    from pathlib import Path
    import vlm_tamp.verifier_trial as trial

    root, record = dataset
    record["metadata"]["bddl_sha256"] = "fixture"
    record["model_inputs"]["candidates"].append(
        {"identifier": "cabinet", "category": "cabinet", "kind": "object", "parent": None}
    )
    record["model_inputs"]["candidates"].append(
        {"identifier": "cabinet_top_region", "category": "region", "kind": "region", "parent": "cabinet"}
    )
    record["model_inputs"]["instruction"] = "open the top drawer and put the bowl inside"
    (root / "records.jsonl").write_text(json.dumps(record))
    facts = facts_fixture()
    facts.update(
        example_id=record["example_id"],
        initial_state_sha256=record["metadata"]["initial_state_sha256"],
        image_sha256=hashlib.sha256((root / "rgb.png").read_bytes()).hexdigest(),
    )
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(facts))
    protocol = {
        "schema_version": "1.2",
        "schema_sha256": hashlib.sha256(
            json.dumps(strict_output_schema(), sort_keys=True).encode()
        ).hexdigest(),
        "prompt_sha256": hashlib.sha256(Path(config[1]["prompt"]).read_bytes()).hexdigest(),
        "models": ["test-model"],
        "example_ids": [record["example_id"]],
        "conditions": ["candidates"],
        "temperature": 0,
        "image_detail": "high",
        "max_output_tokens": 1000,
        "physical_facts_sha256": hashlib.sha256(facts_path.read_bytes()).hexdigest(),
        "verifier_sha256": hashlib.sha256(
            Path(trial.__file__).with_name("verification.py").read_bytes()
        ).hexdigest(),
        "trial_runner_sha256": hashlib.sha256(Path(trial.__file__).read_bytes()).hexdigest(),
    }
    protocol_path = tmp_path / "frozen.json"
    protocol_path.write_text(json.dumps(protocol))
    path, c = config
    c.update(
        provider="openai",
        model="test-model",
        max_requests=2,
        max_retries=0,
        max_budget_usd=1,
        example_ids=protocol["example_ids"],
        conditions=["candidates"],
        temperature=0,
        image_detail="high",
        physical_facts=str(facts_path),
        frozen_protocol=str(protocol_path),
        pricing={"input_per_million": 1, "output_per_million": 1},
    )
    path.write_text(json.dumps(c))

    class Provider:
        calls = 0

        def infer(self, request):
            self.calls += 1
            p = proposal_fixture()
            p["entities"][0]["reference_id"] = "bowl_1"
            if self.calls == 2:
                assert "verification_feedback" in request.payload
                assert "SECRET_REFERENCE" not in request.text
                p["constraints"][0]["parameters"] = [
                    {
                        "name": "axis",
                        "value": [0, 1, 0],
                        "status": "known",
                        "frame": "world",
                        "units": "unit_vector",
                    },
                    {
                        "name": "joint_limits",
                        "value": [-0.16, 0.01],
                        "status": "known",
                        "frame": "joint_coordinate",
                        "units": "m",
                    },
                ]
            return ProviderResponse(json.dumps(p), {"model": "test-model"}, 100, 30)

    backend = Provider()
    with pytest.raises(PermissionError):
        run_verifier_trial(root, path, tmp_path / "forbidden", provider=backend)
    from unittest.mock import patch

    with patch("vlm_tamp.verifier_trial.write_trial_report"):
        manifest = run_verifier_trial(root, path, tmp_path / "trial", allow_paid=True, provider=backend)
    assert backend.calls == 2 and manifest["ledger"]["api_attempts"] == 2
    phases = json.loads((tmp_path / "trial/phases.json").read_text())
    assert all(p["linked_schema_valid"] for p in phases)
    assert phases[1]["verification"]["counts"]["verified"] > phases[0]["verification"]["counts"]["verified"]
    assert all(p["benchmark_assessment"]["feedback_used"] is False for p in phases)
