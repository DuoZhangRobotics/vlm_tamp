"""Bounded complete transitions, provenance, and explicit temporal gaps."""

from vlm_tamp.interaction_memory import InteractionMemory


def record(index: int, before: int, after: int) -> dict:
    """A measured transition independent of any task or object."""
    return {
        "decision": index,
        "before_control": before,
        "after_control": after,
        "before_images": {"agentview": f"before-{index}.png"},
        "after_images": {"agentview": f"after-{index}.png"},
        "operation": "move",
        "receipt": {"ok": True},
    }


def test_anchors_retain_earlier_transitions_and_latest_without_duplicate_images() -> None:
    memory = InteractionMemory("anchored", slots=4, first_control=0, last_control=100)
    for index in range(10):
        memory.add(record(index, index * 10, (index + 1) * 10))
    assert [r["decision"] for r in memory.selected()] == [0, 2, 4, 9]
    assert memory.context()["retained_executed_transitions"] == 4
    assert [r for r in memory.context()["events"] if r["type"] == "TRAJECTORY_GAP"] == [
        {"type": "TRAJECTORY_GAP", "after_control": 10, "before_control": 20},
        {"type": "TRAJECTORY_GAP", "after_control": 30, "before_control": 40},
        {"type": "TRAJECTORY_GAP", "after_control": 50, "before_control": 90},
    ]


def test_anchor_uses_floored_target_and_the_interval_ending_at_that_target() -> None:
    """The post-reset target is 60 + floor(540/8) = 127, inclusive at its endpoint."""
    memory = InteractionMemory("anchored", slots=8, first_control=60, last_control=600)
    memory.add(record(1, 60, 127))
    memory.add(record(2, 127, 128))
    memory.add(record(3, 128, 129))
    assert memory.anchors[1] == 0
    assert [row["decision"] for row in memory.selected()] == [1, 3]


def test_unobserved_interval_does_not_fill_a_temporal_anchor() -> None:
    """An anchor in a gap must not be assigned to an unrelated later transition."""
    memory = InteractionMemory("anchored", slots=4, first_control=0, last_control=100)
    memory.add(record(1, 0, 10))
    memory.add(record(2, 30, 40))
    memory.add(record(3, 50, 60))
    assert 1 not in memory.anchors and 2 not in memory.anchors
    assert [row["decision"] for row in memory.selected()] == [1, 3]


def test_recent_comparator_has_the_same_budget_and_retains_initial_evidence() -> None:
    memory = InteractionMemory("recent", slots=4, first_control=0, last_control=100)
    for index in range(10):
        memory.add(record(index, index * 10, (index + 1) * 10))
    assert [r["decision"] for r in memory.selected()] == [0, 7, 8, 9]


def test_failed_preview_does_not_become_an_executed_memory_anchor() -> None:
    memory = InteractionMemory("anchored", slots=4, first_control=0, last_control=100)
    assert not memory.add(record(0, 0, 0))
    assert not memory.records and len(memory.pending) == 1
    memory.add(record(1, 0, 70))
    assert len(memory.selected()) == 1
    assert "preceding_nonmotion" not in memory.selected()[0]
    assert not memory.pending


def test_new_episode_inherits_no_previous_memory() -> None:
    first = InteractionMemory("anchored")
    first.add(record(1, 60, 100))
    assert not InteractionMemory("anchored").selected()
