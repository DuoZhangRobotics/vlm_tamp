import os

import pytest

from vlm_tamp.dataset import validate_dataset
from vlm_tamp.libero import collect


@pytest.mark.integration
def test_official_task_renders_reproducibly(tmp_path):
    if os.environ.get("RUN_LIBERO_INTEGRATION") != "1":
        pytest.skip("Set RUN_LIBERO_INTEGRATION=1 for real rendering")
    a = collect(tmp_path / "first", tasks=[0], states=[0], gl="egl")
    b = collect(tmp_path / "second", tasks=[0], states=[0], gl="egl")
    assert a[0]["reference"]["goal_predicates"] == b[0]["reference"]["goal_predicates"]
    eid = a[0]["example_id"]
    assert (tmp_path / "first" / eid / "rgb.png").read_bytes() == (
        tmp_path / "second" / eid / "rgb.png"
    ).read_bytes()
    assert validate_dataset(tmp_path / "first")["visible_instance_masks"] > 0


@pytest.mark.integration
def test_real_drawer_part_reference(tmp_path):
    if os.environ.get("RUN_LIBERO_INTEGRATION") != "1":
        pytest.skip("Set RUN_LIBERO_INTEGRATION=1 for simulator reference checks")
    from vlm_tamp.references import generate_simulator_references, check_reference
    from vlm_tamp.dataset import read_records

    collect(tmp_path / "data", tasks=[0], states=[0], gl="egl")
    refs = generate_simulator_references(tmp_path / "data", tmp_path / "references")
    records, root = read_records(tmp_path / "data")
    check_reference(records[0], root, refs[0])
    assert len(refs[0].parts) == 3
    assert all(p.joint_type == "prismatic" and p.visible_pixels > 0 for p in refs[0].parts)
    assert sum(p.goal_related for p in refs[0].parts) == 1
