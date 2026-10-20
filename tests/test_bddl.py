import pytest

from vlm_tamp.bddl import goal_atoms, normalize_predicate, parse_bddl, parse_sexpr
from vlm_tamp.libero import ROOT


def test_exact_vocabulary_order_and_logic(tmp_path):
    path = tmp_path / "task.bddl"
    path.write_text("""(define (problem p) (:language "Put cup inside cabinet")
       (:objects cup - mug cabinet - cabinet)
       (:goal (And (In cup cabinet) (Open cabinet))))""")
    parsed = parse_bddl(path)
    assert parsed["goal_expression"] == ["And", ["In", "cup", "cabinet"], ["Open", "cabinet"]]
    assert parsed["goal_predicates"][0]["arguments"] == ["cup", "cabinet"]
    assert parsed["goal_predicates"][0]["name"] == "In"
    assert parsed["goal_predicates"][0]["normalized_name"] == "in"
    assert parsed["instruction"] == "Put cup inside cabinet"
    assert normalize_predicate("Contains") == "Contains"
    assert normalize_predicate("On") != normalize_predicate("In")


@pytest.mark.parametrize(
    "goal",
    [
        ["Or", ["On", "a", "b"], ["In", "a", "c"]],
        ["Not", ["Open", "a"]],
        ["forall", ["?x", "-", "object"], ["Open", "?x"]],
    ],
)
def test_unsupported_logic_is_preserved(goal):
    atoms, unsupported = goal_atoms(goal)
    assert atoms == [] and unsupported == [goal]


@pytest.mark.parametrize("text", ["(define", "(define))", "(a)(b)", "a"])
def test_reject_malformed(text):
    with pytest.raises(ValueError):
        parse_sexpr(text)


def test_actual_libero_files_and_official_parser():
    base = ROOT / "third_party/LIBERO/libero/libero/bddl_files"
    if not base.exists():
        pytest.skip("Official checkout not installed; run scripts/setup.sh")
    # This imports only upstream BDDL parsing, not an environment or renderer.
    import importlib.util

    path = base.parent / "envs/bddl_utils.py"
    spec = importlib.util.spec_from_file_location("upstream_bddl_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tasks = list(base.glob("*/*.bddl"))
    assert len(tasks) >= 100
    for path in tasks:
        parsed = parse_bddl(path)
        official = module.robosuite_parse_problem(str(path))
        assert parsed["goal_expression"] == _restore_case(parsed["goal_expression"], official["goal_state"])
        assert [p["normalized_name"] for p in parsed["goal_predicates"]] == [
            normalize_predicate(p[0]) for p in official["goal_state"]
        ]
    known = parse_bddl(base / "libero_goal/open_the_middle_drawer_of_the_cabinet.bddl")
    assert known["goal_expression"] == ["And", ["Open", "wooden_cabinet_1_middle_region"]]


def _restore_case(expression, official):
    # Official parser lowercases atoms and flattens positive conjunctions.
    leaves, unsupported = goal_atoms(expression)
    assert not unsupported
    assert [[p["name"].lower(), *[a.lower() for a in p["arguments"]]] for p in leaves] == official
    return expression
