"""The intervention changes instructions, not evidence or robot configuration."""
import importlib.util
from pathlib import Path


def study_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/run_verification_prompt_study.py'
    spec = importlib.util.spec_from_file_location('verification_study_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scaffold_appends_to_both_original_prompt_types_without_a_task_solution():
    module = study_module()
    scaffold = (module.ROOT / 'prompts/generic_result_verification.txt').read_text()
    for base in [module.PROMPT, (module.ROOT / 'prompts/recovery_learning_transfer.txt').read_text()]:
        augmented = module.augment(base, scaffold)
        assert augmented.startswith(base.rstrip() + '\n\n')
        assert augmented[len(base.rstrip()) + 2:] == scaffold.strip() + '\n'
    for forbidden in ['microwave', 'hinge', 'handle', 'regrasp', 'state045', 'ground(', 'degrees']:
        assert forbidden not in scaffold.lower()
    assert not any(character.isdigit() for character in scaffold)


def test_physical_schedule_balances_condition_order_without_retries():
    module = study_module()
    groups = {state: [job['condition'] for job in module.PHYSICAL if job['state'] == state]
              for state in [45, 0]}
    assert groups == {45: ['baseline', 'verification'], 0: ['verification', 'baseline']}
    assert len({row['case_id'] for row in module.PHYSICAL}) == 4
    assert [row[2] for row in module.CASES].count('calibration') == 2
    assert [row[2] for row in module.CASES].count('additional_development') == 4
