"""Separate valid selections, unresolved numerical geometry and unscored replies."""

import copy
import sys

from vlm_tamp.libero import ROOT

sys.path.insert(0, str(ROOT / 'scripts'))
from report_drawer_geometry_stress import aggregate  # noqa: E402


def case(geometry='estimated'):
    resolved = geometry == 'estimated'
    return {'status': 'valid',
            'response_context': {'status': 'completed', 'selection_scored': True},
            'metrics': {'role_binding_correct': True, 'front_plane_estimated': resolved,
                        'two_point_geometry_criterion': resolved},
            'raw_row': {'attempts': 1, 'cost_usd': .001,
                        'numericals': {'drawer_front_plane': {'status': geometry}}}}


def test_completed_valid_selection_with_unresolved_geometry_remains_scored():
    result = aggregate([case('unresolved')])
    assert result['scored_selection_cases'] == 1
    assert result['scored_geometry_unresolved_cases'] == 1
    assert result['boolean_metric_counts']['role_binding_correct'] == 1
    assert result['boolean_metric_counts']['two_point_geometry_criterion'] == 0
    assert result['geometry_status_counts'] == {'unresolved': 1}


def test_truncated_form_does_not_become_a_visual_selection_failure():
    truncated = case()
    truncated.update(status='response_truncated',
                     response_context={'status': 'incomplete', 'selection_scored': False})
    truncated['raw_row']['numericals'] = None
    result = aggregate([case(), truncated])
    assert result['planned_cases'] == 2 and result['scored_selection_cases'] == 1
    assert result['boolean_metric_counts']['role_binding_correct'] == 1
    assert result['unscored_status_counts'] == {'response_truncated': 1}
    assert result['geometry_status_counts'] == {'estimated': 1, 'unavailable': 1}


def test_unrequested_case_keeps_planned_denominator_without_usage():
    skipped = case()
    skipped.update(status='not_run',
                   response_context={'status': 'not_run', 'selection_scored': False},
                   raw_row={'attempts': 0, 'cost_usd': 0, 'numericals': None})
    result = aggregate([case(), skipped])
    assert result['planned_cases'] == 2 and result['api_attempts'] == 1
    assert result['estimated_usage_cost_usd'] == .001
    assert result['boolean_metric_counts']['two_point_geometry_criterion'] == 1


def test_bound_selection_and_correct_axis_do_not_imply_pure_support():
    selected = case()
    selected['metrics'].update(axis_within_tolerance=True,
                              support_purity_above_diagnostic_threshold=False,
                              two_point_geometry_and_reference_pure_support=False)
    value = copy.deepcopy(selected)
    result = aggregate([value])
    assert result['boolean_metric_counts']['two_point_geometry_criterion'] == 1
    assert result['boolean_metric_counts']['two_point_geometry_and_reference_pure_support'] == 0
