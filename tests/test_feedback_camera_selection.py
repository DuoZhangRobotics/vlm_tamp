from types import SimpleNamespace
from vlm_tamp.contact_skill_service import feedback_camera_names


def tracker(counts):
    return SimpleNamespace(tracker=SimpleNamespace(references={
        name:{'pixels':[None]*count} for name,count in counts.items()}))


def test_feedback_views_follow_current_visibility_after_motion():
    available={'requested','old_a','old_b','old_c','clear_a','clear_b','clear_c'}
    first=feedback_camera_names(tracker({'old_a':50,'old_b':40,'old_c':30,'clear_a':2}),
        'requested',available)
    assert first==['requested','old_a','old_b','old_c']
    current=feedback_camera_names(tracker({'old_a':1,'old_b':2,'old_c':3,'clear_a':50,'clear_b':40,'clear_c':30}),
        'requested',available,first)
    assert current==['requested','clear_a','clear_b','clear_c']


def test_camera_selection_cannot_use_uncaptured_reference_views():
    value=tracker({'unavailable':100,'requested':20,'recorded_a':30,'recorded_b':40})
    selected=feedback_camera_names(value,'requested',{'requested','recorded_a','recorded_b'})
    assert selected==['requested','recorded_b','recorded_a']
    assert len(selected)==len(set(selected))


def test_existing_view_order_is_preserved_when_membership_is_unchanged():
    previous=['requested','a','b','c'];value=tracker({'a':10,'b':20,'c':30,'requested':100})
    assert feedback_camera_names(value,'requested',set(previous),previous)==previous
