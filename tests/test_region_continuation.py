from continue_region_recovery import tracking_history_matches

def item(step,value):return {'control_step':step,'tracking':{'accepted':True,'translation_world_m':[value,0,0]}}

def test_policy_history_matches_without_post_episode_observation():
    actual=[item(10,.01),item(20,.02)]
    assert tracking_history_matches(actual,actual,20)==(True,False)
    assert tracking_history_matches(actual,actual+[item(20,.00001)],20)==(True,True)

def test_guard_does_not_ignore_policy_changes_or_extra_physical_steps():
    actual=[item(10,.01),item(20,.02)]
    assert tracking_history_matches(actual,[item(10,.02),item(20,.02),item(20,0)],20)==(False,True)
    assert tracking_history_matches(actual,actual+[item(21,0)],20)==(False,False)
    assert tracking_history_matches(actual,actual+[item(20,0),item(20,0)],20)==(False,False)
