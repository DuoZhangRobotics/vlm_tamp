"""Bounded complete transitions, provenance, and explicit temporal gaps."""

from vlm_tamp.interaction_memory import InteractionMemory


def record(index: int, before: int, after: int) -> dict:
    """A measured transition independent of any task or object."""
    return {'decision':index,'before_control':before,'after_control':after,
            'before_image':f'before-{index}.png','after_image':f'after-{index}.png',
            'operation':'move','receipt':{'ok':True}}


def test_anchors_retain_earlier_transitions_and_latest_without_duplicate_images() -> None:
    memory=InteractionMemory('anchored',slots=4,first_control=0,last_control=100)
    for index in range(10):memory.add(record(index,index*10,(index+1)*10))
    assert [r['decision'] for r in memory.selected()]==[0,2,5,9]
    assert memory.context()['retained_executed_transitions']==4
    assert [r for r in memory.context()['events'] if r['type']=='TRAJECTORY_GAP']==[
        {'type':'TRAJECTORY_GAP','after_control':10,'before_control':20},
        {'type':'TRAJECTORY_GAP','after_control':30,'before_control':50},
        {'type':'TRAJECTORY_GAP','after_control':60,'before_control':90}]


def test_recent_comparator_has_the_same_budget_and_retains_initial_evidence() -> None:
    memory=InteractionMemory('recent',slots=4,first_control=0,last_control=100)
    for index in range(10):memory.add(record(index,index*10,(index+1)*10))
    assert [r['decision'] for r in memory.selected()]==[0,7,8,9]


def test_failed_preview_does_not_become_an_executed_memory_anchor() -> None:
    memory=InteractionMemory('anchored',slots=4,first_control=0,last_control=100)
    assert not memory.add(record(0,0,0))
    assert not memory.records and len(memory.pending)==1
    memory.add(record(1,0,70))
    assert len(memory.selected())==1
    assert memory.selected()[0]['preceding_nonmotion'][0]['decision']==0
    assert not memory.pending


def test_new_episode_inherits_no_previous_memory() -> None:
    first=InteractionMemory('anchored');first.add(record(1,60,100))
    assert not InteractionMemory('anchored').selected()
