import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from vlm_tamp.subscription_scene_variants import transform_pose,apply_scene_variant
from vlm_tamp.subscription_agent import codex_arguments

ROOT=Path(__file__).resolve().parents[1]

def test_rigid_scene_transform_preserves_support_and_local_orientation():
    r=Rotation.from_euler('z',12,degrees=True);origin=np.array([.1,-.2,.9]);shift=np.array([.04,.02,0.])
    p,q=transform_pose(origin+[.03,.02,.2],[1,0,0,0],origin,r,shift)
    recovered=r.inv().apply(p-origin-shift)
    np.testing.assert_allclose(recovered,[.03,.02,.2],atol=1e-12)
    np.testing.assert_allclose(Rotation.from_quat(q[[1,2,3,0]]).as_matrix(),r.as_matrix())

def test_nominal_variant_never_reads_hidden_geometry():
    policy={'position_world_m':[0,0,1]}
    record,camera=apply_scene_variant(None,'nominal',policy)
    assert record['changed_bodies']==[] and camera==policy and camera is not policy
    with pytest.raises(ValueError):apply_scene_variant(None,'agent-invented',policy)

def test_consolidation_corpus_readonly_profile(tmp_path):
    args=codex_arguments(tmp_path,tmp_path/'out',readonly_paths=[tmp_path/'corpus'])
    fs=next(x for x in args if x.startswith('permissions.research-agent.filesystem='))
    assert json.dumps(str(tmp_path/'corpus'))+'="read"' in fs
    assert 'forced_login_method="chatgpt"' in args

def test_guide_rejects_nonexistent_tool_evidence(tmp_path):
    spec=importlib.util.spec_from_file_location('consolidator',ROOT/'scripts/consolidate_subscription_experience.py')
    import sys
    sys.path.insert(0,str(ROOT/'scripts'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    corpus=tmp_path/'corpus';(corpus/'case').mkdir(parents=True)
    (corpus/'case/tool-events.jsonl').write_text('{"sequence": 3}\n')
    guide=tmp_path/'guide.json';rule=dict(id='a',condition='x',guideline='y',origin='z',limitations='uncertain',evidence=[{'case':'case','events':[9],'claim':'x'}])
    guide.write_text(json.dumps({'event_id_convention':'sequence','guidelines':[rule]}))
    with pytest.raises(ValueError,match='Missing cited event'):m.validate_guide(guide,corpus)
    rule['evidence'][0]['events']=[3];guide.write_text(json.dumps({'event_id_convention':'sequence','guidelines':[rule]}))
    assert m.validate_guide(guide,corpus)['evidence_references_exist']
