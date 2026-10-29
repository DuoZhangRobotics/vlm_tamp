"""Adapter for the pinned upstream ArtiPoint factor-graph estimator.

This benchmarks the authors' estimation core on shared measured 3D tracks. It
does not claim to reproduce their full hand-segmentation/tracking pipeline.
"""
from pathlib import Path
from functools import lru_cache
import importlib.util
import sys
import time
import types

import numpy as np

from .part_motion import pair_points


@lru_cache(maxsize=1)
def factor_graph_class():
    """Load the unmodified core without executing the full-pipeline initializer.

    Upstream __init__ eagerly imports human segmentation and dataset adapters.
    Temporary namespaces keep those unrelated dependencies out of this core-only
    comparison. The original modules and their defaults are not patched.
    """
    root=Path(__file__).resolve().parents[2]
    source=root/'.runtime/artipoint/artipoint';deps=root/'.runtime/artipoint-deps'
    if not source.is_dir() or not deps.is_dir():raise ValueError('ArtiPoint estimation runtime is unavailable')
    sys.path.insert(0,str(deps))
    names=['artipoint','artipoint.utils','artipoint.utils.articulation_helper']
    previous={name:sys.modules.get(name) for name in names}
    try:
        for name,folder in [('artipoint',source),('artipoint.utils',source/'utils')]:
            package=types.ModuleType(name);package.__path__=[str(folder)];sys.modules[name]=package
        helper_spec=importlib.util.spec_from_file_location('artipoint.utils.articulation_helper',source/'utils/articulation_helper.py')
        helper=importlib.util.module_from_spec(helper_spec);sys.modules[names[-1]]=helper;helper_spec.loader.exec_module(helper)
        core_spec=importlib.util.spec_from_file_location('_vlm_artipoint_factor_graph',source/'factor_graph/pose_est.py')
        core=importlib.util.module_from_spec(core_spec);core_spec.loader.exec_module(core)
        return core.PoseEstFactorGraph
    finally:
        for name,value in previous.items():
            if value is None:sys.modules.pop(name,None)
            else:sys.modules[name]=value


def estimate_from_tracks(world,valid,intervals):
    started=time.perf_counter()
    PoseEstFactorGraph=factor_graph_class()
    loaded=time.perf_counter();pairs=[];counts=[]
    for first,second in intervals:
        a,b=pair_points(world,valid,first,second)
        displacement=np.linalg.norm(b-a,axis=1)
        good=(displacement>.0015)&(displacement<.15)
        pair=np.stack([a[good],b[good]],axis=1);counts.append(len(pair))
        if len(pair)<6:
            return {'status':'insufficient_evidence','reason':'insufficient_visible_moving_pairs','pair_counts':counts,
                    'backend':'artipoint_factor_graph','runtime_seconds':time.perf_counter()-started}
        pairs.append(pair)
    graph=PoseEstFactorGraph(len(pairs))
    for i,pair in enumerate(pairs):
        graph.add_point_pair_factor(pair,i)
        graph.add_articulation_factor(i,theta_prior=None)
        graph.add_world_frame_factor(i)
    if not graph.solve_factor_graph():
        return {'status':'solver_failed','backend':'artipoint_factor_graph','pair_counts':counts,
                'runtime_seconds':time.perf_counter()-started}
    raw=graph.estimate_joint_type_and_parameters()
    axis=np.asarray(raw.get('axis'),float);origin=np.asarray(raw.get('q',raw.get('center')),float)
    if axis.shape!=(3,) or origin.shape!=(3,) or not np.isfinite(axis).all() or not np.isfinite(origin).all() or np.linalg.norm(axis)<1e-9:
        return {'status':'invalid_estimate','reason':'nonfinite_or_invalid_axis','backend':'artipoint_factor_graph',
                'pair_counts':counts,'runtime_seconds':time.perf_counter()-started}
    return {'status':'estimate_returned','backend':'artipoint_factor_graph','joint_type_candidate':raw['type'],
        'joint_origin_world_m':origin.tolist(),'axis_world_unsigned':(axis/np.linalg.norm(axis)).tolist(),
        'twist':np.asarray(raw['twist']).tolist(),'thetas':np.asarray(raw['thetas']).tolist(),
        'pitch':float(raw['pitch']),'pair_counts':counts,'old_hinge_prior_used':False,
        'import_seconds':loaded-started,'estimation_seconds':time.perf_counter()-loaded,
        'runtime_seconds':time.perf_counter()-started,
        'scope':'Unmodified upstream factor-graph core, author defaults and theta_prior=None; common visible measured tracks with 1.5mm..15cm displacement. No automatic planner admission or semantic correctness claim.'}
