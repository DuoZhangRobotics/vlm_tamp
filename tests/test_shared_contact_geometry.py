import numpy as np
from vlm_tamp.shared_contact_geometry import upright_plane_candidates,closing_target


def test_large_table_does_not_define_an_upright_drawer_axis():
    x,y=np.meshgrid(np.linspace(-.2,.2,70),np.linspace(-.3,.1,70))
    table=np.c_[x.ravel(),y.ravel(),np.full(x.size,.9)]
    x,z=np.meshgrid(np.linspace(-.09,.09,30),np.linspace(.92,1.04,30))
    face=np.c_[x.ravel(),np.full(x.size,-.2),z.ravel()]
    plane=upright_plane_candidates(np.vstack([table,face]),[0,1,1.2],np.array([0,-.175,.97]))[0]
    assert plane['normal']@np.array([0,1,0])>.999
    assert plane['support']>=len(face)


def test_table_only_region_does_not_invent_horizontal_axis():
    x,y=np.meshgrid(np.linspace(-.2,.2,40),np.linspace(-.3,.1,40))
    table=np.c_[x.ravel(),y.ravel(),np.full(x.size,.9)]
    assert upright_plane_candidates(table,[0,1,1.2],np.array([0,-.175,.95]))==[]


def test_closing_target_can_use_cabinet_above_bottom_drawer(monkeypatch):
    from vlm_tamp import shared_contact_geometry as module
    x,z=np.meshgrid(np.linspace(-.12,.12,80),np.linspace(.05,.25,80))
    cabinet=np.c_[x.ravel(),np.full(x.size,-.15),z.ravel()]
    x,y=np.meshgrid(np.linspace(-.15,.15,80),np.linspace(-.24,-.03,80))
    table=np.c_[x.ravel(),y.ravel(),np.full(x.size,-.05)]
    monkeypatch.setattr(module,'observed_points',lambda *a,**k:(np.vstack([cabinet,table]),{}))
    spec={'target_progress':.12,'front_plane':{'point_world_m':[0,0,0]},'panel_normal_world':[0,1,0]}
    out=closing_target({}, {},spec,None,{'robot_self_depth_tolerance_m':.001})
    assert abs(out['target_progress']-.16)<.002
    assert out['closing_target_estimate']['support_vertical_span_m']>.15


def test_contact_seed_keeps_handle_and_discards_disconnected_table():
    from vlm_tamp.shared_contact_geometry import seed_component
    handle=np.c_[np.linspace(-.04,.04,100),np.zeros(100),np.full(100,.95)]
    table=handle.copy();table[:,2]=.9
    selected=seed_component(np.vstack([handle,table]),[0,0,.95])
    assert len(selected)==100 and np.all(selected[:,2]==.95)
