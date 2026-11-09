"""Native GPU queries for shared point geometry and contact permissions.

Run in the cuRobo environment. These exercise public collision queries rather
than calling Warp-decorated functions directly.
"""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('curobo')
if not torch.cuda.is_available():
    pytest.skip('A CUDA device is required for native collision queries', allow_module_level=True)

from curobo.types import DeviceCfg
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from vlm_tamp.rekep_libero.curobo_scene_points import ObservedPointScene


@pytest.fixture
def native_query():
    import warp as wp
    wp.init()
    device = DeviceCfg()
    scene = ObservedPointScene(device, list(range(59)), 8)
    spheres = torch.zeros((1, 1, 59, 4), device=device.device)
    spheres[..., :3] = 10.
    spheres[..., 3] = .024
    buffer = CollisionBuffer.from_shape(spheres.shape, device)
    weight, activation = device.to_device([1.]), device.to_device([0.])

    def query():
        return scene.get_sphere_distance(SimpleNamespace(robot_spheres=spheres), buffer, weight, activation)

    return scene, spheres, query


def clouds(role, points):
    return {name: np.asarray(points, np.float32).reshape(-1, 3) if name == role else np.empty((0, 3), np.float32)
            for name in ('stationary', 'moving', 'contact')}


@pytest.mark.parametrize('role', ['stationary', 'moving', 'contact'])
@pytest.mark.parametrize('sphere', [0, 14, 52, 56])
def test_same_point_radius_and_finger_permissions(native_query, role, sphere):
    scene, spheres, query = native_query
    spheres[0, 0, sphere, :3] = torch.tensor([.02, 0., 0.], device=spheres.device)
    scene.update(clouds(role, [[0., 0., 0.]]), palm_contact=False)
    result = query().detach().cpu().numpy()
    assert result.max() == pytest.approx(.004 if 1 <= sphere < 55 else 0., abs=1e-7)


def test_palm_permission_and_target_updates_survive_cuda_graph(native_query):
    scene, spheres, query = native_query
    spheres[0, 0, 52, :3] = torch.tensor([.02, 0., 0.], device=spheres.device)
    points = clouds('contact', [[0., 0., 0.]])
    scene.update(points, palm_contact=False)
    for _ in range(3):query()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = query()
    graph.replay()
    assert result.max().item() == pytest.approx(.004, abs=1e-7)
    scene.update(points, palm_contact=True)
    graph.replay()
    assert result.max().item() == 0.
    scene.update(points, palm_contact=False)
    transform = np.eye(4);transform[0, 3] = .1
    scene.move_target(transform)
    graph.replay()
    assert result.max().item() == 0.
    scene.update(clouds('stationary', [[0., 0., 0.]]), palm_contact=True)
    scene.move_target(transform)
    graph.replay()
    assert result.max().item() == pytest.approx(.004, abs=1e-7)


def test_native_distance_gradient_matches_point_geometry(native_query):
    scene, spheres, query = native_query
    spheres[0, 0, 14, :3] = torch.tensor([.01, .01, 0.], device=spheres.device)
    spheres.requires_grad_(True)
    scene.update(clouds('moving', [[0., 0., 0.]]), palm_contact=False)
    result = query()
    result.sum().backward()
    assert result.max().item() == pytest.approx(.024-np.sqrt(.0002), abs=1e-7)
    np.testing.assert_allclose(spheres.grad[0, 0, 14, :3].cpu().numpy(),
                               [-np.sqrt(.5), -np.sqrt(.5), 0.], atol=1e-6)


@pytest.mark.parametrize('role', ['stationary', 'moving', 'contact'])
@pytest.mark.parametrize('finger', [0, 1, 2, 3])
def test_target_fingertip_permission_survives_cuda_graph(native_query, role, finger):
    """Distal spheres may touch only target points, with warm query masks updated in place."""
    scene, spheres, query = native_query
    spheres[0, 0, 37, :3] = torch.tensor([10., 0, 0], device=spheres.device)
    spheres[0, 0, 40, :3] = torch.tensor([11., 0, 0], device=spheres.device)
    spheres[0, 0, 45, :3] = torch.tensor([10., 1, 0], device=spheres.device)
    coefficients = [[0., 0., 0., .024] for _ in range(4)]
    coefficients[finger] = [-9.98, 0., 0., .024]
    model = {'coefficients': coefficients}
    points = clouds(role, [[0., 0., 0.]])
    scene.update(points, palm_contact=True, finger_model=model)
    for _ in range(3):
        query()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = query()
    graph.replay()
    assert result.max().item() == pytest.approx(.004, abs=1e-6)
    scene.update(points, palm_contact=True, finger_model=model, fingertip_contact=True)
    graph.replay()
    exempt = role == 'contact' and finger in (1, 3)
    assert result.max().item() == pytest.approx(0. if exempt else .004, abs=1e-6)
    scene.update(points, palm_contact=True, finger_model=model, fingertip_contact=False)
    graph.replay()
    assert result.max().item() == pytest.approx(.004, abs=1e-6)


@pytest.mark.parametrize('role',['stationary','moving','contact'])
def test_finger_phase_and_measured_geometry_updates_survive_warm_cuda_graph(native_query,role):
    """A warm optimizer sees changed finger permissions and aperture, without recreation."""
    scene,spheres,query=native_query
    spheres[0,0,37,:3]=torch.tensor([.02,0,0],device=spheres.device)
    spheres[0,0,40,:3]=torch.tensor([.02,1,0],device=spheres.device)
    spheres[0,0,45,:3]=torch.tensor([.02,0,1],device=spheres.device)
    spheres[0,0,[37,40,45],3]=.001
    model={'coefficients':[[0,0,0,.024],[0,0,0,.024],[1,1,0,.024],[1,1,0,.024]]}
    points=clouds(role,[[0.,0.,0.]])
    scene.update(points,palm_contact=False,finger_model=model)
    for _ in range(3):query()
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):result=query()
    graph.replay()
    assert result.max().item()==pytest.approx(.004,abs=1e-7)
    scene.update(points,palm_contact=False)
    graph.replay();assert result.max().item()==0.
    scene.update(points,palm_contact=False,finger_model=model)
    graph.replay();assert result.max().item()==pytest.approx(.004,abs=1e-7)
    model['coefficients'][0][0]=1.;model['coefficients'][1][0]=1.
    scene.update(points,palm_contact=False,finger_model=model)
    graph.replay();assert result.max().item()==0.
