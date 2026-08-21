import numpy as np
import pytest
import torch

from tmm_fast import coh_tmm, inc_tmm

POLARIZATIONS = ['s', 'p']
SOLVERS = ['coherent', 'incoherent']

# Thicknesses are carried in nanometer throughout this module. In meter the layer values are ~1e-7,
# so the default finite difference step of gradcheck would be a 1000 nm perturbation.
EDGE_THICKNESS = torch.full((1, 1), float('inf'), dtype=torch.float64)

# the coherent substack of the incoherent stack below, i.e. layer 2 of four
COHERENT_MASK = [[2]]


def spectrum():
    wl = torch.linspace(500e-9, 700e-9, 3, dtype=torch.float64)
    theta = torch.tensor([0.0, 0.5], dtype=torch.float64)
    return theta, wl


def full_index(inner_n, num_wl):
    """
    Clads the two inner layers with air. Only the inner indices are ever differentiated: the first
    and last layer have to stay real, which check_inputs enforces, so perturbing them is not a
    meaningful direction.
    """
    edge = torch.ones((1, 1, num_wl), dtype=torch.complex128)
    return torch.cat([edge, inner_n, edge], dim=1)


def inner_index(absorbing_layer, num_wl):
    n = [1.46, 2.56]
    n[absorbing_layer] += 0.01j
    return torch.tensor([n], dtype=torch.complex128)[:, :, None].repeat(1, 1, num_wl)


def response(name, inner_n, thickness_nm, pol, theta, wl):
    """Reflectivity of the stack, as a function of the two quantities under test."""
    N = full_index(inner_n, wl.shape[0])
    D = torch.cat([EDGE_THICKNESS, thickness_nm * 1e-9, EDGE_THICKNESS], dim=1)
    if name == 'coherent':
        return coh_tmm(pol, N, D, theta, wl)['R']
    return inc_tmm(pol, N, D, COHERENT_MASK, theta, wl)['R']


def setup(name):
    """The stack each solver is exercised on, and a sensible starting point."""
    theta, wl = spectrum()
    if name == 'coherent':
        inner_n = inner_index(absorbing_layer=1, num_wl=wl.shape[0])
        thickness = torch.tensor([[120.0, 90.0]], dtype=torch.float64)
    else:
        # a thick incoherent layer and a thin coherent one, so both branches carry a gradient
        inner_n = inner_index(absorbing_layer=0, num_wl=wl.shape[0])
        thickness = torch.tensor([[5000.0, 200.0]], dtype=torch.float64)
    return inner_n, thickness, theta, wl


def central_difference(f, x, index, step):
    perturbed = x.clone()
    perturbed[index] += step
    plus = f(perturbed).sum()
    perturbed[index] -= 2 * step
    minus = f(perturbed).sum()
    return ((plus - minus) / (2 * step)).item()


@pytest.mark.parametrize('name', SOLVERS)
@pytest.mark.parametrize('pol', POLARIZATIONS)
def test_thickness_gradient_matches_central_differences(pol, name):
    inner_n, thickness, theta, wl = setup(name)

    def f(t):
        return response(name, inner_n, t, pol, theta, wl)

    tracked = thickness.clone().requires_grad_(True)
    f(tracked).sum().backward()
    assert torch.isfinite(tracked.grad).all(), tracked.grad

    for index in [(0, 0), (0, 1)]:
        numeric = central_difference(f, thickness, index, step=1e-4)
        analytic = tracked.grad[index].item()
        assert abs(analytic) > 0, 'gradient vanished at %s' % (index,)
        assert np.isclose(analytic, numeric, rtol=1e-6, atol=1e-12), (index, analytic, numeric)


@pytest.mark.parametrize('name', SOLVERS)
@pytest.mark.parametrize('pol', POLARIZATIONS)
def test_thickness_gradcheck(pol, name):
    # gradcheck compares the whole Jacobian against finite differences, not just a scalar loss
    inner_n, thickness, theta, wl = setup(name)
    tracked = thickness.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda t: response(name, inner_n, t, pol, theta, wl),
                                    (tracked,), eps=1e-6, atol=1e-8, rtol=1e-4)


@pytest.mark.parametrize('name', SOLVERS)
@pytest.mark.parametrize('pol', POLARIZATIONS)
def test_refractive_index_gradient(pol, name):
    # the in-place clamp of N.imag used to make this raise instead of differentiating, and the
    # infinite edge thicknesses used to leave nan on the edge layers
    inner_n, thickness, theta, wl = setup(name)
    tracked = inner_n.clone().requires_grad_(True)
    response(name, tracked, thickness, pol, theta, wl).sum().backward()
    assert torch.isfinite(tracked.grad).all(), tracked.grad
    assert (tracked.grad.abs().amax(dim=-1) > 0).all(), tracked.grad


@pytest.mark.parametrize('pol', POLARIZATIONS)
def test_refractive_index_gradcheck(pol):
    inner_n, thickness, theta, wl = setup('coherent')
    tracked = inner_n.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda n: response('coherent', n, thickness, pol, theta, wl),
                                    (tracked,), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_refractive_index_gradcheck_incoherent():
    # gradcheck perturbs one wavelength at a time, so this also verifies that the coherent
    # substack receives the wavelength-dependent angle from its absorbing injection layer
    inner_n, thickness, theta, wl = setup('incoherent')
    tracked = inner_n.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda n: response('incoherent', n, thickness, 's', theta, wl),
                                    (tracked,), eps=1e-6, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize('name', SOLVERS)
def test_inputs_are_left_alone(name):
    # converter2torch hands back the caller's own tensor when it is already complex128 on the
    # right device, so anything applied in place would edit their array
    inner_n, thickness, theta, wl = setup(name)
    # a large k exercises the opacity clamp, which is the in-place site that used to write
    # into the caller. The incoherent solver returns nan above about k=10, so it gets less
    opaque = inner_n.clone()
    opaque[0, 0] += 100j if name == 'coherent' else 1j

    N = full_index(opaque, wl.shape[0])
    D = torch.cat([EDGE_THICKNESS, thickness * 1e-9, EDGE_THICKNESS], dim=1)
    N_before, D_before = N.clone(), D.clone()

    if name == 'coherent':
        out = coh_tmm('s', N, D, theta, wl)
    else:
        out = inc_tmm('s', N, D, COHERENT_MASK, theta, wl)

    assert torch.equal(N_before, N), 'the refractive indices were modified'
    assert torch.equal(D_before, D), 'the thicknesses were modified'
    assert torch.isfinite(out['R']).all()


def test_gradient_descent_reduces_the_objective():
    # the point of having gradients: a few steps have to actually improve the coating
    inner_n, thickness, theta, wl = setup('coherent')
    tracked = thickness.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([tracked], lr=2.0)

    def loss():
        return (response('coherent', inner_n, tracked, 's', theta, wl) ** 2).mean()

    before = loss().item()
    for _ in range(50):
        optimizer.zero_grad()
        loss().backward()
        optimizer.step()
    assert loss().item() < before, (before, loss().item())
