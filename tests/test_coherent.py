import inspect

import numpy as np
import pytest
import torch

from tmm import coh_tmm

from tmm_fast import coh_tmm as coh_tmm_fast, inc_tmm

POLARIZATIONS = ['s', 'p']


def reference(pol, N, T, theta, wl):
    """R and T from the scalar tmm package, for every stack, angle and wavelength."""
    R = torch.zeros((N.shape[0], theta.shape[0], wl.shape[0]), dtype=torch.double)
    transmission = torch.zeros_like(R)
    for stack in range(N.shape[0]):
        for i, t in enumerate(theta.tolist()):
            for j, w in enumerate(wl.tolist()):
                result = coh_tmm(pol, N[stack][:, j].tolist(), T[stack].tolist(), t, w)
                R[stack, i, j] = result['R']
                transmission[stack, i, j] = result['T']
    return R, transmission


def check_against_reference(pol, N, T, theta, wl, rtol=1e-10, atol=1e-12):
    """
    Compares coh_tmm against the scalar reference, on the GPU too where there is one.

    The tolerances are tight on purpose. Agreement across every configuration in this module is
    6.8e-14 absolute and 5.9e-13 relative, so anything looser stops being a regression test: the
    single precision M_r0 that used to cost seven digits still passed at 1e-6.
    """
    R_reference, T_reference = reference(pol, N, T, theta, wl)
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    for device in devices:
        fast = coh_tmm_fast(pol, N, T, theta, wl, device=device)
        assert fast['R'].shape == R_reference.shape, (device, fast['R'].shape)
        torch.testing.assert_close(R_reference, fast['R'].cpu(), rtol=rtol, atol=atol)
        torch.testing.assert_close(T_reference, fast['T'].cpu(), rtol=rtol, atol=atol)


def random_stacks(num_layers, num_stacks, absorbing=False):
    """A batch of stacks with random indices and thicknesses, seeded to stay reproducible."""
    np.random.seed(111)
    torch.manual_seed(111)
    wl = torch.linspace(400, 1200, 65) * (10**(-9))
    theta = torch.linspace(0, 89, 45) * (np.pi/180)

    M = torch.ones((num_stacks, num_layers, wl.shape[0])).type(torch.complex128)
    for i in range(1, num_layers - 1):
        M[:, i, :] *= np.random.uniform(0, 3, [1])[0]
        if absorbing:
            M[:, i, :] += np.random.uniform(0, 1, [1])[0] * 1j

    max_t = 150 * (10**(-9))
    min_t = 10 * (10**(-9))
    T = (max_t - min_t) * np.random.uniform(0, 1, (num_stacks, num_layers)) + min_t
    T[:, 0] = np.inf
    T[:, -1] = np.inf
    return M, torch.from_numpy(T), theta, wl


@pytest.mark.slow
@pytest.mark.parametrize('pol', POLARIZATIONS)
def test_input_output_medium(pol):
    np.random.seed(111)
    torch.manual_seed(111)
    wl = torch.linspace(400, 1200, 65) * (10**(-9))
    theta = torch.linspace(0, 89, 45) * (np.pi/180)

    # two semi-infinite media and nothing in between
    M = torch.ones((2, 2, wl.shape[0])).type(torch.complex128)
    M[:, 0] = 1.3
    T = np.full((2, 2), np.inf)

    check_against_reference(pol, M, torch.from_numpy(T), theta, wl)


@pytest.mark.slow
@pytest.mark.parametrize('pol', POLARIZATIONS)
def test_basic_coherent_stack(pol):
    M, T, theta, wl = random_stacks(num_layers=8, num_stacks=3)
    check_against_reference(pol, M, T, theta, wl)


@pytest.mark.slow
@pytest.mark.parametrize('pol', POLARIZATIONS)
def test_absorbing_coherent_stack(pol):
    M, T, theta, wl = random_stacks(num_layers=8, num_stacks=3, absorbing=True)
    check_against_reference(pol, M, T, theta, wl)


@pytest.mark.parametrize('pol', POLARIZATIONS)
def test_total_internal_reflection(pol):
    # a dense injection layer radiating into air, with layers in between: past the critical angle
    # of 16.6 deg the exit wave is evanescent, and a lossless stack has to reflect everything
    wl = torch.linspace(500, 900, 5) * (10**(-9))
    theta = torch.deg2rad(torch.tensor([0., 5., 10., 15., 20., 30., 40., 45.], dtype=torch.double))
    M = torch.tensor([3.5, 1.8, 2.4, 1.0], dtype=torch.complex128)[None, :, None].repeat(1, 1, wl.shape[0])
    T = torch.tensor([[np.inf, 120e-9, 80e-9, np.inf]], dtype=torch.double)

    check_against_reference(pol, M, T, theta, wl)

    critical = np.arcsin(1.0 / 3.5)
    beyond = theta > critical
    fast = coh_tmm_fast(pol, M, T, theta, wl)
    torch.testing.assert_close(fast['R'][0][beyond], torch.ones_like(fast['R'][0][beyond]),
                               rtol=0, atol=1e-9)
    assert fast['T'][0][beyond].abs().max() < 1e-12, fast['T'][0][beyond].abs().max()


def dispersionless_reference(pol, n, d, theta, wl):
    R = torch.zeros((theta.shape[0], wl.shape[0]), dtype=torch.double)
    T = torch.zeros((theta.shape[0], wl.shape[0]), dtype=torch.double)
    for i, t in enumerate(theta.tolist()):
        for j, w in enumerate(wl.tolist()):
            res = coh_tmm(pol, n, d, t, w)
            R[i, j] = res['R']
            T[i, j] = res['T']
    return R, T


@pytest.mark.parametrize('pol', POLARIZATIONS)
def test_single_wavelength_single_stack(pol):
    # a single stack of dispersionless materials has neither a stack nor a wavelength axis
    # to pass, which is what tmm_fast is supposed to accept for constant refractive indices
    wl = torch.tensor([600e-9], dtype=torch.double)
    theta = torch.linspace(0, 80, 9) * (np.pi / 180)
    M = torch.tensor([1.0, 1.46, 2.56, 1.52], dtype=torch.complex128)
    T = torch.tensor([np.inf, 120e-9, 90e-9, np.inf], dtype=torch.double)

    O_fast = coh_tmm_fast(pol, M, T, theta, wl)
    assert O_fast['R'].shape == (theta.shape[0], wl.shape[0]), O_fast['R'].shape
    R_tmm, T_tmm = dispersionless_reference(pol, M.tolist(), T.tolist(), theta, wl)
    torch.testing.assert_close(R_tmm, O_fast['R'], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(T_tmm, O_fast['T'], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('pol', POLARIZATIONS)
def test_dispersionless_single_stack(pol):
    # same, but over a spectrum: the constant index still carries no wavelength axis
    wl = torch.linspace(500, 700, 5) * (10**(-9))
    theta = torch.linspace(0, 80, 9) * (np.pi / 180)
    M = torch.tensor([1.0, 1.46, 2.56, 1.52], dtype=torch.complex128)
    T = torch.tensor([np.inf, 120e-9, 90e-9, np.inf], dtype=torch.double)

    O_fast = coh_tmm_fast(pol, M, T, theta, wl)
    assert O_fast['R'].shape == (theta.shape[0], wl.shape[0]), O_fast['R'].shape
    R_tmm, T_tmm = dispersionless_reference(pol, M.tolist(), T.tolist(), theta, wl)
    torch.testing.assert_close(R_tmm, O_fast['R'], rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(T_tmm, O_fast['T'], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('pol', POLARIZATIONS)
def test_dispersionless_multiple_stacks(pol):
    # a batch of dispersionless stacks is [S x L]; the thicknesses being [S x L] is what
    # tells it apart from the [L x W] of a single dispersive stack
    wl = torch.linspace(500, 700, 5) * (10**(-9))
    theta = torch.linspace(0, 80, 5) * (np.pi / 180)
    M = torch.tensor([[1.0, 1.46, 2.56, 1.52],
                      [1.0, 2.20, 1.38, 1.52]], dtype=torch.complex128)
    T = torch.tensor([[np.inf, 120e-9, 90e-9, np.inf],
                      [np.inf, 60e-9, 140e-9, np.inf]], dtype=torch.double)

    O_fast = coh_tmm_fast(pol, M, T, theta, wl)
    assert O_fast['R'].shape == (M.shape[0], theta.shape[0], wl.shape[0]), O_fast['R'].shape
    for stack in range(M.shape[0]):
        R_tmm, T_tmm = dispersionless_reference(pol, M[stack].tolist(), T[stack].tolist(), theta, wl)
        torch.testing.assert_close(R_tmm, O_fast['R'][stack], rtol=1e-10, atol=1e-12)
        torch.testing.assert_close(T_tmm, O_fast['T'][stack], rtol=1e-10, atol=1e-12)


def test_mixed_inputs_and_scalar_angle():
    wl = np.linspace(500e-9, 700e-9, 3)
    theta = 0.3
    N = torch.tensor([1.0, 2.1 + 0.02j, 1.5], dtype=torch.complex128)
    T = [np.inf, 120e-9, np.inf]

    mixed = coh_tmm_fast('s', N, T, theta, wl)
    expected = coh_tmm_fast(
        's', N, torch.tensor(T), torch.tensor([theta]), torch.from_numpy(wl)
    )

    assert isinstance(mixed['R'], torch.Tensor)
    assert mixed['R'].shape == (1, wl.size)
    torch.testing.assert_close(mixed['R'], expected['R'])
    torch.testing.assert_close(mixed['T'], expected['T'])


def test_plain_lists_and_scalar_wavelength_return_numpy():
    result = coh_tmm_fast(
        'p', [1.0, 1.8, 1.5], [np.inf, 100e-9, np.inf], 0.0, 600e-9
    )

    assert isinstance(result['R'], np.ndarray)
    assert result['R'].shape == (1, 1)
    assert np.isfinite(result['R']).all()
    assert np.isfinite(result['T']).all()


def test_device_is_inferred_from_refractive_indices():
    assert inspect.signature(coh_tmm_fast).parameters['device'].default is None
    assert inspect.signature(inc_tmm).parameters['device'].default is None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = torch.tensor([1.0, 1.8, 1.5], dtype=torch.complex128, device=device)
    result = coh_tmm_fast('s', N, [np.inf, 100e-9, np.inf], 0.0, 600e-9)
    assert result['R'].device == device


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
