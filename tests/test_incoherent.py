import numpy as np
import pytest
import torch

from tmm import inc_tmm

from tmm_fast import inc_tmm as inc_tmm_fast

POLARIZATIONS = ['s', 'p']

# every test here loops the scalar reference over a full angle x wavelength grid
pytestmark = pytest.mark.slow


def grid(n_wl, n_theta, max_angle):
    wl = torch.linspace(400, 1200, n_wl) * (10**(-9))
    theta = torch.linspace(0, max_angle, n_theta) * (np.pi/180)
    return wl, theta


def alternating_stack(num_layers, num_stacks, wl, k_odd=0.0, k_even=0.0):
    """A batch of identical stacks alternating between n=1.46 and n=2.56, optionally absorbing."""
    M = torch.ones((num_stacks, num_layers, wl.shape[0])).type(torch.complex128)
    for i in range(1, num_layers - 1):
        if i % 2 == 1:
            M[:, i, :] *= 1.46
            M[:, i, :] += k_odd * 1j
        else:
            M[:, i, :] *= 2.56
            M[:, i, :] += k_even * 1j
    return M


def thicknesses(values, num_stacks):
    return torch.tensor([list(values)] * num_stacks, dtype=torch.double)


def reference(pol, N, T, imask, theta, wl):
    """R and T from the scalar tmm package; the stacks of a batch are identical here."""
    R = torch.zeros((theta.shape[0], wl.shape[0]), dtype=torch.double)
    transmission = torch.zeros((theta.shape[0], wl.shape[0]), dtype=torch.double)
    thickness = T[0].tolist()
    for i, t in enumerate(theta.tolist()):
        for j, w in enumerate(wl.tolist()):
            result = inc_tmm(pol, N[0][:, j].tolist(), thickness, imask, t, w)
            R[i, j] = result['R']
            transmission[i, j] = result['T']
    return R, transmission


def check_against_reference(pol, N, T, mask, imask, theta, wl, numpy_input=False,
                            check_cuda=True, rtol=1e-10, atol=1e-12):
    """
    Compares inc_tmm against the scalar reference for every stack of the batch.

    The tolerances used to be 1e-5, which is what hid the single precision the whole path ran
    in. Agreement is now 8.3e-15 at worst across these configurations.
    """
    R_reference, T_reference = reference(pol, N, T, imask, theta, wl)
    devices = ['cpu']
    if check_cuda and torch.cuda.is_available():
        devices.append('cuda')

    for device in devices:
        if numpy_input:
            fast = inc_tmm_fast(pol, N.numpy(), T.numpy(), mask, theta, wl, device=device)
        else:
            fast = inc_tmm_fast(pol, N, T, mask, theta, wl, device=device)
        R = torch.as_tensor(fast['R']).cpu()
        transmission = torch.as_tensor(fast['T']).cpu()
        assert R.shape == (N.shape[0],) + R_reference.shape, (device, R.shape)

        for stack in range(N.shape[0]):
            assert (R[stack].isnan() == R_reference.isnan()).all(), (device, stack)
            assert (transmission[stack].isnan() == T_reference.isnan()).all(), (device, stack)
            torch.testing.assert_close(R_reference, R[stack], rtol=rtol, atol=atol, equal_nan=True)
            torch.testing.assert_close(T_reference, transmission[stack], rtol=rtol, atol=atol, equal_nan=True)


@pytest.mark.parametrize('pol', POLARIZATIONS)
def test_incoherent_input_output_medium(pol):
    wl, theta = grid(n_wl=65, n_theta=45, max_angle=89)
    M = alternating_stack(num_layers=2, num_stacks=2, wl=wl)
    T = thicknesses([np.inf, np.inf], num_stacks=2)
    check_against_reference(pol, M, T, [], ['i', 'i'], theta, wl)


@pytest.mark.parametrize('pol', POLARIZATIONS)
def test_incoherent_numpy_input_output_medium(pol):
    wl, theta = grid(n_wl=65, n_theta=45, max_angle=89)
    M = alternating_stack(num_layers=2, num_stacks=2, wl=wl)
    T = thicknesses([np.inf, np.inf], num_stacks=2)
    check_against_reference(pol, M, T, [], ['i', 'i'], theta, wl,
                            numpy_input=True, check_cuda=False)


@pytest.mark.parametrize('pol', POLARIZATIONS)
def test_fully_incoherent_stack(pol):
    wl, theta = grid(n_wl=65, n_theta=45, max_angle=89)
    M = alternating_stack(num_layers=5, num_stacks=2, wl=wl)
    T = thicknesses([np.inf, 10000e-9, 2000e-9, 5000e-9, np.inf], num_stacks=2)
    check_against_reference(pol, M, T, [], ['i'] * 5, theta, wl)


@pytest.mark.parametrize('pol', POLARIZATIONS)
def test_coherent_stack_with_incoherent_surrounding(pol):
    wl, theta = grid(n_wl=20, n_theta=45, max_angle=85)
    M = alternating_stack(num_layers=5, num_stacks=2, wl=wl)
    T = thicknesses([np.inf, 200e-9, 100e-9, 300e-9, np.inf], num_stacks=2)
    check_against_reference(pol, M, T, [[1, 2, 3]], ['i', 'c', 'c', 'c', 'i'], theta, wl)


@pytest.mark.parametrize('pol', POLARIZATIONS)
def test_coherent_incoherent(pol):
    wl, theta = grid(n_wl=20, n_theta=45, max_angle=89)
    M = alternating_stack(num_layers=5, num_stacks=2, wl=wl)
    T = thicknesses([np.inf, 10000e-9, 100e-9, 300e-9, np.inf], num_stacks=2)
    check_against_reference(pol, M, T, [[2, 3]], ['i', 'i', 'c', 'c', 'i'], theta, wl)


@pytest.mark.parametrize('pol', POLARIZATIONS)
def test_absorbing_fully_incoherent(pol):
    wl, theta = grid(n_wl=65, n_theta=45, max_angle=89)
    M = alternating_stack(num_layers=5, num_stacks=2, wl=wl, k_odd=.005, k_even=.002)
    T = thicknesses([np.inf, 10000e-9, 2000e-9, 5000e-9, np.inf], num_stacks=2)
    check_against_reference(pol, M, T, [], ['i'] * 5, theta, wl)


@pytest.mark.parametrize('pol', POLARIZATIONS)
def test_absorbing_coherent_incoherent(pol):
    wl, theta = grid(n_wl=20, n_theta=45, max_angle=85)
    M = alternating_stack(num_layers=5, num_stacks=2, wl=wl, k_odd=.0005, k_even=.002)
    T = thicknesses([np.inf, 10000e-9, 100e-9, 300e-9, np.inf], num_stacks=2)
    check_against_reference(pol, M, T, [[2, 3]], ['i', 'i', 'c', 'c', 'i'], theta, wl)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
