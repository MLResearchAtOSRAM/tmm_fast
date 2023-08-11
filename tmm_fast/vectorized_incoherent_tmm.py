import torch
import numpy as np
from .vectorized_tmm_dispersive_multistack import coh_vec_tmm_disp_mstack as coh_tmm
from .vectorized_tmm_dispersive_multistack import (
    SnellLaw_vectorized,
    interface_r_vec,
    interface_t_vec,
    T_from_t_vec,
    R_from_r_vec,
)

from typing import Union


def inc_vec_tmm_disp_lstack(
    pol: str,
    N: torch.Tensor,
    L: torch.Tensor,
    mask: list,
    theta: Union[np.ndarray, torch.Tensor],
    lambda_vacuum: Union[np.ndarray, torch.Tensor],
    device: str = "cpu",
    timer: bool = False,
) -> dict:
    """
    Parallelized computation of reflection and transmission for incoherent and coherent
    light spectra that traverse a bunch of multilayer thin-films with dispersive materials.
    This implementation in PyTorch naturally allows:
     - GPU accelerated computations
     - To compute gradients regarding the multilayer thin-film (i.e. N, T) thanks to Pytorch Autograd

    However, the input can also be a numpy array format.
    Although all internal computations are processed via PyTorch, the output data is converted to numpy arrays again.
    Hence, the use of numpy input may increase computation time due to data type conversions.

    Parameters:
    -----------
    pol : str
        Polarization of the light, accepts only 's' or 'p'
    N : torch.Tensor

    L : torch.Tensor


    mask : list
        [[2,3,4], [6,7], [10, 11, 12]]
    """
    n_lambda = len(lambda_vacuum)
    n_theta = len(theta)
    n_layers = L.shape[1]
    n_stack = L.shape[0]
    imask = get_imask(mask, n_layers)
    
    L_coh_loc = np.argwhere(np.diff(imask) != 1).flatten()
    L_inc_loc = np.argwhere(np.diff(imask) == 1).flatten()

    n_L_ = len(imask) -1
    # matrix of Reflectivity and Transmissivity of the layer interfaces
    requires_grad = True if (L.requires_grad or N.requires_grad) else False
    L_ = torch.empty((n_stack, n_L_, n_theta, n_lambda, 2, 2)).requires_grad_(
        requires_grad
    )

    snell_theta = SnellLaw_vectorized(
        N.type(torch.complex128), theta.type(torch.complex128)
    )  # propagation angle in every layer

    # test
    # from tmm import list_snell
    # th_list = np.zeros((45, 3, 20))
    # for w in range(20):
    #     for t in range(45):
    #         th_list[t, :, w] = list_snell(N[0, :, w], theta[t])

    # first, the coherent substacks are evaluated
    for i, m in zip(L_coh_loc, mask):  # eg m = [4,5,6]
        m_ = np.arange(m[0]-1, m[-1]+2, 1, dtype=int)
        N_ = N[:, m_]
        N_[:, 0].imag = N_[:, -1].imag = 0.
        forward = coh_tmm(
            pol, N_, L[:, m_], snell_theta[0, :, m_[0], 0].real, lambda_vacuum, device
        )
        # the substack must be evaluated in both directions since we can have an incoming wave from the output side
        # (a reflection from an incoherent layer) and Reflectivit/Transmissivity can be different depending on the direction
        backward = coh_tmm(
            pol,
            N_.flip([1]),
            L[:, m_].flip([1]),
            snell_theta[0, :, m_[-1], 0].real,
            lambda_vacuum,
            device,
        )
        T_f = forward["T"]  # [n_stack, n_lambda, n_theta]
        T_b = backward["T"]
        R_f = forward["R"]
        R_b = backward["R"]

        sanity_checker(T_f)
        sanity_checker(T_b)
        sanity_checker(R_f)
        sanity_checker(R_b)

        L_[:, i, :, :, 0, 0] = 1.0 / T_f
        L_[:, i, :, :, 0, 1] = -R_b / T_f
        L_[:, i, :, :, 1, 0] = R_f / T_f
        L_[:, i, :, :, 1, 1] = ( T_b * T_f - R_b * R_f ) / T_f

    # [2, 3, 4, 7, 10, 11, 12]
    # [0, 1, 5, 8, 9, 13, 14] inc
    # Now, the incoherent layers are evaluated. In principle, the treatment is identical
    # to a coherent layer but the phase dependency is lost at each interface.

    for i, (k, m) in enumerate(zip(imask[:-1], np.diff(imask))):
        if m == 1:
            tf = interface_t_vec(
                pol,
                N[:, k][:, None],
                N[:, k + 1][:, None],
                snell_theta[:, :, k][:, :, None],
                snell_theta[:, :, k + 1][:, :, None],
            )[:, :, :, 0]
            T_f = T_from_t_vec(
                pol,
                tf,
                N[:, k],
                N[:, k + 1],
                snell_theta[:, :, k],
                snell_theta[:, :, k + 1],
            )
            tb = interface_t_vec(
                pol,
                N[:, k + 1][:, None],
                N[:, k][:, None],
                snell_theta[:, :, k + 1][:, :, None],
                snell_theta[:, :, k][:, :, None],
            )[:, :, :, 0]
            T_b = T_from_t_vec(
                pol,
                tb,
                N[:, k + 1],
                N[:, k],
                snell_theta[:, :, k + 1],
                snell_theta[:, :, k],
            )
            rf = interface_r_vec(
                pol,
                N[:, k][:, None],
                N[:, k + 1][:, None],
                snell_theta[:, :, k][:, :, None],
                snell_theta[:, :, k + 1][:, :, None],
            )[:, :, :, 0]
            R_f = R_from_r_vec(rf)
            rb = interface_r_vec(
                pol,
                N[:, k + 1][:, None],
                N[:, k][:, None],
                snell_theta[:, :, k + 1][:, :, None],
                snell_theta[:, :, k][:, :, None],
            )[:, :, :, 0]
            R_b = R_from_r_vec(rb)

            L_[:, i, :, :, 0, 0] = 1.0 / T_f
            L_[:, i, :, :, 0, 1] = -R_b / T_f
            L_[:, i, :, :, 1, 0] = R_f / T_f
            L_[:, i, :, :, 1, 1] = ( T_b * T_f - R_b * R_f ) / T_f

    for i, k in enumerate(imask[1:-1], 1):
        n_costheta = torch.einsum(
            "ik,ijk->ijk", N[:, k], torch.cos(snell_theta[:, :, k])
        ).imag  # [n_stack, n_theta, n_lambda]
        P = torch.exp(
            -4.
            * np.pi
            * (torch.einsum("ijk,k,i->ijk", n_costheta, 1 / lambda_vacuum, L[:, k]))
        )
        P_ = torch.zeros((*P.shape, 2, 2)) # [n_stack, n_th, n_wl, 2, 2]
        P_[..., 0, 0] = 1/P
        P_[..., 1, 1] = P 
        L_[:, i] = torch.einsum("ijklm,ijkmn->ijkln", P_, L_[:, i])

    #[0, 1, 2]
    L_tilde = L_[:, 0]
    for i in range(1, n_L_):
        L_tilde = torch.einsum("ijklm,ijkmn->ijkln", L_tilde, L_[:, i])

    R = L_tilde[..., 1, 0] / (L_tilde[..., 0, 0] + np.finfo(float).eps)

    T = 1 / (L_tilde[..., 0, 0] + np.finfo(float).eps)

    return {"R": R, "T": T, "L": L_}


def sanity_checker(input):
    assert (
        1.0 >= input.any() >= 0.0
    ).item(), "Some values are out of the accepted range of [0,1]"

def get_imask(mask, n_layers):
    mask = [item for sublist in mask for item in sublist]
    imask = np.isin(np.arange(n_layers, dtype=int), mask, invert=True)
    return np.arange(n_layers, dtype=int)[imask]
