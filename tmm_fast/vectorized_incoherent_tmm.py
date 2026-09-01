import torch
import numpy as np
from .vectorized_tmm_dispersive_multistack import coh_vec_tmm_disp_mstack as coh_tmm
from .vectorized_tmm_dispersive_multistack import (
    SnellLaw_vectorized,
    SnellLaw_cosines_vectorized,
    converter2torch,
    converter2numpy,
    resolve_device,
    check_inputs,
    _complex_abs_squared,
)

from typing import Union


def inc_vec_tmm_disp_lstack(
    pol: str,
    N: torch.Tensor,
    D: torch.Tensor,
    mask: list,
    theta: Union[np.ndarray, torch.Tensor],
    lambda_vacuum: Union[np.ndarray, torch.Tensor],
    device: Union[str, torch.device, None] = None,
    timer: bool = False,
    return_intermediates: bool = True,
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
        Complex refractive indices for all layers. The tensor must have shape 
        [n_stacks, n_layers] for dispersionless materials or [n_stacks, n_layers, n_lambda]
        for dispersive materials. If only 
    D : torch.Tensor
        Layer thicknesses in [m] for all incoherent and coherent layers. Must have shape 
        [n_stacks, n_layers]
    mask : list
        Specifies all the coherent substack. A coherent substack must be adjacent to an incoherent layer.
        Eg. mask = [[2,3,4], [6,7], [10, 11, 12]] specifies 3 coherent substacks for a stack of total 
        length >= 14. the incoherent layers are 0, 1, 5, 8, 9, 13 and any further layers. 
        Note that the function can handle parallel stacks but the mask must be identical for all parallel
        stacks
    theta : torch.tensor    
        Angles of incidence in [rad] of the incoming light in the first layer. Must have shape 
        [n_theta]
    lambda_vacuum : torch.tensor
        Vacuum wavelengths of the light in [m]. Must have shape
        [n_wl]
    device : str, torch.device or None
        Computation device. When omitted, the device is inferred from N if N is a tensor and
        otherwise defaults to CPU.
    return_intermediates : bool
        Return interface matrices, coherent-substack results, propagation matrices, and angles in
        addition to `R` and `T`. Set this to `False` when only the final powers are needed.

    Returns:
    --------
    dict : 
        "R": torch.Tensor or np.ndarray
            Reflectivity of the entire stack of incoherent and coherent layers
        "T": torch.Tensor or np.ndarray
            Transmissivity of the entire stack of incoherent and coherent layers
        The remaining entries are included only when `return_intermediates=True`:
        "L": torch.Tensor or np.ndarray
            Interface matrices see Byrnes Eq. 28
        'coh_tmm_f': dict
            Forward result for the coherent substacks in order. The dict contains the
            results of a normal coherent stack
        'coh_tmm_b': torch.Tensor
            Backward result for the coherent substacks in order. The dict contains the
            results of a normal coherent stack
        'P': torch.Tensor or np.ndarray
            Absorption in the incoherent layers
        'th_list': torch.Tensor or np.ndarray
            Complex angles according to snells law in all layers

    Example:
    --------

    num_layers = 6
    num_stacks = 2
    n_wl = 75
    n_th = 45
    pol = polarization = 's'
    wl = wavelengths = torch.linspace(400, 1200, n_wl) * (10**(-9))
    th = incidence_angles = torch.linspace(0, 89, n_th) * (np.pi/180)
    
    # the mask specifies that layer 1 and 2 form a coherent substack and 
    # layer 4 forms another coherent substack
    mask = [[1, 2], [4]]

    N = refracive_indices = torch.ones(
        (num_stacks, num_layers, wl.shape[0]), 
        dtype=torch.complex128
    )

    N[:, 1] = 1.3 + .003j
    N[:, 2] = 2.2 + .0j
    N[:, 3] = 1.3 + .003j
    N[:, 4] = 1.1 + .0j

    D = layer_thicknesses = torch.empty((n_stacks, n_layers), dtype=torch.float64)
    D[:, 0] = np.inf
    # test how a a change of the first layer thickness changes the result
    D[0, 1] = 200e-9
    D[1, 1] = 400e-9

    D[:, 2] = 200e-9
    D[:, 3] = 15000e-9
    D[:, 4] = 300e-9
    D[:, -1] = np.inf

    result_dict = inc_tmm_fast(pol, N, D, mask, th, wl, device='cpu')

    """
    return_numpy = not any(
        torch.is_tensor(value) for value in (N, D, theta, lambda_vacuum)
    )
    device = resolve_device(N, device)
    N = converter2torch(N, device)
    D = converter2torch(D, device, dtype=torch.float64)
    theta = torch.atleast_1d(converter2torch(theta, device))
    lambda_vacuum = torch.atleast_1d(
        converter2torch(lambda_vacuum, device, dtype=torch.float64)
    )


    n_lambda = len(lambda_vacuum)
    n_theta = len(theta)
    n_layers = D.shape[1]
    n_stack = D.shape[0]
    if N.ndim == 2:
        N = N.unsqueeze(-1).repeat(1, 1, n_lambda)
    check_inputs(N, D, lambda_vacuum, theta)
    imask = get_imask(mask, n_layers)

    coh_res_f = []
    coh_res_b = []
    
    L_coh_loc = np.argwhere(np.diff(imask) != 1).flatten()

    n_L_ = len(imask) -1
    # matrix of Reflectivity and Transmissivity of the layer interfaces
    # no requires_grad_ here: that would make L_ a leaf, and filling a leaf by assignment is
    # what autograd forbids. Assigning tracked values into an ordinary tensor is enough for
    # gradients to flow back to N and D.
    L_ = torch.empty(
        (n_stack, n_L_, n_theta, n_lambda, 2, 2),
        dtype=torch.float64,
        device=N.device,
    )

    complex_N = N.type(torch.complex128)
    complex_theta = theta.type(torch.complex128)
    # th_list is an optional diagnostic. A compact R/T result can stay on the cosine-only
    # path, while the full result constructs angles once and shares the accompanying cosines
    # with every coherent substack.
    if return_intermediates:
        snell_theta, cos_snell_theta = SnellLaw_vectorized(
            complex_N, complex_theta, return_cosines=True
        )
    else:
        snell_theta = None
        cos_snell_theta = SnellLaw_cosines_vectorized(complex_N, complex_theta)

    # first, the coherent substacks are evaluated with the adjacent incoherent stacks as input 
    # and output layer. Therefore, Im(N) of the incoherent layers are set to zero for the 
    # coherent evaluation. The absorption for the incoherent layers are calculated later
    for i, m in zip(L_coh_loc, mask):  # eg m = [4,5,6]
        m_ = np.arange(m[0]-1, m[-1]+2, 1, dtype=int)
        N_ = N[:, m_]
        d = D[:, m_]
        d[:, 0] = d[:, -1] = np.inf
        forward = coh_tmm(
            pol, N_, d, theta, lambda_vacuum, device,
            _validate=False,
            _snell_cosines=cos_snell_theta[:, :, m_, :],
        )
        # the substack must be evaluated in both directions since we can have an incoming wave from the output side
        # (a reflection from an incoherent layer) and Reflectivit/Transmissivity can be different depending on the direction
        backward = coh_tmm(
            pol,
            N_.flip([1]),
            d.flip([1]),
            theta,
            lambda_vacuum,
            device,
            _validate=False,
            _snell_cosines=cos_snell_theta[:, :, m_, :].flip([2]),
        )
        T_f = forward["T"]  # [n_stack, n_lambda, n_theta]
        T_b = backward["T"]
        R_f = forward["R"]
        R_b = backward["R"]

        if return_intermediates:
            coh_res_f.append(forward)
            coh_res_b.append(backward)
        # sanity_checker(T_f)
        # sanity_checker(T_b)
        # sanity_checker(R_f)
        # sanity_checker(R_b)

        L_[:, i, :, :, 0, 0] = 1.0 / T_f
        L_[:, i, :, :, 0, 1] = -R_b / T_f
        L_[:, i, :, :, 1, 0] = R_f / T_f
        L_[:, i, :, :, 1, 1] = ( T_b * T_f - R_b * R_f ) / T_f

    differences = np.diff(imask)
    interface_positions = np.flatnonzero(differences == 1)
    if interface_positions.size:
        interface_layers = imask[:-1][interface_positions]
        n_i = N[:, interface_layers]
        n_f = N[:, interface_layers + 1]
        cos_th_i = cos_snell_theta[:, :, interface_layers].permute(0, 2, 1, 3)
        cos_th_f = cos_snell_theta[:, :, interface_layers + 1].permute(0, 2, 1, 3)
        T_f, T_b, R_f, R_b = interface_powers(pol, n_i, n_f, cos_th_i, cos_th_f)

        inverse_T_f = 1 / T_f
        interface_matrices = torch.stack(
            (
                torch.stack((inverse_T_f, -R_b * inverse_T_f), dim=-1),
                torch.stack(
                    (R_f * inverse_T_f, (T_b * T_f - R_b * R_f) * inverse_T_f),
                    dim=-1,
                ),
            ),
            dim=-2,
        )
        L_[:, interface_positions] = interface_matrices

    P_ = None
    propagation_layers = imask[1:-1]
    if propagation_layers.size:
        n_costheta = (
            N[:, propagation_layers][:, None] * cos_snell_theta[:, :, propagation_layers]
        ).imag
        P = torch.exp(
            -4
            * np.pi
            * n_costheta
            * (1 / lambda_vacuum)[None, None, None]
            * D[:, propagation_layers].real[:, None, :, None]
        ).clamp_min(1e-30)
        P = P.permute(0, 2, 1, 3)
        zeros = torch.zeros_like(P)
        propagation_matrices = torch.stack(
            (
                torch.stack((1 / P, zeros), dim=-1),
                torch.stack((zeros, P), dim=-1),
            ),
            dim=-2,
        )
        L_[:, 1:] = torch.matmul(propagation_matrices, L_[:, 1:].clone())
        P_ = propagation_matrices[:, -1]

    # multiply all interfaces together
    L_tilde = L_[:, 0]
    for i in range(1, n_L_):
        L_tilde = torch.matmul(L_tilde, L_[:, i])

    R = L_tilde[..., 1, 0] / (L_tilde[..., 0, 0] + np.finfo(float).eps)

    T = 1 / (L_tilde[..., 0, 0] + np.finfo(float).eps)

    result = {"R": R, "T": T}
    if return_intermediates:
        result.update({
            "L": L_,
            'coh_tmm_f': coh_res_f,
            'coh_tmm_b': coh_res_b,
            'P': P_,
            'th_list': snell_theta,
        })
    return _to_numpy(result) if return_numpy else result


def interface_powers(pol, n_i, n_f, cos_th_i, cos_th_f):
    n_i = n_i[:, :, None]
    n_f = n_f[:, :, None]
    if pol == 's':
        incoming = n_i * cos_th_i
        outgoing = n_f * cos_th_f
        denominator = incoming + outgoing
        tf = 2 * incoming / denominator
        tb = 2 * outgoing / denominator
        rf = (incoming - outgoing) / denominator
        forward_flux = incoming.real
        backward_flux = outgoing.real
    elif pol == 'p':
        incoming = n_i * cos_th_f
        outgoing = n_f * cos_th_i
        denominator = incoming + outgoing
        tf = 2 * n_i * cos_th_i / denominator
        tb = 2 * n_f * cos_th_f / denominator
        rf = (outgoing - incoming) / denominator
        forward_flux = (n_i * torch.conj(cos_th_i)).real
        backward_flux = (n_f * torch.conj(cos_th_f)).real
    else:
        raise ValueError("Polarization must be 's' or 'p'")

    T_f = _complex_abs_squared(tf) * backward_flux / forward_flux
    T_b = _complex_abs_squared(tb) * forward_flux / backward_flux
    R_f = _complex_abs_squared(rf)
    return T_f, T_b, R_f, R_f


def sanity_checker(input):
    assert (
        1.0 >= input.any() >= 0.0
    ).item(), "Some values are out of the accepted range of [0,1]"


def get_imask(mask, n_layers):
    if not isinstance(mask, (list, tuple)):
        raise ValueError('mask must be a sequence of coherent substacks')

    coherent_layers = []
    previous_end = None
    for substack in mask:
        if not isinstance(substack, (list, tuple, np.ndarray)) or len(substack) == 0:
            raise ValueError('mask substacks must be non-empty sequences of layer indices')
        if any(isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
               for index in substack):
            raise ValueError('mask layer indices must be integers')

        substack = list(substack)
        if substack != list(range(substack[0], substack[-1] + 1)):
            raise ValueError('mask substacks must contain contiguous, increasing layer indices')
        if substack[0] <= 0 or substack[-1] >= n_layers - 1:
            raise ValueError('mask may contain only interior layer indices')
        if previous_end is not None and substack[0] <= previous_end + 1:
            raise ValueError('mask substacks must be ordered, disjoint and separated')

        coherent_layers.extend(substack)
        previous_end = substack[-1]

    imask = np.isin(np.arange(n_layers, dtype=int), coherent_layers, invert=True)
    return np.arange(n_layers, dtype=int)[imask]


def _to_numpy(value):
    if torch.is_tensor(value):
        return converter2numpy(value)
    if isinstance(value, dict):
        return {key: _to_numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_numpy(item) for item in value]
    return value
