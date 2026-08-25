import numpy as np
from numpy import pi
import torch

from typing import Optional, Union
import sys
from warnings import warn
EPSILON = sys.float_info.epsilon

def coh_vec_tmm_disp_mstack(pol:str,
                            N:Union[np.ndarray, torch.Tensor, list, tuple],
                            T:Union[np.ndarray, torch.Tensor, list, tuple],
                            Theta:Union[np.ndarray, torch.Tensor, list, tuple, float],
                            lambda_vacuum:Union[np.ndarray, torch.Tensor, list, tuple, float],
                            device:Optional[Union[str, torch.device]]=None,
                            timer:bool=False,
                            *,
                            _validate:bool=True,
                            _snell_thetas:Optional[torch.Tensor]=None,
                            _snell_cosines:Optional[torch.Tensor]=None) -> dict:
    """
    Parallelized computation of reflection and transmission for coherent light spectra that traverse
    a bunch of multilayer thin-films with dispersive materials.
    This implementation in PyTorch naturally allows:
     - GPU accelerated computations
     - To compute gradients regarding the multilayer thin-film (i.e. N, T) thanks to Pytorch Autograd

    Inputs may mix tensors, numpy arrays and plain array-like values. Outputs remain tensors when
    any input is a tensor; otherwise they are converted to numpy arrays.

    Parameters:
    -----------
    pol : str
        Polarization of the light, accepts only 's' or 'p'
    N : Tensor or array
        PyTorch Tensor or numpy array of shape [S x L x W] with complex or real entries which contain the refractive
        indices at the wavelengths of interest:
        S is the number of multi-layer thin films, L is the number of layers for each thin film, W is the number of
        wavelength considered. Note that the first and last layer must feature real valued refractive indicies, i.e.
        imag(N[:, 0, :]) = 0 and imag(N[:, -1, :]) = 0.
        
    T : Tensor or array
        Layer thicknesses in metres for the individual thin-film stacks.
        T is of shape [S x L] with real-valued entries; infinite values are allowed for the first and last layers only!
    Theta : Tensor or array
        Theta determines the angles with which the light propagates in the injection layer.
        It normally has shape [A]. A precomputed angle grid of shape [S x A x W] is also
        accepted, which is useful when this solver evaluates a substack entered from a
        dispersive medium.
    lambda_vacuum : Tensor or numpy array
        Vacuum wavelengths for which reflection and transmission are computed given a bunch of thin films.
        It is of shape [W] and holds the wavelengths in metres.
    device : str, torch.device or None
        Computation device. When omitted, the device is inferred from N if N is a tensor and
        otherwise defaults to CPU.
    timer: Boolean
        Determines whether to track times for data pushing on CPU or GPU and total computation time; see output
        information for details on how to read out time

    Returns:
    --------
    output : Dict
        Keys:
            'r' : Tensor or array of Fresnel coefficients of reflection for each stack (over angle and wavelength)
            't' : Tensor or array of Fresnel coefficients of transmission for each stack (over angle and wavelength)
            'R' : Tensor or array of Reflectivity / Reflectance for each stack (over angle and wavelength)
            'T' : Tensor or array of Transmissivity / Transmittance for each stack (over angle and wavelength)
            Each of these tensors or arrays is of shape [S x A x W]
    optional output: list of two floats if timer=True
            first entry holds the push time [sec] that is the time required to push the input data on the specified
            device (i.e. cpu oder cuda), the second entry holds the total computation time [sec] (push time + tmm)

    Remarks and prior work from Byrnes:
    -----------------------------------
    Main "coherent transfer matrix method" calc. Given parameters of a stack,
    calculates everything you could ever want to know about how light
    propagates in it. (If performance is an issue, you can delete some of the
    calculations without affecting the rest.)
    pol is light polarization, "s" or "p".
    n_list is the list of refractive indices, in the order that the light would
    pass through them. The 0'th element of the list should be the semi-infinite
    medium from which the light enters, the last element should be the semi-
    infinite medium to which the light exits (if any exits).
    th_0 is the angle of incidence: 0 for normal, pi/2 for glancing.
    Remember, for a dissipative incoming medium (n_list[0] is not real), th_0
    should be complex so that n0 sin(th0) is real (intensity is constant as
    a function of lateral position).
    d_list is the list of layer thicknesses (front to back). Should correspond
    one-to-one with elements of n_list. First and last elements should be "inf".
    lam_vac is vacuum wavelength of the light.
    Outputs the following as a dictionary (see manual for details)
    * r--reflection amplitude
    * t--transmission amplitude
    * R--reflected wave power (as fraction of incident)
    * T--transmitted wave power (as fraction of incident)
    * power_entering--Power entering the first layer, usually (but not always)
      equal to 1-R (see manual).
    * vw_list-- n'th element is [v_n,w_n], the forward- and backward-traveling
      amplitudes, respectively, in the n'th medium just after interface with
      (n-1)st medium.
    * kz_list--normal component of complex angular wavenumber for
      forward-traveling wave in each layer.
    * th_list--(complex) propagation angle (in radians) in each layer
    * pol, n_list, d_list, th_0, lam_vac--same as input
    """

    if timer:
        import time
        starttime = time.time()
    return_numpy = not any(torch.is_tensor(value) for value in (N, T, Theta, lambda_vacuum))
    device = resolve_device(N, device)
    N = converter2torch(N, device)
    T = converter2torch(T, device)
    lambda_vacuum = torch.atleast_1d(converter2torch(lambda_vacuum, device))
    Theta = torch.atleast_1d(converter2torch(Theta, device))
    # T tells a single stack, of shape [L], apart from a batch of them, of shape [S x L].
    # N follows suit and may additionally come without the wavelength axis if the materials
    # are dispersionless, i.e. [L] or [S x L] instead of [L x W] or [S x L x W].
    assert T.ndim in (1, 2), 'T is not of shape [L] (1d) or [S x L] (2d), as it is of dimension ' + str(T.ndim)
    squeezed = T.ndim == 1
    if squeezed:
        T = T.unsqueeze(0)
        N = N.unsqueeze(0)
    assert N.ndim in (2, 3), 'N is not of shape [L], [L x W], [S x L] or [S x L x W], as it is of shape ' + str(tuple(N.shape))
    if timer:
        push_time = time.time() - starttime
    num_layers = T.shape[1]
    num_stacks = T.shape[0]
    num_angles = Theta.shape[0] if Theta.ndim == 1 else Theta.shape[1]
    num_wavelengths = lambda_vacuum.shape[0]
    # a dispersionless N holds no wavelength axis yet, repeat it across the spectrum. This has
    # to happen before check_inputs, which expects the full [S x L x W].
    if N.ndim == 2:
        N = N.unsqueeze(-1).repeat(1, 1, num_wavelengths)
    if _validate:
        check_inputs(N, T, lambda_vacuum, Theta)

    # SnellThetas is a tensor, for each stack and layer, the angle that the light travels
    # through the layer. Computed with Snell's law. Note that the "angles" may be complex!
    if _snell_thetas is None:
        SnellThetas = SnellLaw_vectorized(N, Theta, validate=_validate)
        cos_SnellThetas = torch.cos(SnellThetas)
    else:
        assert _snell_thetas.shape == (num_stacks, num_angles, num_layers, num_wavelengths)
        SnellThetas, cos_SnellThetas = select_forward_angles(
            N, _snell_thetas, _snell_cosines, validate=_validate
        )
        if cos_SnellThetas is None:
            cos_SnellThetas = torch.cos(SnellThetas)


    theta = 2 * np.pi * torch.einsum('skij,sij->skij', cos_SnellThetas, N)  # [theta,d, lambda]
    kz_list = torch.einsum('sijk,k->skij', theta, 1 / lambda_vacuum)  # [lambda, theta, d]

    # kz is the z-component of (complex) angular wavevector for the forward-moving
    # wave. Positive imaginary part means decaying.

    # delta is the total phase accrued by traveling through a given layer.
    # Only the inner layers accumulate phase. Forming it for the semi-infinite edges as well
    # would multiply a finite wavevector by an infinite thickness, and while the forward pass
    # drops those entries, the backward pass of the einsum computes 0 * inf = nan for the
    # refractive indices of the edge layers.
    delta = torch.einsum('skij,sj->skij', kz_list[:, :, :, 1:-1], T[:, 1:-1])

    # check for opacity. If too much of the optical power is absorbed in a layer
    # it can lead to numerical instability.
    if _validate and torch.any(delta.imag > 35.):
        delta = torch.complex(delta.real, delta.imag.clamp(max=35.))
        warn('Opacity warning. The imaginary part of the phase thickness is clamped to 35 for numerical stability.\n'+
             'You might encounter problems with gradient computation...')
    elif not _validate:
        delta = torch.complex(delta.real, delta.imag.clamp(max=35.))


    # t_list and r_list hold the transmission and reflection coefficients from
    # the Fresnel Equations

    t_list = interface_t_vec(
        pol, N[:, :-1, :], N[:, 1:, :], SnellThetas[:, :, :-1, :],
        SnellThetas[:, :, 1:, :], cos_SnellThetas[:, :, :-1, :],
        cos_SnellThetas[:, :, 1:, :]
    )
    r_list = interface_r_vec(
        pol, N[:, :-1, :], N[:, 1:, :], SnellThetas[:, :, :-1, :],
        SnellThetas[:, :, 1:, :], cos_SnellThetas[:, :, :-1, :],
        cos_SnellThetas[:, :, 1:, :]
    )
    
    # A ist the propagation term for matrix optic and holds the appropriate accumulated phase for the thickness
    # of each layer
    A = torch.exp(1j * delta)
    F = r_list[:, :, :, 1:]
    A = A.permute(0, 2, 1, 3)
    inverse_A = 1 / (A + np.finfo(float).eps)
    inverse_t = 1 / t_list[:, :, :, 1:]
    F_over_t = F * inverse_t
    
    # M_list holds the transmission and reflection matrices from matrix-optics 
    
    M_list = torch.zeros((num_stacks, num_angles, num_wavelengths, num_layers, 2, 2), dtype=torch.complex128, device=device)
    M_list[:, :, :, 1:-1, 0, 0] = inverse_A * inverse_t
    M_list[:, :, :, 1:-1, 0, 1] = inverse_A * F_over_t
    M_list[:, :, :, 1:-1, 1, 0] = A * F_over_t
    M_list[:, :, :, 1:-1, 1, 1] = A * inverse_t
    Mtilde = torch.empty((num_stacks, num_angles, num_wavelengths, 2, 2), dtype=torch.complex128, device=device)
    Mtilde[:, :, :] = make_2x2_tensor(1, 0, 0, 1, dtype=torch.complex128)

    # contract the M_list matrix along the dimension of the layers, all
    for i in range(1, num_layers - 1):
        Mtilde = torch.einsum('sijkl,sijlm->sijkm', Mtilde, M_list[:, :, :, i])

    # M_r0 accounts for the first and last stack where the translation coefficients are 1
    # todo: why compute separately?
    M_r0 = torch.empty((num_stacks, num_angles, num_wavelengths, 2, 2), dtype=torch.complex128, device=device)
    M_r0[:, :, :, 0, 0] = 1
    M_r0[:, :, :, 0, 1] = r_list[:, :, :, 0]
    M_r0[:, :, :, 1, 0] = r_list[:, :, :, 0]
    M_r0[:, :, :, 1, 1] = 1
    M_r0 *= (1 / t_list[:, :, :, 0])[..., None, None]

    Mtilde = torch.einsum('shijk,shikl->shijl', M_r0, Mtilde)

    # Net complex transmission and reflection amplitudes
    r = Mtilde[:, :, :, 1, 0] / (Mtilde[:, :, :, 0, 0] + np.finfo(float).eps)
    t = 1 / (Mtilde[:, :, :, 0, 0] + np.finfo(float).eps)

    # Net transmitted and reflected power, as a proportion of the incoming light
    # power.
    R = R_from_r_vec(r)
    T = T_from_t_vec(
        pol, t, N[:, 0], N[:, -1], SnellThetas[:, :, 0], SnellThetas[:, :, -1],
        cos_SnellThetas[:, :, 0], cos_SnellThetas[:, :, -1]
    )

    if squeezed and r.shape[0] == 1:
        r = torch.reshape(r, (r.shape[1], r.shape[2]))
        R = torch.reshape(R, (R.shape[1], R.shape[2]))
        T = torch.reshape(T, (T.shape[1], T.shape[2]))
        t = torch.reshape(t, (t.shape[1], t.shape[2]))

    if return_numpy:
        r = converter2numpy(r)
        t = converter2numpy(t)
        R = converter2numpy(R)
        T = converter2numpy(T)

    if timer:
        total_time = time.time() - starttime
        return {'r': r, 't': t, 'R': R, 'T': T}, [push_time, total_time]
    else:
        return {'r': r, 't': t, 'R': R, 'T': T}

def SnellLaw_vectorized(n, th, validate=True):
    """
    return list of angle theta in each layer based on angle th_0 in layer 0,
    using Snell's law. n_list is index of refraction of each layer. Note that
    "angles" may be complex!!
    """
    # Important that the arcsin here is numpy.lib.scimath.arcsin, not
    # numpy.arcsin! (They give different results e.g. for arcsin(2).)
    if th.dtype != torch.complex128:
        warn('there is some problem with theta, the dtype is not complex')
    if n.dtype != torch.complex128:
        warn('there is some problem with n, the dtype is not conplex')
    th = th if th.dtype == torch.complex128 else th.type(torch.complex128)
    n = n if n.dtype == torch.complex128 else n.type(torch.complex128)

    if th.ndim == 1:
        n0_ = torch.einsum('hk,j,hik->hjik', n[:,0], torch.sin(th), 1/n)
    elif th.ndim == 3:
        # A substack embedded in a dispersive multilayer has a different incident angle for
        # every stack and wavelength. Preserve that [S x A x W] grid instead of silently
        # reusing stack 0 / wavelength 0.
        n0_ = n[:, 0, None, None, :] * torch.sin(th[:, :, None, :]) / n[:, None, :, :]
    else:
        raise AssertionError(
            'Theta is not of shape [A] (1d) or [S x A x W] (3d), as it is of shape '
            + str(tuple(th.shape))
        )
    angles = torch.asin(n0_)
    
    # The first and last entry need to be the forward angle (the intermediate
    # layers don't matter, see https://arxiv.org/abs/1603.02720 Section 5)

    angles, _ = select_forward_angles(n, angles, validate=validate)
    return angles

def select_forward_angles(n, angles, cosines=None, validate=True):
    angles = angles.clone()
    cosines = None if cosines is None else cosines.clone()
    for layer in (0, -1):
        layer_cosines = None if cosines is None else cosines[:, :, layer]
        backward = is_not_forward_angle(
            n[:, layer], angles[:, :, layer], layer_cosines, validate=validate
        ).bool()
        angles[:, :, layer] = torch.where(
            backward, pi - angles[:, :, layer], angles[:, :, layer]
        )
        if cosines is not None:
            cosines[:, :, layer] = torch.where(
                backward, -cosines[:, :, layer], cosines[:, :, layer]
            )
    return angles, cosines


def is_not_forward_angle(n, theta, cos_theta=None, validate=True):
    """
    if a wave is traveling at angle theta from normal in a medium with index n,
    calculate whether or not this is the forward-traveling wave (i.e., the one
    going from front to back of the stack, like the incoming or outgoing waves,
    but unlike the reflected wave). For real n & theta, the criterion is simply
    -pi/2 < theta < pi/2, but for complex n & theta, it's more complicated.
    See https://arxiv.org/abs/1603.02720 appendix D. If theta is the forward
    angle, then (pi-theta) is the backward angle and vice-versa.
    """
    # n = [lambda]
    # theta = [theta, lambda]

    if validate and not (n.real * n.imag >= 0).all():
        raise AssertionError(
            "For materials with gain, it's ambiguous which beam is incoming vs outgoing. See "
            "https://arxiv.org/abs/1603.02720 Appendix C.\n"
            "n: " + str(n) + "   angle: " + str(theta)
        )
    n = n.unsqueeze(1)
    cos_theta = torch.cos(theta) if cos_theta is None else cos_theta
    ncostheta = cos_theta * n
    assert ncostheta.shape == theta.shape, 'ncostheta and theta shape doesnt match'
    # For evanescent decay or a lossy medium the decaying wave is the forward-moving one,
    # everywhere else it is the one with a positive Poynting vector. The Poynting vector is
    # Re[n cos(theta)] for s-polarization and Re[n cos(theta*)] for p-polarization, but the
    # two agree, so assume s here and check both in the assertions below.
    # Note that the criterion is the magnitude of the imaginary part, not its sign: a wave
    # decaying with Im[n cos(theta)] < 0 is evanescent just the same, and testing the signed
    # value instead sends it down the propagating branch and trips the assertions.
    evanescent = abs(ncostheta.imag) > 100 * EPSILON
    answer = torch.where(evanescent, ncostheta.imag > 0, ncostheta.real > 0)

    if validate:
        tolerance = 100 * EPSILON
        conjugate_ncostheta = (n * torch.conj(cos_theta)).real
        valid_forward = ~answer | (
            (ncostheta.imag > -tolerance)
            & (ncostheta.real > -tolerance)
            & (conjugate_ncostheta > -tolerance)
        )
        valid_backward = answer | (
            (ncostheta.imag < tolerance)
            & (ncostheta.real < tolerance)
            & (conjugate_ncostheta < tolerance)
        )
        if not (valid_forward & valid_backward).all():
            raise AssertionError(
                "It's not clear which beam is incoming vs outgoing. Weird index maybe?\n"
                "n: " + str(n.squeeze(1)) + "   angle: " + str(theta)
            )
    answer = (~answer).clone().detach().type(torch.float)

    # for cross checking of the answer
    # answer_tmm = torch.empty_like(answer, dtype=torch.bool)
    # for i, _ in enumerate(answer_tmm):
    #     for j, _ in enumerate(answer_tmm[i]):
    #         for k, _ in enumerate(answer_tmm[i,j]):

    #             m, t = n[i,0,k].numpy(), theta[i,j,k].numpy()
    #             assert m.real * m.imag >= 0, ("For materials with gain, it's ambiguous which "
    #                                     "beam is incoming vs outgoing. See "
    #                                     "https://arxiv.org/abs/1603.02720 Appendix C.\n"
    #                                     "n: " + str(m) + "   angle: " + str(t))
    #             ncostheta2 = m * np.cos(t)
    #             if abs(ncostheta2.imag) > 100 * EPSILON:
    #                 # Either evanescent decay or lossy medium. Either way, the one that
    #                 # decays is the forward-moving wave
    #                 answer2 = (ncostheta2.imag > 0)
    #             else:
    #                 # Forward is the one with positive Poynting vector
    #                 # Poynting vector is Re[n cos(theta)] for s-polarization or
    #                 # Re[n cos(theta*)] for p-polarization, but it turns out they're consistent
    #                 # so I'll just assume s then check both below
    #                 answer2 = (ncostheta2.real > 0)
    #             # convert from numpy boolean to the normal Python boolean
    #             answer2 = bool(answer2)
    #             # double-check the answer ... can't be too careful!
    #             error_string = ("It's not clear which beam is incoming vs outgoing. Weird"
    #                             " index maybe?\n"
    #                             "n: " + str(m) + "   angle: " + str(t))
    #             if answer2 is True:
    #                 assert ncostheta2.imag > -100 * EPSILON, error_string
    #                 assert ncostheta2.real > -100 * EPSILON, error_string
    #                 assert (m * np.cos(t.conjugate())).real > -100 * EPSILON, error_string
    #             else:
    #                 assert ncostheta2.imag < 100 * EPSILON, error_string
    #                 assert ncostheta2.real < 100 * EPSILON, error_string
    #                 assert (m * np.cos(t.conjugate())).real < 100 * EPSILON, error_string

    #             answer_tmm[i,j,k] = answer2
    
    # torch.testing.assert_close((~answer_tmm).type(torch.float), answer)
    return answer

def interface_r_vec(polarization, n_i, n_f, th_i, th_f, cos_th_i=None, cos_th_f=None):
    """
    reflection amplitude (from Fresnel equations)
    polarization is either "s" or "p" for polarization
    n_i, n_f are (complex) refractive index for incident and final
    th_i, th_f are (complex) propegation angle for incident and final
    (in radians, where 0=normal). "th" stands for "theta".
    """
    cos_th_i = torch.cos(th_i) if cos_th_i is None else cos_th_i
    cos_th_f = torch.cos(th_f) if cos_th_f is None else cos_th_f
    if polarization == 's':
        ni_thi = torch.einsum('sij,skij->skji', n_i, cos_th_i)
        nf_thf = torch.einsum('sij,skij->skji', n_f, cos_th_f)
        return (ni_thi - nf_thf) / (ni_thi + nf_thf)
    elif polarization == 'p':
        nf_thi = torch.einsum('sij,skij->skji', n_f, cos_th_i)
        ni_thf = torch.einsum('sij,skij->skji', n_i, cos_th_f)
        return (nf_thi - ni_thf) / (nf_thi + ni_thf)
    else:
        raise ValueError("Polarization must be 's' or 'p'")

def interface_t_vec(polarization, n_i, n_f, th_i, th_f, cos_th_i=None, cos_th_f=None):
    """
    transmission amplitude (frem Fresnel equations)
    polarization is either "s" or "p" for polarization
    n_i, n_f are (complex) refractive index for incident and final
    th_i, th_f are (complex) propegation angle for incident and final
    (in radians, where 0=normal). "th" stands for "theta".
    """
    cos_th_i = torch.cos(th_i) if cos_th_i is None else cos_th_i
    cos_th_f = torch.cos(th_f) if cos_th_f is None else cos_th_f
    if polarization == 's':
        ni_thi = torch.einsum('sij,skij->skji', n_i, cos_th_i)
        nf_thf = torch.einsum('sij,skij->skji', n_f, cos_th_f)
        return 2 * ni_thi / (ni_thi + nf_thf)
    elif polarization == 'p':
        nf_thi = torch.einsum('sij,skij->skji', n_f, cos_th_i)
        ni_thf = torch.einsum('sij,skij->skji', n_i, cos_th_f)
        ni_thi = torch.einsum('sij,skij->skji', n_i, cos_th_i)
        return 2 * ni_thi / (nf_thi + ni_thf)
    else:
        raise ValueError("Polarization must be 's' or 'p'")

def R_from_r_vec(r):
    """
    Calculate reflected power R, starting with reflection amplitude r.
    """
    return abs(r) ** 2

def T_from_t_vec(pol, t, n_i, n_f, th_i, th_f, cos_th_i=None, cos_th_f=None):
    """
    Calculate transmitted power T, starting with transmission amplitude t.

    Parameters:
    -----------
    pol : str
        polarization, either 's' or 'p'
    t : torch.Tensor 
        transmission coefficients. Expects shape []
    
    n_i, n_f are refractive indices of incident and final medium.
    th_i, th_f are (complex) propagation angles through incident & final medium
    (in radians, where 0=normal). "th" stands for "theta".
    In the case that n_i, n_f, th_i, th_f are real, formulas simplify to
    T=|t|^2 * (n_f cos(th_f)) / (n_i cos(th_i)).
    See manual for discussion of formulas
    """

    cos_th_i = torch.cos(th_i) if cos_th_i is None else cos_th_i
    cos_th_f = torch.cos(th_f) if cos_th_f is None else cos_th_f
    if pol == 's':
        ni_thi = torch.real(cos_th_i * n_i.unsqueeze(1))
        nf_thf = torch.real(cos_th_f * n_f.unsqueeze(1))
        return (abs(t ** 2) * ((nf_thf) / (ni_thi)))

    elif pol == 'p':
        ni_thi = torch.real(torch.conj(cos_th_i) * n_i.unsqueeze(1))
        nf_thf = torch.real(torch.conj(cos_th_f) * n_f.unsqueeze(1))
        return (abs(t ** 2) * ((nf_thf) / (ni_thi)))

    else:
        raise ValueError("Polarization must be 's' or 'p'")

def resolve_device(data, device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.is_tensor(data):
        return data.device
    return torch.device('cpu')


def converter2torch(data, device: Union[str, torch.device]) -> torch.Tensor:
    '''
    Checks the datatype of data to torch.tensor and moves the tensor to the device.

    Parameters:
    -----------
    data : array_like
        data that should be converted to torch.Tensor
    device : str
        either 'cpu' or 'cuda'
    '''
    if torch.is_tensor(data):
        return data.to(device=device, dtype=torch.complex128)
    try:
        array = np.asarray(data)
        return torch.as_tensor(array.copy(), dtype=torch.complex128, device=device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError('Inputs must be tensors, numpy arrays, scalars, or array-like values') from error

def converter2numpy(data:torch.Tensor)->np.ndarray:
    data = data.detach().cpu().numpy()
    return data

def check_inputs(N, T, lambda_vacuum, theta):
    # check the dimensionalities of N:
    assert N.ndim == 3, 'N is not of shape [S x L x W] (3d), as it is of dimension ' + str(N.ndim)
    # check the dimensionalities of T:
    assert T.ndim == 2, 'T is not of shape [S x L] (2d), as it is of dimension ' + str(T.ndim)
    assert T.shape[0] == N.shape[0], 'The number of thin-films (first dimension) of N and T must coincide, \
    \nfound N.shape=' + str(N.shape) + ' and T.shape=' + str(T.shape) + ' instead!'
    assert T.shape[1] == N.shape[1], 'The number of thin-film layers (second dimension) of N and T must coincide, \
    \nfound N.shape=' + str(N.shape) + ' and T.shape=' + str(T.shape) + ' instead!'
    # check the dimensionality of Theta. The full grid is used internally for coherent
    # substacks whose injection medium is dispersive.
    assert theta.ndim in (1, 3), (
        'Theta is not of shape [A] (1d) or [S x A x W] (3d), as it is of shape '
        + str(tuple(theta.shape))
    )
    if theta.ndim == 3:
        assert theta.shape[0] == N.shape[0], (
            'The first dimension of a Theta grid must match the number of stacks, found '
            + str(tuple(theta.shape)) + ' and N.shape=' + str(tuple(N.shape))
        )
        assert theta.shape[2] == N.shape[2], (
            'The last dimension of a Theta grid must match the wavelengths, found '
            + str(tuple(theta.shape)) + ' and N.shape=' + str(tuple(N.shape))
        )
    # check the dimensionality of lambda_vacuum:
    assert lambda_vacuum.ndim == 1, 'lambda_vacuum is not of shape [W] (1d), as it is of dimension ' + str(lambda_vacuum.ndim)
    assert N.shape[-1] == lambda_vacuum.shape[0], 'The last dimension of N must coincide with the dimension of lambda_vacuum (W),\nfound N.shape[-1]=' + str(N.shape[-1]) + ' and lambda_vacuum.shape[0]=' + str(lambda_vacuum.shape[0]) + ' instead!'
    # check well defined property of refractive indicies for the first and last layer:
    # n * sin(theta) is the same in every layer by Snell's law, so this cancels to rounding
    # rather than exactly; comparing against a single epsilon makes the check fire on noise
    if theta.ndim == 1:
        injection = torch.einsum('ij,k->ijk', N[:, 0], torch.sin(theta)).imag.abs()
    else:
        injection = (N[:, 0, None, :] * torch.sin(theta)).imag.abs()
    assert torch.all(injection < 100 * EPSILON), (
        'Non well-defined refractive indicies detected for the first layer at index '
        + str(torch.argwhere(injection >= 100 * EPSILON).tolist()))
    
    
    


def make_2x2_tensor(a, b, c, d, dtype=float):
    """
    Makes a 2x2 numpy array of [[a,b],[c,d]]
    Same as "numpy.array([[a,b],[c,d]], dtype=float)", but ten times faster
    """
    my_array = torch.empty((2, 2), dtype=dtype)
    my_array[0, 0] = a
    my_array[0, 1] = b
    my_array[1, 0] = c
    my_array[1, 1] = d
    return my_array
