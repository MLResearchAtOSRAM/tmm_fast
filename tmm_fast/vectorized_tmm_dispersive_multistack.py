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
    T = converter2torch(T, device, dtype=torch.float64)
    lambda_vacuum = torch.atleast_1d(
        converter2torch(lambda_vacuum, device, dtype=torch.float64)
    )
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

    # The numerical core uses cos(theta) everywhere: in kz, the Fresnel coefficients, and
    # transmitted power. It therefore does not need to construct theta with asin. The private
    # angle argument remains as a compatibility fallback for older internal callers.
    snell_shape = (num_stacks, num_angles, num_layers, num_wavelengths)
    if _snell_thetas is not None:
        assert _snell_thetas.shape == snell_shape
    if _snell_cosines is None:
        if _snell_thetas is None:
            cos_SnellThetas = SnellLaw_cosines_vectorized(N, Theta, validate=_validate)
        else:
            cos_SnellThetas = select_forward_cosines(
                N, torch.cos(_snell_thetas), validate=_validate
            )
    else:
        assert _snell_cosines.shape == snell_shape
        cos_SnellThetas = select_forward_cosines(
            N, _snell_cosines, validate=_validate
        )


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
        pol, N[:, :-1, :], N[:, 1:, :], None,
        None, cos_SnellThetas[:, :, :-1, :],
        cos_SnellThetas[:, :, 1:, :]
    )
    r_list = interface_r_vec(
        pol, N[:, :-1, :], N[:, 1:, :], None,
        None, cos_SnellThetas[:, :, :-1, :],
        cos_SnellThetas[:, :, 1:, :]
    )
    
    # A ist the propagation term for matrix optic and holds the appropriate accumulated phase for the thickness
    # of each layer
    A = torch.exp(1j * delta).permute(0, 2, 1, 3)
    F = r_list[:, :, :, 1:]

    inverse_t0 = 1 / t_list[..., 0]
    reflected_t0 = r_list[..., 0] * inverse_t0

    if num_layers > 2:
        m00, m01, m10, m11 = _coherent_layer_components(
            A[..., 0], F[..., 0], t_list[..., 1]
        )
        for i in range(1, num_layers - 2):
            l00, l01, l10, l11 = _coherent_layer_components(
                A[..., i], F[..., i], t_list[..., i + 1]
            )
            m00, m01, m10, m11 = (
                m00 * l00 + m01 * l10,
                m00 * l01 + m01 * l11,
                m10 * l00 + m11 * l10,
                m10 * l01 + m11 * l11,
            )

        transfer00, transfer10 = (
            inverse_t0 * m00 + reflected_t0 * m10,
            reflected_t0 * m00 + inverse_t0 * m10,
        )
    else:
        transfer00 = inverse_t0
        transfer10 = reflected_t0

    # Net complex transmission and reflection amplitudes
    r = transfer10 / (transfer00 + np.finfo(float).eps)
    t = 1 / (transfer00 + np.finfo(float).eps)

    # Net transmitted and reflected power, as a proportion of the incoming light
    # power.
    R = R_from_r_vec(r)
    T = T_from_t_vec(
        pol, t, N[:, 0], N[:, -1], None, None,
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


def _coherent_layer_components(propagation, reflection, transmission):
    """Return one coherent layer's four transfer-matrix entries without stacking them.

    Keeping the entries as separate ``[stack, angle, wavelength]`` tensors avoids allocating
    a trailing 2x2 dimension for every layer. The caller can then apply the fixed 2x2 product
    explicitly, which is cheaper than constructing a matrix only to pass it to ``matmul``.
    """
    inverse_propagation = 1 / (propagation + np.finfo(float).eps)
    inverse_transmission = 1 / transmission
    reflected_transmission = reflection * inverse_transmission
    return (
        inverse_propagation * inverse_transmission,
        inverse_propagation * reflected_transmission,
        propagation * reflected_transmission,
        propagation * inverse_transmission,
    )

def _snell_sines(n, th):
    """Apply Snell's law and return ``sin(theta)`` in every layer.

    Snell's law states ``n_0 sin(theta_0) = n_j sin(theta_j)``. The left-hand side is
    fixed by the injection medium, so division by each layer index produces the complete
    ``[stack, angle, layer, wavelength]`` grid without calculating any angles.

    ``th`` is normally the one-dimensional incident-angle grid. The three-dimensional form
    ``[stack, angle, wavelength]`` is retained for an embedded substack whose injection angle
    is dispersive.
    """
    if th.ndim == 1:
        return torch.einsum('hk,j,hik->hjik', n[:, 0], torch.sin(th), 1 / n)
    if th.ndim == 3:
        return n[:, 0, None, None, :] * torch.sin(th[:, :, None, :]) / n[:, None]
    raise AssertionError(
        'Theta is not of shape [A] (1d) or [S x A x W] (3d), as it is of shape '
        + str(tuple(th.shape))
    )


def _snell_quantities(n, th, validate, return_angles):
    """Build the Snell quantities needed by either the solver or its diagnostic output.

    The solver needs only ``cos(theta_j)``. Once :func:`_snell_sines` has provided
    ``sin(theta_j)``, the identity ``cos(theta_j)**2 = 1 - sin(theta_j)**2`` gives the
    cosine through one complex square root. This avoids constructing ``theta_j`` with
    ``asin`` and immediately evaluating ``cos(theta_j)`` again.

    Angles are still constructed when ``return_angles`` is true because the full incoherent
    result exposes them as ``th_list``. Both paths return a tuple ``(angles, cosines)``;
    ``angles`` is ``None`` on the response-only path.
    """
    if th.dtype != torch.complex128:
        warn('there is some problem with theta, the dtype is not complex')
    if n.dtype != torch.complex128:
        warn('there is some problem with n, the dtype is not conplex')
    th = th if th.dtype == torch.complex128 else th.type(torch.complex128)
    n = n if n.dtype == torch.complex128 else n.type(torch.complex128)

    sines = _snell_sines(n, th)
    cosine_squared = 1 - sines * sines
    # Beyond a critical angle, cosine_squared is a negative real number. Its complex square
    # root can be +ij or -ij, and PyTorch selects between them from the sign of the zero
    # imaginary component. Reconstruct the same signed zero as cos(asin(sines)) so that the
    # direct path preserves the established evanescent-wave branch, including negative indices.
    zero = torch.zeros_like(cosine_squared.imag)
    negative_cut = torch.signbit(sines.real) == torch.signbit(sines.imag)
    cosine_squared = torch.complex(
        cosine_squared.real,
        torch.where(
            (cosine_squared.imag == 0) & (cosine_squared.real < 0),
            torch.where(negative_cut, -zero, zero),
            cosine_squared.imag,
        ),
    )
    cosines = torch.sqrt(cosine_squared)
    if not return_angles:
        return None, select_forward_cosines(n, cosines, validate=validate)

    angles = torch.asin(sines)
    return select_forward_angles(n, angles, cosines, validate=validate)


def SnellLaw_vectorized(n, th, validate=True, return_cosines=False):
    """
    return list of angle theta in each layer based on angle th_0 in layer 0,
    using Snell's law. n_list is index of refraction of each layer. Note that
    "angles" may be complex!!
    """
    angles, cosines = _snell_quantities(n, th, validate, return_angles=True)
    return (angles, cosines) if return_cosines else angles


def SnellLaw_cosines_vectorized(n, th, validate=True):
    """Return the Snell cosine grid without paying for an intermediate angle grid."""
    _, cosines = _snell_quantities(n, th, validate, return_angles=False)
    return cosines

def select_forward_angles(n, angles, cosines=None, validate=True):
    """Select the physically forward branches at the two semi-infinite boundaries.

    The transfer product is invariant to the branch selected in a finite interior layer, but
    the injection and exit media must describe waves travelling away from their respective
    boundaries. When cosines are already available, update them with the angle branch so both
    representations remain consistent.
    """
    angles = angles.clone()
    cosines = None if cosines is None else cosines.clone()
    for layer in (0, -1):
        layer_cosines = None if cosines is None else cosines[:, :, layer]
        backward = is_not_forward_angle(
            n[:, layer], angles[:, :, layer], layer_cosines, validate=validate
        )
        angles[:, :, layer] = torch.where(
            backward, pi - angles[:, :, layer], angles[:, :, layer]
        )
        if cosines is not None:
            cosines[:, :, layer] = torch.where(
                backward, -cosines[:, :, layer], cosines[:, :, layer]
            )
    return angles, cosines


def select_forward_cosines(n, cosines, validate=True):
    """Cosine-only counterpart of :func:`select_forward_angles`.

    Reversing a coherent substack can make a previously forward boundary cosine point
    backwards. Checking the first and last layer here lets parent Snell cosines be reused for
    forward and backward substack calculations without recreating angles.
    """
    cosines = cosines.clone()
    for layer in (0, -1):
        backward = is_not_forward_angle(
            n[:, layer], None, cosines[:, :, layer], validate=validate
        )
        cosines[:, :, layer] = torch.where(
            backward, -cosines[:, :, layer], cosines[:, :, layer]
        )
    return cosines


def is_not_forward_angle(n, theta, cos_theta=None, validate=True):
    """
    Return whether a propagation branch points backwards.

    The decision depends on ``n cos(theta)``, not on the angle itself. Callers may therefore
    pass ``theta=None`` with a precomputed ``cos_theta`` on the fast path. For evanescent or
    lossy waves the forward branch decays along +z; for propagating waves it has a positive
    Poynting vector. See Byrnes, arXiv:1603.02720, appendix D.
    """
    # n = [lambda]
    # theta = [theta, lambda]

    diagnostic_angle = theta
    if validate and not (n.real * n.imag >= 0).all():
        if diagnostic_angle is None:
            diagnostic_angle = torch.acos(cos_theta)
        raise AssertionError(
            "For materials with gain, it's ambiguous which beam is incoming vs outgoing. See "
            "https://arxiv.org/abs/1603.02720 Appendix C.\n"
            "n: " + str(n) + "   angle: " + str(diagnostic_angle)
        )
    n = n.unsqueeze(1)
    cos_theta = torch.cos(theta) if cos_theta is None else cos_theta
    ncostheta = cos_theta * n
    expected_shape = theta.shape if theta is not None else cos_theta.shape
    assert ncostheta.shape == expected_shape, 'ncostheta and theta shape doesnt match'
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
            if diagnostic_angle is None:
                diagnostic_angle = torch.acos(cos_theta)
            raise AssertionError(
                "It's not clear which beam is incoming vs outgoing. Weird index maybe?\n"
                "n: " + str(n.squeeze(1)) + "   angle: " + str(diagnostic_angle)
            )
    answer = ~answer

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
    return _complex_abs_squared(r)


def _complex_abs_squared(value: torch.Tensor) -> torch.Tensor:
    """Return ``|value|²`` without forming a magnitude or taking a square root."""
    return value.real.square() + value.imag.square()


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
        return _complex_abs_squared(t) * nf_thf / ni_thi

    elif pol == 'p':
        ni_thi = torch.real(torch.conj(cos_th_i) * n_i.unsqueeze(1))
        nf_thf = torch.real(torch.conj(cos_th_f) * n_f.unsqueeze(1))
        return _complex_abs_squared(t) * nf_thf / ni_thi

    else:
        raise ValueError("Polarization must be 's' or 'p'")

def resolve_device(data, device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.is_tensor(data):
        return data.device
    return torch.device('cpu')


def converter2torch(
    data, device: Union[str, torch.device], dtype: torch.dtype = torch.complex128
) -> torch.Tensor:
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
        if dtype == torch.float64 and torch.is_complex(data):
            data = data.real
        return data.to(device=device, dtype=dtype)
    try:
        array = np.asarray(data)
        if dtype == torch.float64 and np.iscomplexobj(array):
            array = array.real
        try:
            return torch.as_tensor(array, dtype=dtype, device=device)
        except (ValueError, RuntimeError):
            return torch.as_tensor(array.copy(), dtype=dtype, device=device)
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
    assert N.shape[1] >= 2, ('A stack must contain at least an injection and an '
                             'exit medium')
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
    
    
    
