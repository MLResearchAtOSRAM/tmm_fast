import numpy as np
import sys
from numpy import pi, seterr
from torch import cos, exp, conj, asin
import torch
EPSILON = sys.float_info.epsilon

def coh_vec_tmm_disp_mstack(pol, N, T, Theta, lambda_vacuum, device='cpu', timer=False):
    """
    Parallelized computation of reflection and transmission for coherent light spectra that traverse
    a bunch of multilayer thin-films with dispersive materials.
    This implementation in PyTorch naturally allows:
     - GPU accelerated computations
     - To compute gradients regarding the multilayer thin-film (i.e. N, T) thanks to Pytorch Autograd

    However, the input can also be a numpy array format.
    Although all internal computations are processed via PyTorch, the output data is converted to numpy arrays again.
    Hence, the use of numpy input may increase computation time due to data type conversions.

    Parameters
    ----------
    pol : Str
        Polarization of the light, accepts only 's' or 'p'
    N : Tensor or array
        PyTorch Tensor or numpy array of shape [S x L x W] with complex or real entries which contain the refractive
        indices at the wavelengths of interest:
        S is the number of multi-layer thin films, L is the number of layers for each thin film, W is the number of
        wavelength considered. Note that the first and last layer must feature real valued ()
        refractive indicies, i.e. imag(N[:, 0, :]) = 0 and imag(N[:, -1, :]) = 0.
    T : Tensor or array
        Holds the layer thicknesses of the individual layers for a bunch of thin films in nanometer.
        T is of shape [S x L] with real-valued entries; infinite values are allowed for the first and last layers only!
    Theta : Tensor or array
        Theta is a tensor or array that determines the angles with which the light propagates in the injection layer.
        Theta is of shape [A] and holds the incidence angles [rad] in its entries.
    lambda_vacuum : Tensor or numpy array
        Vacuum wavelengths for which reflection and transmission are computed given a bunch of thin films.
        It is of shape [W] and holds the wavelengths in nanometer.
    device : Str
        Computation device, accepts ether 'cuda' or 'cpu'; GPU acceleration can lower the computational time especially
        for computation involving large tensors
    timer: Boolean
        Determines whether to track times for data pushing on CPU or GPU and total computation time; see output
        information for details on how to read out time
    Returns
    output : Dict
        Keys:
            'r' : Tensor or array of Fresnel coefficients of reflection for each stack (over angle and wavelength)
            't' : Tensor or array of Fresnel coefficients of transmission for each stack (over angle and wavelength)
            'R' : Tensor or array of Reflectivity / Reflectance for each stack (over angle and wavelength)
            'T' : Tensor or array of Transmissivity / Transmittance for each stack (over angle and wavelength)
            Each of these tensors or arrays is of shape [S x A x W]
    optional output: list of two floats if timer=True
            first entry holds the pushtime [sec] that is the time required to push the input data on the specified
            device (i.e. cpu oder cuda), the second entry holds the total computation time [sec] (pushtime + tmm)

    Remarks and prior work from Byrnes:
    Upgrade to the regular coh_tmm from sbyrnes method. Does not perform checks and should
    be cross validated with coh_tmm in case of doubt.
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
    datatype = check_datatype(N, T, lambda_vacuum, Theta)
    # check uniform data types (e.g. only np.array or torch.tensor) -> save this type
    N = converter(N, device)
    T = converter(T, device)
    lambda_vacuum = converter(lambda_vacuum, device)
    Theta = converter(Theta, device)
    squeezed_N = False
    squeezed_T = False
    if N.ndim < 3:
        squeezed_N = True
        N = N.unsqueeze(0)
    if T.ndim < 2:
        squeezed_T = True
        T = T.unsqueeze(0)
    assert squeezed_N == squeezed_T, 'N and T are not of same shape, as they are of dimensions ' + str(N.ndim) + ' and ' + str(T.ndim)
    if timer:
        push_time = time.time() - starttime
    num_layers = T.shape[1]
    num_stacks = T.shape[0]
    num_angles = Theta.shape[0]
    num_wavelengths = lambda_vacuum.shape[0]
    check_inputs(N, T, lambda_vacuum, Theta)

    # if a constant refractive index is used (no dispersion) extend the tensor
    if N.ndim == 2:
       N = torch.tile(N, (num_wavelengths, 1)).T

    # SnellThetas is a tensor, for each stack and layer, the angle that the light travels
    # through the layer. Computed with Snell's law. Note that the "angles" may be complex!
    SnellThetas = SnellLaw_vectorized(N, Theta)

    theta = 2 * np.pi * torch.einsum('skij,sij->skij', cos(SnellThetas), N)  # [theta,d, lambda]
    kz_list = torch.einsum('sijk,k->skij', theta, 1 / lambda_vacuum)  # [lambda, theta, d]

    # kz is the z-component of (complex) angular wavevector for forward-moving
    # wave. Positive imaginary part means decaying.

    # delta is the total phase accrued by traveling through a given layer.
    # Ignore warning about inf multiplication
    olderr = seterr(invalid='ignore')
    delta = torch.einsum('skij,sj->skij', kz_list, T)
    seterr(**olderr)

    # t_list and r_list hold the transmission and reflection coefficients from
    # the Fresnel Equations

    t_list = interface_t_vec(pol, N[:, :-1, :], N[:, 1:, :], SnellThetas[:, :, :-1, :], SnellThetas[:, :, 1:, :])
    r_list = interface_r_vec(pol, N[:, :-1, :], N[:, 1:, :], SnellThetas[:, :, :-1, :], SnellThetas[:, :, 1:, :])

    A = exp(1j * delta[:, :, :, 1:-1])
    F = r_list[:, :, :, 1:]
    #     # A ist the propagation term for matrix optic and holds the appropriate accumulated phase for the thickness
    #     # of each layer

    # M_list holds the transmission and reflection matrices from matrix-optics
    # alex:
    M_list = torch.zeros((num_stacks, num_angles, num_wavelengths, num_layers, 2, 2), dtype=torch.cfloat, device=device)
    M_list[:, :, :, 1:-1, 0, 0] = torch.einsum('shji,sjhi->sjhi', 1 / A, 1 / t_list[:, :, :, 1:])
    M_list[:, :, :, 1:-1, 0, 1] = torch.einsum('shji,sjhi->sjhi', 1 / A, F / t_list[:, :, :, 1:])
    M_list[:, :, :, 1:-1, 1, 0] = torch.einsum('shji,sjhi->sjhi', A, F / t_list[:, :, :, 1:])
    M_list[:, :, :, 1:-1, 1, 1] = torch.einsum('shji,sjhi->sjhi', A, 1 / t_list[:, :, :, 1:])
    Mtilde = torch.empty((num_stacks, num_angles, num_wavelengths, 2, 2), dtype=torch.cfloat, device=device)
    Mtilde[:, :, :] = make_2x2_tensor(1, 0, 0, 1, dtype=torch.cfloat)

    # contract the M_list matrix along the dimension of the layers, all
    for i in range(1, num_layers - 1):
        Mtilde = torch.einsum('sijkl,sijlm->sijkm', Mtilde, M_list[:, :, :, i])

    # M_r0 accounts for the first and last stack where the translation coefficients are 1
    # todo: why compute separately?
    M_r0 = torch.empty((num_stacks, num_angles, num_wavelengths, 2, 2), dtype=torch.cfloat, device=device)
    M_r0[:, :, :, 0, 0] = 1
    M_r0[:, :, :, 0, 1] = r_list[:, :, :, 0]
    M_r0[:, :, :, 1, 0] = r_list[:, :, :, 0]
    M_r0[:, :, :, 1, 1] = 1
    M_r0 = torch.einsum('sijkl,sij->sijkl', M_r0, 1 / t_list[:, :, :, 0])

    Mtilde = torch.einsum('shijk,shikl->shijl', M_r0, Mtilde)

    # Net complex transmission and reflection amplitudes
    r = Mtilde[:, :, :, 1, 0] / Mtilde[:, :, :, 0, 0]
    t = 1 / Mtilde[:, :, :, 0, 0]

    # Net transmitted and reflected power, as a proportion of the incoming light
    # power.
    R = R_from_r(r)
    T = T_from_t_vec(pol, t, N[:, 0], N[:, -1], SnellThetas[:, :, 0], SnellThetas[:, :, -1])

    if squeezed_T and r.shape[0] == 1:
        r = torch.reshape(r, (r.shape[1], r.shape[2]))
        R = torch.reshape(R, (R.shape[1], R.shape[2]))
        T = torch.reshape(T, (T.shape[1], T.shape[2]))
        t = torch.reshape(t, (t.shape[1], t.shape[2]))

    if datatype is np.ndarray:
        r = numpy_converter(r)
        t = numpy_converter(t)
        R = numpy_converter(R)
        T = numpy_converter(T)

    if timer:
        total_time = time.time() - starttime
        return {'r': r, 't': t, 'R': R, 'T': T}, [push_time, total_time]
    else:
        return {'r': r, 't': t, 'R': R, 'T': T}

def SnellLaw_vectorized(n, th):
    """
    return list of angle theta in each layer based on angle th_0 in layer 0,
    using Snell's law. n_list is index of refraction of each layer. Note that
    "angles" may be complex!!
    """
    # Important that the arcsin here is numpy.lib.scimath.arcsin, not
    # numpy.arcsin! (They give different results e.g. for arcsin(2).)

    sin_th = torch.unsqueeze(torch.sin(th), dim=0)
    n0 = torch.unsqueeze(n[:, 0], dim=-1)
    n0th = torch.matmul(n0, sin_th)
    assert n0th.shape == (n.shape[0], n.shape[-1], th.shape[0]), (n.shape[-1], th.shape[0])
    angles = asin(torch.einsum('sij,ski->sjki', n0th, 1/n))
    assert angles.shape == (n.shape[0], th.shape[0], n.shape[1], n.shape[2]), (th.shape[0], n.shape[0], n.shape[1])

    # dim(angles) = [dim_theta, dim_d, dim_lambda]
    # The first and last entry need to be the forward angle (the intermediate
    # layers don't matter, see https://arxiv.org/abs/1603.02720 Section 5)

    angles[:, :, 0] = -is_not_forward_angle(n[:, 0], angles[:, :,  0]) * pi + angles[:, :, 0]
    angles[:, :, -1] = -is_not_forward_angle(n[:, -1], angles[:, :, -1]) * pi + angles[:, :, -1]
    return angles

def is_not_forward_angle(n, theta):
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

    assert (n.real * n.imag >= 0).all(), ("For materials with gain, it's ambiguous which "
                                          "beam is incoming vs outgoing. See "
                                          "https://arxiv.org/abs/1603.02720 Appendix C.\n"
                                          "n: " + str(n) + "   angle: " + str(theta))
    n = n.unsqueeze(1)
    ncostheta = cos(theta) * n
    assert ncostheta.shape == theta.shape, 'ncostheta and theta shape doesnt match'
    answer = (~(abs(ncostheta.imag) > 100 * EPSILON)) * (ncostheta.real > 0)
    error_string = ("It's not clear which beam is incoming vs outgoing. Weird index maybe?\n"
                    "n: " + str(n) + "   angle: " + str(theta))

    assert (ncostheta.imag > -100 * EPSILON)[answer].all(), error_string
    assert (ncostheta.real > -100 * EPSILON)[answer].all(), error_string
    assert ((n * cos(torch.conj(theta))).real > -100 * EPSILON)[answer].all(), error_string
    assert (ncostheta.imag < 100 * EPSILON)[~answer].all(), error_string
    assert (ncostheta.real < 100 * EPSILON)[~answer].all(), error_string
    assert ((n * cos(torch.conj(theta))).real < 100 * EPSILON)[~answer].all(), error_string
    answer = ~answer.clone().detach().long()
    return answer

def is_forward_angle(n, theta):
    """
    if a wave is traveling at angle theta from normal in a medium with index n,
    calculate whether or not this is the forward-traveling wave (i.e., the one
    going from front to back of the stack, like the incoming or outgoing waves,
    but unlike the reflected wave). For real n & theta, the criterion is simply
    -pi/2 < theta < pi/2, but for complex n & theta, it's more complicated.
    See https://arxiv.org/abs/1603.02720 appendix D. If theta is the forward
    angle, then (pi-theta) is the backward angle and vice-versa.
    """
    n = n.clone().detach().to(torch.cfloat)  # torch.tensor(n, dtype=torch.cfloat)
    assert torch.all(n.real * n.imag >= 0), ("For materials with gain, it's ambiguous which "
                                  "beam is incoming vs outgoing. See "
                                  "https://arxiv.org/abs/1603.02720 Appendix C.\n"
                                  "n: " + str(n) + "   angle: " + str(theta))
    # assert n.dtype is not complex,  ("For materials with gain, it's ambiguous which "
    #                               "beam is incoming vs outgoing. See "
    #                               "https://arxiv.org/abs/1603.02720 Appendix C.\n"
    #                               "n: " + str(n) + "   angle: " + str(theta))

    ncostheta = n * cos(theta)
    ncostheta = ncostheta.clone().detach().to(torch.cfloat)  # torch.tensor(ncostheta, dtype=torch.cfloat)
    if torch.all(abs(ncostheta.imag) > 100 * EPSILON):
        # Either evanescent decay or lossy medium. Either way, the one that
        # decays is the forward-moving wave
        answer = (ncostheta.imag > 0)
    else:
        # Forward is the one with positive Poynting vector
        # Poynting vector is Re[n cos(theta)] for s-polarization or
        # Re[n cos(theta*)] for p-polarization, but it turns out they're consistent
        # so I'll just assume s then check both below
        answer = torch.any((ncostheta.real > 0))
    # convert from numpy boolean to the normal Python boolean
    answer = bool(answer)
    # double-check the answer ... can't be too careful!
    error_string = ("It's not clear which beam is incoming vs outgoing. Weird"
                    " index maybe?\n"
                    "n: " + str(n) + "   angle: " + str(theta))
    if answer is True:
        assert torch.all(ncostheta.imag > -100 * EPSILON), error_string
        assert torch.all(ncostheta.real > -100 * EPSILON), error_string
        assert torch.all((n * cos(theta.conj())).real > -100 * EPSILON), error_string
    else:
        assert torch.all(ncostheta.imag < 100 * EPSILON), error_string
        assert torch.all(ncostheta.real < 100 * EPSILON), error_string
        assert torch.all((n * cos(theta.conjugate())).real < 100 * EPSILON), error_string
    return answer

def interface_r_vec(polarization, n_i, n_f, th_i, th_f):
    """
    reflection amplitude (from Fresnel equations)
    polarization is either "s" or "p" for polarization
    n_i, n_f are (complex) refractive index for incident and final
    th_i, th_f are (complex) propegation angle for incident and final
    (in radians, where 0=normal). "th" stands for "theta".
    """
    if polarization == 's':
        ni_thi = torch.einsum('sij,skij->skji', n_i, torch.cos(th_i))
        nf_thf = torch.einsum('sij,skij->skji', n_f, torch.cos(th_f))
        return (ni_thi - nf_thf) / (ni_thi + nf_thf)
    elif polarization == 'p':
        nf_thi = torch.einsum('sij,skij->skji', n_f, torch.cos(th_i))
        ni_thf = torch.einsum('sij,skij->skji', n_i, torch.cos(th_f))
        return (nf_thi - ni_thf) / (nf_thi + ni_thf)
    else:
        raise ValueError("Polarization must be 's' or 'p'")

def interface_t_vec(polarization, n_i, n_f, th_i, th_f):
    """
    transmission amplitude (frem Fresnel equations)
    polarization is either "s" or "p" for polarization
    n_i, n_f are (complex) refractive index for incident and final
    th_i, th_f are (complex) propegation angle for incident and final
    (in radians, where 0=normal). "th" stands for "theta".
    """
    if polarization == 's':
        ni_thi = torch.einsum('sij,skij->skji', n_i, torch.cos(th_i))
        nf_thf = torch.einsum('sij,skij->skji', n_f, torch.cos(th_f))
        return 2 * ni_thi / (ni_thi + nf_thf)
    elif polarization == 'p':
        nf_thi = torch.einsum('sij,skij->skji', n_f, torch.cos(th_i))
        ni_thf = torch.einsum('sij,skij->skji', n_i, torch.cos(th_f))
        ni_thi = torch.einsum('sij,skij->skji', n_i, torch.cos(th_i))
        return 2 * ni_thi / (nf_thi + ni_thf)
    else:
        raise ValueError("Polarization must be 's' or 'p'")

def R_from_r(r):
    """
    Calculate reflected power R, starting with reflection amplitude r.
    """
    return abs(r) ** 2

def T_from_t_vec(pol, t, n_i, n_f, th_i, th_f):
    """
    Calculate transmitted power T, starting with transmission amplitude t.
    n_i, n_f are refractive indices of incident and final medium.
    th_i, th_f are (complex) propagation angles through incident & final medium
    (in radians, where 0=normal). "th" stands for "theta".
    In the case that n_i, n_f, th_i, th_f are real, formulas simplify to
    T=|t|^2 * (n_f cos(th_f)) / (n_i cos(th_i)).
    See manual for discussion of formulas
    """

    if pol == 's':
        ni_thi = torch.real(cos(th_i) * n_i.unsqueeze(1))
        nf_thf = torch.real(cos(th_f) * n_f.unsqueeze(1))
        return (abs(t ** 2) * ((nf_thf) / (ni_thi)))

    elif pol == 'p':
        ni_thi = torch.real(conj(cos(th_i)) * n_i.unsqueeze(1))
        nf_thf = torch.real(conj(cos(th_f)) * n_f.unsqueeze(1))
        return (abs(t ** 2) * ((nf_thf) / (ni_thi)))

    else:
        raise ValueError("Polarization must be 's' or 'p'")

def converter(data, device):
    if type(data) is not torch.Tensor:
        if type(data) is np.ndarray:
            data = torch.from_numpy(data.copy())
        else:
            raise ValueError('At least one of the inputs (i.e. N, Theta, ...) is not of type numpy.array or torch.Tensor!')
    data = data.type(torch.cfloat).to(device)
    return data.squeeze()

def numpy_converter(data):
    data = data.detach().cpu().numpy()
    return data

def check_datatype(N, T, lambda_vacuum, Theta):
    assert type(N) == type(T) == type(lambda_vacuum) == type(Theta), ValueError('All inputs (i.e. N, Theta, ...) must be of the same data type, i.e. numpy.array or torch.Tensor!')
    return type(N)

def check_inputs(N, T, lambda_vacuum, Theta):
    # check the dimensionalities of N:
    assert N.ndim == 3, 'N is not of shape [S x L x W] (3d), as it is of dimension ' + str(N.ndim)
    # check the dimensionalities of T:
    assert T.ndim == 2, 'T is not of shape [S x L] (2d), as it is of dimension ' + str(T.ndim)
    assert T.shape[0] == N.shape[0] and T.shape[1] == N.shape[1], 'First and second dimension of N and T must coincide,\nfound N.shape=' + str(N.shape) + ' and T.shape=' + str(T.shape) + ' instead!'
    # check the dimensionality of Theta:
    assert Theta.ndim == 1, 'Theta is not of shape [A] (1d), as it is of dimension ' + str(Theta.ndim)
    # check the dimensionality of lambda_vacuum:
    assert lambda_vacuum.ndim == 1, 'lambda_vacuum is not of shape [W] (1d), as it is of dimension ' + str(lambda_vacuum.ndim)
    assert N.shape[-1] == lambda_vacuum.shape[0], 'The last dimension of N must coincide with the dimension of lambda_vacuum (W),\nfound N.shape[-1]=' + str(N.shape[-1]) + ' and lambda_vacuum.shape[0]=' + str(lambda_vacuum.shape[0]) + ' instead!'
    # check non-imaginary property of refractive indicies for the first and last layer:
    answer = torch.any((N[:, 0, :].imag > 0))
    assert not answer, 'Non real-valued refractive indicies detected for first layer, i.g. N[:, 0, w].imag > 0 for some valid w!'
    answer = torch.any((N[:, -1, :].imag > 0))
    assert not answer, 'Non real-valued refractive indicies detected for last layer, i.g. N[:, -1, w].imag > 0 for some valid w!'

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
