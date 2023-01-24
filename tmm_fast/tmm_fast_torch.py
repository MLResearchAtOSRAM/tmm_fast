from tmm_fast_core import interface_t, interface_r
import numpy as np
from numpy import inf, nan, isnan, pi, seterr

from torch import cos, zeros, exp, conj, sin, asin
import torch

from warnings import warn
import sys
EPSILON = sys.float_info.epsilon

def make_2x2_tensor(a, b, c, d, dtype=float):
    """
    Creates a 2x2 torch array of [[a,b],[c,d]]
    Same as "torch.array([[a,b],[c,d]], dtype=float)", but ten times faster

    Parameters
    ----------
    a : int
        upper left entry
    b : int
        upper right entry
    c : int
        lower left entry
    d : int
        lower right entry
    dtype :
         (Default value = float)

    Returns
    -------
    my_array : torch tensor

    """
    my_array = torch.empty((2,2), dtype=dtype)
    my_array[0,0] = a
    my_array[0,1] = b
    my_array[1,0] = c
    my_array[1,1] = d
    return my_array

def is_not_forward_angle(n, theta):
    """
    Checks if the traveling waves are propagating in forward direction or not

    Parameters
    ----------
    n : torch tensor
        refractive indices 
    theta : torch tensor
        angles of incidence

    Returns
    -------
    answer : torch tensor (dtype=int)
        if any value is zero, an error is displayed

    Notes
    -----
    if a wave is traveling at angle theta from normal in a medium with index n,
    calculate whether or not this is the forward-traveling wave (i.e., the one
    going from front to back of the stack, like the incoming or outgoing waves,
    but unlike the reflected wave). For real n & theta, the criterion is simply
    -pi/2 < theta < pi/2, but for complex n & theta, it's more complicated.
    See https://arxiv.org/abs/1603.02720 appendix D. If theta is the forward
    angle, then (pi-theta) is the backward angle and vice-versa.

    """
    #n = [lambda]
    #theta = [theta, lambda]
    
    assert (n.real * n.imag >= 0).all(), ("For materials with gain, it's ambiguous which "
                                          "beam is incoming vs outgoing. See "
                                          "https://arxiv.org/abs/1603.02720 Appendix C.\n"
                                          "n: " + str(n) + "   angle: " + str(theta))
    
    
    ncostheta = cos(theta) * n
    assert ncostheta.shape == theta.shape, 'ncostheta and theta shape doesnt match'
    
    answer = torch.zeros(theta.shape, dtype=bool)
#         # Either evanescent decay or lossy medium. Either way, the one that
#         # decays is the forward-moving wave
    answer = (abs(ncostheta.imag) > 100 * EPSILON) * (ncostheta.imag > 0)
#         # Forward is the one with positive Poynting vector
#         # Poynting vector is Re[n cos(theta)] for s-polarization or
#         # Re[n cos(theta*)] for p-polarization, but it turns out they're consistent
#         # so I'll just assume s then check both below
    answer = (~(abs(ncostheta.imag) > 100 * EPSILON)) * (ncostheta.real > 0)
    

    # double-check the answer ... can't be too careful!
    error_string = ("It's not clear which beam is incoming vs outgoing. Weird"
                    " index maybe?\n"
                    "n: " + str(n) + "   angle: " + str(theta))

    assert (ncostheta.imag > -100 * EPSILON)[answer].all(), error_string
    assert (ncostheta.real > -100 * EPSILON)[answer].all(), error_string
    assert ((n * cos(theta.conjugate())).real > -100 * EPSILON)[answer].all(), error_string
    assert (ncostheta.imag < 100 * EPSILON)[~answer].all(), error_string
    assert (ncostheta.real < 100 * EPSILON)[~answer].all(), error_string
    assert ((n * cos(theta.conjugate())).real < 100 * EPSILON)[~answer].all(), error_string
    return torch.tensor(~answer, dtype=int)

def is_forward_angle(n, theta):
    """
    Checks if the traveling waves are propagating in forward direction or not

    Parameters
    ----------
    n : torch tensor
        refractive indices 
    theta : torch tensor
        angles of incidence

    Returns
    -------
    answer : torch tensor (dtype=int)
        if any value is zero, an error is displayed
        

    Notes 
    -----
    if a wave is traveling at angle theta from normal in a medium with index n,
    calculate whether or not this is the forward-traveling wave (i.e., the one
    going from front to back of the stack, like the incoming or outgoing waves,
    but unlike the reflected wave). For real n & theta, the criterion is simply
    -pi/2 < theta < pi/2, but for complex n & theta, it's more complicated.
    See https://arxiv.org/abs/1603.02720 appendix D. If theta is the forward
    angle, then (pi-theta) is the backward angle and vice-versa.
    """
    n = n.clone().detach().to(torch.cfloat) # torch.tensor(n, dtype=torch.cfloat)
    assert n.real * n.imag >= 0, ("For materials with gain, it's ambiguous which "
                                  "beam is incoming vs outgoing. See "
                                  "https://arxiv.org/abs/1603.02720 Appendix C.\n"
                                  "n: " + str(n) + "   angle: " + str(theta))
    # assert n.dtype is not complex,  ("For materials with gain, it's ambiguous which "
    #                               "beam is incoming vs outgoing. See "
    #                               "https://arxiv.org/abs/1603.02720 Appendix C.\n"
    #                               "n: " + str(n) + "   angle: " + str(theta))

    ncostheta = n * cos(theta)
    ncostheta = ncostheta.clone().detach().to(torch.cfloat) # torch.tensor(ncostheta, dtype=torch.cfloat)
    if abs(ncostheta.imag) > 100 * EPSILON:
        # Either evanescent decay or lossy medium. Either way, the one that
        # decays is the forward-moving wave
        answer = (ncostheta.imag > 0)
    else:
        # Forward is the one with positive Poynting vector
        # Poynting vector is Re[n cos(theta)] for s-polarization or
        # Re[n cos(theta*)] for p-polarization, but it turns out they're consistent
        # so I'll just assume s then check both below
        answer = (ncostheta.real > 0)
    # convert from numpy boolean to the normal Python boolean
    answer = bool(answer)
    # double-check the answer ... can't be too careful!
    error_string = ("It's not clear which beam is incoming vs outgoing. Weird"
                    " index maybe?\n"
                    "n: " + str(n) + "   angle: " + str(theta))
    if answer is True:
        assert ncostheta.imag > -100 * EPSILON, error_string
        assert ncostheta.real > -100 * EPSILON, error_string
        assert (n * cos(theta.conj())).real > -100 * EPSILON, error_string
    else:
        assert ncostheta.imag < 100 * EPSILON, error_string
        assert ncostheta.real < 100 * EPSILON, error_string
        assert (n * cos(theta.conjugate())).real < 100 * EPSILON, error_string
    return answer

def list_snell(n_list, th_0):
    """
    Computes Snell's law for a given angle of incidence throughout all layers

    Parameters
    ----------
    n_list : torch tensor
        refractive indices of all layers
    th_0 : float
        angle of incidence of the incoming light, 0 means normal incidence
        unit radian        

    Returns
    -------
    angles : torch tensor
        propagation angle of the light beam in all layers 
    
    Notes
    -----
        Computes angles of refraction using Snell's law. n_list is index of 
        refraction of each layer. Note that "angles" may be complex!!

    """
    # Important that the arcsin here is numpy.lib.scimath.arcsin, not
    # numpy.arcsin! (They give different results e.g. for arcsin(2).)
    angles = asin(n_list[0]*sin(th_0) / n_list)
    # The first and last entry need to be the forward angle (the intermediate
    # layers don't matter, see https://arxiv.org/abs/1603.02720 Section 5)
    if not is_forward_angle(n_list[0], angles[0]):
        angles[0] = pi - angles[0]
    if not is_forward_angle(n_list[-1], angles[-1]):
        angles[-1] = pi - angles[-1]
    return angles


def list_snell_vec(n_list, th):

    """

    Parameters
    ----------
    n_list : torch tensor
        refractive indices of all layers
    th : torch tensor
        angle of incidence of all incoming light rays, 0 means normal incidence
        unit radian        

    Returns
    -------
    angles : torch tensor
        propagation angle of the light beam in all layers 

    Notes
    -----
        Computes angles of refraction using Snell's law. n_list is index of 
        refraction of each layer. Note that "angles" may be complex!!

    """
    # Important that the arcsin here is numpy.lib.scimath.arcsin, not
    # numpy.arcsin! (They give different results e.g. for arcsin(2).)
    
    sin_th = torch.unsqueeze(torch.sin(theta_incidence), dim=0)
    n0 = torch.unsqueeze(n_list[0], dim=-1)
    n0th = torch.matmul(n0, sin_th)
    assert n0th.shape == (n_list.shape[-1], theta_incidence.shape[0]), (n_list.shape[-1], theta_incidence.shape[0]) # [lambda, theta]
    n = 1/n_list # [lambda, d]    
    angles = asin(torch.einsum('ij,ki->jki', n0th, n))
    assert angles.shape == (theta_incidence.shape[0], n_list.shape[0], n_list.shape[1]), (theta_incidence.shape[0], n_list.shape[0], n_list.shape[1])
    
    # dim(angles) = [dim_theta, dim_d, dim_lambda]
    # The first and last entry need to be the forward angle (the intermediate
    # layers don't matter, see https://arxiv.org/abs/1603.02720 Section 5)

    angles[:, 0] = -is_not_forward_angle(n_list[0], angles[:, 0]) * pi + angles[:, 0]
    angles[:, -1] = -is_not_forward_angle(n_list[-1], angles[:, -1]) * pi + angles[:, -1]
    return angles

def interface_r_vec(polarization, n_i, n_f, th_i, th_f):
    """
    Computes the reflection amplitude (from Fresnel equations) at 
    the interface between two flat planes with different refractive index
    
    Parameters
    ----------
    polarization : str
        's' or 'p' polarization
    n_i : torch tensor
        refractive indices of the incident layers
    n_f : torch tensor
        refractive indices of the final layers
    th_i : torch tensor
        propagation angle in the incident layer
    th_f : torch tensor
        propagation angle in the final layer

    Returns
    -------
    r : torch tensor
        coefficient of reflection depending on the polarization

    Notes
    -----
    polarization is either "s" or "p" for polarization
    
    n_i, n_f are (complex) refractive index for incident and final
    
    th_i, th_f are (complex) propegation angle for incident and final
    (in radians, where 0=normal). "th" stands for "theta".
    """
    if polarization == 's':
        ni_thi = torch.einsum('ij,kij->kji', n_i, torch.cos(th_i))
        nf_thf = torch.einsum('ij,kij->kji', n_f, torch.cos(th_f))
        return (ni_thi - nf_thf)/(ni_thi + nf_thf)
    elif polarization == 'p':
        nf_thi = torch.einsum('ij,kij->kji', n_f, torch.cos(th_i))
        ni_thf = torch.einsum('ij,kij->kji', n_i, torch.cos(th_f))
        return (nf_thi - ni_thf)/(nf_thi + ni_thf)
    else:
        raise ValueError("Polarization must be 's' or 'p'")

def interface_t_vec(polarization, n_i, n_f, th_i, th_f):
    """
    Computes the transmission amplitude (frem Fresnel equations) at 
    the interface between two flat planes with different refractive index

    Parameters
    ----------
    polarization : str
        's' or 'p' polarization
    n_i : torch tensor
        refractive indices of the incident layers
    n_f : torch tensor
        refractive indices of the final layers
    th_i : torch tensor
        propagation angle in the incident layer
    th_f : torch tensor
        propagation angle in the final layer        

    Returns
    -------
    t : torch tensor
        coefficient of transmission depending on the polarization

    Notes
    -----
    polarization is either "s" or "p" for polarization
    
    n_i, n_f are (complex) refractive index for incident and final
    
    th_i, th_f are (complex) propegation angle for incident and final
    (in radians, where 0=normal). "th" stands for "theta".

    """
    if polarization == 's':
        ni_thi = torch.einsum('ij,kij->kji', n_i, torch.cos(th_i))
        nf_thf = torch.einsum('ij,kij->kji', n_f, torch.cos(th_f))        
        return 2 * ni_thi / (ni_thi + nf_thf)
    elif polarization == 'p':
        nf_thi = torch.einsum('ij,kij->kji', n_f, torch.cos(th_i))
        ni_thf = torch.einsum('ij,kij->kji', n_i, torch.cos(th_f))
        ni_thi = torch.einsum('ij,kij->kji', n_i, torch.cos(th_i))
        return 2 * ni_thi / (nf_thi + ni_thf)
    else:
        raise ValueError("Polarization must be 's' or 'p'")

def R_from_r(r):
    """
    Calculate reflected power R from the reflection amplitude r
    of the Fresnel equations.

    Parameters
    ----------
    r : array like
        Fresnel reflection coefficients

    Returns
    -------
    R : array like
        Reflectivity
    """
    return abs(r)**2


def coh_tmm_fast_disp(pol, n_list, d_list, theta_incidence, lambda_vacuum):
    """
    Computes coherent Reflection and Transmission spectra for a given multilayer thin-film
    over a broad range of wavelengths and angles of incidence. 
    Requires dispersive materials. 
    The injection- and outcoupling layer MUST have a complex refractive 
    index of zero (ie. no light absorption or amplification) since the in- and 
    outgoing light wave become ambiguous. 

    This implementaion uses PyTorch functions and can be used with Pytorch 
    Autograd. 
    
    The method is based of the initial implementaion of Steven Byrnes.

    Parameters
    ----------
    pol : str 
        's' or 'p'
    n_list : torch tensor 
        refractive indices of the layer, starting with the injection layer
        This method can handle dispersive materials. 
        Note that the first and last layer must 
        be real valued (imag(n_list[0 and -1]) must be 0 for all wavelenghts). 
        shape [<number of layers>, <number of wavelength points>]
    d_list : torch tensor
        list of layer thicknesses of all layers, starting with the injection layer
        unit meter, shape [<number of layers>]
        For optimization, it can be beneficial to use the thickness in µm since 
        many optimizers will terminate if the gradients become to small.
    theta_incidence : torch tensor
        concidered angles of incidence of the incoming light at the firts interface
        unit radian, shape [<number of concidered angles of incidence>]       
    lambda_vacuum : torch tensor
        concidered wavelengths for the computation
        unit meter, shape [<number of concidered wavelength points>]
        For optimization, it can be beneficial to give the wavelengths in µm. 

    Returns
    -------
    res : dict
        'R' : torch tensor 
            Reflectivity of the computed multilayer thin-film over all concidered 
            wavelengths and angles of incidence
            unit absolut reflected power, 
            shape [<number of concidered angles of incidence>, <number of concidered wavelength points>]  
        'T' :  torch tensor - TODO
            Reflectivity of the computed multilayer thin-film over all concidered 
            wavelengths and angles of incidence
            unit absolut transmitted power, 
            shape [<number of concidered angles of incidence>, <number of concidered wavelength points>]
        'r' : torch tensor
            Fresnel coefficient of reflection of the thin-film
            shape [<number of concidered angles of incidence>, <number of concidered wavelength points>]
        't' : torch tensor
            Fresnel coefficient of transmission of the thin-film
            shape [<number of concidered angles of incidence>, <number of concidered wavelength points>]

    Notes
    -----
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
    
    lambda_vacuum is vacuum wavelength of the light.
    
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
    * pol, n_list, d_list, th_0, lambda_vacuum--same as input
    """
    warn()

    # n_list holds refractive indices of every layer, beginning with the layer where the light enters the stack
    # d_list holds the thickness of every layer, same order as n_list
    # lambda_vacuum holds the vacuum wavelength of all wavelengths of interest
    n_list = torch.tensor(n_list)
    d_list = torch.tensor(d_list, dtype=float)
    lambda_vacuum = torch.tensor(lambda_vacuum)

    num_layers = d_list.size
    num_angles = theta_incidence.size
    num_lambda = lambda_vacuum.size

    # th_list is a list with, for each layer, the angle that the light travels
    # through the layer. Computed with Snell's law. Note that the "angles" may be
    # complex!
        
    # if a constant refractive index is used (no dispersion) extend the tensor
    # TODO: call coh_tmm_fast for n(lambda)=const
    if n_list.ndim == 1:
        n_list = torch.tile(n_list, (num_lambda,1)).T
        
    th_list = list_snell_vec(n_list, theta_incidence)

    optical_pathlength = 2 * np.pi * torch.einsum('kij,ij->kij', cos(th_list), n_list)   # [theta,d, lambda]
    # kz is the z-component of (complex) angular wavevector for forward-moving
    # wave. Positive imaginary part means decaying.
    kz_list = torch.einsum('ijk,k->kij', optical_pathlength, 1/lambda_vacuum) #[lambda, theta, d]

    # delta is the total phase accrued by traveling through a given layer.
    # Ignore warning about inf multiplication
    olderr = seterr(invalid='ignore')
    delta = kz_list * d_list
    seterr(**olderr)
    
    # t_list and r_list hold the transmission and reflection coefficients from 
    # the Fresnel Equations
    t_list = interface_t_vec(pol, n_list[:-1, :], n_list[1:, :], th_list[:, :-1, :], th_list[:, 1:, :])
    r_list = interface_r_vec(pol, n_list[:-1, :], n_list[1:, :], th_list[:, :-1, :], th_list[:, 1:, :])

    # A ist the propagation term for matrix optic and holds the appropriate accumulated phase for the thickness
    # of each layer
    A = exp(1j*delta[:,:, 1:-1]) # [lambda, theta, n], n without the first and last layer as they function as injection 
    # and measurement layer
    F = r_list[:, :, 1:]

    
    # M_list holds the transmission and reflection matrices from matrix-optics
    M_list = torch.zeros((num_angles, num_lambda, num_layers, 2, 2), dtype=torch.cfloat)
    M_list[:, :, 1:-1, 0, 0] = torch.einsum('hji,jhi->jhi', 1 / A, 1/t_list[:, :, 1:] )   
    M_list[:, :, 1:-1, 0, 1] = torch.einsum('hji,jhi->jhi', 1 / A, F / t_list[:, :, 1:]) 
    M_list[:, :, 1:-1, 1, 0] = torch.einsum('hji,jhi->jhi', A, F / t_list[:, :, 1:])  
    M_list[:, :, 1:-1, 1, 1] = torch.einsum('hji,jhi->jhi', A, 1 / t_list[:, :, 1:]) 

    Mtilde = torch.empty((num_angles, num_lambda, 2, 2), dtype=torch.cfloat)
    Mtilde[:, :] = make_2x2_tensor(1, 0, 0, 1, dtype=torch.cfloat)
    
    # contract the M_list matrix along the dimension of the layers, all
    for i in range(1, num_layers-1):
        Mtilde = torch.einsum('ijkl,ijlm->ijkm', Mtilde, M_list[:,:,i])

    
    # M_r0 accounts for the first and last stack where the translation coefficients are 1 
    # TODO: why compute separately?
    M_r0 = torch.empty((num_angles, num_lambda, 2, 2), dtype=torch.cfloat)
    M_r0[:, :,0, 0] = 1
    M_r0[:, :, 0, 1] = r_list[:, :, 0]
    M_r0[:, :, 1, 0] = r_list[:, :, 0]
    M_r0[:, :, 1, 1] = 1
    M_r0 = torch.einsum('ijkl,ij->ijkl', M_r0, 1/t_list[:,:,0])
    Mtilde = torch.einsum('hijk,hikl->hijl', M_r0 , Mtilde)

    # Net complex transmission and reflection amplitudes
    r = Mtilde[:, :, 1,0] / Mtilde[:, :, 0,0]
    t = 1 / Mtilde[:, :, 0,0]
    
    # Net transmitted and reflected power, as a proportion of the incoming light
    # power.
    R = R_from_r(r)
    # TODO: Implement Transmission
    # T=None
    T = T_from_t_vec(pol, t, n_list[0], n_list[-1], th_list[:, 0], th_list[:, -1])

    
    # power_entering = power_entering_from_r(pol, r, n_list[0], th_0)
    power_entering = None

    return {'r': r, 't': t, 'R': R, 'T': T}



def T_from_t_vec(pol, t, n_i, n_f, th_i, th_f):
    """
    Calculate transmitted power T, starting with transmission amplitude t
    from the Fresnel Equations.

    Parameters
    ----------
    polarization : str
        's' or 'p' polarization
    t : torch tensor 
        Fresnel coefficient of transmission 
    n_i : torch tensor
        refractive indices of the incident layers
    n_f : torch tensor
        refractive indices of the final layers
    th_i : torch tensor
        propagation angle in the incident layer
    th_f : torch tensor
        propagation angle in the final layer 
        

    Returns
    -------
    T : torch tensor
        Transmissivity of the multilayer thin-film

    Notes
    -----
     n_i, n_f are refractive indices of incident and final medium.
    
    th_i, th_f are (complex) propagation angles through incident & final medium
    (in radians, where 0=normal). "th" stands for "theta".
    
    In the case that n_i, n_f, th_i, th_f are real, formulas simplify to
    T=|t|^2 * (n_f cos(th_f)) / (n_i cos(th_i)).
    """
    if polarization == 's':
        ni_thi = torch.real(cos(th_i)*n_i)
        nf_thf = torch.real(cos(th_f)*n_f) 
        return (abs(t**2) * ((nf_thf) / (ni_thi)))
    
    elif polarization == 'p':
        ni_thi = torch.real(conj(cos(th_i))*n_i)
        nf_thf = torch.real(conj(cos(th_f))*n_f)
        return (abs(t**2) * ((nf_thf) / (ni_thi)))
    
    else:
        raise ValueError("Polarization must be 's' or 'p'")


def coh_tmm_fast(pol, n_list, d_list, th_0, lambda_vacuum):
    """
    Computes coherent Reflection and Transmission spectra for a given multilayer thin-film
    over a broad range of wavelengths and angles of incidence. 
    Requires constant refractive index. 
    The injection- and outcoupling layer MUST have a complex refractive 
    index of zero (ie. no light absorption or amplification) since the in- and 
    outgoing light wave become ambiguous. 

    This implementaion uses PyTorch functions and can be used with Pytorch 
    Autograd. 
    
    The method is based of the initial implementaion of Steven Byrnes.
    
    Parameters
    ----------
    pol : str 
        's' or 'p'
    n_list : torch tensor 
        refractive indices of the layer, starting with the injection layer
        This method can't handle dispersive materials. Note that the first and last 
        layer must be real valued (imag(n_list[0 and -1]) must be 0 for all wavelenghts).
        shape [<number of layers>]
    d_list : torch tensor
        list of layer thicknesses of all layers, starting with the injection layer
        unit meter, shape [<number of layers>]
        For optimization, it can be beneficial to use the thickness in µm since 
        many optimizers will terminate if the gradients become to small.
    th : torch tensor
        concidered angles of incidence of the incoming light at the firts interface
        unit radian, shape [<number of concidered angles of incidence>]       
    lambda_vacuum : torch tensor
        concidered wavelengths for the computation
        unit meter, shape [<number of concidered wavelength points>] 
        For optimization, it can be beneficial to give the wavelengths in µm.

    Returns
    -------
    res : dict
        'R' : torch tensor 
            Reflectivity of the computed multilayer thin-film over all concidered 
            wavelengths and angles of incidence
            unit absolut reflected power, 
            shape [<number of concidered angles of incidence>, <number of concidered wavelength points>]  
        'T' :  torch tensor - TODO
            Reflectivity of the computed multilayer thin-film over all concidered 
            wavelengths and angles of incidence
            unit absolut transmitted power, 
            shape [<number of concidered angles of incidence>, <number of concidered wavelength points>]
        'r' : torch tensor
            Fresnel coefficient of reflection of the thin-film
            shape [<number of concidered angles of incidence>, <number of concidered wavelength points>]
        't' : torch tensor
            Fresnel coefficient of transmission of the thin-film
            shape [<number of concidered angles of incidence>, <number of concidered wavelength points>]

    Notes
    -----
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
    
    lambda_vacuum is vacuum wavelength of the light.
    
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
    * pol, n_list, d_list, th_0, lambda_vacuum--same as input
    """
    warn('This method is deprecated.', DeprecationWarning, stacklevel=2)

    # n_list holds refracitve indices of every layer, beginning with the layer where the light enters the stack
    # d_list holds the thickness of every layer, same order as n_list
    # lambda_vacuum holds the vacuum wavelength of all wavelegths of interest
    if type(n_list) is not torch.Tensor:
        n_list = torch.from_numpy(n_list.copy())
    if type(d_list) is not torch.Tensor:
        d_list = torch.from_numpy(d_list.copy())
    if type(lambda_vacuum) is not torch.Tensor:
        lambda_vacuum = torch.from_numpy(lambda_vacuum.copy())
    if type(th_0) is not torch.Tensor:
        th_0 = torch.from_numpy(th_0.copy())

    num_layers = n_list.numpy().size
    num_angles = th_0.numpy().size
    num_lambda = lambda_vacuum.numpy().size

    # th_list is a list with, for each layer, the angle that the light travels
    # through the layer. Computed with Snell's law. Note that the "angles" may be
    # complex!
    th_list = torch.empty((num_angles, num_layers), dtype=torch.cfloat)
    
    # TODO: vectorize list_snell
    # th_list = list_snell_vec(n_list, th_0)
    for i, th in enumerate(th_0):
        th_list[i] = list_snell(n_list, th)
    
    # kz is the z-component of (complex) angular wavevector for forward-moving
    # wave. Positive imaginary part means decaying.
    theta = 2 * np.pi * torch.einsum('ij,j->ij', cos(th_list), n_list )   
    kz_list = torch.empty((num_lambda, num_angles, num_layers), dtype=torch.cfloat)  # dimensions: [lambda, theta, n]
    kz_list[:] = theta
    kz_list = torch.transpose(kz_list.T * 1/lambda_vacuum, -1, 0)
    
    # delta is the total phase accrued by traveling through a given layer.
    # Ignore warning about inf multiplication
    olderr = seterr(invalid='ignore')
    delta = kz_list * d_list
    seterr(**olderr)
    
    # t_list and r_list hold the transmission and reflection coefficients from 
    # the Fresnel Equations
    # TODO: vectorize interface_t & _r and add unpolarized option for efficient calculation 
    t_list = zeros((num_angles, num_layers-1), dtype=torch.cfloat)  
    r_list = zeros((num_angles, num_layers-1), dtype=torch.cfloat)
    
    for i, th in enumerate(th_list):
        t_list[i] = interface_t(pol, n_list[:-1], n_list[1:], th_list[i, :-1], th_list[i, 1:] )
        r_list[i] = interface_r(pol, n_list[:-1], n_list[1:], th_list[i, :-1], th_list[i, 1:] )
    
    
    # A ist the propagation term for matrix optic and holds the appropriate accumulated phase for the thickness
    # of each layer
    A = exp(1j*delta[:,:, 1:-1])   # [lambda, theta, n], n without the first and last layer as they function as injection 
    # and measurement layer
    F = r_list[:, 1:]
    
    # M_list holds the transmission and reflection matrices from matrix-optics
    M_list = torch.zeros((num_angles, num_lambda, num_layers, 2, 2), dtype=torch.cfloat)
    M_list[:, :, 1:-1, 0, 0] = torch.einsum('hji,ji->jhi', 1 / A, 1/t_list[:, 1:] )   
    M_list[:, :, 1:-1, 0, 1] = torch.einsum('hji,ji->jhi', 1 / A, F / t_list[:, 1:]) 
    M_list[:, :, 1:-1, 1, 0] = torch.einsum('hji,ji->jhi', A, F / t_list[:, 1:])  
    M_list[:, :, 1:-1, 1, 1] = torch.einsum('hji,ji->jhi', A, 1 / t_list[:, 1:]) 

    Mtilde = torch.empty((num_angles, num_lambda, 2, 2), dtype=torch.cfloat)
    Mtilde[:, :] = make_2x2_tensor(1, 0, 0, 1, dtype=torch.cfloat)
    
    # contract the M_list matrix along the dimension of the layers, all
    for i in range(1, num_layers-1):
        Mtilde = Mtilde @ M_list[:,:,i]
    
    # M_r0 accounts for the first and last stack where the translation coefficients are 1 
    # TODO: why compute separately?
    M_r0 = torch.empty((num_angles, 2, 2), dtype=torch.cfloat)
    M_r0[:, 0, 0] = 1
    M_r0[:, 0, 1] = r_list[:, 0]
    M_r0[:, 1, 0] = r_list[:, 0]
    M_r0[:, 1, 1] = 1
    M_r0 = torch.einsum('ijk,i->ijk', M_r0, 1/t_list[:,0])
    Mtilde = torch.einsum('hjk,hikl->hijl', M_r0 , Mtilde)

    # Net complex transmission and reflection amplitudes
    r = Mtilde[:, :, 1,0] / Mtilde[:, :, 0,0]
    t = 1 / Mtilde[:, :, 0,0]

    # Net transmitted and reflected power, as a proportion of the incoming light
    # power.
    R = R_from_r(r)
    T=None
    # T = T_from_t_vec(pol, t.T, n_list[0], n_list[-1], th_0, th_list[:, -1]).T
    
    return {'r': r, 't': t, 'R': R, 'T': T, 'kz_list': kz_list, 'th_list': th_list,
            'pol': pol, 'n_list': n_list, 'd_list': d_list, 'th_0': th_0,
            'lambda_vacuum':lambda_vacuum}




if __name__ == '__main__':
    from numpy.core.fromnumeric import reshape
    from numpy.lib.function_base import gradient

    from numpy.testing._private.utils import requires_memory

    # import tmm_fast_torch as tmmt
    import tmm_fast_core as tmmc
    from plotting_helper import plot_stacks
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    from pytictoc import TicToc
    from scipy.optimize import minimize

    tt = TicToc()

    def log_score_torch(input):
        return -0.434 * torch.log(input) - 0.523


    def merit_function(d, n, wl, th):
        """

        Parameters
        ----------
        d :
            
        n :
            
        wl :
            
        th :
            

        Returns
        -------

        """
        d = torch.tensor(d, requires_grad=True)
        mse = torch.nn.MSELoss()
        rest = coh_tmm_fast('s', n[::-1], d, th, wl)['R']
        target = torch.ones_like(rest[0])
        target[:len(target)//2] = 0
        error = -log_score_torch(mse(rest[0], target))
        error.backward()
        gradients = d.grad #* 1e-3 # * 1e-12
        d = d.detach()
        return error.detach(), gradients

    n_layers = 12
    stack_layers = np.random.uniform(20, 250, n_layers)#*1e-9
    # stack_layers = np.array([60]*n_layers)*1e-9
    stack_layers[0] = stack_layers[-1] = 0 # np.inf 
    optical_index = np.random.uniform(1.2, 3, n_layers) # + np.random.uniform(0.5, 1, n_layers)*0j
    optical_index[0:24] = np.array([2.3,1.5]*6)
    # optical_index = np.sort(optical_index)
    optical_index[-1] = 1
    stack_layers2 = stack_layers
    optical_index2 = optical_index

    N_lambda = 300
    P = 2
    #stack_layers[0] = stack_layers[-1] = np.inf
    wavelength = np.linspace(400, 900, N_lambda)#*1e-9
    theta = np.deg2rad(np.linspace(0, P, 2))

    # rest = tmmc.coh_tmm_fast_disp('s', optical_index[::-1], stack_layers, theta, wavelength)['R']
    # stack_layers = torch.tensor(stack_layers, requires_grad=True)
    # tt.tic()
    # rest = tmmt.coh_tmm_fast('s', optical_index[::-1], stack_layers, theta, wavelength)['R']
    # tt.toc()

    # tt.tic()
    # tmmc.coh_tmm_fast('s', optical_index2[::-1], stack_layers2, theta, wavelength)['R']
    # tt.toc()
    bnds = np.array([(-1,1)] + [(5e2, 1e4)]*10 + [(-1,1)])

    print('\nStart optimization...')
    res = minimize(merit_function,
                x0 = stack_layers,
                args = (optical_index, wavelength, theta),
                jac=True,
                method='BFGS',
                bounds= bnds,
                tol = 1e-5,
                options={'maxiter': 250}
                )

    print(res.x)
    print(res)

    wavelength = np.linspace(400, 900, 500)*1e-9

    ref = coh_tmm_fast('s', optical_index2[::-1], res.x*1e-9, theta, wavelength)['R']

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5), dpi=200)
    ax1, cmap = plot_stacks(ax1, optical_index,res.x)

    ax2.plot(wavelength, ref[0])
    target = np.ones_like(ref[0])
    target[:len(target)//2] = 0
    ax2.plot(wavelength, target)
    plt.show()

    a=1
