from tmm import *
# import math
# from numpy.lib.scimath import arcsin
import numpy as np
# import dask

import sys

from torch._C import dtype
EPSILON = sys.float_info.epsilon
from numpy import inf, nan, isnan, pi, seterr

from torch import cos, zeros, tensor, exp, conj, sin, asin
import torch

def make_2x2_tensor(a, b, c, d, dtype=float):
    """
    Makes a 2x2 numpy array of [[a,b],[c,d]]

    Same as "numpy.array([[a,b],[c,d]], dtype=float)", but ten times faster
    """
    my_array = torch.empty((2,2), dtype=dtype)
    my_array[0,0] = a
    my_array[0,1] = b
    my_array[1,0] = c
    my_array[1,1] = d
    return my_array

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
    if a wave is traveling at angle theta from normal in a medium with index n,
    calculate whether or not this is the forward-traveling wave (i.e., the one
    going from front to back of the stack, like the incoming or outgoing waves,
    but unlike the reflected wave). For real n & theta, the criterion is simply
    -pi/2 < theta < pi/2, but for complex n & theta, it's more complicated.
    See https://arxiv.org/abs/1603.02720 appendix D. If theta is the forward
    angle, then (pi-theta) is the backward angle and vice-versa.
    """
    n = torch.tensor(n, dtype=torch.cfloat)
    assert n.real * n.imag >= 0, ("For materials with gain, it's ambiguous which "
                                  "beam is incoming vs outgoing. See "
                                  "https://arxiv.org/abs/1603.02720 Appendix C.\n"
                                  "n: " + str(n) + "   angle: " + str(theta))
    # assert n.dtype is not complex,  ("For materials with gain, it's ambiguous which "
    #                               "beam is incoming vs outgoing. See "
    #                               "https://arxiv.org/abs/1603.02720 Appendix C.\n"
    #                               "n: " + str(n) + "   angle: " + str(theta))

    ncostheta = n * cos(theta)
    ncostheta = torch.tensor(ncostheta, dtype=torch.cfloat)
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
    return list of angle theta in each layer based on angle th_0 in layer 0,
    using Snell's law. n_list is index of refraction of each layer. Note that
    "angles" may be complex!!
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

def list_snell_new(n_list, th):
    """
    return list of angle theta in each layer based on angle th_0 in layer 0,
    using Snell's law. n_list is index of refraction of each layer. Note that
    "angles" may be complex!!
    """
    # Important that the arcsin here is numpy.lib.scimath.arcsin, not
    # numpy.arcsin! (They give different results e.g. for arcsin(2).)
    
    sin_th = torch.unsqueeze(torch.sin(th), dim=0)
    n0 = torch.unsqueeze(n_list[0], dim=-1)
    n0th = torch.matmul(n0, sin_th)
    assert n0th.shape == (n_list.shape[-1], th.shape[0]), (n_list.shape[-1], th.shape[0]) # [lambda, theta]
    n = 1/n_list # [lambda, d]    
    angles = asin(torch.einsum('ij,ki->jki', n0th, n))
    assert angles.shape == (th.shape[0], n_list.shape[0], n_list.shape[1]), (th.shape[0], n_list.shape[0], n_list.shape[1])
    
    # dim(angles) = [dim_theta, dim_d, dim_lambda]
    # The first and last entry need to be the forward angle (the intermediate
    # layers don't matter, see https://arxiv.org/abs/1603.02720 Section 5)

    angles[:, 0] = -is_not_forward_angle(n_list[0], angles[:, 0]) * pi + angles[:, 0]
    angles[:, -1] = -is_not_forward_angle(n_list[-1], angles[:, -1]) * pi + angles[:, -1]
    return angles


def interface_r_new(polarization, n_i, n_f, th_i, th_f):
    """
    reflection amplitude (from Fresnel equations)

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

def interface_t_new(polarization, n_i, n_f, th_i, th_f):
    """
    transmission amplitude (frem Fresnel equations)

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
    Calculate reflected power R, starting with reflection amplitude r.
    """
    return abs(r)**2

def matmul_complex(t1,t2):
    return torch.view_as_complex(torch.stack((t1.real @ t2.real - t1.imag @ t2.imag, t1.real @ t2.imag + t1.imag @ t2.real),dim=2))


def coh_tmm_fast_disp(pol, n_list, d_list, th, lam_vac):
    """
    Upgrade to the regular coh_tmm method. Does not perform checks and should
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
    
    # n_list holds refracitve indices of every layer, beginning with the layer where the light enters the stack
    # d_list holds the thickness of every layer, same order as n_list
    # lam_vac holds the vacuum wavelength of all wavelegths of interest
    n_list = torch.tensor(n_list)
    d_list = torch.tensor(d_list, dtype=float)
    lam_vac = torch.tensor(lam_vac)

    num_layers = d_list.size
    num_angles = th.size
    num_lambda = lam_vac.size

    # th_list is a list with, for each layer, the angle that the light travels
    # through the layer. Computed with Snell's law. Note that the "angles" may be
    # complex!
        
    # if a constant refractive index is used (no dispersion) extend the tensor
    # todo: call coh_tmm_fast for n=const
    if n_list.ndim==1:
        n_list = torch.tile(n_list, (num_lambda,1)).T
        
    th_list = list_snell_new(n_list, th)

    theta = 2 * np.pi * torch.einsum('kij,ij->kij', cos(th_list), n_list)   # [theta,d, lambda]
    kz_list = torch.einsum('ijk,k->kij', theta, 1/lam_vac) #[lambda, theta, d]
    
    
    # kz is the z-component of (complex) angular wavevector for forward-moving
    # wave. Positive imaginary part means decaying.

    # delta is the total phase accrued by traveling through a given layer.
    # Ignore warning about inf multiplication
    olderr = seterr(invalid='ignore')
    delta = kz_list * d_list
    seterr(**olderr)
    
    # t_list and r_list hold the transmission and reflection coefficients from 
    # the Fresnel Equations
    
    t_list = interface_t_new(pol, n_list[:-1, :], n_list[1:, :], th_list[:, :-1, :], th_list[:, 1:, :])
    r_list = interface_r_new(pol, n_list[:-1, :], n_list[1:, :], th_list[:, :-1, :], th_list[:, 1:, :])
    
    A = exp(1j*delta[:,:, 1:-1]) 
    F = r_list[:, :, 1:]

    
#     # A ist the propagation term for matrix optic and holds the appropriate accumulated phase for the thickness
#     # of each layer
#     A = exp(1j*delta[:,:, 1:-1])   # [lambda, theta, n], n without the first and last layer as they function as injection 
#     # and measurement layer
#     F = r_list[:, 1:]
#     #print('F: ', F.shape)
    
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
    # todo: why compute separately?
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
    
    # vw_list holds the net reflected and transmitted waves in each layer, not required for 
    # total reflectivity calculation
    # vw_list[n] = [v_n, w_n]. v_0 and w_0 are undefined because the 0th medium
    # has no left interface.
    # vw_list = zeros((num_layers, 2), dtype=torch.cfloat)
    # vw = tensor([[t],[0]])
    # vw_list[-1,:] = torch.transpose(vw)
    # for i in range(num_layers-2, 0, -1):
    #     vw = torch.dot(M_list[i], vw)
    #     vw_list[i,:] = torch.transpose(vw)
    vw_list = None
    
    # Net transmitted and reflected power, as a proportion of the incoming light
    # power.
    R = R_from_r(r)

    T=None
    # T = T_from_t_new(pol, t, n_list[0], n_list[-1], th_list[:, 0], th_list[:, -1])
    
    # power_entering = power_entering_from_r(pol, r, n_list[0], th_0)
    power_entering = None

    return {'r': r, 't': t, 'R': R, 'T': T, 'power_entering': power_entering,
            'vw_list': vw_list, 'kz_list': kz_list, 'th_list': th_list,
            'pol': pol, 'n_list': n_list, 'd_list': d_list, 'th': th,
            'lam_vac':lam_vac}


def T_from_t_new(pol, t, n_i, n_f, th_i, th_f):
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
        ni_thi = torch.real(cos(th_i)*n_i)
        nf_thf = torch.real(cos(th_f)*n_f) 
        return (abs(t**2) * ((nf_thf) / (ni_thi)))
    
    elif pol == 'p':
        ni_thi = torch.real(conj(cos(th_i))*n_i)
        nf_thf = torch.real(conj(cos(th_f))*n_f)
        return (abs(t**2) * ((nf_thf) / (ni_thi)))
    
    else:
        raise ValueError("Polarization must be 's' or 'p'")


def coh_tmm_fast(pol, n_list, d_list, th_0, lam_vac):
    """
    Upgrade to the regular coh_tmm method. Does not perform checks and should
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
    # tictoc = TicToc() # uncomment for computing time measurment
    # tictoc.tic()
    
    # n_list holds refracitve indices of every layer, beginning with the layer where the light enters the stack
    # d_list holds the thickness of every layer, same order as n_list
    # lam_vac holds the vacuum wavelength of all wavelegths of interest
    if type(n_list) is not torch.Tensor:
        n_list = torch.from_numpy(n_list.copy())
    if type(d_list) is not torch.Tensor:
        d_list = torch.from_numpy(d_list.copy())
    if type(lam_vac) is not torch.Tensor:
        lam_vac = torch.from_numpy(lam_vac.copy())
    if type(th_0) is not torch.Tensor:
        th_0 = torch.from_numpy(th_0.copy())

    num_layers = n_list.numpy().size
    num_angles = th_0.numpy().size
    num_lambda = lam_vac.numpy().size

    # th_list is a list with, for each layer, the angle that the light travels
    # through the layer. Computed with Snell's law. Note that the "angles" may be
    # complex!
    
    th_list = torch.empty((num_angles, num_layers), dtype=torch.cfloat)
    
    # todo vectorize list_snell
    for i, th in enumerate(th_0):
        th_list[i] = list_snell(n_list, th)

    # th_list = list_snell_new(n_list, th_0)
    
    # kz is the z-component of (complex) angular wavevector for forward-moving
    # wave. Positive imaginary part means decaying.

    theta = 2 * np.pi * torch.einsum('ij,j->ij', cos(th_list), n_list )   
    kz_list = torch.empty((num_lambda, num_angles, num_layers), dtype=torch.cfloat)  # dimensions: [lambda, theta, n]
    kz_list[:] = theta
    kz_list = torch.transpose(kz_list.T * 1/lam_vac, -1, 0)
    

    # delta is the total phase accrued by traveling through a given layer.
    # Ignore warning about inf multiplication
    olderr = seterr(invalid='ignore')
    delta = kz_list * d_list
    seterr(**olderr)
    
    # t_list and r_list hold the transmission and reflection coefficients from 
    # the Fresnel Equations
    
    # todo: vectorize interface_t & _r and add unpolarized option for efficient calculation 
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
    #print('F: ', F.shape)
    
    # M_list holds the transmission and reflection matrices from matrix-optics
    
    M_list = torch.zeros((num_angles, num_lambda, num_layers, 2, 2), dtype=torch.cfloat)
    M_list[:, :, 1:-1, 0, 0] = torch.einsum('hji,ji->jhi', 1 / A, 1/t_list[:, 1:] )   
    M_list[:, :, 1:-1, 0, 1] = torch.einsum('hji,ji->jhi', 1 / A, F / t_list[:, 1:]) 
    M_list[:, :, 1:-1, 1, 0] = torch.einsum('hji,ji->jhi', A, F / t_list[:, 1:])  
    M_list[:, :, 1:-1, 1, 1] = torch.einsum('hji,ji->jhi', A, 1 / t_list[:, 1:]) 

    Mtilde = torch.empty((num_angles, num_lambda, 2, 2), dtype=torch.cfloat)
    Mtilde[:, :] = make_2x2_tensor(1, 0, 0, 1, dtype=torch.cfloat)

    #print('M_list: ', M_list.shape)
    #tictoc.tic()
    
    # contract the M_list matrix along the dimension of the layers, all

    # M = np.copy(Mtilde.numpy())
    # for i in range(1, num_layers-1):
    #     M = np.matmul(M, M_list[:,:,i].numpy())

    for i in range(1, num_layers-1):
        Mtilde = Mtilde @ M_list[:,:,i]
    # np.testing.assert_almost_equal(Mtilde.numpy(), M)
    # for i in range(1, num_layers-1):
    #     Mtilde = torch.einsum('ijkl,ijlm->ijkm', Mtilde, M_list[:,:,i])


    # tictoc.toc()
    
    # M_r0 accounts for the first and last stack where the translation coefficients are 1 
    # todo: why compute separately?
    M_r0 = torch.empty((num_angles, 2, 2), dtype=torch.cfloat)
    M_r0[:, 0, 0] = 1
    M_r0[:, 0, 1] = r_list[:, 0]
    M_r0[:, 1, 0] = r_list[:, 0]
    M_r0[:, 1, 1] = 1
    M_r0 = torch.einsum('ijk,i->ijk', M_r0, 1/t_list[:,0])

    # M_r0 = torch.tensor(M_r0, dtype=torch.float)
    # Mtilde = torch.tensor(Mtilde, dtype=torch.float)
    
    Mtilde = torch.einsum('hjk,hikl->hijl', M_r0 , Mtilde)

    # tictoc.toc()
    # Net complex transmission and reflection amplitudes
    r = Mtilde[:, :, 1,0] / Mtilde[:, :, 0,0]
    
    t = 1 / Mtilde[:, :, 0,0]
    
    # vw_list holds the net reflected and transmitted waves in each layer, not required for 
    # total reflectivity calculation
    # vw_list[n] = [v_n, w_n]. v_0 and w_0 are undefined because the 0th medium
    # has no left interface.
    # vw_list = zeros((num_layers, 2), dtype=torch.cfloat)
    # vw = tensor([[t],[0]])
    # vw_list[-1,:] = torch.transpose(vw)
    # for i in range(num_layers-2, 0, -1):
    #     vw = torch.dot(M_list[i], vw)
    #     vw_list[i,:] = torch.transpose(vw)
    vw_list = 'not calculated'
    
    # Net transmitted and reflected power, as a proportion of the incoming light
    # power.
    R = R_from_r(r)
    T=None
    # T = T_from_t_new(pol, t.T, n_list[0], n_list[-1], th_0, th_list[:, -1]).T
    
    # power_entering = power_entering_from_r(pol, r, n_list[0], th_0)
    power_entering = 'not calculated'
    # print('out')
    # tictoc.toc()
    # print('you have reached rock bottom')
    return {'r': r, 't': t, 'R': R, 'T': T, 'power_entering': power_entering,
            'vw_list': vw_list, 'kz_list': kz_list, 'th_list': th_list,
            'pol': pol, 'n_list': n_list, 'd_list': d_list, 'th_0': th_0,
            'lam_vac':lam_vac}


