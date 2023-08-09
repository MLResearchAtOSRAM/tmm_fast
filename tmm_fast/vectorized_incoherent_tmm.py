import torch
import numpy as np
from .vectorized_tmm_dispersive_multistack import coh_vec_tmm_disp_mstack as coh_tmm
from .vectorized_tmm_dispersive_multistack import SnellLaw_vectorized

from typing import Union

def inc_vec_tmm_disp_lstack(pol:str, 
                            N:torch.Tensor, 
                            L:torch.Tensor, 
                            mask:list,
                            theta:Union[np.ndarray, torch.Tensor],
                            lambda_vacuum:Union[np.ndarray, torch.Tensor], 
                            device:str='cpu', 
                            timer:bool=False) -> dict:

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
    i = 0
    L_coh = sum([len(m) for m in mask])
    n_L_ = int(n_layers - L_coh - 1)
    # matrix of Reflectivity and Transmissivity of the layer interfaces
    requires_grad = True if (L.requires_grad or N.requires_grad) else False
    L_ = torch.empty((n_stack, n_L_, n_theta, n_lambda, 2, 2)).requires_grad_(requires_grad)

    snell_theta = SnellLaw_vectorized(N, theta.type(torch.complex128)).real # propagation angle in every layer 

    # first, the coherent substacks are evaluated
    for m in mask: # eg m = [4,5,6]
        forward = coh_tmm(pol, N[:, m], L[:, m], snell_theta[0, :, m[0]-1, 0], lambda_vacuum, device)
        # the substack must be evaluated in both directions since we can have an incoming wave from the output side 
        # (a reflection from an incoherent layer) and Reflectivit/Transmissivity can be different depending on the direction
        backward = coh_tmm(pol, N[:, m].flip([1]), L[:, m].flip([1]), snell_theta[0, :, m[-1]+1, 0], lambda_vacuum, device)
        T_forward = forward['T'] #[n_stack, n_lambda, n_theta]
        T_backward = backward['T']
        R_forward = forward['R']
        R_backward = backward['R']

        L_[:, m[0]-1, :, :, 0, 0] = 1./T_forward
        L_[:, m[0]-1, :, :, 0, 1] = -R_backward / T_forward
        L_[:, m[0]-1, :, :, 1, 0] = R_forward / T_forward
        L_[:, m[0]-1, :, :, 1, 1] = (T_forward*T_backward - R_forward*R_backward) / T_forward

    # [2, 3, 4, 7, 10, 11, 12]
    # [0, 1, 5, 8, 9, 13, 14] inc
    # Now, the incoherent layers are evaluated. In principle, the treatment is identical 
    # to a coherent layer but the phase dependency is lost at each interface. 

    inc_mask = np.isin(np.arange(n_layers, dtype=int), mask, invert=True)
    inc_mask = np.arange(n_layers, dtype=int)[inc_mask]
    
    for k, m in zip(inc_mask[:-1], np.diff(inc_mask)):
        if m == 1:
            forward = coh_tmm(pol, N[:, k:k+2].real, L[:, k:k+2], snell_theta[0, :, k, 0], lambda_vacuum, device)
            backward = coh_tmm(pol, N[:, k:k+2].flip([1]).real, L[:, k:k+2].flip([1]), snell_theta[0, :, k+1, 0], lambda_vacuum, device)
            T_forward = forward['T'] #[n_stack, n_lambda, n_theta]
            T_backward = backward['T']
            R_forward = forward['R']
            R_backward = backward['R']

            L_[:, k, :, :, 0, 0] = 1./T_forward
            L_[:, k, :, :, 0, 1] = -R_backward / T_forward
            L_[:, k, :, :, 1, 0] = R_forward / T_forward
            L_[:, k, :, :, 1, 1] = (T_forward*T_backward - R_forward*R_backward) / T_forward

    for k in inc_mask[1:-1]:
        n_costheta = torch.einsum('ik,ijk->ijk', N[:,k], torch.cos(snell_theta[:, :, k])).imag # [n_stack, n_theta, n_lambda]
        P = torch.exp(-4 * np.pi * (torch.einsum('ijk,k,i->ijk', n_costheta, 1/lambda_vacuum, L[:,k])))
        L_[:, k] = torch.einsum('ijklm,ijk->ijklm', L_[:, k], P)

    L_tilde = torch.empty((n_stack, n_theta, n_lambda, 2, 2))
    for i in range(0, n_L_):
        L_tilde = torch.einsum('ijklm,ijklm->ijklm', L_tilde, L_[:, i])


    R = L_tilde[:,:,:,1,0] / L_tilde[:,:,:,0,0]

    T = 1 / L_tilde[:,:,:,0,0]

    return {'R': R, 'T':T}

#TODO
def R_interface(pol:str, N:torch.Tensor, T:torch.Tensor, theta:torch.Tensor, wl:torch.tensor)->torch.Tensor:
    pass
def T_interface(pol:str, N:torch.Tensor, T:torch.Tensor, theta:torch.Tensor, wl:torch.tensor)->torch.Tensor:
    pass