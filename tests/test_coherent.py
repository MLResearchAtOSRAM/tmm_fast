import numpy as np
import torch
from tmm_fast import coh_tmm as coh_tmm_fast
import matplotlib.pyplot as plt

from tmm import coh_tmm 


def test_input_output_medium():
    np.random.seed(111)
    torch.manual_seed(111)
    n_wl = 65
    n_th = 45
    wl = torch.linspace(400, 1200, n_wl) * (10**(-9))
    theta = torch.linspace(0, 89, n_th) * (np.pi/180)
    num_layers = 2
    num_stacks = 2

    #create m
    M = torch.ones((num_stacks, num_layers, wl.shape[0])).type(torch.complex128)
    M[:, 0] = 2.5
    M[:, 0] = 1.3

    #create t
    max_t = 150 * (10**(-9))
    min_t = 10 * (10**(-9))
    T = (max_t - min_t) * np.random.uniform(0, 1, (M.shape[0], M.shape[1])) + min_t

    T[:, 0] = np.inf
    T[:, 1] = np.inf

    T = torch.from_numpy(T)
    O_fast_s = coh_tmm_fast('s', M, T, theta, wl, device='cpu')
    O_fast_p = coh_tmm_fast('p', M, T, theta, wl, device='cpu')

    R_tmm_s = torch.zeros((num_stacks, n_th, n_wl), dtype=torch.double)
    R_tmm_p = torch.zeros((num_stacks, n_th, n_wl), dtype=torch.double)
    T_tmm_s = torch.zeros((num_stacks, n_th, n_wl), dtype=torch.double)
    T_tmm_p = torch.zeros((num_stacks, n_th, n_wl), dtype=torch.double)

    for h in range(num_stacks):
        for i, t in enumerate(theta.tolist()):
            for j, w in enumerate(wl.tolist()):
                res_s = coh_tmm('s', M[0][:, j].tolist(), T[h].tolist(), t, w)
                res_p = coh_tmm('p', M[0][:, j].tolist(), T[h].tolist(), t, w)
                R_tmm_s[h, i, j] = res_s['R']
                R_tmm_p[h, i, j] = res_p['R']
                T_tmm_s[h, i, j] = res_s['T']
                T_tmm_p[h, i, j] = res_p['T']

    if torch.cuda.is_available():
        O_fast_s_gpu = coh_tmm_fast('s', M, T, theta, wl, device='cuda')
        O_fast_p_gpu = coh_tmm_fast('p', M, T, theta, wl, device='cuda')

        assert O_fast_s_gpu, 'gpu computation not available'
        torch.testing.assert_close(R_tmm_s, O_fast_s_gpu['R'], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(R_tmm_p, O_fast_p_gpu['R'], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(T_tmm_s, O_fast_s_gpu['T'], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(T_tmm_p, O_fast_p_gpu['T'], rtol=1e-6, atol=1e-6)

    torch.testing.assert_close(R_tmm_s, O_fast_s['R'], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(R_tmm_p, O_fast_p['R'], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(T_tmm_s, O_fast_s['T'], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(T_tmm_p, O_fast_p['T'], rtol=1e-6, atol=1e-6)

    # check dtypes, it appears that pytorch uses float16
    # at some point in the computation
    # torch.testing.assert_close(O_fast_s['T'][0], T_tmm_s)


def test_basic_coherent_stack():
    np.random.seed(111)
    torch.manual_seed(111)
    n_wl = 65
    n_th = 45
    wl = torch.linspace(400, 1200, n_wl) * (10**(-9))
    theta = torch.linspace(0, 89, n_th) * (np.pi/180)
    num_layers = 8
    num_stacks = 3

    #create m
    M = torch.ones((num_stacks, num_layers, wl.shape[0])).type(torch.complex128)
    for i in range(1, M.shape[1]-1):
        if np.mod(i, 2) == 1:
            M[:, i, :] *= np.random.uniform(0, 3, [1])[0]
        else:
            M[:, i, :] *= np.random.uniform(0, 3, [1])[0]

    #create t
    max_t = 150 * (10**(-9))
    min_t = 10 * (10**(-9))
    T = (max_t - min_t) * np.random.uniform(0, 1, (M.shape[0], M.shape[1])) + min_t

    T[:, 0] = np.inf
    T[:, -1] = np.inf

    T = torch.from_numpy(T)
    O_fast_s = coh_tmm_fast('s', M, T, theta, wl, device='cpu')
    O_fast_p = coh_tmm_fast('p', M, T, theta, wl, device='cpu')

    R_tmm_s = torch.zeros((num_stacks, n_th, n_wl), dtype=torch.double)
    R_tmm_p = torch.zeros((num_stacks, n_th, n_wl), dtype=torch.double)
    T_tmm_s = torch.zeros((num_stacks, n_th, n_wl), dtype=torch.double)
    T_tmm_p = torch.zeros((num_stacks, n_th, n_wl), dtype=torch.double)

    for h in range(num_stacks):
        for i, t in enumerate(theta.tolist()):
            for j, w in enumerate(wl.tolist()):
                res_s = coh_tmm('s', M[0][:, j].tolist(), T[h].tolist(), t, w)
                res_p = coh_tmm('p', M[0][:, j].tolist(), T[h].tolist(), t, w)
                R_tmm_s[h, i, j] = res_s['R']
                R_tmm_p[h, i, j] = res_p['R']
                T_tmm_s[h, i, j] = res_s['T']
                T_tmm_p[h, i, j] = res_p['T']

    if torch.cuda.is_available():
        O_fast_s_gpu = coh_tmm_fast('s', M, T, theta, wl, device='cuda')
        O_fast_p_gpu = coh_tmm_fast('p', M, T, theta, wl, device='cuda')

        assert O_fast_s_gpu, 'gpu computation not available'
        torch.testing.assert_close(R_tmm_s, O_fast_s_gpu['R'], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(R_tmm_p, O_fast_p_gpu['R'], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(T_tmm_s, O_fast_s_gpu['T'], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(T_tmm_p, O_fast_p_gpu['T'], rtol=1e-6, atol=1e-6)

    torch.testing.assert_close(R_tmm_s, O_fast_s['R'], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(R_tmm_p, O_fast_p['R'], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(T_tmm_s, O_fast_s['T'], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(T_tmm_p, O_fast_p['T'], rtol=1e-6, atol=1e-6)



def test_absorbing_coherent_stack():
    np.random.seed(111)
    torch.manual_seed(111)
    n_wl = 65
    n_th = 45
    wl = torch.linspace(400, 1200, n_wl) * (10**(-9))
    theta = torch.linspace(0, 89, n_th) * (np.pi/180)
    num_layers = 8
    num_stacks = 3

    #create m
    M = torch.ones((num_stacks, num_layers, wl.shape[0])).type(torch.complex128)
    for i in range(1, M.shape[1]-1):
        if np.mod(i, 2) == 1:
            M[:, i, :] *= np.random.uniform(0,3,[1])[0]
            M[:, i, :] += np.random.uniform(0, 1, [1])[0]*1j
        else:
            M[:, i, :] *= np.random.uniform(0,3,[1])[0]
            M[:, i, :] += np.random.uniform(0, 1, [1])[0]*1j

    #create t
    max_t = 150 * (10**(-9))
    min_t = 10 * (10**(-9))
    T = (max_t - min_t) * np.random.uniform(0, 1, (M.shape[0], M.shape[1])) + min_t

    T[:, 0] = np.inf
    T[:, -1] = np.inf

    T = torch.from_numpy(T)
    O_fast_s = coh_tmm_fast('s', M, T, theta, wl, device='cpu')
    O_fast_p = coh_tmm_fast('p', M, T, theta, wl, device='cpu')

    R_tmm_s = torch.zeros((num_stacks, n_th, n_wl), dtype=torch.double)
    R_tmm_p = torch.zeros((num_stacks, n_th, n_wl), dtype=torch.double)
    T_tmm_s = torch.zeros((num_stacks, n_th, n_wl), dtype=torch.double)
    T_tmm_p = torch.zeros((num_stacks, n_th, n_wl), dtype=torch.double)

    for h in range(num_stacks):
        for i, t in enumerate(theta.tolist()):
            for j, w in enumerate(wl.tolist()):
                res_s = coh_tmm('s', M[0][:, j].tolist(), T[h].tolist(), t, w)
                res_p = coh_tmm('p', M[0][:, j].tolist(), T[h].tolist(), t, w)
                R_tmm_s[h, i, j] = res_s['R']
                R_tmm_p[h, i, j] = res_p['R']
                T_tmm_s[h, i, j] = res_s['T']
                T_tmm_p[h, i, j] = res_p['T']

    if torch.cuda.is_available():
        O_fast_s_gpu = coh_tmm_fast('s', M, T, theta, wl, device='cuda')
        O_fast_p_gpu = coh_tmm_fast('p', M, T, theta, wl, device='cuda')

        assert O_fast_s_gpu, 'gpu computation not available'
        torch.testing.assert_close(R_tmm_s, O_fast_s_gpu['R'], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(R_tmm_p, O_fast_p_gpu['R'], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(T_tmm_s, O_fast_s_gpu['T'], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(T_tmm_p, O_fast_p_gpu['T'], rtol=1e-6, atol=1e-6)

    torch.testing.assert_close(R_tmm_s, O_fast_s['R'], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(R_tmm_p, O_fast_p['R'], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(T_tmm_s, O_fast_s['T'], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(T_tmm_p, O_fast_p['T'], rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    test_input_output_medium()
    test_basic_coherent_stack()
    test_absorbing_coherent_stack()