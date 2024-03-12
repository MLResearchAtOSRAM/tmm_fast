import numpy as np
import torch
from tmm_fast import inc_tmm as inc_tmm_fast
import matplotlib.pyplot as plt

from tmm import inc_tmm 



def test_incoherent_input_output_medium():
    n_wl = 65
    n_th = 45
    wl = torch.linspace(400, 1200, n_wl) * (10**(-9))
    theta = torch.linspace(0, 89, n_th) * (np.pi/180)
    num_layers = 2
    num_stacks = 2
    mask = []

    #create m
    M = torch.ones((num_stacks, num_layers, wl.shape[0])).type(torch.complex128)
    for i in range(1, M.shape[1]-1):
        if np.mod(i, 2) == 1:
            M[:, i, :] *= 1.46
        else:
            M[:, i, :] *= 2.56

    #create t
    max_t = 150 * (10**(-9))
    min_t = 10 * (10**(-9))
    T = (max_t - min_t) * np.random.uniform(0, 1, (M.shape[0], M.shape[1])) + min_t

    T[:, 0] = np.inf
    T[:, 1] = np.inf

    T = torch.from_numpy(T)
    O_fast_s = inc_tmm_fast('s', M, T, mask, theta, wl, device='cpu')
    O_fast_p = inc_tmm_fast('p', M, T, mask, theta, wl, device='cpu')

    R_tmm_s = torch.zeros((n_th, n_wl))
    R_tmm_p = torch.zeros((n_th, n_wl))
    T_tmm_s = torch.zeros((n_th, n_wl))
    T_tmm_p = torch.zeros((n_th, n_wl))

    T_list = T[0].tolist()

    for i, t in enumerate(theta.tolist()):
        for j, w in enumerate(wl.tolist()):
            res_s = inc_tmm('s', M[0][:, j].tolist(), T_list, ['i', 'i'], t, w)
            res_p = inc_tmm('p', M[0][:, j].tolist(), T_list, ['i', 'i'], t, w)
            R_tmm_s[i, j] = res_s['R']
            R_tmm_p[i, j] = res_p['R']
            T_tmm_s[i, j] = res_s['T']
            T_tmm_p[i, j] = res_p['T']

    if torch.cuda.is_available():
        O_fast_s_gpu = inc_tmm_fast('s', M, T, mask, theta, wl, device='cuda')
        O_fast_p_gpu = inc_tmm_fast('p', M, T, mask, theta, wl, device='cuda')

        assert O_fast_s_gpu, 'gpu computation not availabla'
        assert torch.allclose(R_tmm_s, O_fast_s_gpu['R'])
        assert torch.allclose(R_tmm_p, O_fast_p_gpu['R'])
        assert torch.allclose(T_tmm_s, O_fast_s_gpu['T'])
        assert torch.allclose(T_tmm_p, O_fast_p_gpu['T'])

    assert torch.allclose(R_tmm_s, O_fast_s['R'])
    assert torch.allclose(R_tmm_p, O_fast_p['R'])
    assert torch.allclose(T_tmm_s, O_fast_s['T'])
    assert torch.allclose(T_tmm_p, O_fast_p['T'])



def test_fully_incoherent_stack():
    n_wl = 65
    n_th = 45
    wl = torch.linspace(400, 1200, n_wl) * (10**(-9))
    theta = torch.linspace(0, 89, n_th) * (np.pi/180)
    num_layers = 5
    num_stacks = 2
    mask = []

    #create m
    M = torch.ones((num_stacks, num_layers, wl.shape[0])).type(torch.complex128)
    for i in range(1, M.shape[1]-1):
        if np.mod(i, 2) == 1:
            M[:, i, :] *= 1.46
        else:
            M[:, i, :] *= 2.56

    #create t
    max_t = 150000 * (10**(-9))
    min_t = 10000 * (10**(-9))
    T = (max_t - min_t) * np.random.uniform(0, 1, (M.shape[0], M.shape[1])) + min_t

    T[:, 0] = np.inf
    T[:, 1] = 10000e-9
    T[:, 2] = 2000e-9
    T[:, 3] = 5000e-9
    T[:, -1] = np.inf

    T = torch.from_numpy(T)
    O_fast_s = inc_tmm_fast('s', M, T, mask, theta, wl, device='cpu')
    O_fast_p = inc_tmm_fast('p', M, T, mask, theta, wl, device='cpu')

    R_tmm_s = torch.zeros((n_th, n_wl))
    R_tmm_p = torch.zeros((n_th, n_wl))
    T_tmm_s = torch.zeros((n_th, n_wl))
    T_tmm_p = torch.zeros((n_th, n_wl))

    T_list = T[0].tolist()

    for i, t in enumerate(theta.tolist()):
        for j, w in enumerate(wl.tolist()):
            res_s = inc_tmm('s', M[0][:, j].tolist(), T_list, ['i', 'i', 'i', 'i', 'i'], t, w)
            res_p = inc_tmm('p', M[0][:, j].tolist(), T_list, ['i', 'i', 'i', 'i', 'i'], t, w)
            R_tmm_s[i, j] = res_s['R']
            R_tmm_p[i, j] = res_p['R']
            T_tmm_s[i, j] = res_s['T']
            T_tmm_p[i, j] = res_p['T']

    if torch.cuda.is_available():
        O_fast_s_gpu = inc_tmm_fast('s', M, T, mask, theta, wl, device='cuda')
        O_fast_p_gpu = inc_tmm_fast('p', M, T, mask, theta, wl, device='cuda')

        assert O_fast_s_gpu, 'gpu computation not availabla'
        assert torch.allclose(R_tmm_s, O_fast_s_gpu['R'])
        assert torch.allclose(R_tmm_p, O_fast_p_gpu['R'])
        assert torch.allclose(T_tmm_s, O_fast_s_gpu['T'])
        assert torch.allclose(T_tmm_p, O_fast_p_gpu['T'])

    assert torch.allclose(R_tmm_s, O_fast_s['R'])
    assert torch.allclose(R_tmm_p, O_fast_p['R'])
    assert torch.allclose(T_tmm_s, O_fast_s['T'])
    assert torch.allclose(T_tmm_p, O_fast_p['T'])


def test_incoherent_numpy_input_output_medium():
    n_wl = 65
    n_th = 45
    wl = torch.linspace(400, 1200, n_wl) * (10**(-9))
    theta = torch.linspace(0, 89, n_th) * (np.pi/180)
    num_layers = 2
    num_stacks = 2
    mask = []

    #create m
    M = torch.ones((num_stacks, num_layers, wl.shape[0])).type(torch.complex128)
    for i in range(1, M.shape[1]-1):
        if np.mod(i, 2) == 1:
            M[:, i, :] *= 1.46
        else:
            M[:, i, :] *= 2.56

    #create t
    max_t = 150 * (10**(-9))
    min_t = 10 * (10**(-9))
    T = (max_t - min_t) * np.random.uniform(0, 1, (M.shape[0], M.shape[1])) + min_t

    T[:, 0] = np.inf
    T[:, 1] = np.inf

    T = torch.from_numpy(T)
    O_fast_s = inc_tmm_fast('s', M.numpy(), T.numpy(), mask, theta, wl, device='cpu')
    O_fast_p = inc_tmm_fast('p', M.numpy(), T.numpy(), mask, theta, wl, device='cpu')

    R_tmm_s = torch.zeros((n_th, n_wl))
    R_tmm_p = torch.zeros((n_th, n_wl))
    T_tmm_s = torch.zeros((n_th, n_wl))
    T_tmm_p = torch.zeros((n_th, n_wl))

    T_list = T[0].tolist()

    for i, t in enumerate(theta.tolist()):
        for j, w in enumerate(wl.tolist()):
            res_s = inc_tmm('s', M[0][:, j].tolist(), T_list, ['i', 'i'], t, w)
            res_p = inc_tmm('p', M[0][:, j].tolist(), T_list, ['i', 'i'], t, w)
            R_tmm_s[i, j] = res_s['R']
            R_tmm_p[i, j] = res_p['R']
            T_tmm_s[i, j] = res_s['T']
            T_tmm_p[i, j] = res_p['T']


    assert torch.allclose(R_tmm_s, O_fast_s['R'])
    assert torch.allclose(R_tmm_p, O_fast_p['R'])
    assert torch.allclose(T_tmm_s, O_fast_s['T'])
    assert torch.allclose(T_tmm_p, O_fast_p['T'])

def test_coherent_stack_with_incoherent_surrounding():
    n_wl = 20
    n_th = 45
    wl = torch.linspace(400, 1200, n_wl) * (10**(-9))
    theta = torch.linspace(0, 85, n_th) * (np.pi/180)
    num_layers = 5
    num_stacks = 2
    mask = [[1, 2, 3]]

    #create m
    M = torch.ones((num_stacks, num_layers, wl.shape[0])).type(torch.complex128)
    for i in range(1, M.shape[1]-1):
        if np.mod(i, 2) == 1:
            M[:, i, :] *= 1.46
        else:
            M[:, i, :] *= 2.56

    #create t
    max_t = 150 * (10**(-9))
    min_t = 10 * (10**(-9))
    T = (max_t - min_t) * np.random.uniform(0, 1, (M.shape[0], M.shape[1])) + min_t

    T[:, 0] = np.inf
    T[:, 1] = 200e-9
    T[:, 2] = 100e-9
    T[:, 3] = 300e-9
    T[:, 4] = np.inf

    T = torch.from_numpy(T)
    O_fast_s = inc_tmm_fast('s', M, T, mask, theta, wl, device='cpu')
    O_fast_p = inc_tmm_fast('p', M, T, mask, theta, wl, device='cpu')

    R_tmm_s = torch.zeros((n_th, n_wl))
    R_tmm_p = torch.zeros((n_th, n_wl))
    T_tmm_s = torch.zeros((n_th, n_wl))
    T_tmm_p = torch.zeros((n_th, n_wl))

    T_list = T[0].tolist()

    for i, t in enumerate(theta.tolist()):
        for j, w in enumerate(wl.tolist()):
            res_s = inc_tmm('s', M[0][:, j].tolist(), T_list, ['i', 'c', 'c', 'c', 'i'], t, w)
            res_p = inc_tmm('p', M[0][:, j].tolist(), T_list, ['i', 'c', 'c', 'c', 'i'], t, w)
            R_tmm_s[i, j] = res_s['R']
            R_tmm_p[i, j] = res_p['R']
            T_tmm_s[i, j] = res_s['T']
            T_tmm_p[i, j] = res_p['T']

    if torch.cuda.is_available():
        O_fast_s_gpu = inc_tmm_fast('s', M, T, mask, theta, wl, device='cuda')
        O_fast_p_gpu = inc_tmm_fast('p', M, T, mask, theta, wl, device='cuda')

        assert O_fast_s_gpu, 'gpu computation not available'
        assert torch.allclose(R_tmm_s, O_fast_s_gpu['R'])
        assert torch.allclose(R_tmm_p, O_fast_p_gpu['R'])
        assert torch.allclose(T_tmm_s, O_fast_s_gpu['T'])
        assert torch.allclose(T_tmm_p, O_fast_p_gpu['T'])

    assert torch.allclose(R_tmm_s, O_fast_s['R'])
    assert torch.allclose(R_tmm_p, O_fast_p['R'])
    assert torch.allclose(T_tmm_s, O_fast_s['T'])
    assert torch.allclose(T_tmm_p, O_fast_p['T'])
    assert (torch.abs(O_fast_s['R'][0]-R_tmm_s) <1e-5).all().item()
    assert (torch.abs(O_fast_p['R'][0]-R_tmm_p) <1e-5).all().item()
    assert (torch.abs(O_fast_s['T'][0]-T_tmm_s) <1e-5).all().item()
    assert (torch.abs(O_fast_p['T'][0]-T_tmm_p) <1e-5).all().item()

def test_coherent_incoherent():
    n_wl = 20
    n_th = 45
    wl = torch.linspace(400, 1200, n_wl) * (10**(-9))
    theta = torch.linspace(0, 89, n_th) * (np.pi/180)
    num_layers = 5
    num_stacks = 2
    mask = [[2, 3]]

    #create m
    M = torch.ones((num_stacks, num_layers, wl.shape[0])).type(torch.complex128)
    for i in range(1, M.shape[1]-1):
        if np.mod(i, 2) == 1:
            M[:, i, :] *= 1.46
        else:
            M[:, i, :] *= 2.56

    #create t
    max_t = 150 * (10**(-9))
    min_t = 10 * (10**(-9))
    T = (max_t - min_t) * np.random.uniform(0, 1, (M.shape[0], M.shape[1])) + min_t

    T[:, 0] = np.inf
    T[:, 1] = 10000e-9
    T[:, 2] = 100e-9
    T[:, 3] = 300e-9
    T[:, -1] = np.inf

    T = torch.from_numpy(T)
    O_fast_s = inc_tmm_fast('s', M, T, mask, theta, wl, device='cpu')
    O_fast_p = inc_tmm_fast('p', M, T, mask, theta, wl, device='cpu')

    R_tmm_s = torch.zeros((n_th, n_wl))
    R_tmm_p = torch.zeros((n_th, n_wl))
    T_tmm_s = torch.zeros((n_th, n_wl))
    T_tmm_p = torch.zeros((n_th, n_wl))

    T_list = T[0].tolist()

    for i, t in enumerate(theta.tolist()):
        for j, w in enumerate(wl.tolist()):
            res_s = inc_tmm('s', M[0][:, j].tolist(), T_list, ['i', 'i', 'c', 'c', 'i'], t, w)
            res_p = inc_tmm('p', M[0][:, j].tolist(), T_list, ['i', 'i', 'c', 'c', 'i'], t, w)
            R_tmm_s[i, j] = res_s['R']
            R_tmm_p[i, j] = res_p['R']
            T_tmm_s[i, j] = res_s['T']
            T_tmm_p[i, j] = res_p['T']

    if torch.cuda.is_available():
        O_fast_s_gpu = inc_tmm_fast('s', M, T, mask, theta, wl, device='cuda')
        O_fast_p_gpu = inc_tmm_fast('p', M, T, mask, theta, wl, device='cuda')

        assert O_fast_s_gpu, 'gpu computation not available'
        assert torch.allclose(R_tmm_s, O_fast_s_gpu['R'])
        assert torch.allclose(R_tmm_p, O_fast_p_gpu['R'])
        assert torch.allclose(T_tmm_s, O_fast_s_gpu['T'])
        assert torch.allclose(T_tmm_p, O_fast_p_gpu['T'])

    assert torch.allclose(R_tmm_s, O_fast_s['R'])
    assert torch.allclose(R_tmm_p, O_fast_p['R'])
    assert torch.allclose(T_tmm_s, O_fast_s['T'])
    assert torch.allclose(T_tmm_p, O_fast_p['T'])
    assert (torch.abs(O_fast_s['R'][0]-R_tmm_s) <1e-5).all().item()
    assert (torch.abs(O_fast_p['R'][0]-R_tmm_p) <1e-5).all().item()
    assert (torch.abs(O_fast_s['T'][0]-T_tmm_s) <1e-5).all().item()
    assert (torch.abs(O_fast_p['T'][0]-T_tmm_p) <1e-5).all().item()

def test_absorbing_fully_incoherent():
    n_wl = 65
    n_th = 45
    wl = torch.linspace(400, 1200, n_wl) * (10**(-9))
    theta = torch.linspace(0, 89, n_th) * (np.pi/180)
    num_layers = 5
    num_stacks = 2
    mask = []
    imask = ['i', 'i', 'i', 'i', 'i']

    #create m
    M = torch.ones((num_stacks, num_layers, wl.shape[0])).type(torch.complex128)
    for i in range(1, M.shape[1]-1):
        if np.mod(i, 2) == 1:
            M[:, i, :] *= 1.46
            M[:, i, :] += .005j
        else:
            M[:, i, :] *= 2.56
            M[:, i, :] += .002j

    #create t
    max_t = 150000 * (10**(-9))
    min_t = 10000 * (10**(-9))
    T = (max_t - min_t) * np.random.uniform(0, 1, (M.shape[0], M.shape[1])) + min_t

    T[:, 0] = np.inf
    T[:, 1] = 10000e-9
    T[:, 2] = 2000e-9
    T[:, 3] = 5000e-9
    T[:, -1] = np.inf

    T = torch.from_numpy(T)
    O_fast_s = inc_tmm_fast('s', M, T, mask, theta, wl, device='cpu')
    O_fast_p = inc_tmm_fast('p', M, T, mask, theta, wl, device='cpu')

    R_tmm_s = torch.zeros((n_th, n_wl))
    R_tmm_p = torch.zeros((n_th, n_wl))
    T_tmm_s = torch.zeros((n_th, n_wl))
    T_tmm_p = torch.zeros((n_th, n_wl))

    T_list = T[0].tolist()

    for i, t in enumerate(theta.tolist()):
        for j, w in enumerate(wl.tolist()):
            res_s = inc_tmm('s', M[0][:, j].tolist(), T_list, imask, t, w)
            res_p = inc_tmm('p', M[0][:, j].tolist(), T_list, imask, t, w)
            R_tmm_s[i, j] = res_s['R']
            R_tmm_p[i, j] = res_p['R']
            T_tmm_s[i, j] = res_s['T']
            T_tmm_p[i, j] = res_p['T']

    if torch.cuda.is_available():
        O_fast_s_gpu = inc_tmm_fast('s', M, T, mask, theta, wl, device='cuda')
        O_fast_p_gpu = inc_tmm_fast('p', M, T, mask, theta, wl, device='cuda')

        assert O_fast_s_gpu, 'gpu computation not availabla'
        assert torch.allclose(R_tmm_s, O_fast_s_gpu['R'])
        assert torch.allclose(R_tmm_p, O_fast_p_gpu['R'])
        assert torch.allclose(T_tmm_s, O_fast_s_gpu['T'])
        assert torch.allclose(T_tmm_p, O_fast_p_gpu['T'])

    assert torch.allclose(R_tmm_s, O_fast_s['R'])
    assert torch.allclose(R_tmm_p, O_fast_p['R'])
    assert torch.allclose(T_tmm_s, O_fast_s['T'])
    assert torch.allclose(T_tmm_p, O_fast_p['T'])
    
    assert (O_fast_s['R'][0].isnan() == R_tmm_s.isnan()).all()
    assert (O_fast_p['R'][0].isnan() == R_tmm_p.isnan()).all()
    assert (O_fast_s['T'][0].isnan() == T_tmm_s.isnan()).all()
    assert (O_fast_p['T'][0].isnan() == T_tmm_p.isnan()).all()

    assert (torch.abs(O_fast_s['R'][0]-R_tmm_s) <1e-5).all().item()
    assert (torch.abs(O_fast_p['R'][0]-R_tmm_p) <1e-5).all().item()
    assert (torch.abs(O_fast_s['T'][0]-T_tmm_s) <1e-5).all().item()
    assert (torch.abs(O_fast_p['T'][0]-T_tmm_p) <1e-5).all().item()

def test_absorbing_coherent_incoherent():
    n_wl = 20
    n_th = 45
    wl = torch.linspace(400, 1200, n_wl) * (10**(-9))
    theta = torch.linspace(0, 85, n_th) * (np.pi/180)
    num_layers = 5
    num_stacks = 2
    mask = [[2, 3]]
    imask = ['i', 'i', 'c', 'c', 'i']

    #create m
    M = torch.ones((num_stacks, num_layers, wl.shape[0])).type(torch.complex128)
    for i in range(1, M.shape[1]-1):
        if np.mod(i, 2) == 1:
            M[:, i, :] *= 1.46
            M[:, i, :] += .0005j
        else:
            M[:, i, :] *= 2.56
            M[:, i, :] += .002j

    #create t
    max_t = 150 * (10**(-9))
    min_t = 10 * (10**(-9))
    T = (max_t - min_t) * np.random.uniform(0, 1, (M.shape[0], M.shape[1])) + min_t

    T[:, 0] = np.inf
    T[:, 1] = 10000e-9
    T[:, 2] = 100e-9
    T[:, 3] = 300e-9
    T[:, -1] = np.inf

    T = torch.from_numpy(T)
    O_fast_s = inc_tmm_fast('s', M, T, mask, theta, wl, device='cpu')
    O_fast_p = inc_tmm_fast('p', M, T, mask, theta, wl, device='cpu')

    R_tmm_s = torch.zeros((n_th, n_wl))
    R_tmm_p = torch.zeros((n_th, n_wl))
    T_tmm_s = torch.zeros((n_th, n_wl))
    T_tmm_p = torch.zeros((n_th, n_wl))

    T_list = T[0].tolist()

    for i, t in enumerate(theta.tolist()):
        for j, w in enumerate(wl.tolist()):
            res_s = inc_tmm('s', M[0][:, j].tolist(), T_list, imask, t, w)
            res_p = inc_tmm('p', M[0][:, j].tolist(), T_list, imask, t, w)
            R_tmm_s[i, j] = res_s['R']
            R_tmm_p[i, j] = res_p['R']
            T_tmm_s[i, j] = res_s['T']
            T_tmm_p[i, j] = res_p['T']

    if torch.cuda.is_available():
        O_fast_s_gpu = inc_tmm_fast('s', M, T, mask, theta, wl, device='cuda')
        O_fast_p_gpu = inc_tmm_fast('p', M, T, mask, theta, wl, device='cuda')

        assert O_fast_s_gpu, 'gpu computation not available'
        assert torch.allclose(R_tmm_s, O_fast_s_gpu['R'])
        assert torch.allclose(R_tmm_p, O_fast_p_gpu['R'])
        assert torch.allclose(T_tmm_s, O_fast_s_gpu['T'])
        assert torch.allclose(T_tmm_p, O_fast_p_gpu['T'])

    assert torch.allclose(R_tmm_s, O_fast_s['R'])
    assert torch.allclose(R_tmm_p, O_fast_p['R'])
    assert torch.allclose(T_tmm_s, O_fast_s['T'])
    assert torch.allclose(T_tmm_p, O_fast_p['T'])

    assert (O_fast_s['R'][0].isnan() == R_tmm_s.isnan()).all()
    assert (O_fast_p['R'][0].isnan() == R_tmm_p.isnan()).all()
    assert (O_fast_s['T'][0].isnan() == T_tmm_s.isnan()).all()
    assert (O_fast_p['T'][0].isnan() == T_tmm_p.isnan()).all()

    assert (torch.abs(O_fast_s['R'][0]-R_tmm_s) <1e-5).all().item()
    assert (torch.abs(O_fast_p['R'][0]-R_tmm_p) <1e-5).all().item()
    assert (torch.abs(O_fast_s['T'][0]-T_tmm_s) <1e-5).all().item()
    assert (torch.abs(O_fast_p['T'][0]-T_tmm_p) <1e-5).all().item()


if __name__=='__main__':
    # test_incoherent_input_output_medium()
    test_incoherent_numpy_input_output_medium()
    # test_coherent_stack_with_incoherent_surrounding()
    # test_coherent_incoherent()
    # test_absorbing_fully_incoherent()
    # test_absorbing_coherent_incoherent()
