import numpy as np
from tmm_fast.vectorized_tmm_dispersive_multistack import coh_vec_tmm_disp_mstack as tmm

wl = np.linspace(400, 1200, 800) * (10**(-9))
theta = np.linspace(0, 45, 45) * (np.pi/180)
mode = 'T'
num_layers = 4
num_stacks = 128

#create m
M = np.ones((num_stacks, num_layers, wl.shape[0]))
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
T[:, -1] = np.inf

#tmm:
O = tmm('s', M, T, theta, wl, device='cpu')

print(':)')
