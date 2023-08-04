import numpy as np
import torch
from tmm_fast import inc_tmm
import matplotlib.pyplot as plt

wl = torch.linspace(400, 1200, 100) * (10**(-9))
theta = torch.linspace(0, 45, 45) * (np.pi/180)
mode = 'T'
num_layers = 5
num_stacks = 2
mask = [[2,3]]

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
T[:, -1] = np.inf

T = torch.from_numpy(T)


O = inc_tmm('s', M, T, mask, theta, wl, device='cpu')


plt.plot(wl, O['R'][0,0])

a=0
