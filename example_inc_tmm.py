import numpy as np
import torch
from tmm_fast import inc_tmm as inc_tmm_fast
import matplotlib.pyplot as plt

from tmm import inc_tmm 

np.random.seed(111)
torch.manual_seed(111)

n_wl = 150
n_th = 45
pol = 's'
wl = torch.linspace(400, 1200, n_wl) * (10**(-9))
theta = torch.linspace(0, 89, n_th) * (np.pi/180)
num_layers = 5
num_stacks = 2
mask = [[1], [3]]
imask = ['i', 'c', 'i', 'c', 'i']

#create m
M = torch.ones((num_stacks, num_layers, wl.shape[0])).type(torch.complex128)
for i in range(1, M.shape[1]-1):
    if np.mod(i, 2) == 1:
        M[:, i, :] *= np.random.uniform(1,3,[1])[0]
        M[:, i, :] += .05j
    else:
        M[:, i, :] *= np.random.uniform(1,3,[1])[0]
        M[:, i, :] += .02j

#create t
max_t = 350 * (10**(-9))
min_t = 10 * (10**(-9))
T = (max_t - min_t) * np.random.uniform(0, 1, (M.shape[0], M.shape[1])) + min_t

T[:, 0] = np.inf
T[:, 2] = 2000e-9
# T[:, 2] = 100e-9
# T[:, 3] = 300e-9
T[:, -1] = np.inf

T = torch.from_numpy(T)


O_fast = inc_tmm_fast(pol, M, T, mask, theta, wl, device='cpu')

R_tmm = np.zeros((n_th, n_wl))
T_tmm = np.zeros((n_th, n_wl))
# VW = np.zeros((n_th, n_wl, 3, 2))
# L_list = 
M_list = M[0].tolist()
T_list = T[0].tolist()

for i, t in enumerate(theta.tolist()):
    for j, w in enumerate(wl.tolist()):
        O = inc_tmm(pol, M[0][:, j].tolist(), T_list, imask, t, w)
        R_tmm[i, j] = O['R']
        T_tmm[i, j] = O['T']
        # VW[i,j] = O['VW_list']

cbar = plt.imshow(O_fast['T'][0].numpy(), aspect='auto')
plt.colorbar(cbar)

fig, ax = plt.subplots(1,1)
cbar = ax.imshow(T_tmm, aspect='auto')
plt.colorbar(cbar, ax=ax)

fig2, ax2 = plt.subplots(1,1)
cbar2 = ax2.imshow(1 - O_fast['T'][0].numpy() / T_tmm, aspect='auto')
plt.colorbar(cbar2, ax=ax2)
plt.draw()

print('The results for R are identical up to machine precision:', np.allclose(O_fast['R'][0].numpy(), R_tmm))
print('The results for T are identical up to machine precision:', np.allclose(O_fast['T'][0].numpy(), T_tmm))
print("delta R max " + str(np.abs(O_fast['R'][0].numpy()-R_tmm).max()))
print('delta T max ' + str(np.abs(O_fast['T'][0].numpy()-T_tmm).max()))
a=0
