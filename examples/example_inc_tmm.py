import numpy as np
import torch
from tmm_fast import inc_tmm as inc_tmm_fast
import matplotlib.pyplot as plt

np.random.seed(111)
torch.manual_seed(111)

n_wl = 75
n_th = 45
pol = 's'
wl = torch.linspace(400, 1200, n_wl) * (10**(-9))
theta = torch.linspace(0, 89, n_th) * (np.pi/180)
num_layers = 6
num_stacks = 2
mask = [[1, 2], [4]]
imask = ['i', 'c', 'c', 'i', 'c', 'i']

#create m
M = torch.ones((num_stacks, num_layers, wl.shape[0]), dtype=torch.complex128)
# for i in range(1, M.shape[1]-1):
#     if np.mod(i, 2) == 1:
#         M[:, i, :] *= np.random.uniform(1,3,[1])[0]
#         M[:, i, :] += .05j
#     else:
#         M[:, i, :] *= np.random.uniform(1,3,[1])[0]
#         M[:, i, :] += .02j

M[:, 1] = 2.2 + .0j
M[:, 2] = 1.3 + .0j
M[:, 3] = 2.2 + .05j
M[:, 4] = 1.3 + .2j

#create t
max_t = 350 * (10**(-9))
min_t = 10 * (10**(-9))
T = (max_t - min_t) * np.random.uniform(0, 1, (M.shape[0], M.shape[1])) + min_t

T[:, 0] = np.inf
T[:, 1] = 250e-9
T[:, 2] = 2000e-9
T[:, 3] = 150e-9
T[:, -1] = np.inf

T = torch.from_numpy(T)
O_fast = inc_tmm_fast(pol, M, T, mask, theta, wl, device='cpu')

fig, ax = plt.subplots(2,1)
cbar = ax[0].imshow(O_fast['R'][0].numpy(), aspect='auto')
ax[0].set_xlabel('wavelength')
ax[0].set_ylabel('angle of incidence')
plt.colorbar(cbar, ax=ax[0])
ax[0].title.set_text('Reflectivity')
cbar = ax[1].imshow(O_fast['T'][0].numpy(), aspect='auto')
plt.colorbar(cbar, ax=ax[1])
ax[1].title.set_text('Transmissivity')
ax[1].set_xlabel('wavelength')
ax[1].set_ylabel('angle of incidence')
fig.tight_layout(pad=2.)

plt.show()
a=0
