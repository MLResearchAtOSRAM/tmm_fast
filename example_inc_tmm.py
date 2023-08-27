import numpy as np
import torch
from tmm_fast import inc_tmm as inc_tmm_fast
import matplotlib.pyplot as plt

from tmm_core import inc_tmm 

np.random.seed(111)
torch.manual_seed(111)

n_wl = 150
n_th = 45
pol = 's'
wl = torch.linspace(400, 1200, n_wl) * (10**(-9))
theta = torch.linspace(0, 50, n_th) * (np.pi/180)
num_layers = 6
num_stacks = 2
mask = [[1, 2], [4]]
imask = ['i', 'c', 'c', 'i', 'c', 'i']

#create m
M = torch.ones((num_stacks, num_layers, wl.shape[0])).type(torch.complex128)
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
M[:, 4] = 1.3 + .0j

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
L_tmm_fast = O_fast['L'][0]

R_tmm = np.ones((n_th, n_wl)) * 2
T_tmm = np.ones((n_th, n_wl)) * 2
# VW = np.zeros((n_th, n_wl, 3, 2))
L_tmm = np.empty((2, n_th, n_wl, 2, 2))
coh0_tmm_R = np.empty((n_th, n_wl))
coh1_tmm_R = np.empty((n_th, n_wl))
coh0_tmm_T = np.empty((n_th, n_wl))
coh1_tmm_T = np.empty((n_th, n_wl))
P = np.empty((n_th, n_wl, 2, 2))
th_list = np.empty((n_th, num_layers, n_wl), dtype=complex)
M_list = M[0].tolist()
T_list = T[0].tolist()

for i, t in enumerate(theta.tolist()):
    for j, w in enumerate(wl.tolist()):
        O = inc_tmm(pol, M[0][:, j].tolist(), T_list, imask, t, w)
        R_tmm[i, j] = O['R']
        T_tmm[i, j] = O['T']
        L_tmm[:, i, j] = np.array(O['L'])
        coh0_tmm_R[i, j] = O['coh_tmm_data_list'][0]['R']
        coh1_tmm_R[i, j] = O['coh_tmm_data_list'][1]['R']
        coh0_tmm_T[i, j] = O['coh_tmm_data_list'][0]['T']
        coh1_tmm_T[i, j] = O['coh_tmm_data_list'][1]['T']
        P[i, j, 0, 0] = 1/np.array(O['P'])[1]
        P[i, j, 1, 1] = np.array(O['P'])[1]
        th_list[i, :, j] = O['th_list']
        # VW[i,j] = O['VW_list']


fig, ax = plt.subplots(3,1)
cbar = ax[0].imshow(O_fast['T'][0].numpy(), aspect='auto')
plt.colorbar(cbar, ax=ax[0])
cbar = ax[1].imshow(T_tmm, aspect='auto')
plt.colorbar(cbar, ax=ax[1])
cbar = ax[2].imshow(1 - O_fast['T'][0].numpy() / T_tmm, aspect='auto')
plt.colorbar(cbar, ax=ax[2])

fig2, ax2 = plt.subplots(3,1)
cbar = ax2[0].imshow(coh0_tmm_R, aspect='auto')
plt.colorbar(cbar, ax=ax2[0])
cbar = ax2[1].imshow(O_fast['coh_tmm_f'][0]['R'].numpy()[0], aspect='auto')
plt.colorbar(cbar, ax=ax2[1])
cbar = ax2[2].imshow(1 - O_fast['coh_tmm_f'][0]['R'].numpy()[0] / coh0_tmm_R, aspect='auto')
plt.colorbar(cbar, ax=ax2[2])

fig3, ax3 = plt.subplots(3,1)
cbar = ax3[0].imshow(coh1_tmm_R, aspect='auto')
plt.colorbar(cbar, ax=ax3[0])
cbar = ax3[1].imshow(O_fast['coh_tmm_f'][1]['R'].numpy()[0], aspect='auto')
plt.colorbar(cbar, ax=ax3[1])
cbar = ax3[2].imshow(1 - O_fast['coh_tmm_f'][1]['R'].numpy()[0] / coh1_tmm_R, aspect='auto')
plt.colorbar(cbar, ax=ax3[2])

# fig3, ax3 = plt.subplots(3,1)
# cbar3 = ax3[0].imshow(L_tmm_fast[0, :, :, 0, 0].numpy(), aspect='auto')
# plt.colorbar(cbar3, ax=ax3[0])
# cbar3 = ax3[1].imshow(L_tmm[0, :, :, 0, 0], aspect='auto')
# plt.colorbar(cbar3, ax=ax3[1])
# cbar3 = ax3[2].imshow(1 - L_tmm_fast[0, :, :, 0, 0].numpy() / L_tmm[0, :, :, 0, 0], aspect='auto')
# plt.colorbar(cbar3, ax=ax3[2])

# fig4, ax4 = plt.subplots(3,1)
# cbar4 = ax4[0].imshow(L_tmm_fast[0, :, :, 1, 0].numpy(), aspect='auto')
# plt.colorbar(cbar4, ax=ax4[0])
# cbar4 = ax4[1].imshow(L_tmm[0, :, :, 1, 0], aspect='auto')
# plt.colorbar(cbar4, ax=ax4[1])
# cbar4 = ax4[2].imshow(1- L_tmm_fast[0, :, :, 1, 0].numpy() / L_tmm[0, :, :, 1, 0], aspect='auto')
# plt.colorbar(cbar4, ax=ax4[2])

# fig5, ax5 = plt.subplots(3,1)
# cbar5 = ax5[0].imshow(L_tmm_fast[0, :, :, 0, 1].numpy(), aspect='auto')
# plt.colorbar(cbar5, ax=ax5[0])
# cbar5 = ax5[1].imshow(L_tmm[0, :, :, 0, 1], aspect='auto')
# plt.colorbar(cbar5, ax=ax5[1])
# cbar5 = ax5[2].imshow(1- L_tmm_fast[0, :, :, 0, 1].numpy() / L_tmm[0, :, :, 0, 1], aspect='auto')
# plt.colorbar(cbar5, ax=ax5[2])

# fig6, ax6 = plt.subplots(3,1)
# cbar6 = ax6[0].imshow(L_tmm_fast[0, :, :, 1, 1].numpy(), aspect='auto')
# plt.colorbar(cbar6, ax=ax6[0])
# cbar6 = ax6[1].imshow(L_tmm[0, :, :, 1, 1], aspect='auto')
# plt.colorbar(cbar6, ax=ax6[1])
# cbar6 = ax6[2].imshow(1- L_tmm_fast[0, :, :, 1, 1].numpy() / L_tmm[0, :, :, 1, 1], aspect='auto')
# plt.colorbar(cbar6, ax=ax6[2])

# fig7, ax7 = plt.subplots(3,1)
# cbar7 = ax7[0].imshow(L_tmm_fast[1, :, :, 0, 0].numpy(), aspect='auto')
# plt.colorbar(cbar7, ax=ax7[0])
# cbar7 = ax7[1].imshow(L_tmm[1, :, :, 0, 0], aspect='auto')
# plt.colorbar(cbar7, ax=ax7[1])
# cbar7 = ax7[2].imshow(1- L_tmm_fast[1, :, :, 0, 0].numpy()  /L_tmm[1, :, :, 0, 0], aspect='auto')
# plt.colorbar(cbar7, ax=ax7[2])

# fig8, ax8 = plt.subplots(3,1)
# cbar8 = ax8[0].imshow(L_tmm_fast[1, :, :, 1, 0].numpy(), aspect='auto')
# plt.colorbar(cbar8, ax=ax8[0])
# cbar8 = ax8[1].imshow(L_tmm[1, :, :, 1, 0], aspect='auto')
# plt.colorbar(cbar8, ax=ax8[1])
# cbar8 = ax8[2].imshow(1- L_tmm_fast[1, :, :, 1, 0].numpy() / L_tmm[1, :, :, 1, 0], aspect='auto')
# plt.colorbar(cbar8, ax=ax8[2])

# fig9, ax9 = plt.subplots(3,1)
# cbar9 = ax9[0].imshow(L_tmm_fast[1, :, :, 0, 1].numpy(), aspect='auto')
# plt.colorbar(cbar9, ax=ax9[0])
# cbar9 = ax9[1].imshow(L_tmm[1, :, :, 0, 1], aspect='auto')
# plt.colorbar(cbar9, ax=ax9[1])
# cbar9 = ax9[2].imshow(1- L_tmm_fast[1, :, :, 0, 1].numpy() / L_tmm[1, :, :, 0, 1], aspect='auto')
# plt.colorbar(cbar9, ax=ax9[2])

# fig10, ax10 = plt.subplots(3,1)
# cbar10 = ax10[0].imshow(L_tmm_fast[1, :, :, 1, 1].numpy(), aspect='auto')
# plt.colorbar(cbar10, ax=ax10[0])
# cbar10 = ax10[1].imshow(L_tmm[1, :, :, 1, 1], aspect='auto')
# plt.colorbar(cbar10, ax=ax10[1])
# cbar10 = ax10[2].imshow(1- L_tmm_fast[1, :, :, 1, 1].numpy() / L_tmm[1, :, :, 1, 1], aspect='auto')
# plt.colorbar(cbar10, ax=ax10[2])

for i in range(5):
    fig, ax = plt.subplots(3,1)
    cbar = ax[0].imshow(th_list.real[:, i], aspect='auto')
    plt.colorbar(cbar, ax=ax[0])
    cbar = ax[1].imshow(O_fast['th_list'][0].numpy().real[:, i], aspect='auto')
    plt.colorbar(cbar, ax=ax[1])
    cbar = ax[2].imshow(O_fast['th_list'][0].numpy().real[:, i] - th_list.real[:, i], aspect='auto')
    plt.colorbar(cbar, ax=ax[2])

for i in range(5):
    fig, ax = plt.subplots(3,1)
    cbar = ax[0].imshow(th_list.imag[:, i], aspect='auto')
    plt.colorbar(cbar, ax=ax[0])
    cbar = ax[1].imshow(O_fast['th_list'][0].numpy().imag[:, i], aspect='auto')
    plt.colorbar(cbar, ax=ax[1])
    cbar = ax[2].imshow(O_fast['th_list'][0].numpy().imag[:, i] - th_list.imag[:, i], aspect='auto')
    plt.colorbar(cbar, ax=ax[2])

plt.draw()


print('The results for R are identical up to machine precision:', np.allclose(O_fast['R'][0].numpy(), R_tmm))
print('The results for T are identical up to machine precision:', np.allclose(O_fast['T'][0].numpy(), T_tmm))
print("delta R max " + str(np.abs(O_fast['R'][0].numpy()-R_tmm).max()))
print('delta T max ' + str(np.abs(O_fast['T'][0].numpy()-T_tmm).max()))
a=0
