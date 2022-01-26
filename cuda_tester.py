import numpy as np
from vectorized_tmm_dispersive_multistack import coh_vec_tmm_disp_mstack as tmm
import time

wl = np.linspace(600, 1200, 600) * (10**(-9))
theta = np.linspace(0, 45, 45) * (np.pi/180)
mode = 'T'
num_layers = 32
num_stacks = 128

#create m
m = np.ones((num_layers, wl.shape[0]))
for i in range(1, m.shape[0]-1):
    if np.mod(i, 2) == 1:
        m[i, :] *= 1.46
    else:
        m[i, :] *= 2.56
noiser = np.linspace(1.0, 0.5, m.shape[1])
m = m[:] * noiser
#create t
max_t = 150 * (10**(-9))
min_t = 10 * (10**(-9))
t = (max_t - min_t) * np.random.uniform(0, 1, m.shape[0]) + min_t
t[0] = np.inf
t[-1] = np.inf

#device comparison:
# CPU
device = 'cpu'
for stack in range(num_stacks):
    Ocpu, cpu_pt,total_time_cpu = tmm('s', m, t, theta, wl, device, timer=True)
# GPU
device = 'cuda'
for stack in range(num_stacks):
    Ocuda, cuda_pt, total_time_cuda = tmm('s', m, t, theta, wl, device, timer=True)

print('CUDA push time: ' + str(cuda_pt) + '\nCPU push time: ' + str(cpu_pt))
print('CUDA time: ' + str(total_time_cuda) + '\nCPU time: ' + str(total_time_cpu))
print(':)')
