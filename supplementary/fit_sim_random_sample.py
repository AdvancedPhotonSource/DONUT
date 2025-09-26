# Run this script to obtain conventional correlation fitting analysis on simulated scan

import os
import numpy as np
from math import *
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch
import time
from skimage import io

data = np.load('/data/aileen/DONUT_data/sim_random_sample.npy')
data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2], data.shape[3]))
data[np.isnan(data)] = 0
rng = np.random.default_rng()
for i in range(data.shape[0]):
    data[i] = (data[i] / np.max(data[i])) * 7
    data[i] = rng.poisson(data[i])
data = torch.tensor(data, dtype=torch.float32, device='cuda')

# Solve for parameters directly using the simulated data
folder = '/data/aileen/DONUT_data'

sim_mat = np.load(os.path.join(folder, 'sim_data.npy')).astype('float32')

sim_mat[np.isnan(sim_mat)] = 0
sim_mat /= sim_mat.sum(axis=(3, 4), keepdims=True)
sim_mat = torch.tensor(sim_mat, dtype=torch.float32, device='cuda')

strain = torch.zeros((data.shape[0], ), dtype=torch.float32, device='cuda')
tilt_lr = torch.zeros((data.shape[0], ), dtype=torch.float32, device='cuda')
tilt_ud = torch.zeros((data.shape[0], ), dtype=torch.float32, device='cuda')

fit_time = []

i = 0
t0 = time.time()

while i + 8 < data.shape[0]:
    
    t1 = time.time()

    dd_sum = (data[i:i+8, None, None, None]*sim_mat).sum(axis=(4, 5))

    dd_sum_s = dd_sum.sum(axis=(2, 3))
    dd_sum_s -= torch.min(dd_sum_s, dim=1, keepdim=True)[0]

    dd_sum_lr = dd_sum.sum(axis=(1, 3))
    dd_sum_lr -= torch.min(dd_sum_lr, dim=1, keepdim=True)[0]

    dd_sum_ud = dd_sum.sum(axis=(1, 2))
    dd_sum_ud -= torch.min(dd_sum_ud, dim=1, keepdim=True)[0]
    
    i_s = (dd_sum_s*torch.arange(41, dtype=torch.float32, device='cuda')).sum(axis=(1))/dd_sum_s.sum(axis=(1))
    s = (i_s-20)*0.00025

    i_lr = (dd_sum_lr*torch.arange(41, dtype=torch.float32, device='cuda')).sum(axis=(1))/dd_sum_lr.sum(axis=(1))
    lr = (i_lr-20)*0.0025
    
    i_ud = (dd_sum_ud*torch.arange(41, dtype=torch.float32, device='cuda')).sum(axis=(1))/dd_sum_ud.sum(axis=(1))
    ud = (i_ud-20)*0.005

    strain[i:i+8] = s
    tilt_lr[i:i+8] = lr
    tilt_ud[i:i+8] = ud
    
    fit_time.append(time.time()-t1)
    
    i += 8

t1 = time.time()

dd_sum = (data[i:, None, None, None]*sim_mat).sum(axis=(4, 5))

dd_sum_s = dd_sum.sum(axis=(2, 3))
dd_sum_s -= torch.min(dd_sum_s, dim=1, keepdim=True)[0]

dd_sum_lr = dd_sum.sum(axis=(1, 3))
dd_sum_lr -= torch.min(dd_sum_lr, dim=1, keepdim=True)[0]

dd_sum_ud = dd_sum.sum(axis=(1, 2))
dd_sum_ud -= torch.min(dd_sum_ud, dim=1, keepdim=True)[0]

i_s = (dd_sum_s*torch.arange(41, dtype=torch.float32, device='cuda')).sum(axis=(1))/dd_sum_s.sum(axis=(1))
s = (i_s-20)*0.00025

i_lr = (dd_sum_lr*torch.arange(41, dtype=torch.float32, device='cuda')).sum(axis=(1))/dd_sum_lr.sum(axis=(1))
lr = (i_lr - 20)*0.0025

i_ud = (dd_sum_ud*torch.arange(41, dtype=torch.float32, device='cuda')).sum(axis=(1))/dd_sum_ud.sum(axis=(1))
ud = (i_ud-20)*0.005

strain[i:] = s
tilt_lr[i:] = lr
tilt_ud[i:] = ud

fit_time.append(time.time()-t1)

print('Average fit time (s): ', np.mean(fit_time[2:len(fit_time)-1])/8)
print('Fit time error (s): ', np.std(fit_time[2:len(fit_time)-1])/8)
print('Total time (s): ', time.time()-t0)

strain_cpu = strain.cpu()
tilt_lr_cpu = tilt_lr.cpu()
tilt_ud_cpu = tilt_ud.cpu()

np.save(os.path.join(folder, 'strain_fit_random.npy'), strain_cpu)
np.save(os.path.join(folder, 'tilt_lr_fit_random.npy'), tilt_lr_cpu)
np.save(os.path.join(folder, 'tilt_ud_fit_random.npy'), tilt_ud_cpu)