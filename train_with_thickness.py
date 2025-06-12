
import os
import torch
import numpy as np
from math import *
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

data_folder = '/home/aileenluo/benchmark'

sim_mat = np.load(os.path.join(data_folder, 'sim_SIO_4nm_range.npy'))

#print(np.max(sim_mat[0, 10, 10, 10]), np.max(sim_mat[5, 10, 10, 10]), np.max(sim_mat[10, 10, 10, 10]))
sim_mat[np.isnan(sim_mat)]=0
sim_mat /= sim_mat.sum(axis=(4, 5), keepdims=True)
#print(np.max(sim_mat), np.max(sim_mat[5, 10, 10, 10]))
max_avg = np.max(sim_mat[5, 10, 10, 10])

sim_mat = np.reshape(sim_mat, (sim_mat.shape[0]*sim_mat.shape[1]*sim_mat.shape[2]*sim_mat.shape[3], 
                               sim_mat.shape[4], sim_mat.shape[5]))
avg_max = np.mean(sim_mat.max(axis=(1, 2)))
print('Avg. max.: ', avg_max)

print('Data shape: ', sim_mat.shape)

thickness = np.linspace(67, 167, 11)
strain = np.linspace(-0.005, 0.005, 21)
tilt_lr = np.linspace(-0.05, 0.05, 21)
tilt_ud = np.linspace(-0.1, 0.1, 21)

# Normalize data from 0 to average experimental single shot maximum value
sim_mat[np.isnan(sim_mat)] = 0
#sim_mat = (sim_mat / max_avg) * 7
for i in range(sim_mat.shape[0]):
    sim_mat[i] = (sim_mat[i] / avg_max) * 7
print('Normalized min. and max.: ', np.min(sim_mat), np.max(sim_mat))

# Add Poisson noise 
sim_mat_noisy = np.zeros(sim_mat.shape)
rng = np.random.default_rng()
for i in range(sim_mat_noisy.shape[0]):
    sim_mat_noisy[i] = rng.poisson(sim_mat[i])
    # Log scale training and prediction
    #sim_mat_noisy[i] = np.log10(sim_mat_noisy[i]+1e-8)

sim_mat = sim_mat.astype('float32')
sim_mat_noisy = np.rint(sim_mat_noisy).astype('float32')
#sim_mat_noisy = sim_mat_noisy.astype('float32')
print('Data type: ', sim_mat.dtype, sim_mat_noisy.dtype)

# Experimental data
folder = '/home/aileenluo/benchmark'
data = np.load(os.path.join(folder, 'data190.npy')).astype(np.float32)

# Combined data
all_data = np.vstack((sim_mat_noisy, data))
print('Combined data size:', all_data.shape)

class DiffDataset(Dataset):
    """Makes PyTorch Dataset object for diffraction data (images)."""
    
    def __init__(self, data, params=None, transform=None):
        self.data = data
        if isinstance(params, np.ndarray):
            self.params = params
        else:
            self.params = None
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        image = self.data[idx]
        if isinstance(self.params, np.ndarray):
            lattice = self.params[idx]
            sample={'image': image, 'lattice': lattice}
        else:
            sample = {'image': image, 'lattice': None}
            
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class ToTensor(object):
    """Convert numpy arrays to Tensors"""
    def __call__(self, sample):
        image = sample['image']
        if isinstance(sample['lattice'], np.ndarray):
            lattice = sample['lattice']
            return {'image': torch.unsqueeze(torch.from_numpy(image), 0), 'lattice': torch.from_numpy(lattice)}
        return {'image': torch.unsqueeze(torch.from_numpy(image), 0), 'lattice': 0}
    
diff_dataset = DiffDataset(data=sim_mat, params=None, transform=ToTensor())
diff_dataset_noisy = DiffDataset(data=sim_mat_noisy, params=None, transform=ToTensor())
exp_dataset = DiffDataset(data=data, params=None, transform=ToTensor())
all_dataset = DiffDataset(data=all_data, params=None, transform=ToTensor())

generator0 = torch.Generator().manual_seed(8)
subsets = torch.utils.data.random_split(diff_dataset, [0.8, 0.1, 0.1], generator=generator0)
subsets_noisy = torch.utils.data.random_split(diff_dataset_noisy, [0.8, 0.1, 0.1], generator=generator0)
subsets_exp = torch.utils.data.random_split(exp_dataset, [0.8, 0.1, 0.1], generator=generator0)
subsets_all = torch.utils.data.random_split(all_dataset, [0.8, 0.1, 0.1], generator=generator0)

NGPUS = 8 #torch.cuda.device_count() # Use all available GPUs
BATCH_SIZE = 256
LR = 1e-4
print("GPUs:", NGPUS, "| Batch size:", BATCH_SIZE, "| Learning rate:", LR)

EPOCHS = 60
MODEL_SAVE_PATH = '/home/aileenluo/benchmark/model_thickness_6'

# Use a DataLoader to iterate through the Dataset
trainloader = DataLoader(subsets[0], batch_size=BATCH_SIZE, shuffle=True)
validloader = DataLoader(subsets[1], batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(subsets[2], batch_size=BATCH_SIZE, shuffle=False)

trainloader_noisy = DataLoader(subsets_noisy[0], batch_size=BATCH_SIZE, shuffle=True)
validloader_noisy = DataLoader(subsets_noisy[1], batch_size=BATCH_SIZE, shuffle=True)
testloader_noisy = DataLoader(subsets_noisy[2], batch_size=BATCH_SIZE, shuffle=False)

trainloader_exp = DataLoader(subsets_exp[0], batch_size=BATCH_SIZE, shuffle=True)
validloader_exp = DataLoader(subsets_exp[1], batch_size=BATCH_SIZE, shuffle=True)
testloader_exp = DataLoader(subsets_exp[2], batch_size=BATCH_SIZE, shuffle=False)

trainloader_all = DataLoader(subsets_all[0], batch_size=BATCH_SIZE, shuffle=True)
validloader_all = DataLoader(subsets_all[1], batch_size=BATCH_SIZE, shuffle=True)
testloader_all = DataLoader(subsets_all[2], batch_size=BATCH_SIZE, shuffle=False)

class MCDropout(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super().__init__()
        self.p = dropout_rate
       
    def forward(self, x: torch.Tensor):
        return nn.functional.dropout(x, p=self.p, training=True)
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), 
            nn.Dropout(dropout_rate),
            #MCDropout(dropout_rate),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            #MCDropout(dropout_rate)
        )
        
    def forward(self, x: torch.Tensor):
        return self.conv(x)

# Autoencoder building blocks
class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d((2, 2))
        
    def forward(self, x: torch.Tensor):
        return self.pool(x)
    
class UpSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        
    def forward(self, x: torch.Tensor):
        return self.up(x)
    
class DoubleConvUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            #MCDropout(dropout_rate),
            nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), 
            nn.Dropout(dropout_rate)
            #MCDropout(dropout_rate)
        )
        
    def forward(self, x: torch.Tensor):
        return self.conv(x)
    
class Encoder(nn.Module):
    def __init__(self, in_channels: int, dropout_rate=0.0):
        super().__init__()
        self.down_conv = nn.ModuleList([DoubleConv(i, o, dropout_rate) for i, o in 
                                        [(in_channels, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 256)]])
        self.down_sample = nn.ModuleList([DownSample() for _ in range(6)])
        self.fc = nn.Linear(256, 4)
        
    def forward(self, x: torch.Tensor):
        for i in range(len(self.down_conv)):
            x = self.down_conv[i](x)
            x = self.down_sample[i](x)
            #print(x.shape)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, out_channels: int, dropout_rate=0.0):
        super().__init__()
        self.up_sample = UpSample()
        self.up_conv = nn.ModuleList([DoubleConvUp(i, o, dropout_rate) for i, o in 
                                      [(256, 256), (256, 128), (128, 64), (64, 32), (32, 16), (16, out_channels)]])
        self.inv_fc = nn.Linear(4, 256)
        
    def forward(self, x: torch.Tensor):
        x = self.inv_fc(x)
        x = x.view(x.shape[0], 256, 1, 1)
        for i in range(len(self.up_conv)):
            x = self.up_sample(x)
            x = self.up_conv[i](x)
            #print(x.shape)
        return x
    
class DonutNN2(nn.Module):
    """Version with Tao's new forward model."""
    
    def __init__(self, in_channels: int, out_channels: int, dropout_rate=0.0, **kwargs):
        super().__init__()
        
        self.upsampling = kwargs.get('upsampling', 2) # Upsampling factor for reciprocal space
        self.energy = kwargs.get('energy', 11.3) # X-ray energy in keV
        self.c = kwargs.get('c', 4.013) # Lattice constant in Angstroms
        self.l = kwargs.get('l', 2) # l from hkl Bragg peak
        #self.thickness = kwargs.get('thickness', 117) # Film thickness in Angstroms
        self.X0 = kwargs.get('X0', 256) # Pixel coordinate of the center of Bragg peak
        self.Xcen = kwargs.get('Xcen', 256) # Pixel coordinate of the center of ZP reflection on detector
        self.zp_diameter = kwargs.get('zp_diameter', 149e-6) # Zone plate diameter in meters
        self.cs_diameter = kwargs.get('cs_diameter', 77e-6) # Central beam stop diameter in meters
        self.outer_zone_width = kwargs.get('outer_zone_width', 16e-9) # Outermost zone width in meters
        self.distance = kwargs.get('distance', 0.85) # Detector distance from sample in meters
        self.pixelsize = kwargs.get('pixelsize', 55e-6*2) # Detector pixel size in meters (*2 for binning)
        
        # Some more parameters that depend on those above
        self.wavelength = 12.398/self.energy # X-ray wavelength in Angstroms
        self.K = 2*pi/self.wavelength
        self.alf = asin(self.wavelength*self.l/2/self.c) # Incident angle in radians
        self.gam0 = asin(self.wavelength*self.l/self.c-sin(self.alf)) # Exit angle in radians
        self.focal_length = (self.zp_diameter*self.outer_zone_width)/(self.wavelength*1e-10) # ZP-sample distance (m)
        self.outer_angle = self.zp_diameter/2/self.focal_length
        self.inner_angle = self.cs_diameter/2/self.focal_length
        self.wx = self.K*self.pixelsize/self.distance/10 # FWHM of Gaussian function describing Bragg peak in qx
        self.wy = self.K*self.pixelsize/self.distance/10 # FWHM of Gaussian function describing Bragg peak in qy
        
        # Detector and reciprocal space
        gam, det_Qx, det_Qz, det_Qy = self.detector_space()
        self.register_buffer('gam', gam, persistent=False) # Should not need register buffer in PyTorch 2
        self.register_buffer('det_Qx', det_Qx, persistent=False)
        self.register_buffer('det_Qz', det_Qz, persistent=False)
        self.register_buffer('det_Qy', det_Qy, persistent=False)
        O_Qx, O_Qz, O_Qy = self.recip_space()
        self.register_buffer('O_Qx', O_Qx, persistent=False)
        self.register_buffer('O_Qz', O_Qz, persistent=False)
        self.register_buffer('O_Qy', O_Qy, persistent=False)
        
        # Initialize autoencoder
        self.encoder = Encoder(in_channels, dropout_rate)
        self.decoder = Decoder(out_channels, dropout_rate)
        
        self.epoch = 0
        # Parameter-specific scaling, initialized with log values of scaling factors
        self.log_scales = nn.Parameter(torch.tensor([np.log(50), np.log(5e-3), np.log(5e-2), np.log(1e-1)], dtype=torch.float32), requires_grad=False)
        
    def detector_space(self):
        det_x = np.arange(64).astype(np.float32)
        det_y = np.arange(64).astype(np.float32)
        det_x = det_x - det_x.mean() + self.X0 - (self.X0-self.Xcen)
        det_y -= det_y.mean()
        det_xx, det_yy = np.meshgrid(det_x, det_y)
        gam = np.arcsin((det_xx-self.X0)*self.pixelsize/self.distance)+self.gam0
        gam = torch.Tensor(gam)
        det_Qx = torch.Tensor(self.K*(np.cos(self.alf)-np.cos(gam)))
        det_Qz = torch.Tensor(self.K*(np.sin(gam)+np.sin(self.alf)))
        det_Qy = torch.Tensor(det_yy*self.pixelsize/self.distance*self.K)
        return gam, det_Qx, det_Qz, det_Qy
    
    def recip_space(self):
        O_x = np.arange(64*self.upsampling).astype(np.float32)
        O_y = np.arange(64*self.upsampling).astype(np.float32)
        O_x -= O_x.mean()
        O_y -= O_y.mean()
        O_xx, O_yy = np.meshgrid(O_x, O_y)
        O_xx = O_xx[:, :, np.newaxis, np.newaxis]
        O_yy = O_yy[:, :, np.newaxis, np.newaxis]
        O_Qx = -O_xx*self.pixelsize/self.upsampling/self.distance*self.K*sin(self.alf)
        O_Qz = O_xx*self.pixelsize/self.upsampling/self.distance*self.K*cos(self.alf)
        O_Qy = O_yy*self.pixelsize/self.upsampling/self.distance*self.K
        # Zone plate effects
        O_angle = np.sqrt(O_yy**2+O_xx**2)*self.pixelsize/self.upsampling/self.distance
        O_donut = (O_angle < self.outer_angle) * (O_angle > self.inner_angle)
        O_Qx = torch.Tensor(O_Qx[O_donut][:, np.newaxis, np.newaxis])
        O_Qy = torch.Tensor(O_Qy[O_donut][:, np.newaxis, np.newaxis])
        O_Qz = torch.Tensor(O_Qz[O_donut][:, np.newaxis, np.newaxis])
        return O_Qx, O_Qz, O_Qy
    
    def forward(self, x: torch.Tensor):
        
        # Encoder
        x = self.encoder(x)
        
        # Constrain output space
        constrained_params = 1.7159 * torch.tanh((2/3) * x) # LeCun, et al. Efficient BackProp 1998
        scales = torch.exp(self.log_scales)
        strain_tensor = constrained_params * scales
        
        # Decoder
        recon = self.decoder(constrained_params)
        
        # X-ray scattering model
        det_Qz = torch.tile(self.det_Qz, (strain_tensor.shape[0], 1, 1))
        det_Qz = det_Qz[:, None, :, :]
        det_Qx = torch.tile(self.det_Qx, (strain_tensor.shape[0], 1, 1))
        det_Qx = det_Qx[:, None, :, :]
        det_Qy = torch.tile(self.det_Qy, (strain_tensor.shape[0], 1, 1))
        det_Qy = det_Qy[:, None, :, :]
        
        O_Qz = torch.tile(self.O_Qz, (strain_tensor.shape[0], 1, 1, 1))
        O_Qx = torch.tile(self.O_Qx, (strain_tensor.shape[0], 1, 1, 1))
        O_Qy = torch.tile(self.O_Qy, (strain_tensor.shape[0], 1, 1, 1))
        
        thickness = (strain_tensor[:, 0] + 117)
        thickness = thickness.view(strain_tensor.shape[0], 1, 1, 1)
        strain = strain_tensor[:, 1]
        strain = strain.view(strain_tensor.shape[0], 1, 1, 1)
        tilt_lr = torch.deg2rad(strain_tensor[:, 2])
        tilt_lr = tilt_lr.view(strain_tensor.shape[0], 1, 1, 1)
        tilt_ud = torch.deg2rad(strain_tensor[:, 3])
        tilt_ud = tilt_ud.view(strain_tensor.shape[0], 1, 1, 1)
        
        qx = det_Qx+2*pi/self.c*self.l/(1+strain)*tilt_lr-O_Qx
        qy = det_Qy+2*pi/self.c*self.l/(1+strain)*tilt_ud-O_Qy
        qz = det_Qz-2*pi/self.c*self.l/(1+strain)-O_Qz
        
        I = thickness*torch.sinc(thickness*qz/pi/2)**2 *\
            torch.exp(-qx**2/self.wx**2) * torch.exp(-qy**2/self.wy**2)
        intensity = I.sum(1)
        intensity_norm = intensity/intensity.sum(axis=(1, 2), keepdims=True)
        avg_max = torch.mean(intensity_norm.max(1, keepdims=True).values.max(2, keepdims=True).values)
        sim_norm = (intensity_norm / avg_max) * 7
        sim_norm = sim_norm[:, None, :, :]
        #sim_norm_log = torch.log10(sim_norm + 1e-8)
        
        self.epoch += 1
        
        return recon, sim_norm, strain_tensor
        
cnn = DonutNN2(1, 1)
import torch.optim as optim

criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW([{'params': cnn.encoder.parameters()}, 
                         {'params': cnn.decoder.parameters(), 'lr': 3e-5}], lr=LR)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

print('Finding devices')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if NGPUS > 1:
    #print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    cnn = nn.DataParallel(cnn) #Default all devices

print('Moving model to device')
cnn = cnn.to(DEVICE)

from training import *

metrics = {'losses': [], 'decoder_loss': [], 'sim_loss': [], 'val_losses': [np.inf], 'best_val_loss': np.inf, 'enc_lrs': [], 
           'dec_lrs':[]}

trainer = Trainer(cnn, NGPUS, DEVICE, MODEL_SAVE_PATH)
#benchmark = np.zeros((EPOCHS,))

print('Training')
for epoch in range(EPOCHS):    
   # t0 = time.time()    
    #Set model to train mode
    cnn.train()
    
    #Training loop
    if epoch == EPOCHS-1: 
        trainer.train(trainloader_all, criterion, optimizer, (1, 5), metrics)
    else:
        #print('Move data to device')
        trainer.train(trainloader_all, criterion, optimizer, (1, 5), metrics)
    
    #Switch model to eval mode
    cnn.eval()
    
    #Validation loop
    trainer.validate(validloader_all, criterion, optimizer, (1, 5), metrics, scheduler=None)
    
    print('Epoch: %d | Train Loss: %.5f | Val. Loss: %.5f'
        %(epoch, metrics['losses'][-1], metrics['val_losses'][-1]))
   # benchmark[epoch] = time.time() - t0
   # print('Epoch: %d | Train time: %.5f' %(epoch, benchmark[epoch]))
    
trainer.save_model_and_states_checkpoint(epoch, metrics, optimizer, scheduler=None)
#np.save('/home/aileenluo/benchmark/benchmark_amd_16.npy', benchmark)
with open(os.path.join(MODEL_SAVE_PATH, 'metrics_thickness_6.pickle'), 'wb') as file:
    pickle.dump(metrics, file)
    
print('Finished Training')
