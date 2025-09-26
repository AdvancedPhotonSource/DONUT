# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com/AdvancedPhotonSource/DONUT/blob/main/LICENSE

from sim_SIO_gpu import *

constant_thickness = np.full((64, 64), 117, dtype='float32')

wave_t = 0.2
wave_s = 0.4
x = y = np.linspace(-1, 1, 64)
X, Y = np.meshgrid(x, y)
gt_thickness = 117 + 8*np.sin(2*np.pi*Y / wave_t)
gt_thickness_high = 550 + 10*np.sin(2*np.pi*Y / wave_t)
gt_strain = 0.0005 * np.sin(2 * np.pi * (X + Y) / wave_s)

def sigmoid_function(A, x, y, x0, y0, sigma, theta):         
    rot_x = (x-x0)*np.cos(theta) - (y-y0)*np.sin(theta)       
    g = A * (1 / (1 + np.exp(-rot_x/sigma)))
    return g 

# Define the grid coordinates 
x = np.linspace(-3, -1, 64) 
y = np.linspace(-1, 1, 64) 
X, Y = np.meshgrid(x, y)

x1 = np.linspace(-1.5, 0.5, 64)
y1 = np.linspace(-1, 1, 64)
X1, Y1 = np.meshgrid(x1, y1)

x2 = np.linspace(1, 3, 64)
y2 = np.linspace(1, -1, 64)
X2, Y2 = np.meshgrid(x2, y2)

# Create the sigmoid function with adjustable parameters 
lr = sigmoid_function(0.01, X, Y, 0, 0, 0.07, np.pi/3) # rotate by 60 degrees 
lr1 = -sigmoid_function(0.01, X1, Y1, 0, 0, 0.07, np.pi/3)
lr2 = -sigmoid_function(0.01, X2, Y2, 0, 0, 0.07, 2*np.pi/3)
gt_tilt_lr = lr + lr1 + 0.005 + lr2

ud = sigmoid_function(0.02, X, Y, 0, 0, 0.07, np.pi/3) # rotate by 60 degrees 
ud1 = -sigmoid_function(0.02, X1, Y1, 0, 0, 0.07, np.pi/3)
ud2 = -sigmoid_function(0.02, X2, Y2, 0, 0, 0.07, 2*np.pi/3)
gt_tilt_ud = ud + ud1 + 0.01 + ud2

param_arrays = [constant_thickness, gt_strain, gt_tilt_lr, gt_tilt_ud]
param_arrays_thickness = [gt_thickness, gt_strain, gt_tilt_lr, gt_tilt_ud]
param_arrays_thickness_old = [gt_thickness_high, gt_strain, gt_tilt_lr, gt_tilt_ud]

# Generate the simulated scan on a film of constant thickness
total_saved = generate_2d_parameter_dataset(
    param_arrays,
    output_dir="/data/aileen/DONUT_data",
    filename="sim_sample.npy",
    batch_size=16
)

print(f"Dataset generation complete! {total_saved:,} images saved.")

# Generate the simulated scan on a film of varying thickness
total_saved_t = generate_2d_parameter_dataset(
    param_arrays_thickness,
    output_dir="/data/aileen/DONUT_data",
    filename="sim_sample_thickness.npy",
    batch_size=16
)

# Generate the simulated scan on a film of varying thickness
total_saved_t = generate_2d_parameter_dataset(
    param_arrays_thickness_old,
    output_dir="/data/aileen/DONUT_data",
    filename="sim_sample_thickness_high.npy",
    batch_size=16
)

gt_dir="/data/aileen/DONUT_data"
np.save(os.path.join(gt_dir, "constant_thickness.npy"), constant_thickness)
np.save(os.path.join(gt_dir, "gt_thickness.npy"), gt_thickness)
np.save(os.path.join(gt_dir, "gt_thickness_high.npy"), gt_thickness_high)
np.save(os.path.join(gt_dir, "gt_strain.npy"), gt_strain)
np.save(os.path.join(gt_dir, "gt_tilt_lr.npy"), gt_tilt_lr)
np.save(os.path.join(gt_dir, "gt_tilt_ud.npy"), gt_tilt_ud)