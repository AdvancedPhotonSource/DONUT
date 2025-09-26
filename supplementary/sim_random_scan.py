import sys
sys.path.append('..')

from sim_SIO_gpu import *

gt_thickness = np.full((8, 8), 117, dtype='float32')
gt_strain = np.random.uniform(-0.0005, 0.0005, (8, 8))
gt_tilt_lr = np.random.uniform(-0.005, 0.005, (8, 8))
gt_tilt_ud = np.random.uniform(-0.01, 0.01, (8, 8))

param_arrays = [gt_thickness, gt_strain, gt_tilt_lr, gt_tilt_ud]

# Generate the simulated scan on a film of constant thickness
total_saved = generate_2d_parameter_dataset(
    param_arrays,
    output_dir="/data/aileen/DONUT_data",
    filename="sim_random_sample.npy",
    batch_size=16
)

print(f"Dataset generation complete! {total_saved:,} images saved.")

gt_dir="/data/aileen/DONUT_data"
np.save(os.path.join(gt_dir, "gt_thickness_random.npy"), gt_thickness)
np.save(os.path.join(gt_dir, "gt_strain_random.npy"), gt_strain)
np.save(os.path.join(gt_dir, "gt_tilt_lr_random.npy"), gt_tilt_lr)
np.save(os.path.join(gt_dir, "gt_tilt_ud_random.npy"), gt_tilt_ud)
