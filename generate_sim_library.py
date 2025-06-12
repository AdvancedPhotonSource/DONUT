from sim_SIO_gpu import *

param_shapes = [11, 21, 21, 21]

param_ranges = [
    None, # Thickness set as discrete values
    (-0.005, 0.005), # Strain
    (-0.05, 0.05), # In-plane rotation
    (-0.1, 0.1) # Out-of-plane rotation
]

param_discrete_values = [
    [50, 75, 90, 100, 110, 117, 124, 134, 144, 159, 184],
    None,
    None,
    None
]

# Generate the dataset with individual files
total_saved = generate_dataset(
    param_shapes,
    output_dir="/local/aileen",
    param_ranges=param_ranges, # Uniform ranges
    one_file=True,
    param_discrete_values=param_discrete_values, # Discrete values (takes precedence)
    batch_size=64
)

print(f"Dataset generation complete! {total_saved:,} images saved.")