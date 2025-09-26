# Copyright © 2025 UChicago Argonne, LLC All right reserved
# Full license accessible at https://github.com/AdvancedPhotonSource/DONUT/blob/main/LICENSE

# Multi-GPU parallelized simulations

from math import *
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

class ImageGenerator(nn.Module):
    """Generates simulated diffraction patterns of 64x64 pixels from a list of parameters"""
    
    def __init__(self, param_dim):
        super().__init__()
        
        # Constants
        self.upsampling = 2
        self.energy = 11.3 # X-ray energy in keV
        self.c = 4.013 # Lattice constant in Angstroms
        self.l = 2 # l from hkl Bragg peak
        self.X0 = 256 # Pixel coordinate of the center of Bragg peak
        self.Xcen = 256 # Pixel coordinate of the center of ZP reflection on detector
        self.zp_diameter = 149e-6 # Zone plate diameter in meters
        self.cs_diameter = 77e-6 # Central beam stop diameter in meters
        self.outer_zone_width = 16e-9 # Outermost zone width in meters
        self.distance = 0.85 # Detector distance from sample in meters
        self.pixelsize = 55e-6*2 # Detector pixel size in meters (*2 for binning)
        
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
        self.register_buffer('gam', gam, persistent=False)
        self.register_buffer('det_Qx', det_Qx, persistent=False)
        self.register_buffer('det_Qz', det_Qz, persistent=False)
        self.register_buffer('det_Qy', det_Qy, persistent=False)
        O_Qx, O_Qz, O_Qy = self.recip_space()
        self.register_buffer('O_Qx', O_Qx, persistent=False)
        self.register_buffer('O_Qz', O_Qz, persistent=False)
        self.register_buffer('O_Qy', O_Qy, persistent=False)
    
    def detector_space(self):
        det_x = np.arange(64).astype(np.float32)
        det_y = np.arange(64).astype(np.float32)
        det_x = det_x - det_x.mean() + self.X0 - (self.X0-self.Xcen)
        det_y -= det_y.mean()
        det_xx, det_yy = np.meshgrid(det_x, det_y)
        gam = np.arctan((det_xx-self.X0)*self.pixelsize/self.distance)+self.gam0
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
    
    def forward(self, params: torch.Tensor):
        # Generate an image for each parameter set
        
        # Parameters and dimensions
        batch_size = params.shape[0]
        thickness = params[:, 0].view(batch_size, 1, 1, 1)
        strain = params[:, 1].view(batch_size, 1, 1, 1)
        tilt_lr = torch.deg2rad(params[:, 2]).view(batch_size, 1, 1, 1)
        tilt_ud = torch.deg2rad(params[:, 3]).view(batch_size, 1, 1, 1)
        
        # Dimensionality adjustment for batching and PyTorch operations
        det_Qz = torch.tile(self.det_Qz, (batch_size, 1, 1))
        det_Qz = det_Qz[:, None, :, :]
        det_Qx = torch.tile(self.det_Qx, (batch_size, 1, 1))
        det_Qx = det_Qx[:, None, :, :]
        det_Qy = torch.tile(self.det_Qy, (batch_size, 1, 1))
        det_Qy = det_Qy[:, None, :, :]
        
        O_Qz = torch.tile(self.O_Qz, (batch_size, 1, 1, 1))
        O_Qx = torch.tile(self.O_Qx, (batch_size, 1, 1, 1))
        O_Qy = torch.tile(self.O_Qy, (batch_size, 1, 1, 1))
        
        # Position and distribution of Bragg intensity
        qx = det_Qx+2*pi/self.c*self.l/(1+strain)*tilt_lr-O_Qx
        qy = det_Qy+2*pi/self.c*self.l/(1+strain)*tilt_ud-O_Qy
        qz = det_Qz-2*pi/self.c*self.l/(1+strain)-O_Qz
        
        # Shape of Bragg peak
        thin_film = thickness*torch.sinc(thickness*qz/pi/2)**2 # Fourier transform along truncated crystal --> sinc
        gauss_x = torch.exp(-qx**2/self.wx**2) # Infinite crystal in in-plane direction --> sharp Gaussian
        gauss_y = torch.exp(-qy**2/self.wy**2) # Infinite crystal in other in-plane direction --> sharp Gaussian
        
        # Intensity
        intensity = thin_film * gauss_x * gauss_y
        return intensity.sum(1)

class ParameterGridDataset(Dataset):
    """Dataset that yields parameter combinations with their indices"""
    
    def __init__(self, param_shapes, param_ranges=None, param_discrete_values=None):
        """
        param_shapes: List of sizes for each parameter dimension
                    e.g. [5, 11, 11, 11, 21, 21, 21]
        param_ranges: List of tuples (min_val, max_val) for each dimension
                    e.g. [None, (22, 32), (450, 950), ...]
                    If None, defaults to (0, 1) for all dimensions
        param_discrete_values: List of lists containing exact values for each dimension
                    e.g. [[45, 55, 60, 65, 75], None, ...]
                    Overrides param_ranges for specified dimensions
                    Length of each inner list must match the corresponding param_shape
                    Use None for dimensions where you want to use the range-based values
        """
        self.param_shapes = param_shapes
        
        # Create parameter ranges for each dimension
        self.param_ranges = []
        
        for i, shape in enumerate(param_shapes):
            # Case 1: Discrete values provided for this dimension
            if param_discrete_values is not None and i < len(param_discrete_values) and param_discrete_values[i] is not None:
                discrete_values = param_discrete_values[i]
                
                # Validate discrete values match the expected shape
                if len(discrete_values) != shape:
                    raise ValueError(
                        f"Dimension {i}: Discrete values list has length {len(discrete_values)}, "
                        f"but param_shapes specifies {shape} values"
                    )
                
                # Convert to tensor if it's not already
                if not isinstance(discrete_values, torch.Tensor):
                    self.param_ranges.append(torch.tensor(discrete_values, dtype=torch.float32))
                else:
                    self.param_ranges.append(discrete_values)
            
            # Case 2: Custom range provided for this dimension
            elif param_ranges is not None and i < len(param_ranges) and param_ranges[i] is not None:
                min_val, max_val = param_ranges[i]
                self.param_ranges.append(torch.linspace(min_val, max_val, shape))
            
            # Case 3: Default to 0-1 range
            else:
                self.param_ranges.append(torch.linspace(0, 1, shape))
        
        # Calculate total number of combinations
        self.total_combinations = np.prod(param_shapes)
        
        # Create index mapping for reconstruction of parameter space
        self.strides = self._compute_strides(param_shapes)

    def _compute_strides(self, shapes):
        """Compute strides for converting flat indices to multi-dimensional indices"""
        strides = torch.ones(len(shapes), dtype=torch.long)
        for i in range(len(shapes)-2, -1, -1):
            strides[i] = strides[i+1] * shapes[i+1]
        return strides
    
    def flat_to_multi_idx(self, flat_idx):
        """Convert flat index to multi-dimensional indices"""
        indices = []
        for dim, stride in enumerate(self.strides):
            dim_idx = (flat_idx // stride) % self.param_shapes[dim]
            indices.append(dim_idx.item() if isinstance(dim_idx, torch.Tensor) else dim_idx)
        return indices
    
    def __len__(self):
        return self.total_combinations
    
    def __getitem__(self, idx):
        # Convert flat index to multi-dimensional indices
        indices = self.flat_to_multi_idx(idx)
        
        # Get parameter values for each dimension
        params = torch.tensor([self.param_ranges[dim][i] for dim, i in enumerate(indices)])
        
        # Return parameters and original index for reconstruction
        return params, idx
    
class Explicit2DParameterDataset(Dataset):
    """Dataset that works with parameter arrays"""
    
    def __init__(self, param_arrays):
        """
        Initialize dataset with explicit 2D parameter arrays
        Args:
            param_arrays: List of 2D arrays (numpy arrays or torch tensors)
                          Each array contains values for one parameter
                          All arrays must have the same shape
        """
        self.param_arrays = []
        
        # Validate and convert arrays to torch tensors
        if not param_arrays or len(param_arrays) == 0:
            raise ValueError("param_arrays must contain at least one array")
            
        # Get the shape from the first array
        first_array = param_arrays[0]
        if not hasattr(first_array, 'shape') or len(first_array.shape) != 2:
            raise ValueError("Each parameter array must be a 2D array")
            
        self.shape = first_array.shape
        rows, cols = self.shape
        
        # Check all shapes and convert all arrays to torch tensors
        for i, arr in enumerate(param_arrays):
            if not hasattr(arr, 'shape') or arr.shape != self.shape:
                raise ValueError(f"Parameter array {i} has incorrect shape: {arr.shape} != {self.shape}")
                
            # Convert to torch tensor if it's not already
            if not isinstance(arr, torch.Tensor):
                self.param_arrays.append(torch.tensor(arr, dtype=torch.float32))
            else:
                self.param_arrays.append(arr)

        # Calculate total number of parameter combinations
        self.total_combinations = rows * cols
        
        # Store dimensions for index calculations
        self.rows = rows
        self.cols = cols
        self.num_params = len(self.param_arrays)
        
    def flat_to_multi_idx(self, flat_idx):
        """
        Convert flat index to (row, column) position
        
        Args:
            flat_idx: Flat index (0 to total_combinations-1)
            
        Returns:
            Tuple (row, col) corresponding to the position in the 2D arrays
        """
        row = flat_idx // self.cols
        col = flat_idx % self.cols
        
        return (row, col)
    
    def __len__(self):
        """Return the total number of parameter combinations"""
        return self.total_combinations
    
    def __getitem__(self, idx):
        """
        Get parameters and index for a specific flat index
        
        Args:
            idx: Flat index (0 to total_combinations-1)
            
        Returns:
            Tuple(params, idx) where:
                params: Tensor of shape [num_params] containing parameter values
                idx: The original flat index for reconstruction
        """
        row, col = self.flat_to_multi_idx(idx)
        
        # Gather parameters from each array at the (row, col) position
        params = torch.tensor([arr[row, col].item() for arr in self.param_arrays])
        
        return params, idx

# Function to generate dataset spanning a parameter space 
def generate_dataset(param_shapes, output_dir, param_ranges=None, param_discrete_values=None, one_file=False, batch_size=1024, num_workers=4):
    """
    Generate images and save each as a separate .npz file with its parameters
    
    param_shapes: List of sizes for each parameter dimension
    output_dir: Directory to save the .npz files
    param_ranges: Optional list of (min, max) tuples for each parameter dimension
    param_discrete_values: Optional list of lists containing exact values for dimensions
                Use None for dimensions where you want to use param_ranges instead
    one_file: Whether or not to save the images generated in one file (convenient for small data) rather than separate files 
    batch_size: Batch size for processing
    num_workers: Number of dataloader workers
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate total number of parameter combinations
    total_combinations = int(np.prod(param_shapes))
    print(f"Preparing to generate {total_combinations:,} images...")
    
    # Determine number of digits needed for filenames
    num_digits = len(str(total_combinations - 1))
    print(f"Using {num_digits}-digit zero-padded filenames")
    
    # Create dataset and dataloader
    dataset = ParameterGridDataset(param_shapes, param_ranges, param_discrete_values)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # Important: Keep order for mapping to parameters
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create the simulation model and distribute across GPUs
    model = ImageGenerator(param_dim=len(param_shapes))
    num_gpus = torch.cuda.device_count() # Use all GPUs
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for parallel processing")
        model = nn.DataParallel(model)
    else:
        print(f"Using 1 GPU for processing")
    model = model.cuda()
    
    # Setup progress tracking
    total_batches = len(dataloader)
    processed_count = 0
    start_time = time.time()
    
    # Create main progress bar for overall tracking
    main_pbar = tqdm(
        total=total_combinations,
        desc="Total Progress",
        unit="img",
        position=0,
        leave=True,
        bar_format="{desc}: {percentage:3.1f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    # For smaller simulations, save just one file with the entire simulated dataset
    if one_file == True:
        output_shape = param_shapes + [64, 64]
        final_output = np.zeros(output_shape)
    
    # Process all batches
    with torch.no_grad():
        for batch_idx, (params, indices) in enumerate(dataloader):
            # Update batch information
            batch_size_actual = len(indices) # Last batch might be smaller
            
            # Create a nested progress bar for this batch
            batch_desc = f"Batch {batch_idx+1}/{total_batches}"
            batch_pbar = tqdm(
                total=batch_size_actual,
                desc=batch_desc,
                unit="img",
                position=1,
                leave=False,
                bar_format="{desc}: {percentage:3.1f}%|{bar}| {n_fmt}/{total_fmt}"
            )
            
            # Transfer to GPU
            params = params.cuda()
            
            # Generate images
            batch_output = model(params)
            
            # Move results back to CPU
            batch_output = batch_output.cpu().numpy()
            params_cpu = params.cpu().numpy()
            
            # Handle the saving
            for i, idx in enumerate(indices):
                # Get the flat index
                flat_idx = idx.item()
                
                # Convert flat index to multi-indices for parameter reference
                multi_idx = dataset.flat_to_multi_idx(flat_idx)
                
                if one_file == True:
                    final_output[tuple(multi_idx)] = batch_output[i]
                   
                else:
                    # Convert to a zero-padded string
                    idx_str = str(flat_idx).zfill(num_digits)

                    # Create the filename
                    filename = os.path.join(output_dir, f"image_{idx_str}.npz")

                    # Save the image and its parameters as an npz file
                    np.savez(
                        filename,
                        image=batch_output[i],
                        parameters=params_cpu[i], # Raw parameter values
                        indices=np.array(multi_idx) # Parameter indices in the grid
                    )
                
                # Update counters and progress bars
                processed_count += 1
                batch_pbar.update(1)
                main_pbar.update(1)
                
            # Close the batch progress bar
            batch_pbar.close()
            
            # Calculate and display stats every few batches
            if (batch_idx + 1) % max(1, total_batches // 100) == 0:
                elapsed = time.time() - start_time
                images_per_sec = processed_count / elapsed
                eta_seconds = (total_combinations - processed_count) / max(0.1, images_per_sec)
                
                # Convert ETA to a readable format
                eta_hours = int(eta_seconds // 3600)
                eta_minutes = int((eta_seconds % 3600) // 60)
                
                # Update main progress bar description
                main_pbar.set_description(
                    f"Progress: {processed_count:,}/{total_combinations:,} ({images_per_sec:.1f} img/s, "
                    f"ETA: {eta_hours}h {eta_minutes}m)"
                )
                
    # Close the main progress bar
    main_pbar.close()
    
    # Save the one file
    if one_file == True:
        np.save(os.path.join(output_dir, 'sim_SIO_gpu.npy'), final_output)
    
    # Calculate and display final stats
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n✅ Successfully saved {processed_count:,} images to {output_dir}")
    print(f"⏱️ Total time: {hours}h {minutes}m {seconds}s")
    print(f"⚡ Average speed: {processed_count / total_time:.1f} images/second")
    
    if num_gpus > 1:
        print(f"💻 Utilized {num_gpus} GPUs with {batch_size} batch size")
        
    return processed_count

# Function to generate dataset with explicit 2D parameter arrays
def generate_2d_parameter_dataset(param_arrays, output_dir, filename, batch_size=64, num_workers=4):
    """
    Generate images from explicit 2D parameter arrays
    
    Args: 
        param_arrays: List of 2D arrays for parameters
        output_dir: Directory to save the file (str)
        filename: Name of output file (str)
        batch_size: Batch size for processing
        num_workers: Number of dataloader workers
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    total_to_process = param_arrays[0].shape[0] * param_arrays[0].shape[1]
    print(f"Preparing to generate {total_to_process:,} images...")
    
    # Create dataset
    dataset = Explicit2DParameterDataset(param_arrays)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create the simulation model and distribute across GPUs
    model = ImageGenerator(param_dim=len(param_arrays))
    num_gpus = torch.cuda.device_count() # Use all GPUs
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs for parallel processing")
        model = nn.DataParallel(model)
    else:
        print(f"Using 1 GPU for processing")
    model = model.cuda()
    
    # Setup progress tracking
    total_batches = len(dataloader)
    processed_count = 0
    start_time = time.time()
    
    # Create main progress bar
    main_pbar = tqdm(
        total=total_to_process, 
        desc="Progress", 
        unit="img",
        position=0,
        leave=True,
        bar_format="{desc}: {percentage:3.1f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    # Aggregate images to output
    final_output = np.zeros((dataset.param_arrays[0].shape[0], dataset.param_arrays[0].shape[1], 64, 64))
    
    # Process all batches
    with torch.no_grad():
        for batch_idx, (params, indices) in enumerate(dataloader):
            # Update batch information
            batch_size_actual = len(indices) # Last batch may be smaller
            
            # Generate images
            params = params.cuda()
            batch_output = model(params)
            
            # Move results back to CPU
            batch_output = batch_output.cpu().numpy()
            
            # For each item in the batch
            for i, idx in enumerate(indices):
                # Calculate flat index 
                flat_idx = idx.item()
                
                # Convert flat index to (row, col) position
                row, col = dataset.flat_to_multi_idx(flat_idx)
                
                final_output[row, col] = batch_output[i]
                
                # Update counter and progress bar
                processed_count += 1
                main_pbar.update(1)
                
                # Calculate and display stats every few batches
                if (batch_idx + 1) % max(1, total_batches // 100) == 0:
                    elapsed = time.time() - start_time
                    images_per_sec = processed_count / elapsed
                    eta_seconds = (total_to_process - processed_count) / max(0.1, images_per_sec)
                    
                    # Convert ETA to a readable format
                    eta_hours = int(eta_seconds // 3600)
                    eta_minutes = int((eta_seconds % 3600) //60)
                    
                    # Update main progress bar description
                    main_pbar.set_description(
                        f"Progress: {processed_count:,}/{total_to_process:,} ({images_per_sec:.1f} img/s, ETA: {eta_hours}h {eta_minutes}m)"
                    )

    # Close the main progress bar
    main_pbar.close()
    
    np.save(os.path.join(output_dir, filename), final_output)
    
    # Calculate and display final stats
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n✅ Successfully saved {processed_count:,} images to {output_dir}")
    print(f"⏱️ Total time: {hours}h {minutes}m {seconds}s")
    print(f"⚡ Average speed: {processed_count / total_time:.1f} images/second")
    
    if num_gpus > 1:
        print(f"💻 Utilized {num_gpus} GPUs with {batch_size} batch size")
        
    return processed_count