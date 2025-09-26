# DONUT
Diffraction with Optics for Nanobeam by Unsupervised Training

## Overview
DONUT is a physics-aware machine learning framework for analyzing nanobeam X-ray diffraction data. This repository contains neural network models and simulation tools for extracting structural parameters (strain, rotation, thickness) from scanning X-ray diffraction microscopy (SXDM) measurements.

## Key Features
- **Forward simulation models** for generating synthetic nanobeam diffraction patterns
- **Neural network architectures** (encoder-decoder) for parameter extraction
- **Monte Carlo dropout** implementation for uncertainty quantification
- **Correlation fitting** analysis for comparison with traditional methods
- **Visualization tools** for diffraction patterns and parameter maps

## Repository Structure
- **Simulation scripts**: `sim_SIO_gpu.py`, `sim_scans.py`, `generate_sim_library.py`
- **Training scripts**: `train_with_thickness.py`, `correlation_fit_gpu.py`
- **Analysis notebooks**: Jupyter notebooks for loss landscapes, visualizations, and Monte Carlo dropout analysis
- **Supplementary analysis**: Additional notebooks for hyperparameter studies and Q-range analysis
