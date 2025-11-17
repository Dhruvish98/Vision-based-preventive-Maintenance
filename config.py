"""
Configuration file for Vision-Based Preventive Maintenance System
"""

import os

# Dataset Configuration
DATASET_CONFIG = {
    'image_size': (128, 128),
    'num_samples_per_class': 2000,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'data_dir': 'data',
    'seed': 42
}

# Defect Generation Parameters - EXTREMELY prominent defects for clear learning
DEFECT_CONFIG = {
    'scratch_probability': 0.95,   # Almost always add scratches
    'crack_probability': 0.9,      # Almost always add cracks  
    'corrosion_probability': 0.85, # Almost always add corrosion
    'dent_probability': 0.8,       # Almost always add dents
    'stain_probability': 0.75,     # Almost always add stains
    'defect_intensity_range': (0.9, 1.0),  # MAXIMUM intensity defects only
    'noise_level': 0.001           # Almost no noise - pure defects
}

# Model Configuration - Optimized for clear defect learning
MODEL_CONFIG = {
    'input_shape': (128, 128, 3),
    'num_classes': 2,
    'learning_rate': 0.0005,    # Higher learning rate for faster initial learning
    'batch_size': 32,          # Smaller batches for more gradient updates
    'epochs': 50,             # More epochs to ensure proper learning
    'patience': 10,            # More patience to avoid premature stopping
    'model_dir': 'models',
    'checkpoint_path': 'models/best_model.h5'
}

# Training Configuration - Minimal augmentation for clear defect learning
TRAINING_CONFIG = {
    'use_data_augmentation': True,
    'augmentation_params': {
        'rotation_range': 5,         # Minimal rotation to preserve defect patterns
        'width_shift_range': 0.05,   # Minimal shift
        'height_shift_range': 0.05,  # Minimal shift
        'shear_range': 0.02,         # Minimal shear
        'zoom_range': 0.05,          # Minimal zoom
        'horizontal_flip': True,     # Keep horizontal flip
        'vertical_flip': False,      # No vertical flip
        'fill_mode': 'nearest'
    },
    'class_weights': {0: 1.0, 1: 2.0}  # Give more weight to defective class
}

# Visualization Configuration
VIS_CONFIG = {
    'sample_images_per_class': 8,
    'figure_size': (12, 8),
    'dpi': 100,
    'save_plots': True,
    'plots_dir': 'plots'
}

# Create directories if they don't exist
for directory in [DATASET_CONFIG['data_dir'], MODEL_CONFIG['model_dir'], VIS_CONFIG['plots_dir']]:
    os.makedirs(directory, exist_ok=True)
