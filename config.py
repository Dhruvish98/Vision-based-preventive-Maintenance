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

# Defect Generation Parameters
DEFECT_CONFIG = {
    'scratch_probability': 0.4,   # Increased probability
    'crack_probability': 0.35,    # Increased probability
    'corrosion_probability': 0.3, # Increased probability
    'dent_probability': 0.25,     # Increased probability
    'stain_probability': 0.2,     # Increased probability
    'defect_intensity_range': (0.5, 0.9),  # More intense defects
    'noise_level': 0.05           # Reduced noise for clearer defects
}

# Model Configuration - Anti-overfitting settings
MODEL_CONFIG = {
    'input_shape': (128, 128, 3),
    'num_classes': 2,
    'learning_rate': 0.0005,  # Much lower learning rate for better generalization
    'batch_size': 32,         # Larger batch size for more stable gradients
    'epochs': 50,             # More epochs with early stopping
    'patience': 10,           # More patience to find best model
    'model_dir': 'models',
    'checkpoint_path': 'models/best_model.h5'
}

# Training Configuration - Strong augmentation to prevent overfitting
TRAINING_CONFIG = {
    'use_data_augmentation': True,
    'augmentation_params': {
        'rotation_range': 40,        # Increased rotation
        'width_shift_range': 0.2,    # Increased shift
        'height_shift_range': 0.2,   # Increased shift
        'shear_range': 0.2,          # Increased shear
        'zoom_range': 0.2,           # Increased zoom
        'horizontal_flip': True,
        'vertical_flip': True,       # Added vertical flip
        'fill_mode': 'nearest'
    },
    'class_weights': {0: 1.0, 1: 1.0}  # Balanced classes
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
