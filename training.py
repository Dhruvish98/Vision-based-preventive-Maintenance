"""
Training Pipeline for Vision-Based Preventive Maintenance System
Handles data loading, preprocessing, augmentation, and model training
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import pickle

from config import DATASET_CONFIG, MODEL_CONFIG, TRAINING_CONFIG
from model import DefectDetectionCNN


class DataLoader:
    """Handle data loading and preprocessing"""
    
    def __init__(self, config=DATASET_CONFIG):
        self.config = config
        self.image_size = config['image_size']
    
    def load_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            
            # Resize to target size
            img = img.resize(self.image_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            return img_array
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def load_dataset(self):
        """Load the complete dataset"""
        print("Loading dataset...")
        
        data_dir = self.config['data_dir']
        normal_dir = os.path.join(data_dir, 'normal')
        defective_dir = os.path.join(data_dir, 'defective')
        
        images = []
        labels = []
        
        # Load normal images (label = 0)
        if os.path.exists(normal_dir):
            normal_files = [f for f in os.listdir(normal_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Loading {len(normal_files)} normal images...")
            
            for filename in tqdm(normal_files, desc="Normal images"):
                img_path = os.path.join(normal_dir, filename)
                img = self.load_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(0)  # Normal = 0
        
        # Load defective images (label = 1)
        if os.path.exists(defective_dir):
            defective_files = [f for f in os.listdir(defective_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"Loading {len(defective_files)} defective images...")
            
            for filename in tqdm(defective_files, desc="Defective images"):
                img_path = os.path.join(defective_dir, filename)
                img = self.load_image(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(1)  # Defective = 1
        
        if not images:
            raise ValueError("No images found! Please generate dataset first.")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        print(f"Dataset loaded successfully!")
        print(f"Total images: {len(X)}")
        print(f"Normal images: {np.sum(y == 0)}")
        print(f"Defective images: {np.sum(y == 1)}")
        print(f"Image shape: {X[0].shape}")
        
        return X, y
    
    def split_dataset(self, X, y):
        """Split dataset into train, validation, and test sets"""
        print("Splitting dataset...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_split'],
            random_state=self.config['seed'],
            stratify=y
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = self.config['val_split'] / (1 - self.config['test_split'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config['seed'],
            stratify=y_temp
        )
        
        print(f"Training set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        print(f"Test set: {len(X_test)} images")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class DataAugmentation:
    """Handle data augmentation for training"""
    
    def __init__(self, config=TRAINING_CONFIG):
        self.config = config
        self.augmentation_enabled = config['use_data_augmentation']
        self.aug_params = config['augmentation_params']
    
    def create_generators(self, X_train, y_train, X_val, y_val, batch_size=32):
        """Create data generators with augmentation"""
        
        if self.augmentation_enabled:
            # Training generator with augmentation
            # Remove brightness_range as it causes issues with normalized images
            train_datagen = ImageDataGenerator(
                rotation_range=self.aug_params['rotation_range'],
                width_shift_range=self.aug_params['width_shift_range'],
                height_shift_range=self.aug_params['height_shift_range'],
                shear_range=self.aug_params['shear_range'],
                zoom_range=self.aug_params['zoom_range'],
                horizontal_flip=self.aug_params['horizontal_flip'],
                vertical_flip=self.aug_params.get('vertical_flip', False),
                fill_mode=self.aug_params['fill_mode']
                # Removed brightness_range to prevent black images
            )
            
            print("Data augmentation enabled for training")
        else:
            # No augmentation
            train_datagen = ImageDataGenerator()
            print("No data augmentation applied")
        
        # Validation generator (no augmentation)
        val_datagen = ImageDataGenerator()
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=batch_size,
            shuffle=True,
            seed=42
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val,
            batch_size=batch_size,
            shuffle=False,
            seed=42
        )
        
        return train_generator, val_generator
    
    def visualize_augmentation(self, X_train, y_train, num_examples=4):
        """Visualize data augmentation effects"""
        if not self.augmentation_enabled:
            print("Data augmentation is disabled")
            return
        
        # Create augmentation generator
        datagen = ImageDataGenerator(
            rotation_range=self.aug_params['rotation_range'],
            width_shift_range=self.aug_params['width_shift_range'],
            height_shift_range=self.aug_params['height_shift_range'],
            shear_range=self.aug_params['shear_range'],
            zoom_range=self.aug_params['zoom_range'],
            horizontal_flip=self.aug_params['horizontal_flip'],
            vertical_flip=self.aug_params.get('vertical_flip', False),
            fill_mode=self.aug_params['fill_mode']
            # Removed brightness_range to prevent black images
        )
        
        fig, axes = plt.subplots(num_examples, 5, figsize=(15, 3*num_examples))
        fig.suptitle('Data Augmentation Examples', fontsize=16)
        
        for i in range(num_examples):
            # Select a random image
            idx = np.random.randint(0, len(X_train))
            img = X_train[idx:idx+1]  # Keep batch dimension
            label = "Defective" if y_train[idx] == 1 else "Normal"
            
            # Original image
            axes[i, 0].imshow(img[0])
            axes[i, 0].set_title(f'Original ({label})')
            axes[i, 0].axis('off')
            
            # Generate augmented versions
            aug_iter = datagen.flow(img, batch_size=1, seed=42+i)
            for j in range(1, 5):
                aug_img = next(aug_iter)[0]
                # Ensure values are in valid range [0,1]
                aug_img = np.clip(aug_img, 0, 1)
                axes[i, j].imshow(aug_img)
                axes[i, j].set_title(f'Augmented {j}')
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig('plots/data_augmentation_examples.png', dpi=300, bbox_inches='tight')
        plt.show()


class Trainer:
    """Main training class"""
    
    def __init__(self, model_config=MODEL_CONFIG, training_config=TRAINING_CONFIG):
        self.model_config = model_config
        self.training_config = training_config
        self.model = None
        self.history = None
        self.data_loader = DataLoader()
        self.augmentation = DataAugmentation()
    
    def prepare_data(self):
        """Load and prepare data for training"""
        print("Preparing data for training...")
        
        # Load dataset
        X, y = self.data_loader.load_dataset()
        
        # Split dataset
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.data_loader.split_dataset(X, y)
        
        # Calculate class weights for balanced training
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        print(f"Class weights: {class_weight_dict}")
        
        # Store data
        self.train_data = (X_train, y_train)
        self.val_data = (X_val, y_val)
        self.test_data = (X_test, y_test)
        self.class_weights = class_weight_dict
        
        return self.train_data, self.val_data, self.test_data
    
    def build_and_compile_model(self):
        """Build and compile the CNN model"""
        print("Building and compiling model...")
        
        # Create model instance
        cnn = DefectDetectionCNN(self.model_config)
        
        # Build model
        model = cnn.build_model()
        
        # Compile model
        cnn.compile_model()
        
        # Print summary
        cnn.summary()
        
        self.model = cnn
        return cnn
    
    def train_model(self, visualize_augmentation=True):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model must be built before training")
        
        print("Starting model training...")
        
        X_train, y_train = self.train_data
        X_val, y_val = self.val_data
        
        # Visualize data augmentation if requested
        if visualize_augmentation and self.training_config['use_data_augmentation']:
            self.augmentation.visualize_augmentation(X_train, y_train)
        
        # Create data generators
        train_gen, val_gen = self.augmentation.create_generators(
            X_train, y_train, X_val, y_val,
            batch_size=self.model_config['batch_size']
        )
        
        # Get callbacks
        callbacks = self.model.get_callbacks()
        
        # Calculate steps per epoch
        steps_per_epoch = len(X_train) // self.model_config['batch_size']
        validation_steps = len(X_val) // self.model_config['batch_size']
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        
        # Train model
        history = self.model.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.model_config['epochs'],
            validation_data=val_gen,
            validation_steps=validation_steps,
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        print("Training completed!")
        
        return history
    
    def save_training_results(self):
        """Save training history and model info"""
        if self.history is None:
            print("No training history to save")
            return
        
        # Save training history
        history_path = 'models/training_history.json'
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_dict = {}
            for key, values in self.history.history.items():
                history_dict[key] = [float(v) for v in values]
            json.dump(history_dict, f, indent=2)
        
        print(f"Training history saved to {history_path}")
        
        # Save training configuration
        config_path = 'models/training_config.json'
        config_data = {
            'model_config': self.model_config,
            'training_config': self.training_config,
            'class_weights': self.class_weights
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Training configuration saved to {config_path}")
    
    def plot_training_curves(self):
        """Plot training and validation curves"""
        if self.history is None:
            print("No training history available")
            return
        
        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Loss curves
        ax1.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Precision curves
        if 'precision' in history:
            ax3.plot(epochs, history['precision'], 'b-', label='Training Precision', linewidth=2)
            ax3.plot(epochs, history['val_precision'], 'r-', label='Validation Precision', linewidth=2)
            ax3.set_title('Model Precision')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Precision')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Recall curves
        if 'recall' in history:
            ax4.plot(epochs, history['recall'], 'b-', label='Training Recall', linewidth=2)
            ax4.plot(epochs, history['val_recall'], 'r-', label='Validation Recall', linewidth=2)
            ax4.set_title('Model Recall')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Recall')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Example training pipeline
    trainer = Trainer()
    
    # Prepare data
    train_data, val_data, test_data = trainer.prepare_data()
    
    # Build model
    model = trainer.build_and_compile_model()
    
    # Train model
    history = trainer.train_model()
    
    # Save results
    trainer.save_training_results()
    
    # Plot training curves
    trainer.plot_training_curves()
