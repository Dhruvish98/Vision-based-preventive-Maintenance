"""
CNN Model Architecture for Vision-Based Preventive Maintenance
Optimized for defect detection in machine parts
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
from config import MODEL_CONFIG


class DefectDetectionCNN:
    def __init__(self, config=MODEL_CONFIG):
        self.config = config
        self.model = None
        self.history = None
    
    def build_model(self):
        """Build simplified CNN architecture optimized for clear defect detection"""

        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.config['input_shape']),

            # First Convolutional Block - More filters to capture defect features
            layers.Conv2D(32, (5, 5), activation='relu', padding='same', name='conv1_1'),
            layers.BatchNormalization(name='bn1'),
            layers.MaxPooling2D((2, 2), name='pool1'),
            layers.Dropout(0.2, name='dropout1'),  # Reduced dropout for better learning

            # Second Convolutional Block - Focus on defect patterns
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1'),
            layers.BatchNormalization(name='bn2'),
            layers.MaxPooling2D((2, 2), name='pool2'),
            layers.Dropout(0.2, name='dropout2'),  # Reduced dropout

            # Third Convolutional Block - High-level feature detection
            layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1'),
            layers.BatchNormalization(name='bn3'),
            layers.MaxPooling2D((2, 2), name='pool3'),
            layers.Dropout(0.3, name='dropout3'),  # Moderate dropout

            # Flatten and dense layers for classification
            layers.Flatten(name='flatten'),
            
            # Larger dense layer to learn complex defect patterns
            layers.Dense(128, activation='relu', name='dense1'),
            layers.BatchNormalization(name='bn_dense1'),
            layers.Dropout(0.4, name='dropout_dense1'),
            
            # Second dense layer
            layers.Dense(64, activation='relu', name='dense2'),
            layers.Dropout(0.3, name='dropout_dense2'),

            # Output layer (binary classification)
            layers.Dense(1, activation='sigmoid', name='output')
        ])

        self.model = model
        return model
    
    def compile_model(self, learning_rate=None):
        """Compile the model with optimizer and loss function"""
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        lr = learning_rate or self.config['learning_rate']
        
        # Use Adam optimizer with custom learning rate
        optimizer = optimizers.Adam(learning_rate=lr)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        return self.model
    
    def get_callbacks(self):
        """Get training callbacks for better training control"""
        callbacks_list = [
            # Early stopping to prevent overfitting
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint to save best model
            callbacks.ModelCheckpoint(
                filepath=self.config['checkpoint_path'],
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logger for training history
            callbacks.CSVLogger(
                filename='models/training_log.csv',
                append=True
            )
        ]
        
        return callbacks_list
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model must be built before getting summary")
        
        print("Model Architecture Summary:")
        print("=" * 50)
        self.model.summary()
        
        # Calculate and display model parameters
        total_params = self.model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = total_params - trainable_params
        
        print(f"\nModel Parameters:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {non_trainable_params:,}")
    
    def save_model(self, filepath=None):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        filepath = filepath or self.config['checkpoint_path']
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """Load a pre-trained model"""
        filepath = filepath or self.config['checkpoint_path']
        
        try:
            self.model = keras.models.load_model(filepath)
            print(f"Model loaded from {filepath}")
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def predict(self, images, threshold=0.5):
        """Make predictions on new images"""
        if self.model is None:
            raise ValueError("Model must be loaded or trained before prediction")
        
        # Ensure images are in correct format
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        
        # Get predictions
        predictions = self.model.predict(images)
        
        # Convert probabilities to binary predictions
        binary_predictions = (predictions > threshold).astype(int)
        
        return predictions, binary_predictions
    
    def evaluate_model(self, test_data, test_labels):
        """Evaluate model performance on test data"""
        if self.model is None:
            raise ValueError("Model must be loaded or trained before evaluation")
        
        # Get detailed evaluation metrics
        results = self.model.evaluate(test_data, test_labels, verbose=1)
        
        # Create results dictionary
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]
        
        return metrics


class ModelVisualizer:
    """Utility class for model visualization"""
    
    @staticmethod
    def plot_model_architecture(model, filename='plots/model_architecture.png'):
        """Plot and save model architecture"""
        try:
            tf.keras.utils.plot_model(
                model,
                to_file=filename,
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=False,
                dpi=96
            )
            print(f"Model architecture saved to {filename}")
        except Exception as e:
            print(f"Could not save model architecture plot: {e}")
    
    @staticmethod
    def analyze_model_complexity(model):
        """Analyze and display model complexity metrics"""
        print("\nModel Complexity Analysis:")
        print("=" * 40)
        
        # Count different layer types
        layer_types = {}
        for layer in model.layers:
            layer_type = type(layer).__name__
            layer_types[layer_type] = layer_types.get(layer_type, 0) + 1
        
        print("Layer Distribution:")
        for layer_type, count in sorted(layer_types.items()):
            print(f"  {layer_type}: {count}")
        
        # Memory estimation (rough)
        total_params = model.count_params()
        memory_mb = (total_params * 4) / (1024 * 1024)  # Assuming float32
        
        print(f"\nEstimated Model Size:")
        print(f"  Parameters: {total_params:,}")
        print(f"  Memory (approx): {memory_mb:.2f} MB")


if __name__ == "__main__":
    # Example usage
    cnn = DefectDetectionCNN()
    model = cnn.build_model()
    cnn.compile_model()
    cnn.summary()
    
    # Visualize model
    visualizer = ModelVisualizer()
    visualizer.plot_model_architecture(model)
    visualizer.analyze_model_complexity(model)
