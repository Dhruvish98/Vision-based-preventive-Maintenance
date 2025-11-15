"""
Vision-Based Preventive Maintenance System
Main interface for the complete pipeline

This script provides a comprehensive solution for detecting defective machine parts
using synthetic camera images and CNN-based classification.
"""

import os
import sys
import argparse
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
from config import DATASET_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, VIS_CONFIG
from data_generator import SyntheticDataGenerator
from training import Trainer, DataLoader
from model import DefectDetectionCNN, ModelVisualizer
from evaluation import ModelEvaluator, PerformanceAnalyzer


class VisionMaintenanceSystem:
    """Main system class that orchestrates the entire pipeline"""
    
    def __init__(self, config_override=None):
        """Initialize the system with optional configuration override"""
        self.dataset_config = config_override.get('dataset', DATASET_CONFIG) if config_override else DATASET_CONFIG
        self.model_config = config_override.get('model', MODEL_CONFIG) if config_override else MODEL_CONFIG
        self.training_config = config_override.get('training', TRAINING_CONFIG) if config_override else TRAINING_CONFIG
        
        # Initialize components
        self.data_generator = SyntheticDataGenerator(self.dataset_config)
        self.trainer = Trainer(self.model_config, self.training_config)
        self.evaluator = ModelEvaluator()
        
        # System state
        self.dataset_generated = False
        self.model_trained = False
        self.model_evaluated = False
        
        print("Vision-Based Preventive Maintenance System Initialized")
        print("=" * 60)
    
    def generate_dataset(self, visualize_samples=True):
        """Generate synthetic dataset"""
        print("\n🔧 STEP 1: Generating Synthetic Dataset")
        print("-" * 40)
        
        try:
            # Check if dataset already exists
            if (os.path.exists(os.path.join(self.dataset_config['data_dir'], 'normal')) and 
                os.path.exists(os.path.join(self.dataset_config['data_dir'], 'defective'))):
                
                response = input("Dataset already exists. Regenerate? (y/n): ").lower()
                if response != 'y':
                    print("Using existing dataset...")
                    self.dataset_generated = True
                    return True
            
            # Generate dataset
            self.data_generator.generate_dataset()
            
            # Visualize samples if requested
            if visualize_samples:
                print("\nGenerating sample visualizations...")
                self.data_generator.visualize_samples()
            
            self.dataset_generated = True
            print("✅ Dataset generation completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error generating dataset: {e}")
            return False
    
    def train_model(self, retrain=False):
        """Train the CNN model"""
        print("\n🤖 STEP 2: Training CNN Model")
        print("-" * 40)
        
        if not self.dataset_generated:
            print("❌ Dataset must be generated first!")
            return False
        
        try:
            # Check if model already exists
            if os.path.exists(self.model_config['checkpoint_path']) and not retrain:
                response = input("Trained model already exists. Retrain? (y/n): ").lower()
                if response != 'y':
                    print("Using existing model...")
                    self.model_trained = True
                    return True
            
            # Prepare data
            print("Preparing training data...")
            train_data, val_data, test_data = self.trainer.prepare_data()
            
            # Build and compile model
            print("Building CNN model...")
            model = self.trainer.build_and_compile_model()
            
            # Visualize model architecture
            visualizer = ModelVisualizer()
            visualizer.plot_model_architecture(model.model)
            visualizer.analyze_model_complexity(model.model)
            
            # Train model
            print("Starting training process...")
            history = self.trainer.train_model(visualize_augmentation=True)
            
            # Save training results
            self.trainer.save_training_results()
            
            # Plot training curves
            self.trainer.plot_training_curves()
            
            self.model_trained = True
            print("✅ Model training completed successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return False
    
    def evaluate_model(self, detailed_analysis=True):
        """Evaluate the trained model with comprehensive analysis"""
        print("\n📊 STEP 3: Evaluating Model Performance")
        print("-" * 40)
        
        if not self.model_trained:
            print("❌ Model must be trained first!")
            return False
        
        try:
            # Load test data
            data_loader = DataLoader(self.dataset_config)
            X, y = data_loader.load_dataset()
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = data_loader.split_dataset(X, y)
            
            # Load trained model
            self.evaluator.model_path = self.model_config['checkpoint_path']
            if not self.evaluator.load_model():
                print("❌ Failed to load trained model!")
                return False
            
            # Evaluate on test data
            print("Evaluating model performance...")
            metrics = self.evaluator.evaluate_on_test_data(X_test, y_test)
            
            # Analyze overfitting from training history
            overfitting_gap, max_val_acc = self._analyze_overfitting()
            
            if detailed_analysis:
                print("\nGenerating comprehensive analysis...")
                
                # Generate all evaluation plots
                self.evaluator.plot_confusion_matrix(save_path='plots/confusion_matrix.png')
                self.evaluator.plot_confusion_matrix(normalize=True, save_path='plots/confusion_matrix_normalized.png')
                self.evaluator.plot_roc_curve()
                self.evaluator.plot_precision_recall_curve()
                
                # Threshold analysis
                optimal_threshold, threshold_metrics = self.evaluator.plot_threshold_analysis()
                print(f"\n🎯 Optimal threshold found: {optimal_threshold:.3f}")
                
                # Analyze misclassifications
                self.evaluator.analyze_misclassifications(X_test)
                
                # Generate classification report
                self.evaluator.generate_classification_report()
                
                # Save comprehensive report
                self.evaluator.save_evaluation_report(metrics)
                
                # Print detailed performance analysis
                self._print_detailed_analysis(metrics, overfitting_gap, max_val_acc, optimal_threshold)
            
            self.model_evaluated = True
            print("✅ Model evaluation completed successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error evaluating model: {e}")
            return False
    
    def run_inference(self, image_path=None, num_samples=5):
        """Run inference on new images"""
        print("\n🔍 STEP 4: Running Inference")
        print("-" * 40)
        
        if not self.model_evaluated:
            print("❌ Model must be evaluated first!")
            return False
        
        try:
            if image_path:
                # Single image inference
                print(f"Running inference on: {image_path}")
                # Implementation for single image
                pass
            else:
                # Generate and test on new synthetic samples
                print(f"Generating {num_samples} new test samples...")
                
                # Generate new samples
                test_images = []
                true_labels = []
                
                for i in range(num_samples):
                    if i < num_samples // 2:
                        img = self.data_generator.generate_normal_image()
                        label = 0
                    else:
                        img = self.data_generator.generate_defective_image()
                        label = 1
                    
                    # Convert to numpy array and normalize
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    test_images.append(img_array)
                    true_labels.append(label)
                
                test_images = np.array(test_images)
                
                # Run inference
                predictions = self.evaluator.model.predict(test_images)
                binary_preds = (predictions > 0.5).astype(int)
                
                # Visualize results
                self._visualize_inference_results(test_images, true_labels, predictions, binary_preds)
                
                print("✅ Inference completed successfully!")
                return True
                
        except Exception as e:
            print(f"❌ Error running inference: {e}")
            return False
    
    def _visualize_inference_results(self, images, true_labels, predictions, binary_preds):
        """Visualize inference results"""
        num_images = len(images)
        cols = min(5, num_images)
        rows = (num_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
        fig.suptitle('Inference Results on New Samples', fontsize=16, fontweight='bold')
        
        if rows == 1:
            axes = axes.reshape(1, -1) if num_images > 1 else [axes]
        
        class_names = ['Normal', 'Defective']
        
        for i in range(num_images):
            row = i // cols
            col = i % cols
            
            ax = axes[row][col] if rows > 1 else axes[col]
            
            # Display image
            ax.imshow(images[i])
            
            # Create title with prediction info
            true_label = class_names[true_labels[i]]
            pred_label = class_names[binary_preds[i][0]]
            confidence = predictions[i][0]
            
            # Color based on correctness
            color = 'green' if true_labels[i] == binary_preds[i][0] else 'red'
            
            title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}'
            ax.set_title(title, fontsize=10, color=color, fontweight='bold')
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_images, rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row][col].axis('off')
            else:
                axes[col].axis('off')
        
        plt.tight_layout()
        plt.savefig('plots/inference_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_overfitting(self):
        """Analyze overfitting from training history"""
        try:
            import json
            with open('models/training_history.json', 'r') as f:
                history = json.load(f)
            
            train_acc = history['accuracy']
            val_acc = history['val_accuracy']
            
            final_train_acc = train_acc[-1]
            final_val_acc = val_acc[-1]
            max_val_acc = max(val_acc)
            
            overfitting_gap = final_train_acc - final_val_acc
            
            return overfitting_gap, max_val_acc
        except:
            return 0.0, 0.0
    
    def _print_detailed_analysis(self, metrics, overfitting_gap, max_val_acc, optimal_threshold):
        """Print comprehensive performance analysis"""
        print("\n" + "="*70)
        print("🎯 COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("="*70)
        
        # Test Performance
        print(f"📊 Test Set Performance:")
        print(f"   🎯 Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   🔍 Precision: {metrics['precision']:.4f}")
        print(f"   📡 Recall: {metrics['recall']:.4f}")
        print(f"   ⚖️ F1-Score: {metrics['f1_score']:.4f}")
        print(f"   🎪 Specificity: {metrics['specificity']:.4f}")
        print(f"   📊 AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"   📈 AUC-PR: {metrics['auc_pr']:.4f}")
        
        # Overfitting Analysis
        print(f"\n🔬 Overfitting Analysis:")
        print(f"   Overfitting Gap: {overfitting_gap:.4f}")
        print(f"   Best Val Accuracy: {max_val_acc:.4f}")
        print(f"   Optimal Threshold: {optimal_threshold:.3f}")
        
        # Performance Assessment
        print(f"\n🏆 Performance Assessment:")
        if metrics['accuracy'] >= 0.85:
            print("🎉 Excellent performance! Production-ready model.")
        elif metrics['accuracy'] >= 0.75:
            print("✅ Good performance! Model works well.")
        elif metrics['accuracy'] >= 0.65:
            print("👍 Decent performance, some room for improvement.")
        else:
            print("⚠️ Performance needs significant improvement.")
        
        # Overfitting Assessment
        if overfitting_gap < 0.05:
            print("✅ Excellent generalization! Very little overfitting.")
        elif overfitting_gap < 0.10:
            print("✅ Good generalization! Minimal overfitting.")
        elif overfitting_gap < 0.20:
            print("⚠️ Moderate overfitting detected.")
        else:
            print("❌ Severe overfitting! Model needs regularization.")
        
        # Real-world Interpretation
        print(f"\n🏭 Real-World Manufacturing Impact:")
        print(f"   • Out of 100 parts flagged as defective, {metrics['precision']*100:.1f} are actually defective")
        print(f"   • Catches {metrics['recall']*100:.1f}% of all defective parts")
        print(f"   • Only {(1-metrics['specificity'])*100:.1f}% false alarms on good parts")
        print(f"   • Overall system accuracy: {metrics['accuracy']*100:.1f}%")
        
        # Success Criteria
        success_criteria = [
            metrics['accuracy'] > 0.70,
            overfitting_gap < 0.15,
            metrics['f1_score'] > 0.65
        ]
        
        print(f"\n🎯 Success Criteria Met: {sum(success_criteria)}/3")
        if all(success_criteria):
            print("🎉 SUCCESS! All criteria met - Model ready for deployment!")
        elif sum(success_criteria) >= 2:
            print("✅ GOOD! Most criteria met - Model performs well!")
        else:
            print("⚠️ NEEDS WORK! Consider further optimization.")
        
        print("="*70)
    
    def _print_evaluation_summary(self, metrics):
        """Print a formatted evaluation summary (legacy method)"""
        # This method is kept for backward compatibility but detailed analysis is now in _print_detailed_analysis
        pass
    
    def run_complete_pipeline(self):
        """Run the complete pipeline from start to finish"""
        print("🚀 STARTING COMPLETE VISION-BASED MAINTENANCE PIPELINE")
        print("="*70)
        
        start_time = datetime.now()
        
        # Step 1: Generate dataset
        if not self.generate_dataset():
            return False
        
        # Step 2: Train model
        if not self.train_model():
            return False
        
        # Step 3: Evaluate model
        if not self.evaluate_model():
            return False
        
        # Step 4: Run inference demo
        if not self.run_inference():
            return False
        
        # Calculate total time
        end_time = datetime.now()
        total_time = end_time - start_time
        
        print("\n" + "="*70)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total execution time: {total_time}")
        print("📁 Check the 'plots' and 'models' directories for outputs")
        print("="*70)
        
        return True


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Vision-Based Preventive Maintenance System')
    parser.add_argument('--mode', choices=['complete', 'generate', 'train', 'evaluate', 'inference'], 
                       default='complete', help='Mode to run the system in')
    parser.add_argument('--config', type=str, help='Path to custom configuration file')
    parser.add_argument('--retrain', action='store_true', help='Force retrain even if model exists')
    parser.add_argument('--image', type=str, help='Path to image for inference')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples for inference demo')
    
    args = parser.parse_args()
    
    # Load custom configuration if provided
    config_override = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_override = json.load(f)
        print(f"Loaded custom configuration from {args.config}")
    
    # Initialize system
    system = VisionMaintenanceSystem(config_override)
    
    # Run based on mode
    if args.mode == 'complete':
        system.run_complete_pipeline()
    elif args.mode == 'generate':
        system.generate_dataset()
    elif args.mode == 'train':
        system.train_model(retrain=args.retrain)
    elif args.mode == 'evaluate':
        system.evaluate_model()
    elif args.mode == 'inference':
        system.run_inference(args.image, args.samples)


if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Run main function
    main()
