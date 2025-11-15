"""
Evaluation Module for Vision-Based Preventive Maintenance System
Comprehensive evaluation metrics and analysis tools
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd
import json
from datetime import datetime
import os

from model import DefectDetectionCNN
from training import DataLoader


class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.predictions = None
        self.true_labels = None
        self.prediction_probs = None
        self.class_names = ['Normal', 'Defective']
    
    def load_model(self, model_path=None):
        """Load trained model"""
        path = model_path or self.model_path
        if path is None:
            raise ValueError("Model path must be provided")
        
        try:
            cnn = DefectDetectionCNN()
            self.model = cnn.load_model(path)
            print(f"Model loaded successfully from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def evaluate_on_test_data(self, X_test, y_test, threshold=0.5):
        """Evaluate model on test dataset"""
        if self.model is None:
            raise ValueError("Model must be loaded before evaluation")
        
        print("Evaluating model on test data...")
        
        # Get predictions
        self.prediction_probs = self.model.predict(X_test)
        self.predictions = (self.prediction_probs > threshold).astype(int).flatten()
        self.true_labels = y_test
        
        # Calculate basic metrics
        metrics = self.calculate_metrics()
        
        print("Evaluation Results:")
        print("=" * 50)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def calculate_metrics(self):
        """Calculate comprehensive evaluation metrics"""
        if self.predictions is None or self.true_labels is None:
            raise ValueError("Predictions must be generated before calculating metrics")
        
        metrics = {
            'accuracy': accuracy_score(self.true_labels, self.predictions),
            'precision': precision_score(self.true_labels, self.predictions, average='binary'),
            'recall': recall_score(self.true_labels, self.predictions, average='binary'),
            'f1_score': f1_score(self.true_labels, self.predictions, average='binary'),
            'specificity': self._calculate_specificity(),
            'auc_roc': self._calculate_auc_roc(),
            'auc_pr': self._calculate_auc_pr()
        }
        
        return metrics
    
    def _calculate_specificity(self):
        """Calculate specificity (True Negative Rate)"""
        tn, fp, fn, tp = confusion_matrix(self.true_labels, self.predictions).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _calculate_auc_roc(self):
        """Calculate AUC-ROC score"""
        if self.prediction_probs is None:
            return 0.0
        fpr, tpr, _ = roc_curve(self.true_labels, self.prediction_probs)
        return auc(fpr, tpr)
    
    def _calculate_auc_pr(self):
        """Calculate AUC-PR (Precision-Recall) score"""
        if self.prediction_probs is None:
            return 0.0
        return average_precision_score(self.true_labels, self.prediction_probs)
    
    def plot_confusion_matrix(self, normalize=False, save_path='plots/confusion_matrix.png'):
        """Plot confusion matrix"""
        if self.predictions is None:
            raise ValueError("Predictions must be generated before plotting confusion matrix")
        
        # Calculate confusion matrix
        cm = confusion_matrix(self.true_labels, self.predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        # Create plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Add statistics
        if not normalize:
            tn, fp, fn, tp = cm.ravel()
            stats_text = f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}'
            plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, save_path='plots/roc_curve.png'):
        """Plot ROC curve"""
        if self.prediction_probs is None:
            raise ValueError("Prediction probabilities must be available for ROC curve")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(self.true_labels, self.prediction_probs)
        roc_auc = auc(fpr, tpr)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fpr, tpr, roc_auc
    
    def plot_precision_recall_curve(self, save_path='plots/precision_recall_curve.png'):
        """Plot Precision-Recall curve"""
        if self.prediction_probs is None:
            raise ValueError("Prediction probabilities must be available for PR curve")
        
        # Calculate PR curve
        precision, recall, thresholds = precision_recall_curve(self.true_labels, self.prediction_probs)
        pr_auc = average_precision_score(self.true_labels, self.prediction_probs)
        
        # Create plot
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR Curve (AUC = {pr_auc:.3f})')
        
        # Baseline (random classifier)
        baseline = np.sum(self.true_labels) / len(self.true_labels)
        plt.axhline(y=baseline, color='red', linestyle='--', 
                   label=f'Random Classifier (AP = {baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return precision, recall, pr_auc
    
    def plot_threshold_analysis(self, save_path='plots/threshold_analysis.png'):
        """Analyze performance across different thresholds"""
        if self.prediction_probs is None:
            raise ValueError("Prediction probabilities must be available for threshold analysis")
        
        thresholds = np.linspace(0.1, 0.9, 50)
        metrics_by_threshold = {
            'threshold': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'specificity': []
        }
        
        for threshold in thresholds:
            pred_binary = (self.prediction_probs > threshold).astype(int).flatten()
            
            # Calculate metrics for this threshold
            acc = accuracy_score(self.true_labels, pred_binary)
            prec = precision_score(self.true_labels, pred_binary, zero_division=0)
            rec = recall_score(self.true_labels, pred_binary, zero_division=0)
            f1 = f1_score(self.true_labels, pred_binary, zero_division=0)
            
            # Calculate specificity
            tn, fp, fn, tp = confusion_matrix(self.true_labels, pred_binary).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            metrics_by_threshold['threshold'].append(threshold)
            metrics_by_threshold['accuracy'].append(acc)
            metrics_by_threshold['precision'].append(prec)
            metrics_by_threshold['recall'].append(rec)
            metrics_by_threshold['f1_score'].append(f1)
            metrics_by_threshold['specificity'].append(spec)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        plt.plot(thresholds, metrics_by_threshold['accuracy'], 'b-', label='Accuracy', linewidth=2)
        plt.plot(thresholds, metrics_by_threshold['precision'], 'g-', label='Precision', linewidth=2)
        plt.plot(thresholds, metrics_by_threshold['recall'], 'r-', label='Recall', linewidth=2)
        plt.plot(thresholds, metrics_by_threshold['f1_score'], 'm-', label='F1-Score', linewidth=2)
        plt.plot(thresholds, metrics_by_threshold['specificity'], 'c-', label='Specificity', linewidth=2)
        
        # Find optimal threshold (max F1-score)
        optimal_idx = np.argmax(metrics_by_threshold['f1_score'])
        optimal_threshold = thresholds[optimal_idx]
        plt.axvline(x=optimal_threshold, color='black', linestyle='--', 
                   label=f'Optimal Threshold = {optimal_threshold:.3f}')
        
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Performance Metrics vs. Classification Threshold', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim([0.1, 0.9])
        plt.ylim([0.0, 1.0])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return optimal_threshold, metrics_by_threshold
    
    def generate_classification_report(self):
        """Generate detailed classification report"""
        if self.predictions is None:
            raise ValueError("Predictions must be generated before generating report")
        
        report = classification_report(
            self.true_labels, 
            self.predictions,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Convert to DataFrame for better display
        df_report = pd.DataFrame(report).transpose()
        
        print("\nDetailed Classification Report:")
        print("=" * 60)
        print(df_report.round(4))
        
        return report
    
    def analyze_misclassifications(self, X_test, max_examples=8, save_path='plots/misclassifications.png'):
        """Analyze and visualize misclassified examples"""
        if self.predictions is None:
            raise ValueError("Predictions must be generated before analyzing misclassifications")
        
        # Find misclassified examples
        misclassified_indices = np.where(self.predictions != self.true_labels)[0]
        
        if len(misclassified_indices) == 0:
            print("No misclassifications found!")
            return
        
        print(f"Found {len(misclassified_indices)} misclassified examples")
        
        # Select random subset for visualization
        num_examples = min(max_examples, len(misclassified_indices))
        selected_indices = np.random.choice(misclassified_indices, num_examples, replace=False)
        
        # Create visualization
        cols = 4
        rows = (num_examples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        fig.suptitle('Misclassified Examples', fontsize=16, fontweight='bold')
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(selected_indices):
            row = i // cols
            col = i % cols
            
            # Get image and predictions
            img = X_test[idx]
            true_label = self.class_names[self.true_labels[idx]]
            pred_label = self.class_names[self.predictions[idx]]
            confidence = self.prediction_probs[idx][0] if self.prediction_probs is not None else 0.5
            
            # Plot image
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'True: {true_label}\\nPred: {pred_label}\\nConf: {confidence:.3f}',
                                   fontsize=10)
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(num_examples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_evaluation_report(self, metrics, save_path='models/evaluation_report.json'):
        """Save comprehensive evaluation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'metrics': metrics,
            'classification_report': self.generate_classification_report(),
            'dataset_info': {
                'total_samples': len(self.true_labels),
                'positive_samples': int(np.sum(self.true_labels)),
                'negative_samples': int(len(self.true_labels) - np.sum(self.true_labels)),
                'class_distribution': {
                    'normal': float(np.mean(self.true_labels == 0)),
                    'defective': float(np.mean(self.true_labels == 1))
                }
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Evaluation report saved to {save_path}")
        
        return report


class PerformanceAnalyzer:
    """Additional performance analysis tools"""
    
    @staticmethod
    def compare_models(evaluation_reports, save_path='plots/model_comparison.png'):
        """Compare multiple model evaluation reports"""
        if not evaluation_reports:
            print("No evaluation reports provided")
            return
        
        # Extract metrics for comparison
        models = list(evaluation_reports.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        
        comparison_data = {metric: [] for metric in metrics}
        
        for model_name in models:
            report = evaluation_reports[model_name]
            for metric in metrics:
                comparison_data[metric].append(report.get('metrics', {}).get(metric, 0))
        
        # Create comparison plot
        x = np.arange(len(models))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        for i, metric in enumerate(metrics):
            ax.bar(x + i * width, comparison_data[metric], width, 
                  label=metric.replace('_', ' ').title(), color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def analyze_prediction_confidence(prediction_probs, true_labels, save_path='plots/confidence_analysis.png'):
        """Analyze prediction confidence distribution"""
        correct_predictions = prediction_probs[prediction_probs.flatten() == true_labels]
        incorrect_predictions = prediction_probs[prediction_probs.flatten() != true_labels]
        
        plt.figure(figsize=(12, 5))
        
        # Confidence distribution for correct predictions
        plt.subplot(1, 2, 1)
        plt.hist(correct_predictions, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.title('Confidence Distribution - Correct Predictions')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Confidence distribution for incorrect predictions
        plt.subplot(1, 2, 2)
        plt.hist(incorrect_predictions, bins=20, alpha=0.7, color='red', edgecolor='black')
        plt.title('Confidence Distribution - Incorrect Predictions')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Example usage
    evaluator = ModelEvaluator('models/best_model.h5')
    
    # Load test data (you would load your actual test data here)
    data_loader = DataLoader()
    X, y = data_loader.load_dataset()
    
    # For demonstration, use a subset as test data
    X_test, y_test = X[:100], y[:100]
    
    # Load model and evaluate
    if evaluator.load_model():
        metrics = evaluator.evaluate_on_test_data(X_test, y_test)
        
        # Generate visualizations
        evaluator.plot_confusion_matrix()
        evaluator.plot_roc_curve()
        evaluator.plot_precision_recall_curve()
        evaluator.plot_threshold_analysis()
        evaluator.analyze_misclassifications(X_test)
        
        # Save report
        evaluator.save_evaluation_report(metrics)
