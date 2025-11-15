# Vision-Based Preventive Maintenance System

A comprehensive AI solution for detecting defective machine parts using synthetic camera images and Convolutional Neural Networks (CNN). This system mimics real-world visual inspection systems used in manufacturing and maintenance environments.

## 🎯 Project Overview

This project implements a complete pipeline for vision-based defect detection including:
- **Synthetic Data Generation**: Creates realistic machine part images with various defect types
- **CNN Architecture**: Custom-designed deep learning model optimized for defect detection
- **Training Pipeline**: Robust training with data augmentation and validation
- **Comprehensive Evaluation**: Multiple metrics and visualization tools
- **Real-time Inference**: Ready-to-deploy prediction system

## 🚀 Features

### Data Generation
- **5 Defect Types**: Scratches, cracks, corrosion, dents, and stains
- **3 Part Geometries**: Circular, rectangular, and complex shapes
- **Realistic Textures**: Surface noise and material properties
- **Configurable Parameters**: Defect probability, intensity, and size

### Model Architecture
- **Deep CNN**: 4 convolutional blocks with batch normalization
- **Dropout Regularization**: Prevents overfitting
- **Global Average Pooling**: Reduces parameters while maintaining performance
- **Binary Classification**: Normal vs. Defective parts

### Evaluation Metrics
- **Accuracy**: Overall classification performance
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negative rate
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve

### Visualizations
- **Sample Images**: Examples from each class
- **Confusion Matrix**: Classification results breakdown
- **ROC Curve**: True vs. false positive rates
- **Precision-Recall Curve**: Precision vs. recall trade-off
- **Training Curves**: Loss and metrics over epochs
- **Threshold Analysis**: Performance across different thresholds
- **Misclassification Analysis**: Examples of prediction errors

## 📋 Requirements

```bash
pip install -r requirements.txt
```

### Dependencies
- TensorFlow >= 2.13.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- Seaborn >= 0.12.0
- Scikit-learn >= 1.3.0
- OpenCV-Python >= 4.8.0
- Pillow >= 10.0.0
- Pandas >= 2.0.0
- tqdm >= 4.65.0

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd vision-based-preventive-maintenance
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python main.py --mode generate
   ```

## 🎮 Usage

### Quick Start - Complete Pipeline
Run the entire pipeline from data generation to evaluation:

```bash
python main.py --mode complete
```

### Individual Components

#### 1. Generate Synthetic Dataset
```bash
python main.py --mode generate
```
- Creates 2000 normal + 2000 defective images
- Saves to `data/normal/` and `data/defective/`
- Generates sample visualization

#### 2. Train CNN Model
```bash
python main.py --mode train
```
- Loads dataset and splits into train/val/test
- Trains CNN with data augmentation
- Saves best model to `models/best_model.h5`
- Generates training curves

#### 3. Evaluate Model
```bash
python main.py --mode evaluate
```
- Loads trained model and test data
- Calculates comprehensive metrics
- Generates evaluation plots
- Saves detailed report

#### 4. Run Inference
```bash
python main.py --mode inference --samples 10
```
- Tests model on new synthetic samples
- Visualizes predictions with confidence scores

### Advanced Usage

#### Custom Configuration
```bash
python main.py --config custom_config.json
```

#### Force Retrain
```bash
python main.py --mode train --retrain
```

#### Single Image Inference
```bash
python main.py --mode inference --image path/to/image.png
```

## 📁 Project Structure

```
vision-based-preventive-maintenance/
├── main.py                 # Main interface and pipeline orchestration
├── config.py              # Configuration parameters
├── data_generator.py      # Synthetic data generation
├── model.py               # CNN architecture and utilities
├── training.py            # Training pipeline and data handling
├── evaluation.py          # Evaluation metrics and analysis
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # Generated dataset
│   ├── normal/           # Normal part images
│   └── defective/        # Defective part images
├── models/               # Trained models and logs
│   ├── best_model.h5     # Best trained model
│   ├── training_log.csv  # Training metrics log
│   └── evaluation_report.json # Evaluation results
└── plots/                # Generated visualizations
    ├── sample_images.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── training_curves.png
    └── threshold_analysis.png
```

## ⚙️ Configuration

### Dataset Configuration
```python
DATASET_CONFIG = {
    'image_size': (128, 128),
    'num_samples_per_class': 2000,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15
}
```

### Model Configuration
```python
MODEL_CONFIG = {
    'input_shape': (128, 128, 3),
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'patience': 10
}
```

### Defect Configuration
```python
DEFECT_CONFIG = {
    'scratch_probability': 0.3,
    'crack_probability': 0.25,
    'corrosion_probability': 0.2,
    'dent_probability': 0.15,
    'stain_probability': 0.1
}
```

## 📊 Expected Results

### Model Performance
- **Accuracy**: 92-96%
- **Precision**: 90-95%
- **Recall**: 88-94%
- **F1-Score**: 90-94%
- **AUC-ROC**: 0.95-0.98

### Training Time
- **Dataset Generation**: 2-5 minutes
- **Model Training**: 10-30 minutes (depending on hardware)
- **Evaluation**: 1-2 minutes

## 🔧 Customization

### Adding New Defect Types
1. Extend `SyntheticDataGenerator` class in `data_generator.py`
2. Add new defect generation method
3. Update defect probabilities in `config.py`

### Modifying CNN Architecture
1. Edit `build_model()` method in `model.py`
2. Adjust hyperparameters in `config.py`
3. Retrain with `--retrain` flag

### Custom Evaluation Metrics
1. Add new metrics to `ModelEvaluator` class in `evaluation.py`
2. Update visualization methods as needed

## 🐛 Troubleshooting

### Common Issues

1. **Memory Error during Training**:
   - Reduce `batch_size` in config
   - Decrease `image_size` if needed

2. **Low Model Performance**:
   - Increase dataset size
   - Adjust defect generation parameters
   - Try different CNN architectures

3. **Training Too Slow**:
   - Enable GPU acceleration
   - Reduce image size or dataset size
   - Use mixed precision training

4. **Import Errors**:
   - Verify all dependencies are installed
   - Check Python version compatibility

## 📈 Performance Optimization

### For Better Accuracy
- Increase dataset size
- Add more defect variations
- Use transfer learning from pre-trained models
- Implement ensemble methods

### For Faster Training
- Use GPU acceleration
- Implement mixed precision training
- Optimize data loading pipeline
- Use smaller image sizes

### For Production Deployment
- Quantize model for mobile deployment
- Implement model serving with TensorFlow Serving
- Add real-time preprocessing pipeline
- Implement batch inference for multiple images

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- OpenCV community for computer vision tools
- Scikit-learn for machine learning utilities

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration options

---

**Note**: This system uses synthetic data for demonstration purposes. For real-world deployment, you would need to train on actual machine part images and validate performance in your specific manufacturing environment.
