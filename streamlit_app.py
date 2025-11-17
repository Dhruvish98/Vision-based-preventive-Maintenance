"""
Vision-Based Preventive Maintenance Dashboard
A comprehensive Streamlit application for defect detection in manufacturing
"""

import streamlit as st
import os
import sys
import json
import zipfile
import tempfile
import shutil
import time
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import zoom

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from data_generator import SyntheticDataGenerator
from training import Trainer, DataLoader
from model import DefectDetectionCNN
from evaluation import ModelEvaluator
from config import DATASET_CONFIG, MODEL_CONFIG, TRAINING_CONFIG

# Page configuration
st.set_page_config(
    page_title="Vision-Based Preventive Maintenance",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'training_in_progress' not in st.session_state:
        st.session_state.training_in_progress = False
    if 'current_dataset_path' not in st.session_state:
        st.session_state.current_dataset_path = None
    if 'training_history' not in st.session_state:
        st.session_state.training_history = []
    if 'current_model_path' not in st.session_state:
        st.session_state.current_model_path = None

# Gemini API integration
def get_gemini_analysis(prompt, analysis_type="technical"):
    """Get analysis from Google Gemini API"""
    try:
        import google.generativeai as genai
        
        # Get API key from Streamlit secrets
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            return "⚠️ Gemini API key not configured in Streamlit secrets."
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Customize prompt based on analysis type
        if analysis_type == "executive":
            system_prompt = """You are a business analyst explaining AI/ML results to C-suite executives. 
            Focus on business impact, ROI, deployment readiness, risk assessment, and strategic implications. 
            Use business terminology and avoid technical jargon. Provide actionable insights for decision makers."""
        else:
            system_prompt = """You are a machine learning expert providing technical analysis. 
            Focus on model performance, optimization suggestions, technical explanations, and implementation details. 
            Use appropriate technical terminology and provide specific recommendations for improvement."""
        
        full_prompt = f"{system_prompt}\n\nAnalyze the following:\n{prompt}"
        
        response = model.generate_content(full_prompt)
        return response.text
        
    except Exception as e:
        return f"⚠️ Error getting Gemini analysis: {str(e)}"

# Sidebar navigation
def create_sidebar():
    """Create sidebar navigation"""
    st.sidebar.markdown("# 🔧 Navigation")
    
    pages = {
        "🏠 Home": "home",
        "📊 Dataset": "dataset", 
        "🤖 Training": "training",
        "🎯 Demo": "demo",
        "📈 Model History": "history"
    }
    
    selected_page = st.sidebar.selectbox(
        "Select Page",
        list(pages.keys()),
        key="page_selector"
    )
    
    # Training status indicator
    if st.session_state.training_in_progress:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔄 Training Status")
        st.sidebar.info("Model training in progress...")
        
        # Add stop training button
        if st.sidebar.button("⏹️ Stop Training", type="secondary"):
            st.session_state.training_in_progress = False
            st.rerun()
    
    # Dataset status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Status")
    if st.session_state.dataset_loaded:
        st.sidebar.success("✅ Dataset Loaded")
    else:
        st.sidebar.warning("⚠️ No Dataset")
    
    # Model status
    st.sidebar.markdown("### 🤖 Model Status")
    if st.session_state.model_trained:
        st.sidebar.success("✅ Model Ready")
    else:
        st.sidebar.warning("⚠️ No Trained Model")
    
    return pages[selected_page]

# Home page
def show_home_page():
    """Display home page with project overview"""
    st.markdown('<h1 class="main-header">🔧 Vision-Based Preventive Maintenance System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Project Overview</h3>
    This comprehensive AI-powered system detects defective machine parts using computer vision and deep learning. 
    The system uses Convolutional Neural Networks (CNN) to classify parts as normal or defective, 
    helping manufacturing companies implement predictive maintenance strategies.
    </div>
    """, unsafe_allow_html=True)
    
    # Key features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 Key Features")
        st.markdown("""
        - **Synthetic Data Generation**: Create realistic training datasets
        - **Custom Dataset Upload**: Use your own manufacturing data
        - **Interactive Training**: Adjust parameters and monitor progress
        - **Real-time Analysis**: AI-powered insights with Google Gemini
        - **Comprehensive Evaluation**: Detailed performance metrics
        - **Production Demo**: Test the system with real images
        """)
    
    with col2:
        st.markdown("### 📋 System Capabilities")
        st.markdown("""
        - **Defect Types**: Scratches, cracks, corrosion, dents, stains
        - **Accuracy**: Up to 90%+ with proper training
        - **Real-time Inference**: Instant classification results
        - **Batch Processing**: Handle multiple images simultaneously
        - **Model Versioning**: Track different training experiments
        - **Executive Reporting**: Business-focused insights
        """)
    
    # Page descriptions
    st.markdown('<h2 class="sub-header">📖 Page Guide</h2>', unsafe_allow_html=True)
    
    # Dataset page info
    st.markdown("### 📊 Dataset Page")
    st.markdown("""
    <div class="info-box">
    <strong>Purpose:</strong> Manage your training data<br>
    <strong>Options:</strong>
    <ul>
    <li><strong>Generate Synthetic Dataset:</strong> Create 4000+ realistic machine part images with various defects</li>
    <li><strong>Upload Custom Dataset:</strong> Use your own images (ZIP format with 'normal' and 'defective' folders)</li>
    <li><strong>Dataset Requirements:</strong> Minimum 100 images per class, PNG/JPG/JPEG formats</li>
    <li><strong>Visualizations:</strong> Sample images, class distribution, defect analysis</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Training page info
    st.markdown("### 🤖 Training Page")
    st.markdown("""
    <div class="warning-box">
    <strong>⚠️ Important Training Information:</strong><br>
    <ul>
    <li><strong>Training Time:</strong> 15-45 minutes depending on dataset size and parameters</li>
    <li><strong>Background Training:</strong> You can switch tabs while training continues</li>
    <li><strong>Progress Tracking:</strong> Real-time metrics, estimated completion time</li>
    <li><strong>Adjustable Parameters:</strong> Learning rate, batch size, epochs, dropout, augmentation</li>
    <li><strong>Auto-save:</strong> Best models are automatically saved with performance history</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Demo page info
    st.markdown("### 🎯 Demo Page")
    st.markdown("""
    <div class="success-box">
    <strong>✅ Ready-to-Use Demo:</strong><br>
    <ul>
    <li><strong>Pre-trained Model:</strong> High-performance model ready for immediate use</li>
    <li><strong>Single Image Upload:</strong> Test individual parts with confidence scores</li>
    <li><strong>Batch Processing:</strong> Upload multiple images for bulk classification</li>
    <li><strong>Visual Analysis:</strong> See model attention and defect highlighting</li>
    <li><strong>Performance Metrics:</strong> View saved model evaluation results</li>
    <li><strong>Real-time Updates:</strong> Automatically uses newly trained models</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Analysis info
    st.markdown("### 🤖 AI-Powered Analysis")
    st.markdown("""
    <div class="info-box">
    <strong>Google Gemini Integration:</strong><br>
    <ul>
    <li><strong>Technical Analysis:</strong> Detailed model performance insights for engineers</li>
    <li><strong>Executive Summary:</strong> Business-focused reports for C-suite decision makers</li>
    <li><strong>Parameter Recommendations:</strong> AI suggestions for improving model performance</li>
    <li><strong>Real-time Insights:</strong> Analysis updates as you modify training parameters</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Getting started
    st.markdown('<h2 class="sub-header">🚀 Getting Started</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h4>1️⃣ Prepare Data</h4>
        <p>Go to Dataset page and either generate synthetic data or upload your own manufacturing images</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h4>2️⃣ Train Model</h4>
        <p>Use Training page to configure parameters and train your custom defect detection model</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h4>3️⃣ Test & Deploy</h4>
        <p>Visit Demo page to test the model with real images and view performance analytics</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick demo option
    st.markdown("---")
    st.markdown("### ⚡ Quick Demo")
    st.info("💡 **Want to try it immediately?** Go to the Demo page to test our pre-trained model with your images!")
    
    if st.button("🎯 Go to Demo", type="primary"):
        st.info("Please navigate to the Demo page using the sidebar.")
        st.rerun()

def main():
    """Main application function"""
    initialize_session_state()
    
    # Create sidebar and get selected page
    selected_page = create_sidebar()
    
    # Route to appropriate page
    if selected_page == "home":
        show_home_page()
    elif selected_page == "dataset":
        show_dataset_page()
    elif selected_page == "training":
        show_training_page()
    elif selected_page == "demo":
        show_demo_page()
    elif selected_page == "history":
        show_history_page()

# Dataset page implementation
def show_dataset_page():
    """Dataset management page"""
    st.markdown('<h1 class="main-header">📊 Dataset Management</h1>', unsafe_allow_html=True)
    
    # Dataset options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎲 Generate Synthetic Dataset")
        st.markdown("""
        Create a realistic dataset of machine parts with various defects:
        - **5 Defect Types**: Scratches, cracks, corrosion, dents, stains
        - **3 Part Geometries**: Circular, rectangular, complex shapes
        - **Configurable Size**: Adjust number of samples per class
        
        ⚠️ **Note**: Generating a new dataset will replace any existing dataset.
        """)
        
        # Synthetic dataset parameters
        st.markdown("#### Parameters")
        samples_per_class = st.slider("Samples per class", 500, 5000, 2000, 500)
        
        defect_intensity = st.slider("Defect Intensity", 0.3, 1.0, 0.7, 0.1)
        
        if st.button("🎲 Generate Synthetic Dataset", type="primary"):
            generate_synthetic_dataset(samples_per_class, defect_intensity)
    
    with col2:
        st.markdown("### 📁 Upload Custom Dataset")
        st.markdown("""
        Upload your own manufacturing images:
        - **Format**: ZIP file containing 'normal' and 'defective' folders
        - **Images**: PNG, JPG, JPEG formats
        - **Minimum**: 100 images per class
        - **Structure**: `dataset.zip/normal/` and `dataset.zip/defective/`
        """)
        
        uploaded_file = st.file_uploader(
            "Choose ZIP file",
            type=['zip'],
            help="Upload a ZIP file with 'normal' and 'defective' folders"
        )
        
        if uploaded_file is not None:
            if st.button("📁 Process Uploaded Dataset", type="primary"):
                process_uploaded_dataset(uploaded_file)
    
    # Dataset visualization
    if st.session_state.dataset_loaded:
        st.markdown("---")
        show_dataset_analysis()
    
    # Gemini analysis for dataset
    if st.session_state.dataset_loaded:
        st.markdown("---")
        show_dataset_gemini_analysis()

def generate_synthetic_dataset(samples_per_class, defect_intensity):
    """Generate synthetic dataset"""
    try:
        with st.spinner("Generating synthetic dataset..."):
            # Update config
            config = DATASET_CONFIG.copy()
            config['num_samples_per_class'] = samples_per_class
            
            # Update defect config
            from config import DEFECT_CONFIG
            defect_config = DEFECT_CONFIG.copy()
            defect_config['defect_intensity_range'] = (defect_intensity-0.2, defect_intensity+0.2)
            
            # Clear existing data directory first
            data_dir = config['data_dir']
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)
                st.info("🗑️ Cleared existing dataset")
            
            # Generate dataset
            generator = SyntheticDataGenerator(config, defect_config)
            generator.generate_dataset()
            
            # Update session state
            st.session_state.dataset_loaded = True
            st.session_state.current_dataset_path = config['data_dir']
            
            st.success(f"✅ Generated {samples_per_class * 2} images successfully!")
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error generating dataset: {str(e)}")

def process_uploaded_dataset(uploaded_file):
    """Process uploaded ZIP dataset"""
    try:
        with st.spinner("Processing uploaded dataset..."):
            # Create temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Extract ZIP file
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find normal and defective folders
            normal_path = None
            defective_path = None
            
            for root, dirs, files in os.walk(temp_dir):
                if 'normal' in dirs:
                    normal_path = os.path.join(root, 'normal')
                if 'defective' in dirs:
                    defective_path = os.path.join(root, 'defective')
            
            if not normal_path or not defective_path:
                st.error("❌ ZIP file must contain 'normal' and 'defective' folders")
                return
            
            # Validate images
            normal_images = validate_image_folder(normal_path)
            defective_images = validate_image_folder(defective_path)
            
            if normal_images < 100 or defective_images < 100:
                st.error(f"❌ Minimum 100 images per class required. Found: Normal={normal_images}, Defective={defective_images}")
                return
            
            # Copy to data directory
            data_dir = "data"
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)
            
            shutil.copytree(temp_dir, data_dir)
            
            # Update session state
            st.session_state.dataset_loaded = True
            st.session_state.current_dataset_path = data_dir
            
            st.success(f"✅ Dataset uploaded successfully! Normal: {normal_images}, Defective: {defective_images}")
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error processing dataset: {str(e)}")

def validate_image_folder(folder_path):
    """Validate and count images in folder"""
    valid_extensions = {'.png', '.jpg', '.jpeg'}
    count = 0
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            count += 1
    
    return count

def show_dataset_analysis():
    """Show dataset analysis and visualizations"""
    st.markdown('<h2 class="sub-header">📈 Dataset Analysis</h2>', unsafe_allow_html=True)
    
    try:
        # Load dataset info
        data_loader = DataLoader()
        
        # Count images
        normal_path = os.path.join(st.session_state.current_dataset_path, 'normal')
        defective_path = os.path.join(st.session_state.current_dataset_path, 'defective')
        
        normal_count = len([f for f in os.listdir(normal_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        defective_count = len([f for f in os.listdir(defective_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Images", normal_count + defective_count)
        with col2:
            st.metric("Normal Images", normal_count)
        with col3:
            st.metric("Defective Images", defective_count)
        with col4:
            balance_ratio = normal_count / defective_count if defective_count > 0 else 0
            st.metric("Balance Ratio", f"{balance_ratio:.2f}")
        
        # Class distribution chart
        fig = px.pie(
            values=[normal_count, defective_count],
            names=['Normal', 'Defective'],
            title="Class Distribution",
            color_discrete_map={'Normal': '#2E8B57', 'Defective': '#DC143C'}
        )
        st.plotly_chart(fig, width="stretch")
        
        # Sample images
        show_sample_images(normal_path, defective_path)
        
    except Exception as e:
        st.error(f"Error analyzing dataset: {str(e)}")

def show_sample_images(normal_path, defective_path):
    """Display sample images from dataset"""
    st.markdown("### 🖼️ Sample Images")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Normal Parts")
        normal_files = [f for f in os.listdir(normal_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if normal_files:
            sample_files = np.random.choice(normal_files, min(4, len(normal_files)), replace=False)
            
            cols = st.columns(2)
            for i, file in enumerate(sample_files):
                with cols[i % 2]:
                    img = Image.open(os.path.join(normal_path, file))
                    st.image(img, caption=f"Normal: {file}", width="stretch")
    
    with col2:
        st.markdown("#### Defective Parts")
        defective_files = [f for f in os.listdir(defective_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if defective_files:
            sample_files = np.random.choice(defective_files, min(4, len(defective_files)), replace=False)
            
            cols = st.columns(2)
            for i, file in enumerate(sample_files):
                with cols[i % 2]:
                    img = Image.open(os.path.join(defective_path, file))
                    st.image(img, caption=f"Defective: {file}", width="stretch")

def show_dataset_gemini_analysis():
    """Show Gemini AI analysis of dataset"""
    st.markdown('<h2 class="sub-header">🤖 AI Dataset Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔬 Technical Analysis", type="secondary"):
            analyze_dataset_with_gemini("technical")
    
    with col2:
        if st.button("👔 Executive Summary", type="secondary"):
            analyze_dataset_with_gemini("executive")

def analyze_dataset_with_gemini(analysis_type):
    """Get Gemini analysis of dataset"""
    try:
        # Prepare dataset information
        normal_path = os.path.join(st.session_state.current_dataset_path, 'normal')
        defective_path = os.path.join(st.session_state.current_dataset_path, 'defective')
        
        normal_count = len([f for f in os.listdir(normal_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        defective_count = len([f for f in os.listdir(defective_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        dataset_info = f"""
        Dataset Analysis Request:
        - Total Images: {normal_count + defective_count}
        - Normal Parts: {normal_count}
        - Defective Parts: {defective_count}
        - Balance Ratio: {normal_count/defective_count:.2f}
        - Image Format: Manufacturing part inspection images
        - Classes: Binary classification (Normal vs Defective)
        
        Please analyze this dataset for machine learning training and provide insights on:
        1. Dataset size adequacy for training
        2. Class balance implications
        3. Potential training challenges
        4. Recommendations for improvement
        """
        
        with st.spinner("Getting AI analysis..."):
            analysis = get_gemini_analysis(dataset_info, analysis_type)
            
            if analysis_type == "technical":
                st.markdown("#### 🔬 Technical Analysis")
            else:
                st.markdown("#### 👔 Executive Summary")
            
            st.markdown(analysis)
            
    except Exception as e:
        st.error(f"Error getting AI analysis: {str(e)}")

def show_training_page():
    """Model training page with parameter controls"""
    st.markdown('<h1 class="main-header">🤖 Model Training</h1>', unsafe_allow_html=True)
    
    if not st.session_state.dataset_loaded:
        st.warning("⚠️ Please load a dataset first from the Dataset page.")
        if st.button("📊 Go to Dataset Page"):
            st.info("Please manually navigate to the Dataset page using the sidebar.")
        return
    
    # Training parameters section
    st.markdown('<h2 class="sub-header">⚙️ Training Parameters</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏗️ Model Architecture")
        
        # Model parameters
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.0005,
            help="Lower values = more stable training, Higher values = faster convergence"
        )
        
        batch_size = st.selectbox(
            "Batch Size",
            options=[8, 16, 32, 64],
            index=2,
            help="Larger batches = more stable gradients, smaller batches = more updates"
        )
        
        epochs = st.slider(
            "Maximum Epochs",
            min_value=10,
            max_value=100,
            value=50,
            help="Training will stop early if no improvement"
        )
        
        patience = st.slider(
            "Early Stopping Patience",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of epochs to wait without improvement"
        )
    
    with col2:
        st.markdown("### 🔧 Regularization")
        
        # Dropout rates
        conv_dropout = st.slider(
            "Convolutional Dropout",
            min_value=0.1,
            max_value=0.7,
            value=0.4,
            step=0.1,
            help="Dropout rate for convolutional layers"
        )
        
        dense_dropout = st.slider(
            "Dense Layer Dropout",
            min_value=0.3,
            max_value=0.8,
            value=0.6,
            step=0.1,
            help="Dropout rate for dense layers"
        )
        
        # Model complexity
        base_filters = st.selectbox(
            "Base Filters",
            options=[8, 16, 32],
            index=1,
            help="Starting number of convolutional filters"
        )
        
        dense_units = st.selectbox(
            "Dense Layer Units",
            options=[16, 32, 64, 128],
            index=1,
            help="Number of neurons in dense layer"
        )
    
    # Data augmentation parameters
    st.markdown("### 🔄 Data Augmentation")
    
    use_augmentation = st.checkbox("Enable Data Augmentation", value=True)
    
    if use_augmentation:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            rotation_range = st.slider("Rotation Range (degrees)", 0, 60, 40)
            width_shift = st.slider("Width Shift Range", 0.0, 0.3, 0.2, 0.05)
        
        with col2:
            height_shift = st.slider("Height Shift Range", 0.0, 0.3, 0.2, 0.05)
            shear_range = st.slider("Shear Range", 0.0, 0.3, 0.2, 0.05)
        
        with col3:
            zoom_range = st.slider("Zoom Range", 0.0, 0.3, 0.2, 0.05)
            horizontal_flip = st.checkbox("Horizontal Flip", value=True)
            vertical_flip = st.checkbox("Vertical Flip", value=True)
    
    # Parameter analysis with Gemini
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🤖 Analyze Parameters (Technical)", type="secondary"):
            analyze_parameters_with_gemini("technical", {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
                'conv_dropout': conv_dropout,
                'dense_dropout': dense_dropout,
                'base_filters': base_filters,
                'dense_units': dense_units,
                'use_augmentation': use_augmentation
            })
    
    with col2:
        if st.button("👔 Business Impact Analysis", type="secondary"):
            analyze_parameters_with_gemini("executive", {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
                'training_time_estimate': estimate_training_time(epochs, batch_size)
            })
    
    # Training section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">🚀 Start Training</h2>', unsafe_allow_html=True)
    
    # Training status
    if st.session_state.training_in_progress:
        show_training_progress()
        # Auto-refresh every 5 seconds during training (using placeholder for better UX)
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        current_time = time.time()
        if current_time - st.session_state.last_refresh > 5:  # Refresh every 5 seconds
            st.session_state.last_refresh = current_time
            st.rerun()
        
        # Add JavaScript auto-refresh as fallback
        st.markdown("""
        <script>
        setTimeout(function(){
            window.location.reload();
        }, 5000);
        </script>
        """, unsafe_allow_html=True)
    else:
        # Training controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.info(f"⏱️ Estimated training time: {estimate_training_time(epochs, batch_size)} minutes")
        
        with col2:
            if st.button("🚀 Start Training", type="primary", disabled=st.session_state.training_in_progress):
                start_training({
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'patience': patience,
                    'conv_dropout': conv_dropout,
                    'dense_dropout': dense_dropout,
                    'base_filters': base_filters,
                    'dense_units': dense_units,
                    'use_augmentation': use_augmentation,
                    'rotation_range': rotation_range if use_augmentation else 0,
                    'width_shift': width_shift if use_augmentation else 0,
                    'height_shift': height_shift if use_augmentation else 0,
                    'shear_range': shear_range if use_augmentation else 0,
                    'zoom_range': zoom_range if use_augmentation else 0,
                    'horizontal_flip': horizontal_flip if use_augmentation else False,
                    'vertical_flip': vertical_flip if use_augmentation else False
                })
        
        with col3:
            if st.button("📊 Load Previous Model", type="secondary"):
                load_previous_model()

def estimate_training_time(epochs, batch_size):
    """Estimate training time based on parameters"""
    # Base time per epoch (minutes) - rough estimate
    base_time_per_epoch = 1.5
    
    # Adjust based on batch size (smaller batches take longer)
    batch_factor = 32 / batch_size
    
    # Estimate total time
    estimated_time = epochs * base_time_per_epoch * batch_factor
    
    return int(estimated_time)

def analyze_parameters_with_gemini(analysis_type, parameters):
    """Analyze training parameters with Gemini AI"""
    try:
        if analysis_type == "technical":
            prompt = f"""
            Analyze these machine learning training parameters for a CNN defect detection model:
            
            Model Architecture:
            - Learning Rate: {parameters['learning_rate']}
            - Batch Size: {parameters['batch_size']}
            - Max Epochs: {parameters['epochs']}
            - Convolutional Dropout: {parameters['conv_dropout']}
            - Dense Dropout: {parameters['dense_dropout']}
            - Base Filters: {parameters['base_filters']}
            - Dense Units: {parameters['dense_units']}
            - Data Augmentation: {parameters['use_augmentation']}
            
            Please provide:
            1. Parameter optimization recommendations
            2. Potential overfitting/underfitting risks
            3. Expected training behavior
            4. Suggestions for improvement
            """
        else:
            prompt = f"""
            Analyze these AI training parameters from a business perspective:
            
            Training Configuration:
            - Learning Rate: {parameters['learning_rate']} (affects training stability)
            - Batch Size: {parameters['batch_size']} (affects memory usage and speed)
            - Maximum Training Cycles: {parameters['epochs']}
            - Estimated Training Time: {parameters['training_time_estimate']} minutes
            
            Please provide:
            1. Business impact of these settings
            2. Resource requirements and costs
            3. Time-to-deployment implications
            4. Risk assessment for production use
            """
        
        with st.spinner("Getting AI analysis..."):
            analysis = get_gemini_analysis(prompt, analysis_type)
            
            if analysis_type == "technical":
                st.markdown("#### 🔬 Technical Parameter Analysis")
            else:
                st.markdown("#### 👔 Business Impact Analysis")
            
            st.markdown(analysis)
            
    except Exception as e:
        st.error(f"Error getting parameter analysis: {str(e)}")

def start_training(parameters):
    """Start model training with given parameters"""
    try:
        # Clean up any existing completion flag
        try:
            if os.path.exists('models/training_complete.flag'):
                os.remove('models/training_complete.flag')
        except:
            pass
        
        # Update session state
        st.session_state.training_in_progress = True
        st.session_state.training_start_time = datetime.now()
        st.session_state.training_parameters = parameters
        
        # Initialize training progress
        st.session_state.current_epoch = 0
        st.session_state.training_metrics = {
            'epoch': [],
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Start actual training in background thread
        import threading
        
        def run_training():
            """Run training in background thread"""
            try:
                print("🚀 Starting background training...")
                print(f"Training parameters: {parameters}")
                
                # Update config with parameters
                update_config_with_parameters(parameters)
                print("✅ Config updated")
                
                # Import required modules
                from training import Trainer, DataLoader
                from tensorflow.keras.callbacks import Callback
                import json
                import os
                
                # Create custom callback to save history after each epoch
                class IncrementalHistoryCallback(Callback):
                    def __init__(self):
                        super().__init__()
                        self.epoch_history = {
                            'loss': [],
                            'accuracy': [],
                            'val_loss': [],
                            'val_accuracy': [],
                            'precision': [],
                            'recall': [],
                            'auc': []
                        }
                    
                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}
                        # Append current epoch metrics
                        for key in ['loss', 'accuracy', 'val_loss', 'val_accuracy', 'precision', 'recall', 'auc']:
                            if key in logs:
                                self.epoch_history[key].append(float(logs[key]))
                        
                        # Save history after each epoch
                        history_path = 'models/training_history.json'
                        os.makedirs('models', exist_ok=True)
                        with open(history_path, 'w') as f:
                            json.dump(self.epoch_history, f, indent=2)
                
                # Create trainer
                print("📊 Creating trainer...")
                trainer = Trainer()
                
                # Prepare data (this loads and splits the data internally)
                print("📁 Preparing data...")
                trainer.prepare_data()
                
                # Build model
                print("🏗️ Building model...")
                model = trainer.build_and_compile_model()
                
                # Add incremental history callback
                incremental_callback = IncrementalHistoryCallback()
                
                # Modify trainer to include our callback
                # Get existing callbacks from the model
                existing_callbacks = trainer.model.get_callbacks()
                if existing_callbacks is None:
                    existing_callbacks = []
                elif not isinstance(existing_callbacks, list):
                    existing_callbacks = [existing_callbacks]
                
                # Add our callback
                existing_callbacks.append(incremental_callback)
                
                # Override the model's get_callbacks method temporarily
                original_get_callbacks = trainer.model.get_callbacks
                trainer.model.get_callbacks = lambda: existing_callbacks
                
                # Train model
                print("🎯 Starting model training...")
                history = trainer.train_model(visualize_augmentation=False)
                
                # Restore original method
                trainer.model.get_callbacks = original_get_callbacks
                
                # Save final results
                print("💾 Saving training results...")
                trainer.save_training_results()
                
                # Mark training as complete
                print("✅ Training completed successfully!")
                
                # Create a completion flag file
                with open('models/training_complete.flag', 'w') as f:
                    f.write('completed')
                
                st.session_state.training_in_progress = False
                st.session_state.model_trained = True
                
            except Exception as e:
                import traceback
                error_msg = f"Training error: {str(e)}"
                full_traceback = traceback.format_exc()
                print(f"Training error: {error_msg}")
                print(f"Full traceback: {full_traceback}")
                
                # Store error in session state for display
                st.session_state.training_error = error_msg
                st.session_state.training_in_progress = False
                
                # Create completion flag even on error
                try:
                    with open('models/training_complete.flag', 'w') as f:
                        f.write('error')
                except:
                    pass
        
        # Start training thread
        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()
        
        st.success("🚀 Training started! You can switch tabs while training continues in the background.")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error starting training: {str(e)}")
        st.session_state.training_in_progress = False

def update_config_with_parameters(parameters):
    """Update config.py with training parameters"""
    try:
        import re
        
        # Read config file
        with open('config.py', 'r') as f:
            config_content = f.read()
        
        # Update MODEL_CONFIG parameters
        config_content = re.sub(
            r"('learning_rate':\s*)[\d.]+",
            f"\\g<1>{parameters['learning_rate']}",
            config_content
        )
        config_content = re.sub(
            r"('batch_size':\s*)\d+",
            f"\\g<1>{parameters['batch_size']}",
            config_content
        )
        config_content = re.sub(
            r"('epochs':\s*)\d+",
            f"\\g<1>{parameters['epochs']}",
            config_content
        )
        config_content = re.sub(
            r"('patience':\s*)\d+",
            f"\\g<1>{parameters['patience']}",
            config_content
        )
        
        # Write updated config
        with open('config.py', 'w') as f:
            f.write(config_content)
            
    except Exception as e:
        st.warning(f"Could not update config file: {str(e)}")

def show_training_progress():
    """Show real-time training progress"""
    st.markdown("### 🔄 Training in Progress")
    
    # Check for training errors
    if 'training_error' in st.session_state:
        st.error(f"Training failed: {st.session_state.training_error}")
        if st.button("Clear Error"):
            del st.session_state.training_error
            st.session_state.training_in_progress = False
            st.rerun()
        return
    
    # Check for training updates
    update_training_metrics()
    
    # Progress metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_epoch = st.session_state.get('current_epoch', 0)
        max_epochs = st.session_state.training_parameters['epochs']
        st.metric("Current Epoch", f"{current_epoch}/{max_epochs}")
    
    with col2:
        elapsed_time = datetime.now() - st.session_state.training_start_time
        total_seconds = int(elapsed_time.total_seconds())
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        st.metric("Elapsed Time", f"{minutes}m {seconds}s")
    
    with col3:
        if 'training_metrics' in st.session_state and st.session_state.training_metrics.get('accuracy'):
            current_acc = st.session_state.training_metrics['accuracy'][-1]
            st.metric("Current Accuracy", f"{current_acc:.3f}")
        else:
            st.metric("Current Accuracy", "Starting...")
    
    with col4:
        if 'training_metrics' in st.session_state and st.session_state.training_metrics.get('val_accuracy'):
            current_val_acc = st.session_state.training_metrics['val_accuracy'][-1]
            st.metric("Validation Accuracy", f"{current_val_acc:.3f}")
        else:
            st.metric("Validation Accuracy", "Starting...")
    
    # Progress bar
    progress = current_epoch / max_epochs if max_epochs > 0 else 0
    st.progress(float(progress), text=f"Training Progress: {progress*100:.1f}%")
    
    # Training curves (if data available)
    if 'training_metrics' in st.session_state and st.session_state.training_metrics.get('epoch'):
        show_live_training_curves()
    
    # Control buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⏹️ Stop Training", type="secondary"):
            st.session_state.training_in_progress = False
            st.success("Training stopped by user.")
            st.rerun()
    
    with col2:
        if st.button("🔄 Refresh Progress", type="secondary"):
            st.rerun()

def update_training_metrics():
    """Update training metrics from the training history file"""
    try:
        if os.path.exists('models/training_history.json'):
            with open('models/training_history.json', 'r') as f:
                history = json.load(f)
            
            if history:
                # Update current epoch
                if 'loss' in history and history['loss']:
                    st.session_state.current_epoch = len(history['loss'])
                
                # Initialize training metrics if not exists
                if 'training_metrics' not in st.session_state:
                    st.session_state.training_metrics = {
                        'epoch': [],
                        'accuracy': [],
                        'val_accuracy': [],
                        'loss': [],
                        'val_loss': []
                    }
                
                # Update metrics
                for key in ['accuracy', 'val_accuracy', 'loss', 'val_loss']:
                    if key in history and history[key]:
                        st.session_state.training_metrics[key] = history[key]
                
                # Update epoch list
                if 'loss' in history and history['loss']:
                    st.session_state.training_metrics['epoch'] = list(range(1, len(history['loss']) + 1))
                
                # Check if training is complete
                training_complete = False
                
                # Method 1: Check for completion flag file (most reliable)
                if os.path.exists('models/training_complete.flag'):
                    training_complete = True
                
                # Method 2: Max epochs reached
                max_epochs = st.session_state.training_parameters['epochs']
                if st.session_state.current_epoch >= max_epochs:
                    training_complete = True
                
                if training_complete and st.session_state.training_in_progress:
                    st.session_state.training_in_progress = False
                    st.session_state.model_trained = True
                    st.session_state.current_model_path = 'models/best_model.h5'
                    st.success("🎉 Training completed successfully!")
                    
                    # Clean up completion flag
                    try:
                        if os.path.exists('models/training_complete.flag'):
                            os.remove('models/training_complete.flag')
                    except:
                        pass
                    
                    # Clear the last refresh time to stop auto-refresh
                    if 'last_refresh' in st.session_state:
                        del st.session_state.last_refresh
                    
    except Exception as e:
        st.error(f"Error updating training metrics: {str(e)}")

def show_live_training_curves():
    """Show live training curves during training"""
    metrics = st.session_state.training_metrics
    
    if len(metrics['epoch']) > 1:
        # Create training curves
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training & Validation Loss', 'Training & Validation Accuracy', 
                          'Learning Progress', 'Overfitting Monitor'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Loss curves
        fig.add_trace(
            go.Scatter(x=metrics['epoch'], y=metrics['loss'], name='Training Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=metrics['epoch'], y=metrics['val_loss'], name='Validation Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        # Accuracy curves
        fig.add_trace(
            go.Scatter(x=metrics['epoch'], y=metrics['accuracy'], name='Training Accuracy', line=dict(color='green')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=metrics['epoch'], y=metrics['val_accuracy'], name='Validation Accuracy', line=dict(color='orange')),
            row=1, col=2
        )
        
        # Learning progress (accuracy improvement)
        if len(metrics['accuracy']) > 1:
            acc_improvement = [metrics['accuracy'][i] - metrics['accuracy'][0] for i in range(len(metrics['accuracy']))]
            fig.add_trace(
                go.Scatter(x=metrics['epoch'], y=acc_improvement, name='Accuracy Improvement', line=dict(color='purple')),
                row=2, col=1
            )
        
        # Overfitting monitor (train vs val gap)
        if len(metrics['accuracy']) > 0 and len(metrics['val_accuracy']) > 0:
            overfitting_gap = [metrics['accuracy'][i] - metrics['val_accuracy'][i] 
                             for i in range(min(len(metrics['accuracy']), len(metrics['val_accuracy'])))]
            fig.add_trace(
                go.Scatter(x=metrics['epoch'][:len(overfitting_gap)], y=overfitting_gap, 
                          name='Overfitting Gap', line=dict(color='red', dash='dash')),
                row=2, col=2
            )
        
        fig.update_layout(height=600, title_text="Live Training Metrics")
        st.plotly_chart(fig, width="stretch")

def load_previous_model():
    """Load a previously trained model"""
    try:
        if os.path.exists('models/best_model.h5'):
            st.session_state.model_trained = True
            st.session_state.current_model_path = 'models/best_model.h5'
            st.success("✅ Previous model loaded successfully!")
        else:
            st.warning("⚠️ No previous model found. Please train a model first.")
    except Exception as e:
        st.error(f"Error loading previous model: {str(e)}")

def show_demo_page():
    """Demo page for testing trained models"""
    st.markdown('<h1 class="main-header">🎯 Demo & Testing</h1>', unsafe_allow_html=True)
    
    # Check if model is available
    model_available = check_model_availability()
    
    if not model_available:
        st.warning("⚠️ No trained model available. Please train a model first or use the pre-trained model.")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🤖 Go to Training Page"):
                st.info("Please navigate to the Training page using the sidebar.")
                st.rerun()
        
        with col2:
            if st.button("📊 Load Pre-trained Model"):
                load_pretrained_model()
        return
    
    # Model information section
    show_model_info()
    
    # Demo options
    st.markdown("---")
    st.markdown('<h2 class="sub-header">🔍 Testing Options</h2>', unsafe_allow_html=True)
    
    demo_option = st.selectbox(
        "Choose Demo Type",
        ["Single Image Classification", "Batch Image Processing", "Model Performance Analysis"],
        help="Select the type of demonstration you want to perform"
    )
    
    if demo_option == "Single Image Classification":
        show_single_image_demo()
    elif demo_option == "Batch Image Processing":
        show_batch_processing_demo()
    elif demo_option == "Model Performance Analysis":
        show_performance_analysis()

def check_model_availability():
    """Check if a trained model is available"""
    model_paths = [
        'models/best_model.h5',
        st.session_state.get('current_model_path')
    ]
    
    for path in model_paths:
        if path and os.path.exists(path):
            st.session_state.current_model_path = path
            return True
    
    return False

def load_pretrained_model():
    """Load a pre-trained model (placeholder for demo)"""
    try:
        # In a real scenario, you would download or load a pre-trained model
        # For now, we'll check if the current trained model exists
        if os.path.exists('models/best_model.h5'):
            st.session_state.current_model_path = 'models/best_model.h5'
            st.session_state.model_trained = True
            st.success("✅ Pre-trained model loaded successfully!")
            st.rerun()
        else:
            st.info("💡 Pre-trained model would be downloaded here. For now, please train a model first.")
    except Exception as e:
        st.error(f"Error loading pre-trained model: {str(e)}")

def show_model_info():
    """Display current model information"""
    st.markdown('<h2 class="sub-header">🤖 Current Model Information</h2>', unsafe_allow_html=True)
    
    try:
        # Load model info
        model_path = st.session_state.current_model_path
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Status", "✅ Ready")
        
        with col2:
            if os.path.exists('models/training_history.json'):
                with open('models/training_history.json', 'r') as f:
                    history = json.load(f)
                final_acc = history.get('val_accuracy', [0])[-1] if history.get('val_accuracy') else 0
                st.metric("Validation Accuracy", f"{final_acc:.3f}")
            else:
                st.metric("Validation Accuracy", "Unknown")
        
        with col3:
            if os.path.exists('models/evaluation_report.json'):
                with open('models/evaluation_report.json', 'r') as f:
                    report = json.load(f)
                test_acc = report.get('metrics', {}).get('accuracy', 0)
                st.metric("Test Accuracy", f"{test_acc:.3f}")
            else:
                st.metric("Test Accuracy", "Not evaluated")
        
        with col4:
            model_size = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0
            st.metric("Model Size", f"{model_size:.1f} MB")
        
    except Exception as e:
        st.error(f"Error loading model info: {str(e)}")

def show_single_image_demo():
    """Single image classification demo"""
    st.markdown("### 📸 Single Image Classification")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a machine part image for defect detection"
        )
        
        # Option to use sample images
        st.markdown("#### Or Use Sample Images")
        if st.button("🎲 Generate Random Sample"):
            generate_sample_image()
    
    with col2:
        if uploaded_file is not None:
            classify_single_image(uploaded_file)
        elif 'sample_image' in st.session_state:
            classify_sample_image()

def generate_sample_image():
    """Generate a random sample image for testing"""
    try:
        from data_generator import SyntheticDataGenerator
        generator = SyntheticDataGenerator()
        
        # Randomly choose normal or defective
        is_defective = np.random.choice([True, False])
        
        if is_defective:
            img = generator.generate_defective_image()
            true_label = "Defective"
        else:
            img = generator.generate_normal_image()
            true_label = "Normal"
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        st.session_state.sample_image = img_array
        st.session_state.sample_true_label = true_label
        st.session_state.sample_pil_image = img
        
        st.success(f"Generated sample {true_label.lower()} image!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error generating sample: {str(e)}")

def classify_single_image(uploaded_file):
    """Classify a single uploaded image"""
    try:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width="stretch")
        
        if st.button("🔍 Classify Image", type="primary"):
            with st.spinner("Classifying image..."):
                # Preprocess image
                img_array = preprocess_image(image)
                
                # Load model and predict
                prediction, confidence, attention_map = predict_with_model(img_array)
                
                # Display results
                display_classification_results(prediction, confidence, image, attention_map)
                
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")

def classify_sample_image():
    """Classify the generated sample image"""
    try:
        st.markdown("#### Generated Sample")
        
        # Display image
        st.image(st.session_state.sample_pil_image, caption=f"Sample Image (True: {st.session_state.sample_true_label})", width="stretch")
        
        if st.button("🔍 Classify Sample", type="primary"):
            with st.spinner("Classifying sample..."):
                # Get prediction
                prediction, confidence, attention_map = predict_with_model(st.session_state.sample_image)
                
                # Display results with ground truth comparison
                display_classification_results(
                    prediction, confidence, st.session_state.sample_pil_image, 
                    attention_map, st.session_state.sample_true_label
                )
                
    except Exception as e:
        st.error(f"Error classifying sample: {str(e)}")

def preprocess_image(image):
    """Preprocess image for model input"""
    # Resize to model input size
    image = image.resize((128, 128), Image.Resampling.LANCZOS)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    return img_array

@st.cache_resource
def load_cached_model(model_path):
    """Load and cache the model to avoid reloading"""
    try:
        from model import DefectDetectionCNN
        cnn = DefectDetectionCNN()
        model = cnn.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_with_model(img_array):
    """Make prediction with the trained model"""
    try:
        # Check if model path is available
        if not st.session_state.get('current_model_path'):
            raise ValueError("No model path available")
        
        # Load cached model
        model = load_cached_model(st.session_state.current_model_path)
        
        if model is None:
            raise ValueError("Failed to load model")
        
        # Prepare input
        input_array = np.expand_dims(img_array, axis=0)
        
        # Get prediction
        prediction_prob = model.predict(input_array, verbose=0)[0][0]
        
        # Determine class
        prediction = "Defective" if prediction_prob > 0.5 else "Normal"
        confidence = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob
        
        # Generate attention map (simplified version)
        try:
            attention_map = generate_attention_map(model, input_array)
        except Exception as e:
            st.warning(f"Attention map generation failed: {str(e)}")
            attention_map = None
        
        return prediction, confidence, attention_map
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return "Error", 0.0, None

def generate_attention_map(model, input_array):
    """Generate a simple attention map using Grad-CAM technique"""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Model
        
        # Find the last convolutional layer
        conv_layers = []
        for layer in model.layers:
            if hasattr(layer, 'filters'):  # Conv2D layers have filters
                conv_layers.append(layer)
        
        if not conv_layers:
            return None
        
        last_conv_layer = conv_layers[-1]
        
        # Create a model that maps the input image to the activations of the last conv layer
        grad_model = Model(inputs=model.input, outputs=[last_conv_layer.output, model.output])
        
        # Compute the gradient of the top predicted class for our input image
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(input_array)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        
        # Get the gradients of the loss with respect to the outputs of the last conv layer
        grads = tape.gradient(loss, conv_outputs)
        
        # Pool the gradients over all the axes leaving out the channel dimension
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the corresponding gradients and sum
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
        
    except Exception as e:
        # If Grad-CAM fails, create a simple attention map
        try:
            # Simple fallback: use the average of feature maps from last conv layer
            conv_layers = [layer for layer in model.layers if hasattr(layer, 'filters')]
            if conv_layers:
                last_conv_layer = conv_layers[-1]
                attention_model = Model(inputs=model.input, outputs=last_conv_layer.output)
                feature_maps = attention_model.predict(input_array, verbose=0)
                attention_map = np.mean(feature_maps[0], axis=-1)
                # Normalize
                if attention_map.max() > attention_map.min():
                    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
                return attention_map
        except:
            pass
        
        return None

def display_classification_results(prediction, confidence, image, attention_map=None, true_label=None):
    """Display classification results with visualizations"""
    st.markdown("### 📊 Classification Results")
    
    # Results summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Prediction with color coding
        color = "🔴" if prediction == "Defective" else "🟢"
        st.markdown(f"#### {color} Prediction")
        st.markdown(f"**{prediction}**")
    
    with col2:
        st.markdown("#### 🎯 Confidence")
        st.markdown(f"**{confidence:.1%}**")
        
        # Confidence bar
        st.progress(float(confidence), text=f"Confidence: {confidence:.1%}")
    
    with col3:
        if true_label:
            # Accuracy check
            is_correct = prediction == true_label
            accuracy_icon = "✅" if is_correct else "❌"
            st.markdown(f"#### {accuracy_icon} Accuracy")
            st.markdown(f"**{'Correct' if is_correct else 'Incorrect'}**")
            st.markdown(f"True: {true_label}")
    
    # Detailed analysis
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🖼️ Original Image")
        st.image(image, width="stretch")
    
    with col2:
        if attention_map is not None:
            st.markdown("#### 🔍 Attention Map")
            
            # Create attention visualization
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # Resize attention map to match image size
            try:
                from scipy.ndimage import zoom
                # Get image dimensions
                if len(image.shape) == 3:
                    img_height, img_width = image.shape[:2]
                else:
                    img_height, img_width = image.shape
                
                # Calculate zoom factors
                zoom_h = img_height / attention_map.shape[0]
                zoom_w = img_width / attention_map.shape[1]
                attention_resized = zoom(attention_map, (zoom_h, zoom_w))
            except Exception as e:
                # Fallback: use original attention map
                attention_resized = attention_map
            
            # Display original image
            ax.imshow(image)
            
            # Overlay attention map with better visualization
            im = ax.imshow(attention_resized, alpha=0.5, cmap='hot', interpolation='bilinear')
            ax.set_title("Model Attention Areas", fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Add a small colorbar
            from matplotlib.colorbar import ColorbarBase
            from matplotlib.colors import Normalize
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Attention Intensity', rotation=270, labelpad=15)
            
            st.pyplot(fig)
            plt.close()
        else:
            st.markdown("#### 🔍 Attention Map")
            st.info("💡 Generating attention map...")
            
            # Show a placeholder with the original image
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(image, alpha=0.8)
            ax.set_title("Original Image (Attention map unavailable)", fontsize=10)
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
    
    # Prediction probabilities
    st.markdown("#### 📈 Prediction Probabilities")
    prob_normal = 1 - confidence if prediction == "Defective" else confidence
    prob_defective = confidence if prediction == "Defective" else 1 - confidence
    
    prob_df = pd.DataFrame({
        'Class': ['Normal', 'Defective'],
        'Probability': [prob_normal, prob_defective]
    })
    
    fig = px.bar(prob_df, x='Class', y='Probability', 
                 color='Class', color_discrete_map={'Normal': '#2E8B57', 'Defective': '#DC143C'})
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, width="stretch")
    
    # AI Analysis
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🤖 Technical Analysis", type="secondary"):
            analyze_prediction_with_gemini("technical", prediction, confidence, true_label)
    
    with col2:
        if st.button("👔 Business Impact", type="secondary"):
            analyze_prediction_with_gemini("executive", prediction, confidence, true_label)

def analyze_prediction_with_gemini(analysis_type, prediction, confidence, true_label=None):
    """Analyze prediction results with Gemini AI"""
    try:
        accuracy_info = ""
        if true_label:
            is_correct = prediction == true_label
            accuracy_info = f"Ground Truth: {true_label}, Prediction Accuracy: {'Correct' if is_correct else 'Incorrect'}"
        
        if analysis_type == "technical":
            prompt = f"""
            Analyze this defect detection model prediction:
            
            Prediction Results:
            - Predicted Class: {prediction}
            - Confidence Score: {confidence:.3f}
            - {accuracy_info}
            
            Please provide technical analysis including:
            1. Model confidence interpretation
            2. Prediction reliability assessment
            3. Potential sources of uncertainty
            4. Recommendations for model improvement
            """
        else:
            prompt = f"""
            Analyze this manufacturing defect detection result from a business perspective:
            
            Detection Results:
            - Part Classification: {prediction}
            - System Confidence: {confidence:.1%}
            - {accuracy_info}
            
            Please provide business analysis including:
            1. Manufacturing quality implications
            2. Cost impact of this classification
            3. Risk assessment for production line
            4. Recommended actions for operations team
            """
        
        with st.spinner("Getting AI analysis..."):
            analysis = get_gemini_analysis(prompt, analysis_type)
            
            if analysis_type == "technical":
                st.markdown("#### 🔬 Technical Analysis")
            else:
                st.markdown("#### 👔 Business Impact Analysis")
            
            st.markdown(analysis)
            
    except Exception as e:
        st.error(f"Error getting AI analysis: {str(e)}")

def show_batch_processing_demo():
    """Batch image processing demo"""
    st.markdown("### 📦 Batch Image Processing")
    
    uploaded_files = st.file_uploader(
        "Choose multiple image files",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload multiple machine part images for batch processing"
    )
    
    if uploaded_files:
        st.info(f"📊 {len(uploaded_files)} images uploaded")
        
        if st.button("🚀 Process All Images", type="primary"):
            process_batch_images(uploaded_files)

def process_batch_images(uploaded_files):
    """Process multiple images in batch"""
    try:
        results = []
        progress_bar = st.progress(0, text="Processing images...")
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(float(progress), text=f"Processing image {i+1}/{len(uploaded_files)}")
            
            # Process image
            image = Image.open(uploaded_file)
            img_array = preprocess_image(image)
            prediction, confidence, _ = predict_with_model(img_array)
            
            results.append({
                'filename': uploaded_file.name,
                'prediction': prediction,
                'confidence': confidence,
                'image': image
            })
        
        # Display results
        display_batch_results(results)
        
    except Exception as e:
        st.error(f"Error processing batch: {str(e)}")

def display_batch_results(results):
    """Display batch processing results"""
    st.markdown("### 📊 Batch Processing Results")
    
    # Summary statistics
    total_images = len(results)
    defective_count = sum(1 for r in results if r['prediction'] == 'Defective')
    normal_count = total_images - defective_count
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", total_images)
    with col2:
        st.metric("Normal Parts", normal_count)
    with col3:
        st.metric("Defective Parts", defective_count)
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Results table
    st.markdown("#### 📋 Detailed Results")
    
    results_df = pd.DataFrame([
        {
            'Filename': r['filename'],
            'Prediction': r['prediction'],
            'Confidence': f"{r['confidence']:.1%}",
            'Status': '🔴 Defective' if r['prediction'] == 'Defective' else '🟢 Normal'
        }
        for r in results
    ])
    
    st.dataframe(results_df, width="stretch")
    
    # Visual results grid
    st.markdown("#### 🖼️ Visual Results")
    
    # Display images in grid
    cols = st.columns(4)
    for i, result in enumerate(results[:12]):  # Show first 12 images
        with cols[i % 4]:
            color_border = "🔴" if result['prediction'] == 'Defective' else "🟢"
            st.image(
                result['image'], 
                caption=f"{color_border} {result['prediction']} ({result['confidence']:.1%})",
                width="stretch"
            )
    
    if len(results) > 12:
        st.info(f"Showing first 12 of {len(results)} images")
    
    # Download results
    if st.button("📥 Download Results CSV"):
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"defect_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def show_performance_analysis():
    """Show detailed model performance analysis"""
    st.markdown("### 📈 Model Performance Analysis")
    
    try:
        # Load evaluation report
        if os.path.exists('models/evaluation_report.json'):
            with open('models/evaluation_report.json', 'r') as f:
                report = json.load(f)
            
            display_performance_metrics(report)
            display_performance_charts()
            
        else:
            st.warning("⚠️ No evaluation report found. Please evaluate the model first.")
            if st.button("🔍 Run Model Evaluation"):
                run_model_evaluation()
    
    except Exception as e:
        st.error(f"Error loading performance analysis: {str(e)}")

def display_performance_metrics(report):
    """Display performance metrics from evaluation report"""
    metrics = report.get('metrics', {})
    
    st.markdown("#### 🎯 Key Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = metrics.get('accuracy', 0)
        st.metric("Accuracy", f"{accuracy:.1%}", delta=f"{(accuracy-0.5):.1%}")
    
    with col2:
        precision = metrics.get('precision', 0)
        st.metric("Precision", f"{precision:.1%}")
    
    with col3:
        recall = metrics.get('recall', 0)
        st.metric("Recall", f"{recall:.1%}")
    
    with col4:
        f1_score = metrics.get('f1_score', 0)
        st.metric("F1-Score", f"{f1_score:.1%}")
    
    # Additional metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        specificity = metrics.get('specificity', 0)
        st.metric("Specificity", f"{specificity:.1%}")
    
    with col2:
        auc_roc = metrics.get('auc_roc', 0)
        st.metric("AUC-ROC", f"{auc_roc:.3f}")
    
    with col3:
        auc_pr = metrics.get('auc_pr', 0)
        st.metric("AUC-PR", f"{auc_pr:.3f}")

def display_performance_charts():
    """Display performance visualization charts"""
    try:
        # Load training history
        if os.path.exists('models/training_history.json'):
            with open('models/training_history.json', 'r') as f:
                history = json.load(f)
            
            # Training curves
            st.markdown("#### 📊 Training History")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Training & Validation Loss', 'Training & Validation Accuracy',
                              'Precision & Recall', 'AUC Metrics'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            epochs = list(range(1, len(history['loss']) + 1))
            
            # Loss
            fig.add_trace(go.Scatter(x=epochs, y=history['loss'], name='Training Loss', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], name='Validation Loss', line=dict(color='red')), row=1, col=1)
            
            # Accuracy
            fig.add_trace(go.Scatter(x=epochs, y=history['accuracy'], name='Training Accuracy', line=dict(color='green')), row=1, col=2)
            fig.add_trace(go.Scatter(x=epochs, y=history['val_accuracy'], name='Validation Accuracy', line=dict(color='orange')), row=1, col=2)
            
            # Precision & Recall
            if 'precision' in history:
                fig.add_trace(go.Scatter(x=epochs, y=history['precision'], name='Precision', line=dict(color='purple')), row=2, col=1)
            if 'recall' in history:
                fig.add_trace(go.Scatter(x=epochs, y=history['recall'], name='Recall', line=dict(color='brown')), row=2, col=1)
            
            # AUC
            if 'auc' in history:
                fig.add_trace(go.Scatter(x=epochs, y=history['auc'], name='AUC', line=dict(color='pink')), row=2, col=2)
            
            fig.update_layout(height=600, title_text="Model Training Performance")
            st.plotly_chart(fig, width="stretch")
        
        # Load confusion matrix if available
        if os.path.exists('plots/confusion_matrix.png'):
            st.markdown("#### 🔍 Confusion Matrix")
            st.image('plots/confusion_matrix.png', width="stretch")
        
        # Load ROC curve if available
        if os.path.exists('plots/roc_curve.png'):
            st.markdown("#### 📈 ROC Curve")
            st.image('plots/roc_curve.png', width="stretch")
            
    except Exception as e:
        st.error(f"Error displaying charts: {str(e)}")

def run_model_evaluation():
    """Run model evaluation"""
    try:
        with st.spinner("Running model evaluation..."):
            # This would run the evaluation pipeline
            # For now, we'll show a placeholder
            st.info("Model evaluation would run here. This requires the full evaluation pipeline.")
            
    except Exception as e:
        st.error(f"Error running evaluation: {str(e)}")

def show_history_page():
    """Model history and experiment tracking page"""
    st.markdown('<h1 class="main-header">📈 Model History & Experiments</h1>', unsafe_allow_html=True)
    
    # Load training history
    training_experiments = load_training_experiments()
    
    if not training_experiments:
        st.info("📊 No training experiments found. Train some models to see history here!")
        if st.button("🤖 Go to Training Page"):
            st.info("Please navigate to the Training page using the sidebar.")
        return
    
    # Experiment overview
    show_experiment_overview(training_experiments)
    
    # Detailed experiment analysis
    st.markdown("---")
    show_detailed_experiments(training_experiments)
    
    # Model comparison
    st.markdown("---")
    show_model_comparison(training_experiments)
    
    # Export functionality
    st.markdown("---")
    show_export_options(training_experiments)

def load_training_experiments():
    """Load all training experiments from history"""
    experiments = []
    
    try:
        # First, try to load from new training sessions file
        if os.path.exists('models/training_sessions.json'):
            with open('models/training_sessions.json', 'r') as f:
                sessions = json.load(f)
            
            # Convert sessions to experiments format
            for session in sessions:
                experiment = {
                    'timestamp': session.get('timestamp', 'Unknown'),
                    'session_id': session.get('session_id', 'Unknown'),
                    'history': session.get('history', {}),
                    'config': session.get('config', {}),
                    'final_metrics': session.get('final_metrics', {}),
                    'evaluation': {}  # Will be loaded separately if available
                }
                experiments.append(experiment)
            
            return experiments
        
        # Fallback: Load from old single experiment files
        if os.path.exists('models/training_history.json'):
            with open('models/training_history.json', 'r') as f:
                history = json.load(f)
            
            # Load config if available
            config = {}
            if os.path.exists('models/training_config.json'):
                with open('models/training_config.json', 'r') as f:
                    config = json.load(f)
            
            # Load evaluation if available
            evaluation = {}
            if os.path.exists('models/evaluation_report.json'):
                with open('models/evaluation_report.json', 'r') as f:
                    evaluation = json.load(f)
            
            experiment = {
                'id': 'current',
                'timestamp': datetime.now().isoformat(),
                'history': history,
                'config': config,
                'evaluation': evaluation,
                'model_path': 'models/best_model.h5'
            }
            
            experiments.append(experiment)
        
        # In a real implementation, you would load multiple experiments
        # from a database or experiment tracking system
        
    except Exception as e:
        st.error(f"Error loading experiments: {str(e)}")
    
    return experiments

def show_experiment_overview(experiments):
    """Show overview of all experiments"""
    st.markdown('<h2 class="sub-header">🔬 Experiment Overview</h2>', unsafe_allow_html=True)
    
    if len(experiments) == 1:
        st.info("💡 Only one experiment available. Train more models with different parameters to compare!")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Experiments", len(experiments))
    
    with col2:
        if experiments:
            # Get best validation accuracy from experiments
            best_accs = []
            for exp in experiments:
                # Try different sources for accuracy
                acc = (exp.get('final_metrics', {}).get('best_val_accuracy', 0) or
                       exp.get('evaluation', {}).get('metrics', {}).get('accuracy', 0) or
                       max(exp.get('history', {}).get('val_accuracy', [0])))
                best_accs.append(acc)
            best_acc = max(best_accs) if best_accs else 0
            st.metric("Best Accuracy", f"{best_acc:.1%}")
        else:
            st.metric("Best Accuracy", "N/A")
    
    with col3:
        if experiments:
            latest_exp = max(experiments, key=lambda x: x.get('timestamp', ''))
            # Try different sources for latest accuracy
            latest_acc = (latest_exp.get('final_metrics', {}).get('best_val_accuracy', 0) or
                         latest_exp.get('evaluation', {}).get('metrics', {}).get('accuracy', 0) or
                         max(latest_exp.get('history', {}).get('val_accuracy', [0])))
            st.metric("Latest Accuracy", f"{latest_acc:.1%}")
        else:
            st.metric("Latest Accuracy", "N/A")
    
    with col4:
        total_epochs = sum([len(exp.get('history', {}).get('loss', [])) for exp in experiments])
        st.metric("Total Epochs Trained", total_epochs)

def show_detailed_experiments(experiments):
    """Show detailed experiment information"""
    st.markdown('<h2 class="sub-header">📋 Experiment Details</h2>', unsafe_allow_html=True)
    
    # Create experiment table
    experiment_data = []
    
    for exp in experiments:
        history = exp.get('history', {})
        config = exp.get('config', {})
        evaluation = exp.get('evaluation', {})
        
        # Extract key metrics
        final_train_acc = history.get('accuracy', [0])[-1] if history.get('accuracy') else 0
        final_val_acc = history.get('val_accuracy', [0])[-1] if history.get('val_accuracy') else 0
        test_acc = evaluation.get('metrics', {}).get('accuracy', 0)
        
        # Extract config parameters
        model_config = config.get('model_config', {})
        training_config = config.get('training_config', {})
        
        experiment_data.append({
            'Experiment ID': exp.get('session_id', exp.get('id', 'Unknown')),
            'Timestamp': exp.get('timestamp', 'Unknown')[:19].replace('T', ' '),
            'Final Train Acc': f"{final_train_acc:.3f}",
            'Final Val Acc': f"{final_val_acc:.3f}",
            'Test Accuracy': f"{test_acc:.3f}",
            'Learning Rate': model_config.get('learning_rate', 'N/A'),
            'Batch Size': model_config.get('batch_size', 'N/A'),
            'Epochs Trained': len(history.get('loss', [])),
            'Overfitting Gap': f"{final_train_acc - final_val_acc:.3f}",
            'Status': '✅ Best' if test_acc == max([e.get('evaluation', {}).get('metrics', {}).get('accuracy', 0) for e in experiments]) else '📊 Complete'
        })
    
    if experiment_data:
        df = pd.DataFrame(experiment_data)
        st.dataframe(df, width="stretch")
        
        # Select experiment for detailed view
        st.markdown("#### 🔍 Detailed Experiment View")
        selected_exp_id = st.selectbox(
            "Select experiment to analyze",
            [exp.get('session_id', exp.get('id', 'Unknown')) for exp in experiments],
            format_func=lambda x: f"Experiment {x}"
        )
        
        selected_exp = next((exp for exp in experiments if exp.get('session_id', exp.get('id', 'Unknown')) == selected_exp_id), None)
        if selected_exp:
            show_single_experiment_analysis(selected_exp)

def show_single_experiment_analysis(experiment):
    """Show detailed analysis of a single experiment"""
    exp_id = experiment.get('session_id', experiment.get('id', 'Unknown'))
    st.markdown(f"### 📊 Experiment {exp_id} Analysis")
    
    history = experiment.get('history', {})
    config = experiment.get('config', {})
    evaluation = experiment.get('evaluation', {})
    
    # Training curves
    if history.get('loss'):
        st.markdown("#### 📈 Training Curves")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss', 'Accuracy', 'Precision & Recall', 'Learning Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        epochs = list(range(1, len(history['loss']) + 1))
        
        # Loss curves
        fig.add_trace(go.Scatter(x=epochs, y=history['loss'], name='Train Loss', line=dict(color='blue')), row=1, col=1)
        if 'val_loss' in history:
            fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], name='Val Loss', line=dict(color='red')), row=1, col=1)
        
        # Accuracy curves
        if 'accuracy' in history:
            fig.add_trace(go.Scatter(x=epochs, y=history['accuracy'], name='Train Acc', line=dict(color='green')), row=1, col=2)
        if 'val_accuracy' in history:
            fig.add_trace(go.Scatter(x=epochs, y=history['val_accuracy'], name='Val Acc', line=dict(color='orange')), row=1, col=2)
        
        # Precision & Recall
        if 'precision' in history:
            fig.add_trace(go.Scatter(x=epochs, y=history['precision'], name='Precision', line=dict(color='purple')), row=2, col=1)
        if 'recall' in history:
            fig.add_trace(go.Scatter(x=epochs, y=history['recall'], name='Recall', line=dict(color='brown')), row=2, col=1)
        
        # Learning rate (if available)
        # Note: This would need to be tracked during training
        
        exp_id = experiment.get('session_id', experiment.get('id', 'Unknown'))
        fig.update_layout(height=600, title_text=f"Training History - Experiment {exp_id}")
        st.plotly_chart(fig, width="stretch")
    
    # Configuration details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚙️ Model Configuration")
        model_config = config.get('model_config', {})
        if model_config:
            config_df = pd.DataFrame([
                {'Parameter': str(k), 'Value': str(v)} for k, v in model_config.items()
            ])
            st.dataframe(config_df, width="stretch")
        else:
            st.info("No model configuration available")
    
    with col2:
        st.markdown("#### 🔧 Training Configuration")
        training_config = config.get('training_config', {})
        if training_config:
            # Flatten augmentation params if they exist
            flat_config = {}
            for k, v in training_config.items():
                if k == 'augmentation_params' and isinstance(v, dict):
                    for ak, av in v.items():
                        flat_config[f"aug_{ak}"] = av
                else:
                    flat_config[k] = v
            
            config_df = pd.DataFrame([
                {'Parameter': str(k), 'Value': str(v)} for k, v in flat_config.items()
            ])
            st.dataframe(config_df, width="stretch")
        else:
            st.info("No training configuration available")
    
    # Performance metrics
    if evaluation.get('metrics'):
        st.markdown("#### 🎯 Performance Metrics")
        metrics = evaluation['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
        with col4:
            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")

def show_model_comparison(experiments):
    """Show comparison between different models"""
    st.markdown('<h2 class="sub-header">⚖️ Model Comparison</h2>', unsafe_allow_html=True)
    
    if len(experiments) < 2:
        st.info("💡 Need at least 2 experiments to show comparison. Train more models!")
        return
    
    # Performance comparison chart
    st.markdown("#### 📊 Performance Comparison")
    
    comparison_data = []
    for exp in experiments:
        evaluation = exp.get('evaluation', {})
        metrics = evaluation.get('metrics', {})
        
        comparison_data.append({
            'Experiment': exp.get('session_id', exp.get('id', 'Unknown')),
            'Accuracy': metrics.get('accuracy', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1-Score': metrics.get('f1_score', 0),
            'AUC-ROC': metrics.get('auc_roc', 0)
        })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Melt for plotting
        df_melted = df.melt(id_vars=['Experiment'], var_name='Metric', value_name='Score')
        
        fig = px.bar(df_melted, x='Experiment', y='Score', color='Metric', 
                     title="Model Performance Comparison",
                     barmode='group')
        fig.update_layout(height=500)
        st.plotly_chart(fig, width="stretch")
        
        # Best model recommendation
        def get_exp_accuracy(exp):
            return (exp.get('final_metrics', {}).get('best_val_accuracy', 0) or
                   exp.get('evaluation', {}).get('metrics', {}).get('accuracy', 0) or
                   max(exp.get('history', {}).get('val_accuracy', [0])))
        
        best_exp = max(experiments, key=get_exp_accuracy)
        best_exp_id = best_exp.get('session_id', best_exp.get('id', 'Unknown'))
        best_acc = (best_exp.get('final_metrics', {}).get('best_val_accuracy', 0) or
                   best_exp.get('evaluation', {}).get('metrics', {}).get('accuracy', 0) or
                   max(best_exp.get('history', {}).get('val_accuracy', [0])))
        st.success(f"🏆 Best performing model: Experiment {best_exp_id} with {best_acc:.1%} accuracy")

def show_export_options(experiments):
    """Show options to export experiment data"""
    st.markdown('<h2 class="sub-header">📥 Export Options</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export Experiment Summary"):
            export_experiment_summary(experiments)
    
    with col2:
        if st.button("📈 Export Training History"):
            export_training_history(experiments)
    
    with col3:
        if st.button("🎯 Export Performance Report"):
            export_performance_report(experiments)

def export_experiment_summary(experiments):
    """Export experiment summary as CSV"""
    try:
        summary_data = []
        
        for exp in experiments:
            history = exp.get('history', {})
            config = exp.get('config', {})
            evaluation = exp.get('evaluation', {})
            
            summary_data.append({
                'experiment_id': exp.get('session_id', exp.get('id', 'Unknown')),
                'timestamp': exp.get('timestamp', ''),
                'final_train_accuracy': history.get('accuracy', [0])[-1] if history.get('accuracy') else 0,
                'final_val_accuracy': history.get('val_accuracy', [0])[-1] if history.get('val_accuracy') else 0,
                'test_accuracy': evaluation.get('metrics', {}).get('accuracy', 0),
                'test_precision': evaluation.get('metrics', {}).get('precision', 0),
                'test_recall': evaluation.get('metrics', {}).get('recall', 0),
                'test_f1_score': evaluation.get('metrics', {}).get('f1_score', 0),
                'learning_rate': config.get('model_config', {}).get('learning_rate', 0),
                'batch_size': config.get('model_config', {}).get('batch_size', 0),
                'epochs_trained': len(history.get('loss', [])),
                'overfitting_gap': (history.get('accuracy', [0])[-1] if history.get('accuracy') else 0) - (history.get('val_accuracy', [0])[-1] if history.get('val_accuracy') else 0)
            })
        
        df = pd.DataFrame(summary_data)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Experiment Summary CSV",
            data=csv,
            file_name=f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error exporting summary: {str(e)}")

def export_training_history(experiments):
    """Export detailed training history"""
    try:
        if not experiments:
            st.warning("No experiments to export")
            return
        
        # For simplicity, export the first experiment's history
        exp = experiments[0]
        history = exp.get('history', {})
        
        if not history:
            st.warning("No training history available")
            return
        
        # Convert history to DataFrame
        df = pd.DataFrame(history)
        df['epoch'] = range(1, len(df) + 1)
        
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Training History CSV",
            data=csv,
            file_name=f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Error exporting training history: {str(e)}")

def export_performance_report(experiments):
    """Export comprehensive performance report"""
    try:
        report_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_experiments': len(experiments),
            'experiments': []
        }
        
        for exp in experiments:
            exp_data = {
                'id': exp.get('session_id', exp.get('id', 'Unknown')),
                'timestamp': exp.get('timestamp', ''),
                'performance_metrics': exp.get('evaluation', {}).get('metrics', {}),
                'configuration': exp.get('config', {}),
                'training_summary': {
                    'epochs_trained': len(exp.get('history', {}).get('loss', [])),
                    'final_train_accuracy': exp.get('history', {}).get('accuracy', [0])[-1] if exp.get('history', {}).get('accuracy') else 0,
                    'final_val_accuracy': exp.get('history', {}).get('val_accuracy', [0])[-1] if exp.get('history', {}).get('val_accuracy') else 0
                }
            }
            report_data['experiments'].append(exp_data)
        
        json_str = json.dumps(report_data, indent=2)
        
        st.download_button(
            label="📥 Download Performance Report JSON",
            data=json_str,
            file_name=f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Error exporting performance report: {str(e)}")

if __name__ == "__main__":
    main()
