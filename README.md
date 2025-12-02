# 🕵️‍♂️ Deepfake Image Detector

A deep learning-powered application that detects deepfake images using Convolutional Neural Networks (CNN). Built with TensorFlow/Keras and deployed via Streamlit for an interactive web interface.

## 🎯 Project Overview

This project implements a binary image classifier to distinguish between real and AI-generated (deepfake) images. Using a custom CNN architecture, the model analyzes facial images and provides real-time predictions with confidence scores.

## ✨ Features

- **CNN-Based Detection**: Custom 3-layer convolutional neural network architecture
- **Real-Time Predictions**: Instant analysis with confidence scores
- **Interactive Web Interface**: User-friendly Streamlit dashboard
- **Image Preprocessing**: Automatic resizing and normalization (128x128 pixels)
- **Performance Metrics**: Processing time tracking for each inference
- **Visual Feedback**: Clear prediction results with probability percentages

## 📁 Project Structure

```
deepfake-detector-Model/
├── V1/
│   ├── MODELdeepfakedetector.h5      # Trained CNN model (~38 MB)
│   ├── deepfake_app.py                # Streamlit web application
│   ├── deepfake_detector_create_model_file.py  # Model training script
│   └── research paper.pdf             # Reference research paper
└── README.md
```

## 🏗️ Model Architecture

### CNN Architecture

```python
Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])
```

### Model Specifications

| Parameter | Value |
|-----------|-------|
| Input Size | 128 x 128 x 3 (RGB) |
| Optimizer | Adam (lr=0.001) |
| Loss Function | Binary Crossentropy |
| Activation (Output) | Sigmoid |
| Dropout Rate | 0.5 |
| Model Size | ~38 MB |

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy tensorflow matplotlib seaborn scikit-learn streamlit pillow
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deepfake-detector-Model.git
cd deepfake-detector-Model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

#### Streamlit Web App

```bash
cd V1
python -m streamlit run deepfake_app.py
```

The app will open in your default browser at `http://localhost:8501`

#### Training a New Model

To train a new model from scratch:

```bash
cd V1
python deepfake_detector_create_model_file.py
```

**Note**: You'll need to update the dataset paths in the script to point to your data directories:
- Set `fake_dir` to your fake images folder
- Set `real_dir` to your real images folder

## 📊 Usage

### Web Interface

1. **Launch the app** using the Streamlit command
2. **Upload an image** (JPG, JPEG, or PNG format)
3. **View results** including:
   - Prediction: Real or Fake
   - Confidence probability
   - Processing time

### Prediction Process

```python
# Image preprocessing
- Resize to 128x128 pixels
- Normalize pixel values (0-1 range)
- Convert RGBA to RGB if needed
- Predict using trained model
```

## 🔬 Model Training Details

### Data Preprocessing

- **Image Size**: 128x128 pixels
- **Normalization**: Pixel values divided by 255.0
- **Train/Test Split**: 80/20 ratio
- **Random State**: 42 (for reproducibility)

### Training Configuration

- **Epochs**: Configurable (default: 2 for quick testing)
- **Validation**: Real-time validation on test set
- **Metrics**: Accuracy and Binary Crossentropy Loss

### Evaluation Metrics

The model tracks:
- Training accuracy and validation accuracy
- Confusion matrix visualization
- Classification report (Precision, Recall, F1-Score)
- Loss curves over epochs

## 🎨 Web Interface Features

The Streamlit app provides:

- **📤 File Upload**: Drag-and-drop or browse for images
- **🖼️ Image Preview**: View uploaded image before prediction
- **📊 Metrics Dashboard**: Three-column layout showing:
  - Prediction result (Real/Fake)
  - Confidence percentage
  - Processing time in seconds
- **🎨 Professional UI**: Clean, intuitive interface design

## 🛠️ Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **Streamlit**: Web application framework
- **NumPy**: Numerical computations
- **Pillow (PIL)**: Image processing
- **Matplotlib**: Visualization (training)
- **Seaborn**: Statistical visualization
- **Scikit-learn**: Model evaluation metrics

## 📈 Model Performance

The model provides:
- **Binary Classification**: Fake (0) vs Real (1)
- **Threshold**: 0.5 for classification
- **Confidence Scores**: Percentage-based probability
- **Fast Inference**: Typically < 1 second per image

## 🔧 Customization

### Modifying the Model

Edit `deepfake_detector_create_model_file.py` to:
- Add more convolutional layers
- Adjust filter sizes and counts
- Change dropout rates
- Experiment with different optimizers
- Increase training epochs

### Updating the App

Modify `deepfake_app.py` to:
- Change the UI layout
- Add more visualization features
- Implement batch processing
- Add result export functionality

## 📝 Dataset Requirements

For training, organize your dataset as:

```
Data/
├── Fake/
│   ├── fake_image_001.jpg
│   ├── fake_image_002.jpg
│   └── ...
└── Real/
    ├── real_image_001.jpg
    ├── real_image_002.jpg
    └── ...
```

## 🔍 How It Works

1. **Upload Image**: User uploads a facial image
2. **Preprocessing**: Image is resized to 128x128 and normalized
3. **CNN Analysis**: Model processes through convolutional layers to extract features
4. **Classification**: Sigmoid activation produces probability score
5. **Result**: Prediction (Fake/Real) with confidence percentage

## 📚 Research

This project is based on deepfake detection research. See `research paper.pdf` for more details on the underlying concepts and methodologies.

## ⚠️ Limitations

- Model is trained on specific datasets and may not generalize to all deepfake types
- Performance depends on image quality and resolution
- Best results with frontal facial images
- May require retraining for new deepfake generation techniques

## 🚀 Future Enhancements

- [ ] Add support for video deepfake detection
- [ ] Implement ensemble model architecture
- [ ] Create REST API for programmatic access
- [ ] Add multi-face detection in single image
- [ ] Implement GAN-based augmentation for training
- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Add explainability features (Grad-CAM visualization)
- [ ] Create Docker container for easy deployment

## 📄 License

This project is available for educational and research purposes.

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/deepfake-detector-Model/issues).

## 🙏 Acknowledgments

- TensorFlow team for the excellent deep learning framework
- Streamlit for the interactive web framework
- Research community for deepfake detection methodologies

## ⭐ Show Your Support

Give a ⭐️ if this project helped you!

---

**Disclaimer**: This tool is for educational and research purposes. Always verify critical information through multiple sources. Deepfake detection is an evolving field, and no detector is 100% accurate.
