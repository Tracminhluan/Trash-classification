# 🗑️ TrashTerminator.net

A real-time trash classification application powered by machine learning to help identify and sort waste materials correctly.

## 📋 Overview

TrashTerminator.net is an intelligent waste classification system built using **Google's Teachable Machine** for model training and **TensorFlow/Keras** for deployment. The application provides an intuitive interface for classifying trash through uploaded images or live webcam feed, promoting proper waste disposal and environmental awareness.

### Key Features

- **Image Classification**: Upload images of trash items for instant classification
- **Real-time Webcam Detection**: Live classification using your device's camera (~20 FPS)
- **Probability Display**: View confidence scores for all waste categories
- **User-friendly Interface**: Clean, modern GUI built with Tkinter
- **Fast Predictions**: Optimized model inference for quick results

## 🤖 About the Model

The classification model was trained using **Google Teachable Machine**, a web-based tool that makes machine learning accessible without requiring extensive coding knowledge. 

### Model Details

- **Architecture**: Image classification model (224x224 input)
- **Training Platform**: Google Teachable Machine
- **Framework**: TensorFlow/Keras (exported as `.h5` format)
- **Preprocessing**: Images normalized to range [-1, 1] following Teachable Machine standards
- **Output**: Multi-class probability distribution across trash categories

The model identifies various types of waste materials, enabling users to make informed decisions about proper disposal and recycling.

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- Webcam (optional, for real-time detection)

### Required Libraries

```bash
pip install tensorflow
pip install opencv-python
pip install pillow
pip install numpy
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
trash-classifier/
│
├── model/
│   ├── keras_model.h5          # Trained model from Teachable Machine
│   └── labels.txt               # Class labels
│
├── app.py                       # Main application file
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🎯 Usage

### Running the Application

```bash
python app.py
```

### Classification Methods

1. **Image Upload**
   - Click "📁 Open Image" button
   - Select an image file (JPG, JPEG, or PNG)
   - View the prediction and probability scores

2. **Webcam Detection**
   - Click "📷 Start Webcam" to begin live detection
   - Point your camera at the trash item
   - Real-time predictions will appear automatically
   - Click "🛑 Stop Webcam" when finished

## 🔧 Model Configuration

The application expects the following files in the `model/` directory:

- `keras_model.h5`: The trained Teachable Machine model
- `labels.txt`: Text file containing class labels (one per line, format: `index label_name`)

### Training Your Own Model

To train a custom model using Teachable Machine:

1. Visit [Teachable Machine](https://teachablemachine.withgoogle.com/)
2. Create a new **Image Project**
3. Add classes and upload training images for each waste category
4. Train the model
5. Export as **TensorFlow → Keras** format
6. Download and replace files in the `model/` directory

## 🎨 Interface Preview

The application features:
- **Title Bar**: TrashTerminator.net branding
- **Control Buttons**: Image upload and webcam controls
- **Preview Panel**: Displays the current image or webcam feed
- **Prediction Panel**: Shows the classified trash category
- **Probabilities Panel**: Displays confidence scores for all categories

## ⚙️ Technical Details

- **Input Size**: 224x224 pixels (RGB)
- **Normalization**: `(pixel_value / 127.5) - 1.0`
- **Webcam FPS**: ~20 frames per second
- **Display Size**: 350x350 pixels for preview

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Improve the model with additional training data
- Enhance the user interface

## 📝 License

This project is open source and available under the MIT License.

## 🌍 Environmental Impact

By accurately classifying trash, this application helps:
- Reduce contamination in recycling streams
- Promote proper waste disposal habits
- Increase awareness about different waste categories
- Support environmental sustainability efforts

## 🙏 Acknowledgments

- **Google Teachable Machine** for providing an accessible platform for ML model training
- **TensorFlow/Keras** for the deep learning framework
- The open-source community for essential libraries

---

**Made with ♻️ for a cleaner planet**
