# 🚗 Car Defect Detection System

A comprehensive real-time car defect detection system using YOLOv8 and Streamlit.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This system provides end-to-end car defect detection capabilities:

1. **Custom Model Training**: Train YOLOv8 on your car defect dataset
2. **Real-time Detection**: Detect defects using webcam or uploaded images
3. **Comprehensive Testing**: Test your model on real car images

## ✨ Features

### Core Features
- ✅ **Custom Model Training**: Train on your specific car defect dataset
- ✅ **Real-time Webcam Detection**: Live defect detection with start/stop controls
- ✅ **Image Upload Processing**: Analyze uploaded car images
- ✅ **Batch Processing**: Process multiple images at once
- ✅ **Performance Analytics**: Detailed performance reports

### Advanced Features
- 📊 **Real-time Monitoring**: Live detection results and statistics
- 🔧 **Configurable Settings**: Adjustable parameters for different use cases
- 📈 **Performance Reports**: HTML reports with detailed analytics

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│   Streamlit     │───▶│  YOLO Model     │
│   (Webcam/File) │    │   Interface     │    │  (Inference)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Results UI    │◀───│   Streamlit     │◀───│  Detection      │
│   (Display)     │    │   Session       │    │  Results        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Webcam (for real-time detection)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd MLPROJECT
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset
```bash
python prepare_dataset.py
```

### 4. Train Model
```bash
python train.py
```

## 🚀 Quick Start

### Basic Usage

1. **Start the Streamlit App**:
```bash
streamlit run app.py
```

2. **Choose Detection Mode**:
   - **Real-time Webcam Detection**: Live camera feed with start/stop controls
   - **Upload Image Detection**: Upload car images
   - **Show Sample from Dataset**: View training data

## 📖 Usage Guide

### 1. Model Training

#### Prepare Your Dataset
```bash
python prepare_dataset.py
```
This converts your COCO format dataset to YOLO format.

#### Train the Model
```bash
python train.py
```
The model will be saved in `runs/detect/train*/weights/best.pt`

#### Update App Configuration
Edit `app.py` to use your trained model:
```python
model = YOLO('runs/detect/train*/weights/best.pt')
```

### 2. Real-time Detection

#### Webcam Detection
1. Open the Streamlit app
2. Select "Real-time Webcam Detection"
3. Click "Start Webcam Detection"
4. Point camera at cars
5. View live detection results

#### Image Upload
1. Select "Upload Image Detection"
2. Upload car images
3. Click "Detect Objects"
4. View detection results

### 3. Batch Testing

#### Test Single Image
```bash
python test_model.py --model runs/detect/train/weights/best.pt --image path/to/car.jpg
```

#### Test Multiple Images
```bash
python test_model.py --model runs/detect/train/weights/best.pt --batch path/to/images/ --report
```

## 🧪 Testing

### Model Testing Script

The `test_model.py` script provides comprehensive testing capabilities:

```bash
# Test single image
python test_model.py --model your_model.pt --image car.jpg

# Test batch of images
python test_model.py --model your_model.pt --batch images_folder/

# Generate performance report
python test_model.py --model your_model.pt --batch images_folder/ --report
```

### Test Results

The testing script generates:
- **Annotated Images**: Saved in `test_results/` folder
- **JSON Results**: Detailed detection data in `test_results.json`
- **HTML Report**: Performance report in `performance_report.html`

## 🔧 Troubleshooting

### Common Issues

#### 1. Model Not Detecting Defects
- **Cause**: Using pre-trained model instead of custom model
- **Solution**: Train custom model and update path in `app.py`

#### 2. Training Errors
- **Cause**: Dataset format issues
- **Solution**: Run `prepare_dataset.py` to fix format

#### 3. Webcam Not Working
- **Cause**: Camera access issues
- **Solution**: Check camera permissions and connections

## 📊 Performance Metrics

### Expected Performance
- **Inference Time**: 50-200ms per frame
- **FPS**: 5-20 FPS (depending on hardware)
- **Accuracy**: 70-90% (with good training data)

### Model Classes
- `damage`: Car damages (scratches, dents)
- `headlamp`: Car headlights
- `front_bumper`: Front bumper
- `hood`: Car hood
- `door`: Car doors
- `rear_bumper`: Rear bumper

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [OpenCV](https://opencv.org/)

---

**Happy Car Defect Detection! 🚗✨** 