# AI Vision System: Human & Animal Detection + Industrial OCR

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.7+-red)](https://github.com/PaddlePaddle/PaddleOCR)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction

This project implements a complete offline AI system for two distinct tasks:

**Part A: Human & Animal Detection**
- Two-model architecture: Faster R-CNN for object detection + Custom CNN for classification
- Automatically processes videos from `./test_videos/` and outputs annotated videos to `./outputs/`
- Training pipeline with wandb logging

**Part B: Industrial OCR**
- Offline OCR system for extracting stenciled/painted text from industrial/military boxes
- Handles low contrast, faded paint, and surface damage
- Completely offline operation (no cloud APIs)

## Project Structure

```
project/
├── datasets/              # Training datasets
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── models/                # Trained model weights
│   ├── faster_rcnn_detector.pth
│   └── human_animal_classifier.pth
├── test_videos/           # Input videos for processing
├── outputs/               # Processed videos and OCR results
│   └── ocr_results/
├── main.py                # Main script with both parts
├── train_detection.py     # Training script for detection model
├── train_classifier.py    # Training script for classification model
├── streamlit_app.py       # Visualization app
├── requirements.txt       # Python dependencies
└── README.md
```

## Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd ai-vision
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Models (Optional)**
   - Pre-trained models can be downloaded and placed in `./models/`
   - Or train your own models using the training scripts

## Usage

### Part A: Human & Animal Detection

#### Training

1. **Prepare Dataset**
   - Organize images in `./datasets/images/train/` and `./datasets/images/val/`
   - Create COCO-format annotations in `./datasets/annotations/`
   - Ensure annotations include human (category_id=1) and animal (category_id=2) labels

2. **Train Detection Model**
   ```bash
   python train_detection.py
   ```
   - Model will be saved to `./models/faster_rcnn_detector.pth`
   - Training metrics logged to wandb (if configured)

3. **Train Classification Model**
   ```bash
   python train_classifier.py
   ```
   - Organize cropped images in `./datasets/classification/train/human/` and `./datasets/classification/train/animal/`
   - Model will be saved to `./models/human_animal_classifier.pth`

#### Inference

1. **Place videos in `./test_videos/`**
   ```bash
   # Supported formats: .mp4, .avi, .mov, .mkv
   ```

2. **Run main script**
   ```bash
   python main.py
   ```
   - Videos will be processed automatically
   - Annotated videos saved to `./outputs/`

### Part B: Industrial OCR

#### Using OCR

1. **Single Image**
   ```python
   from main import IndustrialOCR
   
   ocr = IndustrialOCR()
   result = ocr.extract_text("path/to/image.jpg")
   print(result["text"])
   ```

2. **Batch Processing**
   ```python
   ocr = IndustrialOCR()
   results = ocr.process_directory("./test_images", "./outputs/ocr_results")
   ```

### Streamlit Visualization App

Launch the interactive visualization app:

```bash
streamlit run streamlit_app.py
```

The app provides:
- Image upload and detection visualization
- Video processing interface
- OCR results display
- Batch processing capabilities

## Model Selection & Justification

### Part A: Detection System

**Object Detection Model: Faster R-CNN ResNet-50 FPN**
- **Why not YOLO?** Task requirements explicitly exclude YOLO
- **Why Faster R-CNN?**
  - Excellent localization accuracy
  - Better than SSD for small objects (important for animals)
  - Pre-trained on ImageNet/COCO, fine-tunable
  - Works offline with torchvision
  - Good balance between accuracy and speed

**Classification Model: Custom CNN**
- **Architecture**: 3 Conv blocks (32→64→128 channels) + 2 Dense layers
- **Why custom?**
  - Fine-grained control for human vs animal distinction
  - Lightweight for offline deployment
  - Trained on cropped detections for better accuracy
  - Transfer learning from ImageNet features

**Dataset: Open Images Dataset V7 (subset)**
- **Why not COCO/ImageNet?** Task requirements exclude widely used datasets
- **Why Open Images?**
  - High-quality annotations
  - Diverse scenarios (indoor, outdoor, various lighting)
  - Better annotation quality for our use case
  - Free for research use

### Part B: OCR System

**OCR Model: PaddleOCR**
- **Why PaddleOCR?**
  - Completely offline operation
  - Excellent on challenging text (low contrast, faded paint)
  - Handles rotated and distorted text
  - Pre-trained models available
  - Better than Tesseract for industrial text

**Preprocessing Pipeline:**
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Denoising (Non-local means)
- Adaptive thresholding for binarization

## Training Details

### Preprocessing
- **Detection**: Resize to 800x800, normalize, augment (flip, rotation, brightness)
- **Classification**: Resize to 224x224, normalize with ImageNet stats, augment

### Training Steps
1. Load pre-trained Faster R-CNN weights
2. Fine-tune on human/animal dataset
3. Train classifier on cropped detections
4. Use transfer learning for faster convergence

### Metrics Logged (wandb)
- **Detection**: Loss (classification + localization), mAP
- **Classification**: Loss, Accuracy (train/val)
- Learning rate schedule
- Validation metrics

## Challenges & Solutions

### Challenges Faced
1. **Model Selection**: Finding YOLO alternatives that work offline
   - Solution: Faster R-CNN with torchvision

2. **Dataset**: Finding suitable dataset (not COCO/ImageNet)
   - Solution: Open Images Dataset V7 subset

3. **Industrial OCR**: Low contrast text extraction
   - Solution: Preprocessing pipeline (CLAHE, denoising, adaptive threshold)

4. **Offline Deployment**: Ensuring all dependencies work offline
   - Solution: torchvision models, PaddleOCR (offline-first)

### Possible Improvements
1. Model optimization (quantization, pruning, TensorRT/ONNX)
2. Enhanced data augmentation for industrial OCR
3. Better text post-processing and validation
4. Real-time processing optimizations (frame skipping, multi-threading)
5. Comprehensive evaluation on standard benchmarks

## Requirements

See `requirements.txt` for full list. Key dependencies:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- PaddleOCR >= 2.7.0
- OpenCV >= 4.8.0
- Streamlit >= 1.28.0
- wandb >= 0.15.0 (for training)

## License

MIT License

## Author

AI Vision Team
