# CV-OCR-Offline-Deployment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue)](https://github.com/ultralytics/ultralytics)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-2.7+-red)](https://github.com/PaddlePaddle/PaddleOCR)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction

**CV-OCR-Offline-Deployment** is a robust, privacy-centric AI system designed for offline environments. It integrates **Real-time Object Detection** and **Optical Character Recognition (OCR)** into a unified pipeline, capable of running efficiently on edge devices without internet dependency.

The system is built to detect humans and animals in video streams while simultaneously extracting text from scenes, making it suitable for security surveillance, wildlife monitoring, and automated data entry systems in remote locations.

## Key Features

- **Fully Offline Operation**: No cloud dependencies or external API calls required. All processing happens locally.
- **Real-time Detection**: Powered by **YOLOv8 Nano**, ensuring high-speed inference even on CPU-only hardware.
- **High-Accuracy OCR**: Utilizes **PaddleOCR** for robust text extraction across multiple languages and angles.
- **Privacy First**: Data never leaves the deployment environment.
- **Modern API**: Built with **FastAPI** for high performance and easy integration with frontend applications.

## Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) - Modern, fast web framework for building APIs with Python.
- **Object Detection**: [YOLOv8n](https://docs.ultralytics.com/) - State-of-the-art implementation for object detection.
- **OCR Engine**: [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - Awesome multilingual OCR toolkits based on PaddlePaddle.
- **Package Manager**: [PDM](https://pdm.fming.dev/) - A modern Python package and dependency manager.
- **Environment**: Python 3.10+

## Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Soumyajit001/CV-OCR-Offline-Deployment-
    cd CV-OCR-Offline-Deployment-/ai-vision
    ```

2.  **Install Dependencies**
    This project uses PDM for dependency management.
    ```bash
    # Install PDM if not already installed
    pip install pdm

    # Install project dependencies
    pdm install
    ```

3.  **Model Setup**
    Ensure the model weights are present in the `models/` directory.
    - `yolov8n.pt` should be in `models/yolov8n.pt`.

## Usage

Start the FastAPI server:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. You can access the automatic documentation at `http://localhost:8000/docs`.

## Model Rationale

### YOLOv8 Nano (Detection)
We selected **YOLOv8n** for its exceptional balance between speed and accuracy. The "Nano" variant is specifically optimized for edge devices and CPUs, allowing for real-time inference rates that heavier models cannot achieve without GPU acceleration. It comes pre-trained on the COCO dataset, providing robust detection for common classes like 'person', 'cat', 'dog', and 'horse'.

### PaddleOCR (Text Extraction)
**PaddleOCR** was chosen for its lightweight architecture and superior performance on challenging text (curved, rotated, or low-resolution). Unlike Tesseract, which often struggles with scene text, PaddleOCR leverages deep learning to provide reliable extraction in diverse environments, while remaining efficient enough for offline deployment.
