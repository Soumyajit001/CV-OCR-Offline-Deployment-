"""
Example Usage Script
====================

This script demonstrates how to use the AI Vision System for both Part A and Part B.
"""

from main import ObjectDetector, HumanAnimalClassifier, VideoProcessor, IndustrialOCR
import cv2
import numpy as np
from pathlib import Path

def example_detection():
    """
    Example: Detect humans and animals in an image.
    """
    print("=" * 70)
    print("Example 1: Human & Animal Detection")
    print("=" * 70)
    
    # Initialize models
    print("\nInitializing models...")
    detector = ObjectDetector(model_path=None)  # Uses pretrained
    classifier = HumanAnimalClassifier(model_path=None)  # Uses pretrained
    
    # Load a test image (replace with your image path)
    image_path = "test_image.jpg"
    if not Path(image_path).exists():
        print(f"\nNote: {image_path} not found. Please provide a test image.")
        print("Example usage:")
        print("  image = cv2.imread('your_image.jpg')")
        print("  detections = detector.detect(image)")
        print("  for det in detections:")
        print("      crop = image[y1:y2, x1:x2]")
        print("      class_name, conf = classifier.classify(crop)")
        return
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run detection
    print("\nRunning detection...")
    detections = detector.detect(image_rgb, confidence_threshold=0.5)
    print(f"Found {len(detections)} objects")
    
    # Classify each detection
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = map(int, det['bbox'])
        crop = image_rgb[y1:y2, x1:x2]
        
        if crop.size > 0:
            class_name, confidence = classifier.classify(crop)
            print(f"\nDetection {i+1}:")
            print(f"  Class: {class_name}")
            print(f"  Confidence: {confidence:.2f}")
            print(f"  BBox: [{x1}, {y1}, {x2}, {y2}]")


def example_video_processing():
    """
    Example: Process a video file.
    """
    print("\n" + "=" * 70)
    print("Example 2: Video Processing")
    print("=" * 70)
    
    # Initialize models
    detector = ObjectDetector(model_path=None)
    classifier = HumanAnimalClassifier(model_path=None)
    processor = VideoProcessor(detector, classifier)
    
    # Process video
    input_video = "./test_videos/sample_video.mp4"
    output_video = "./outputs/annotated_sample_video.mp4"
    
    if not Path(input_video).exists():
        print(f"\nNote: {input_video} not found.")
        print("To process videos:")
        print("  1. Place videos in ./test_videos/")
        print("  2. Run: processor.process_all_videos()")
        return
    
    print(f"\nProcessing {input_video}...")
    processor.process_video(input_video, output_video)
    print(f"Output saved to {output_video}")


def example_ocr():
    """
    Example: Extract text from industrial image.
    """
    print("\n" + "=" * 70)
    print("Example 3: Industrial OCR")
    print("=" * 70)
    
    # Initialize OCR
    print("\nInitializing OCR model...")
    ocr = IndustrialOCR()
    
    # Process an image
    image_path = "industrial_text_image.jpg"
    if not Path(image_path).exists():
        print(f"\nNote: {image_path} not found. Please provide a test image.")
        print("Example usage:")
        print("  result = ocr.extract_text('your_image.jpg')")
        print("  print(result['text'])")
        return
    
    print(f"\nProcessing {image_path}...")
    result = ocr.extract_text(image_path)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("\nExtracted Text:")
        print(result.get("text", ""))
        print(f"\nAverage Confidence: {result.get('average_confidence', 0):.2%}")
        
        if "structured_text" in result:
            print("\nStructured Results:")
            for item in result["structured_text"]:
                print(f"  Text: {item['text']}, Confidence: {item['confidence']:.2f}")


def example_batch_ocr():
    """
    Example: Batch process multiple images.
    """
    print("\n" + "=" * 70)
    print("Example 4: Batch OCR Processing")
    print("=" * 70)
    
    ocr = IndustrialOCR()
    
    input_dir = "./test_images"
    output_dir = "./outputs/ocr_results"
    
    if not Path(input_dir).exists():
        print(f"\nNote: {input_dir} not found.")
        print("To batch process:")
        print("  1. Place images in a directory")
        print("  2. Run: ocr.process_directory(input_dir, output_dir)")
        return
    
    print(f"\nProcessing images from {input_dir}...")
    results = ocr.process_directory(input_dir, output_dir)
    print(f"Processed {len(results)} images")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("AI Vision System - Example Usage")
    print("=" * 70)
    
    # Run examples
    example_detection()
    example_video_processing()
    example_ocr()
    example_batch_ocr()
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nFor interactive visualization, run:")
    print("  streamlit run streamlit_app.py")
