"""
Annotation Converter: Text (YOLO) to JSON
===============================================

This script converts YOLO format text annotations to JSON format.

YOLO Format:
  class_id x_center y_center width height
  (all values normalized 0-1, center-based coordinates)
 Format:
  {
    "images": [...],
    "annotations": [...],
    "categories": [...]
  }
  (absolute pixel coordinates, top-left corner based)

Usage:
    python convert_annotations.py --input_dir ./datasets --output_dir ./datasets/annotations
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple
import yaml


class YOLOT onverter:
    """
    Converts YOLO format annotations to JSON format.
    """
    
    def __init__(self, class_names: List[str]):
        """
        Initialize converter with class names.
        
        Args:
            class_names: List of class names in order (e.g., ['animal', 'human'])
        """
        self.class_names = class_names
        self.categories = [
            {"id": i + 1, "name": name, "supercategory": "object"}
            for i, name in enumerate(class_names)
        ]
        
        # Map YOLO class_id to category_id
        # YOLO: 0-indexed,  1-indexed (0 is background)
        self.class_id_map = {i: i + 1 for i in range(len(class_names))}
    
    def yolo_to bbox(self, yolo_bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """
        Convert YOLO bbox to bbox format.
        
        YOLO: [x_center, y_center, width, height] (normalized 0-1)
      [x, y, width, height] (absolute pixels, top-left corner)
        
        Args:
            yolo_bbox: [x_center, y_center, width, height] normalized
            img_width: Image width in pixels
            img_height: Image height in pixels
        
        Returns:
            [x, y, width, height] in absolute pixels
        """
        x_center, y_center, width, height = yolo_bbox
        
        # Convert to absolute coordinates
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height
        
        # Convert center-based to top-left corner
        x = x_center_abs - (width_abs / 2)
        y = y_center_abs - (height_abs / 2)
        
        # Ensure bbox is within image bounds
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        width_abs = min(width_abs, img_width - x)
        height_abs = min(height_abs, img_height - y)
        
        return [x, y, width_abs, height_abs]
    
    def read_yolo_annotation(self, txt_path: Path) -> List[Dict]:
        """
        Read YOLO format annotation file.
        
        Args:
            txt_path: Path to YOLO annotation file
        
        Returns:
            List of annotations with class_id and bbox
        """
        annotations = []
        
        if not txt_path.exists():
            return annotations
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) != 5:
                print(f"Warning: Invalid line in {txt_path}: {line}")
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            annotations.append({
                "class_id": class_id,
                "bbox": [x_center, y_center, width, height]
            })
        
        return annotations
    
    def get_image_info(self, image_path: Path) -> Tuple[int, int]:
        """
        Get image dimensions.
        
        Args:
            image_path: Path to image file
        
        Returns:
            (width, height)
        """
        try:
            with Image.open(image_path) as img:
                return img.size  # Returns (width, height)
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            return None, None
    
    def convert_split(self, images_dir: Path, labels_dir: Path, split_name: str) -> Dict:
        """
        Convert annotations for a single split (train/val/test).
        
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO annotation files
            split_name: Name of the split (train/val/test)
        
        Returns:
         format dictionary
        """
     data = {
            "images": [],
            "annotations": [],
            "categories": self.categories
        }
        
        image_id = 1
        annotation_id = 1
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(images_dir.glob(f"*{ext}"))
        
        print(f"\nProcessing {split_name} split...")
        print(f"Found {len(image_files)} images")
        
        for image_path in sorted(image_files):
            # Get image dimensions
            img_width, img_height = self.get_image_info(image_path)
            if img_width is None or img_height is None:
                continue
            
            # Add image info
         data["images"].append({
                "id": image_id,
                "file_name": image_path.name,
                "width": img_width,
                "height": img_height
            })
            
            # Find corresponding annotation file
            label_file = labels_dir / f"{image_path.stem}.txt"
            
            if label_file.exists():
                # Read YOLO annotations
                yolo_annotations = self.read_yolo_annotation(label_file)
                
                for yolo_ann in yolo_annotations:
                    # Convert bbox
                 bbox = self.yolo_to bbox(
                        yolo_ann["bbox"],
                        img_width,
                        img_height
                    )
                    
                    # Map class_id (YOLO 0-indexed -> 1-indexed)
                    category_id = self.class_id_map[yolo_ann["class_id"]]
                    
                    # Calculate area
                    area = bbox[2] * bbox[3]
                    
                    # Add annotation
                 data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0
                    })
                    
                    annotation_id += 1
            else:
                print(f"Warning: No annotation file found for {image_path.name}")
            
            image_id += 1
        
        print(f"  Images: {len data['images'])}")
        print(f"  Annotations: {len data['annotations'])}")
        
        return data
    
    def convert_dataset(self, dataset_root: Path, output_dir: Path):
        """
        Convert entire dataset structure.
        
        Supports multiple structures:
        
        Structure 1 (Roboflow style):
        dataset_root/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── train/
        │   └── labels/
        ├── valid/ or val/
        │   └── labels/
        └── test/
            └── labels/
        
        Structure 2 (Standard):
        dataset_root/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
        
        Args:
            dataset_root: Root directory of dataset
            output_dir: Directory to save JSON files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        splits = []
        
        # Try Structure 1: images/train/ and train/labels/
        images_dir = dataset_root / "images"
        if images_dir.exists():
            # Train split
            train_images = images_dir / "train"
            train_labels = dataset_root / "train" / "labels"
            if train_images.exists() and train_labels.exists():
                splits.append(("train", train_images, train_labels))
            
            # Val split
            val_images = images_dir / "val"
            val_labels = dataset_root / "val" / "labels"
            if val_images.exists() and val_labels.exists():
                splits.append(("val", val_images, val_labels))
            else:
                # Try "valid" instead of "val"
                valid_labels = dataset_root / "valid" / "labels"
                if val_images.exists() and valid_labels.exists():
                    splits.append(("val", val_images, valid_labels))
            
            # Test split
            test_images = images_dir / "test"
            test_labels = dataset_root / "test" / "labels"
            if test_images.exists() and test_labels.exists():
                splits.append(("test", test_images, test_labels))
        
        # Try Structure 2: train/images/ and train/labels/
        if not splits:
            # Train split
            train_images = dataset_root / "train" / "images"
            train_labels = dataset_root / "train" / "labels"
            if train_images.exists() and train_labels.exists():
                splits.append(("train", train_images, train_labels))
            
            # Val split
            val_images = dataset_root / "val" / "images"
            val_labels = dataset_root / "val" / "labels"
            if val_images.exists() and val_labels.exists():
                splits.append(("val", val_images, val_labels))
            else:
                # Try "valid" instead of "val"
                valid_images = dataset_root / "valid" / "images"
                valid_labels = dataset_root / "valid" / "labels"
                if valid_images.exists() and valid_labels.exists():
                    splits.append(("val", valid_images, valid_labels))
            
            # Test split
            test_images = dataset_root / "test" / "images"
            test_labels = dataset_root / "test" / "labels"
            if test_images.exists() and test_labels.exists():
                splits.append(("test", test_images, test_labels))
        
        if not splits:
            print("Error: No valid splits found!")
            print("\nExpected structures:")
            print("  Structure 1 (Roboflow):")
            print("    dataset_root/images/train/ and dataset_root/train/labels/")
            print("  Structure 2 (Standard):")
            print("    dataset_root/train/images/ and dataset_root/train/labels/")
            return
        
        # Convert each split
        for split_name, images_dir, labels_dir in splits:
            if not images_dir.exists():
                print(f"Warning: Images directory not found: {images_dir}")
                continue
            
            if not labels_dir.exists():
                print(f"Warning: Labels directory not found: {labels_dir}")
                continue
            
         data = self.convert_split(images_dir, labels_dir, split_name)
            
            # Save JSON
            output_file = output_dir / f"{split_name}.json"
            with open(output_file, 'w') as f:
                json.dump data, f, indent=2)
            
            print(f"Saved: {output_file}")


def load_class_names_from_yaml(yaml_path: Path) -> List[str]:
    """
    Load class names from YAML file (Roboflow format).
    
    Args:
        yaml_path: Path to data.yaml file
    
    Returns:
        List of class names
    """
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if 'names' in data:
            return data['names']
        elif 'nc' in data:
            # If only number of classes, create generic names
            num_classes = data['nc']
            return [f"class_{i}" for i in range(num_classes)]
    except Exception as e:
        print(f"Error loading YAML: {e}")
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert YOLO format annotations to JSON format"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./datasets",
        help="Root directory of dataset (default: ./datasets)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets/annotations",
        help="Output directory for JSON files (default: ./datasets/annotations)"
    )
    parser.add_argument(
        "--yaml_file",
        type=str,
        default=None,
        help="Path to data.yaml file (for class names). If not provided, will look for data.yaml in input_dir"
    )
    parser.add_argument(
        "--class_names",
        type=str,
        nargs='+',
        default=None,
        help="Class names in order (e.g., --class_names animal human). Overrides YAML file."
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Determine class names
    class_names = None
    
    if args.class_names:
        class_names = args.class_names
    else:
        # Try to load from YAML
        yaml_path = None
        if args.yaml_file:
            yaml_path = Path(args.yaml_file)
        else:
            yaml_path = input_dir / "data.yaml"
        
        if yaml_path.exists():
            print(f"Loading class names from {yaml_path}")
            class_names = load_class_names_from_yaml(yaml_path)
        
        if not class_names:
            # Default class names
            print("Warning: Could not determine class names. Using defaults: ['animal', 'human']")
            class_names = ['animal', 'human']
    
    print(f"Class names: {class_names}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create converter
    converter = YOLOT onverter(class_names)
    
    # Convert dataset
    converter.convert_dataset(input_dir, output_dir)
    
    print("\n" + "=" * 70)
    print("Conversion complete!")
    print("=" * 70)
    print(f"\ JSON files saved to: {output_dir}")
    print("\nYou can now use these files for training:")
    print("  python train_detection.py")


if __name__ == "__main__":
    main()
