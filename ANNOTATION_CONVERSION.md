# Annotation Conversion Guide

This guide explains how to convert YOLO format text annotations to COCO JSON format.

## Overview

The `convert_annotations.py` script converts YOLO format annotations (text files) to COCO JSON format, which is required for training the Faster R-CNN detection model.

## YOLO Format

YOLO annotations are stored in `.txt` files with the following format:
```
class_id x_center y_center width height
```

Where:
- `class_id`: Integer class ID (0-indexed)
- `x_center`, `y_center`: Normalized center coordinates (0-1)
- `width`, `height`: Normalized dimensions (0-1)

Example:
```
0 0.33125 0.63515625 0.40703125 0.4640625
1 0.54453125 0.596875 0.546875 0.77265625
```

## COCO Format

COCO annotations are stored in JSON files with the following structure:
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 150, 200, 300],
      "area": 60000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "animal"},
    {"id": 2, "name": "human"}
  ]
}
```

Where:
- `bbox`: [x, y, width, height] in absolute pixels (top-left corner)
- `category_id`: 1-indexed (1=animal, 2=human in our case)

## Usage

### Basic Usage

```bash
python convert_annotations.py --input_dir ./datasets --output_dir ./datasets/annotations
```

### With Custom Class Names

```bash
python convert_annotations.py \
  --input_dir ./datasets \
  --output_dir ./datasets/annotations \
  --class_names animal human
```

### With Custom YAML File

```bash
python convert_annotations.py \
  --input_dir ./datasets \
  --output_dir ./datasets/annotations \
  --yaml_file ./custom_data.yaml
```

## Supported Directory Structures

The script supports two common directory structures:

### Structure 1: Roboflow Style
```
datasets/
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
```

### Structure 2: Standard
```
datasets/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Class Name Mapping

The script automatically loads class names from `data.yaml` if present. The YAML file should have:
```yaml
nc: 2
names: ['animal', 'human']
```

**Important**: 
- YOLO uses 0-indexed class IDs (0=animal, 1=human)
- COCO uses 1-indexed category IDs (1=animal, 2=human)
- The script automatically handles this conversion

## Output

The script generates three JSON files:
- `train.json`: Training set annotations
- `val.json`: Validation set annotations  
- `test.json`: Test set annotations

These files are saved to the specified output directory (default: `./datasets/annotations/`).

## Example Output

After running the conversion, you'll see:
```
Processing train split...
Found 55 images
  Images: 55
  Annotations: 108
Saved: datasets/annotations/train.json

Processing val split...
Found 16 images
  Images: 16
  Annotations: 32
Saved: datasets/annotations/val.json

Processing test split...
Found 7 images
  Images: 7
  Annotations: 16
Saved: datasets/annotations/test.json
```

## Next Steps

After conversion, you can use the COCO JSON files for training:

```bash
python train_detection.py
```

The training script will automatically load annotations from `./datasets/annotations/train.json` and `./datasets/annotations/val.json`.

## Troubleshooting

### Error: "No valid splits found!"
- Check that your directory structure matches one of the supported formats
- Ensure images and labels directories exist for at least one split

### Error: "Could not determine class names"
- Provide class names via `--class_names` argument
- Or ensure `data.yaml` exists in the input directory

### Warning: "No annotation file found for..."
- Some images may not have corresponding annotation files
- These images will be included in the JSON but without annotations

## Notes

- The script automatically reads image dimensions from image files
- Bounding boxes are converted from normalized center-based (YOLO) to absolute top-left corner (COCO)
- Empty annotation files are handled gracefully
- The script preserves image file names exactly as they appear
