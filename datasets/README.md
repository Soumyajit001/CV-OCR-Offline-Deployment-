# Dataset Structure Guide

This directory should contain your training datasets for Part A (Human & Animal Detection).

## Detection Dataset Structure

For training the Faster R-CNN detection model, organize your data as follows:

```
datasets/
├── images/
│   ├── train/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── val/
│   │   ├── img_101.jpg
│   │   └── ...
│   └── test/
│       └── ...
└── annotations/
    ├── train.json  (COCO format)
    ├── val.json    (COCO format)
    └── test.json   (COCO format)
```

### COCO Format Annotation Example

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "img_001.jpg",
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
    {"id": 1, "name": "human"},
    {"id": 2, "name": "animal"}
  ]
}
```

**Note**: 
- `category_id`: 1 = human, 2 = animal
- `bbox`: [x, y, width, height] format (COCO standard)

## Classification Dataset Structure

For training the human/animal classifier, organize cropped images:

```
datasets/
└── classification/
    ├── train/
    │   ├── human/
    │   │   ├── crop_001.jpg
    │   │   └── ...
    │   └── animal/
    │       ├── crop_001.jpg
    │       └── ...
    └── val/
        ├── human/
        └── animal/
```

## Recommended Datasets

- **Open Images Dataset V7**: https://storage.googleapis.com/openimages/web/index.html
- **Pascal VOC** (subset with humans and animals)
- **Custom dataset**: Annotate your own images using tools like LabelImg or CVAT

## Dataset Preparation Tips

1. **Image Quality**: Use high-resolution images (minimum 640x480)
2. **Diversity**: Include various scenarios (indoor, outdoor, different lighting)
3. **Balance**: Maintain roughly equal number of human and animal samples
4. **Validation Split**: Use 80/20 or 70/30 train/val split
5. **Augmentation**: The training scripts include augmentation, but diverse source data is better
