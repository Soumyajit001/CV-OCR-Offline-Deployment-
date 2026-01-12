# Class Mapping Fix

## Issue
The class mappings were reversed, causing human images to be classified as animals and vice versa.

## Root Cause
The class order in the code didn't match the actual dataset mapping:

**Dataset (data.yaml):**
- YOLO class 0 = animal
- YOLO class 1 = human

**COCO conversion (convert_annotations.py):**
- category_id 1 = animal (from YOLO class 0)
- category_id 2 = human (from YOLO class 1)

**Previous (incorrect) code:**
- Detection model: label 1 → 'human', label 2 → 'animal' ❌
- Training script: category_id 1 → label 1 (human) ❌

**Fixed code:**
- Detection model: label 1 → 'animal', label 2 → 'human' ✅
- Training script: category_id 1 → label 1 (animal) ✅

## Files Fixed

1. **main.py** - ObjectDetector class
   - Changed: `'animal' if label == 1 else 'human'`
   - Updated comment to reflect correct mapping

2. **train_detection.py** - DetectionDataset class
   - Fixed category_id to label mapping
   - Updated comments

## Class Mappings (Correct)

### Detection Model (Faster R-CNN)
- Label 0: Background (filtered out)
- Label 1: Animal
- Label 2: Human

### Classification Model (Custom CNN)
- Label 0: Human
- Label 1: Animal

## Important Notes

⚠️ **If you have already trained models with the old (incorrect) mapping:**
- You will need to retrain the models for the fix to take effect
- The trained model weights contain the wrong class mappings
- Simply updating the code won't fix pre-trained models

### To Retrain:

1. **Detection Model:**
   ```bash
   python train_detection.py
   ```
   This will create a new model with correct mappings.

2. **Classification Model:**
   ```bash
   python train_classifier.py
   ```
   The classifier mapping was already correct, but retraining ensures consistency.

## Verification

After retraining, test with known images:
- Human image should show "human" class
- Animal image should show "animal" class

The fix ensures consistency across:
- YOLO format (data.yaml)
- COCO format (convert_annotations.py)
- Detection model (main.py, train_detection.py)
- Classification model (main.py, train_classifier.py)
