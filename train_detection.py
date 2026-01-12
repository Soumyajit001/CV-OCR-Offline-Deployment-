"""
Training Script for Object Detection Model (Part A)
====================================================

This script trains the Faster R-CNN model for human and animal detection.
It includes:
- Dataset loading and preprocessing
- Training loop with wandb logging
- Model checkpointing
- Validation metrics
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import json
from pathlib import Path
from PIL import Image
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Training will continue without logging.")


class DetectionDataset(Dataset):
    """
    Dataset class for object detection.
    Expects COCO-format annotations.
    """
    
    def __init__(self, images_dir: str, annotations_file: str, transforms=None):
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        
        # Load annotations (COCO format)
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Create image_id to annotations mapping
        self.image_info = {img['id']: img for img in self.annotations['images']}
        self.annotations_by_image = {}
        
        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []
            self.annotations_by_image[image_id].append(ann)
        
        # Filter: only images with animal (category_id=1) or human (category_id=2) annotations
        self.image_ids = [
            img_id for img_id in self.image_info.keys()
            if img_id in self.annotations_by_image
        ]
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_info[image_id]
        
        # Load image
        image_path = self.images_dir / image_info['file_name']
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
        
        # Get annotations for this image
        anns = self.annotations_by_image[image_id]
        boxes = []
        labels = []
        
        for ann in anns:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            
            # Map to our classes: 1=animal, 2=human
            # From convert_annotations.py: category_id 1=animal, category_id 2=human
            category_id = ann['category_id']
            if category_id == 1:
                labels.append(1)  # animal (from YOLO class 0)
            else:
                labels.append(2)  # human (from YOLO class 1)
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
        }
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, target


def get_model(num_classes=3):
    """
    Initialize Faster R-CNN model.
    num_classes: background + human + animal = 3
    """
    model = fasterrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Replace classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def collate_fn(batch):
    """
    Custom collate function for batching.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Train for one epoch.
    """
    model.train()
    total_loss = 0
    
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def validate(model, data_loader, device):
    """
    Validate model.
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss


def main():
    """
    Main training function.
    """
    # Configuration
    dataset_path = "./datasets"
    train_images = f"{dataset_path}/images/train"
    train_annotations = f"{dataset_path}/annotations/train.json"
    val_images = f"{dataset_path}/images/val"
    val_annotations = f"{dataset_path}/annotations/val.json"
    
    model_save_path = "./models/faster_rcnn_detector.pth"
    num_epochs = 10
    batch_size = 4
    learning_rate = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    if WANDB_AVAILABLE:
        wandb.init(
            project="human-animal-detection",
            name="faster-rcnn-training",
            config={
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "model": "Faster R-CNN ResNet-50 FPN"
            }
        )
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = DetectionDataset(train_images, train_annotations, transforms=T.ToTensor())
    val_dataset = DetectionDataset(val_images, val_annotations, transforms=T.ToTensor())
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2
    )
    
    # Initialize model
    print("Initializing model...")
    model = get_model(num_classes=3)
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}")
        
        # Log to wandb
        if WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": lr_scheduler.get_last_lr()[0]
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        lr_scheduler.step()
    
    print("\nTraining complete!")
    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
