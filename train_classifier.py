"""
Training Script for Classification Model (Part A)
==================================================

This script trains the custom CNN classifier to distinguish humans from animals.
It uses cropped detections from the detection model.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Training will continue without logging.")


class CustomCNN(nn.Module):
    """
    Custom CNN for human/animal classification.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ClassificationDataset(Dataset):
    """
    Dataset for classification (human vs animal).
    Expects directory structure:
    dataset/
    ├── human/
    │   ├── img1.jpg
    │   └── ...
    └── animal/
        ├── img1.jpg
        └── ...
    """
    
    def __init__(self, data_dir: str, transforms=None):
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        
        # Load images
        self.images = []
        self.labels = []
        
        # Human images (label 0)
        human_dir = self.data_dir / "human"
        if human_dir.exists():
            for img_path in human_dir.glob("*.jpg"):
                self.images.append(str(img_path))
                self.labels.append(0)
        
        # Animal images (label 1)
        animal_dir = self.data_dir / "animal"
        if animal_dir.exists():
            for img_path in animal_dir.glob("*.jpg"):
                self.images.append(str(img_path))
                self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, label


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Validate model.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def main():
    """
    Main training function.
    """
    # Configuration
    train_dir = "./datasets/classification/train"
    val_dir = "./datasets/classification/val"
    model_save_path = "./models/human_animal_classifier.pth"
    
    num_epochs = 20
    batch_size = 32
    learning_rate = 0.001
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    if WANDB_AVAILABLE:
        wandb.init(
            project="human-animal-detection",
            name="classifier-training",
            config={
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "model": "Custom CNN"
            }
        )
    
    # Data transforms
    train_transforms = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = ClassificationDataset(train_dir, transforms=train_transforms)
    val_dataset = ClassificationDataset(val_dir, transforms=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize model
    print("Initializing model...")
    model = CustomCNN(num_classes=2)
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training loop
    print("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Log to wandb
        if WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": scheduler.get_last_lr()[0]
            })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model (val_acc: {val_acc:.2f}%)")
        
        scheduler.step()
    
    print("\nTraining complete!")
    if WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
