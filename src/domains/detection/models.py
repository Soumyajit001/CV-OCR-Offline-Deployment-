from abc import ABC, abstractmethod
from typing import Any, List, Dict
from loguru import logger

class DetectionModel(ABC):
    """Abstract Base Class for Detection Models."""
    
    @abstractmethod
    def load_model(self):
        """Load model weights."""
        pass
    
    @abstractmethod
    def predict(self, image: Any) -> List[Dict]:
        """Perform detection on an image."""
        pass

class YOLOv8DetectionModel(DetectionModel):
    """YOLOv8 Implementation for Human & Animal Detection."""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        
    def load_model(self):
        if self.model is None:
            from ultralytics import YOLO
            logger.info(f"Loading YOLOv8 model from {self.model_path}")
            # This will download the model to the current directory if not found, 
            # then cache it. For strict offline, the .pt file must be present.
            self.model = YOLO(self.model_path)

    def predict(self, image: Any) -> List[Dict]:
        if self.model is None:
            self.load_model()
        
        # Run inference
        results = self.model(image, conf=self.confidence_threshold)
        
        detections = []
        for result in results:
            for box in result.boxes:
                if box.conf < self.confidence_threshold:
                    continue
                    
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                
                detections.append({
                    "label": class_name,
                    "class_id": cls_id,
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist()
                })
        return detections
