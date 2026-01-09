from typing import List, Dict
import io
from PIL import Image
from src.common.config.settings import get_settings
from src.domains.detection.models import YOLOv8DetectionModel

settings = get_settings()

class DetectionService:
    def __init__(self):
        self.model = YOLOv8DetectionModel(
            model_path=settings.detection_model_path,
            confidence_threshold=settings.confidence_threshold
        )
        # Class IDs for YOLOv8 (COCO dataset)
        # 0: person
        # 14: bird
        # 15: cat, 16: dog, 17: horse, 18: sheep, 19: cow
        # 20: elephant, 21: bear, 22: zebra, 23: giraffe
        self.target_classes = {0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}

    async def detect_objects(self, image_bytes: bytes) -> Dict:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Run prediction
        raw_detections = self.model.predict(image)
        
        # Filter for humans and animals
        filtered_detections = []
        has_human = False
        has_animal = False
        
        for det in raw_detections:
            cls_id = det["class_id"]
            if cls_id in self.target_classes:
                filtered_detections.append(det)
                if cls_id == 0:
                    has_human = True
                else:
                    has_animal = True
        
        return {
            "has_human": has_human,
            "has_animal": has_animal,
            "detections": filtered_detections
        }
