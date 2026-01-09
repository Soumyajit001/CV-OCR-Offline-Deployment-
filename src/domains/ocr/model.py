from abc import ABC, abstractmethod
from typing import Any, str
from loguru import logger

class OCRModel(ABC):
    """Abstract Base Class for OCR Models."""
    
    @abstractmethod
    def load_model(self):
        """Load model weights."""
        pass
    
    @abstractmethod
    def extract_text(self, image: Any) -> str:
        """Extract text from an image."""
        pass

class PaddleOCRModel(OCRModel):
    """PaddleOCR Implementation for Industrial Text."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        if self.model is None:
            logger.info(f"Loading OCR model from {self.model_path}")
            # from paddleocr import PaddleOCR
            # self.model = PaddleOCR(use_angle_cls=True, lang='en')
            self.model = "LOADED_MOCK_OCR"

    def extract_text(self, image: Any) -> str:
        if self.model is None:
            self.load_model()
        
        # Mock OCR logic
        return "MOCK_INDUSTRIAL_TEXT_123"
