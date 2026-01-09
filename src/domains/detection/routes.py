from fastapi import APIRouter, File, UploadFile, HTTPException

from src.domains.detection.services import DetectionService

router = APIRouter()
service = DetectionService()

@router.post("/predict")
async def detect_objects(file: UploadFile = File(...)):
    """
    Detect humans and animals in an image.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        result = await service.detect_objects(contents)
        return {
            "filename": file.filename,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
