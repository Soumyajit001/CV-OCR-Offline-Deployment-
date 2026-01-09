from fastapi import APIRouter, File, UploadFile, HTTPException

router = APIRouter()

@router.post("/extract")
async def extract_text(file: UploadFile = File(...)):
    """
    Extract text from an industrial image using OCR.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Placeholder for actual OCR logic
    return {
        "filename": file.filename,
        "text": ""
    }
