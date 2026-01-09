import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.common.config.settings import get_settings
from src.domains.detection.routes import router as detection_router
from src.domains.ocr.routes import router as ocr_router

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name}...")
    yield
    logger.info(f"Shutting down {settings.app_name}...")

app = FastAPI(
    title=settings.app_name,
    description="Offline AI Vision Pipeline for Detection and OCR",
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.include_router(detection_router, prefix="/api/v1/detection", tags=["Detection"])
app.include_router(ocr_router, prefix="/api/v1/ocr", tags=["OCR"])

@app.get("/health")
async def health_check():
    logger.debug("Health check requested")
    return {"status": "ok", "app": settings.app_name}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
