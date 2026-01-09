from typing import List
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

load_dotenv(override=True)


class Settings(BaseSettings):
    """Application settings for AI Vision Pipeline."""

    # Application settings
    app_name: str = "ai-vision-pipeline"
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    timezone: str = Field(default="Asia/Kolkata")

    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # Security settings
    secret_key: str = Field(default="unsafe-secret-key-change-me")
    access_token_expire_minutes: int = Field(default=30)

    # CORS settings
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])

    # Inference Settings
    detection_model_path: str = Field(default="models/yolov8n.pt")
    ocr_model_path: str = Field(default="models/ocr_model")
    
    confidence_threshold: float = Field(default=0.35)

    def validate_settings(self):
        """Validate settings after initialization."""
        pass

    model_config = SettingsConfigDict(
        env_file="src/.env",
        case_sensitive=False,
        extra="ignore"
    )


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.validate_settings()
    return _settings

