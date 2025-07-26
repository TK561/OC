import os
from typing import List
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Depth Estimation API"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = ENVIRONMENT == "development"
    
    # CORS設定 - 全て許可（デバッグ用）
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]
    
    MODEL_CACHE_DIR: str = os.getenv("MODEL_CACHE_DIR", "./models")
    TEMP_DIR: str = os.getenv("TEMP_DIR", "./temp")
    
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
    
    # Memory-optimized model settings for free tier
    LIGHTWEIGHT_MODEL: str = "Intel/dpt-hybrid-midas"  # Smallest DPT model
    DEFAULT_DEPTH_MODEL: str = "Intel/dpt-hybrid-midas"
    DEFAULT_RESOLUTION: int = 384  # Lower resolution for memory efficiency
    MAX_RESOLUTION: int = 512  # Maximum allowed resolution
    
    AVAILABLE_MODELS: List[str] = [
        "Intel/dpt-hybrid-midas",  # Lightweight option first
        "Intel/dpt-large",
        "LiheYoung/depth-anything-large-hf"
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()