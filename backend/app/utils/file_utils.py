import os
import tempfile
import aiofiles
from typing import BinaryIO
import logging

from ..config import settings

logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure required directories exist"""
    directories = [
        settings.MODEL_CACHE_DIR,
        settings.TEMP_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

async def save_temp_file(file_content: bytes, suffix: str = ".tmp") -> str:
    """Save file content to temporary file and return path"""
    try:
        ensure_directories()
        
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=suffix, 
            dir=settings.TEMP_DIR
        ) as tmp_file:
            tmp_file.write(file_content)
            return tmp_file.name
            
    except Exception as e:
        logger.error(f"Failed to save temporary file: {e}")
        raise

async def cleanup_temp_files():
    """Clean up old temporary files"""
    try:
        if not os.path.exists(settings.TEMP_DIR):
            return
            
        import time
        current_time = time.time()
        
        for filename in os.listdir(settings.TEMP_DIR):
            filepath = os.path.join(settings.TEMP_DIR, filename)
            
            if os.path.isfile(filepath):
                # Delete files older than 1 hour
                if current_time - os.path.getctime(filepath) > 3600:
                    try:
                        os.remove(filepath)
                        logger.info(f"Cleaned up old temp file: {filename}")
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {filename}: {e}")
                        
    except Exception as e:
        logger.error(f"Temp file cleanup failed: {e}")

def get_file_size(filepath: str) -> int:
    """Get file size in bytes"""
    try:
        return os.path.getsize(filepath)
    except OSError:
        return 0