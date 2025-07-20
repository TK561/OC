import os
import hashlib
import json
from typing import Dict, List, Optional
from pathlib import Path
import logging

from ..config import settings

logger = logging.getLogger(__name__)

class ModelManager:
    """Manage model downloads, caching, and metadata"""
    
    def __init__(self):
        self.cache_dir = Path(settings.MODEL_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "model_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load model metadata from cache"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save model metadata to cache"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get model information"""
        model_info = {
            "Intel/dpt-large": {
                "name": "DPT-Large", 
                "size": "1.3GB",
                "description": "High accuracy depth estimation",
                "features": ["High precision", "Good for detailed scenes"],
                "input_size": 384,
                "type": "dpt"
            },
            "Intel/dpt-hybrid-midas": {
                "name": "MiDaS v3.1",
                "size": "470MB", 
                "description": "Fast depth estimation",
                "features": ["Fast processing", "Good balance"],
                "input_size": 384,
                "type": "dpt"
            },
            "LiheYoung/depth-anything-large-hf": {
                "name": "DepthAnything-Large",
                "size": "1.4GB",
                "description": "Universal depth estimation",
                "features": ["High generalization", "Works on any image"],
                "input_size": 518,
                "type": "depth_anything"
            }
        }
        
        return model_info.get(model_name, {
            "name": model_name,
            "size": "Unknown",
            "description": "Custom model",
            "features": [],
            "input_size": 384,
            "type": "unknown"
        })
    
    def is_model_cached(self, model_name: str) -> bool:
        """Check if model is cached locally"""
        return model_name in self.metadata.get("cached_models", {})
    
    def mark_model_cached(self, model_name: str, cache_path: str):
        """Mark model as cached"""
        if "cached_models" not in self.metadata:
            self.metadata["cached_models"] = {}
        
        self.metadata["cached_models"][model_name] = {
            "path": cache_path,
            "cached_at": str(Path(cache_path).stat().st_mtime),
            "info": self.get_model_info(model_name)
        }
        self._save_metadata()
    
    def get_cached_models(self) -> List[str]:
        """Get list of cached model names"""
        return list(self.metadata.get("cached_models", {}).keys())
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear model cache"""
        if model_name:
            # Clear specific model
            if model_name in self.metadata.get("cached_models", {}):
                del self.metadata["cached_models"][model_name]
                self._save_metadata()
                logger.info(f"Cleared cache for model: {model_name}")
        else:
            # Clear all models
            self.metadata["cached_models"] = {}
            self._save_metadata()
            
            # Remove cache directory contents
            for file_path in self.cache_dir.iterdir():
                if file_path.is_file() and file_path != self.metadata_file:
                    try:
                        file_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove {file_path}: {e}")
            
            logger.info("Cleared all model cache")
    
    def get_cache_size(self) -> int:
        """Get total cache size in bytes"""
        total_size = 0
        for file_path in self.cache_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def get_recommended_model(self, use_case: str = "general") -> str:
        """Get recommended model for specific use case"""
        recommendations = {
            "speed": "Intel/dpt-hybrid-midas",
            "accuracy": "Intel/dpt-large", 
            "general": "LiheYoung/depth-anything-large-hf",
            "mobile": "Intel/dpt-hybrid-midas"
        }
        
        return recommendations.get(use_case, settings.DEFAULT_DEPTH_MODEL)
    
    def validate_model_compatibility(self, model_name: str) -> bool:
        """Validate if model is compatible with current system"""
        try:
            model_info = self.get_model_info(model_name)
            
            # Check if model type is supported
            supported_types = ["dpt", "depth_anything"]
            if model_info.get("type") not in supported_types:
                return False
            
            # Check available memory (simplified)
            import psutil
            available_memory = psutil.virtual_memory().available
            
            # Rough memory requirement estimation
            model_size_gb = float(model_info.get("size", "1GB").replace("GB", "").replace("MB", "0.001"))
            required_memory = model_size_gb * 1024 * 1024 * 1024 * 2  # 2x model size
            
            if available_memory < required_memory:
                logger.warning(f"Insufficient memory for model {model_name}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Model compatibility check failed: {e}")
            return False