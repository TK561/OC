import torch
import numpy as np
from PIL import Image
import io
import logging
import gc
import psutil
from typing import Tuple, Optional
from transformers import DPTImageProcessor, DPTForDepthEstimation, pipeline
import cv2

from ..config import settings

logger = logging.getLogger(__name__)

class DepthEstimator:
    def __init__(self):
        self.models = {}
        self.processors = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self._log_memory_usage("Initializing DepthEstimator")
        
    def _load_model(self, model_name: str):
        """Load depth estimation model and processor with memory optimization"""
        self._log_memory_usage(f"Before loading model: {model_name}")
        
        # Clear cache before loading new model
        self.clear_cache()
        
        try:
            logger.info(f"Loading model: {model_name}")
            
            if "dpt" in model_name.lower() and "depth" not in model_name.lower():
                # Load with maximum memory optimization
                processor = DPTImageProcessor.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    do_rescale=True,
                    size_divisor=32
                )
                model = DPTForDepthEstimation.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    revision="main"
                )
            elif "depth-anything" in model_name.lower() or "depthpro" in model_name.lower() or "zoedepth" in model_name.lower():
                # Use pipeline for DepthAnything, DepthPro, and ZoeDepth with memory optimization
                pipe = pipeline(
                    task="depth-estimation",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self._log_memory_usage(f"After loading pipeline: {model_name}")
                return pipe, None
            else:
                # Fallback to lightweight DPT
                processor = DPTImageProcessor.from_pretrained(
                    settings.LIGHTWEIGHT_MODEL,
                    torch_dtype=torch.float32
                )
                model = DPTForDepthEstimation.from_pretrained(
                    settings.LIGHTWEIGHT_MODEL,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
            
            # Move to device with memory optimization
            with torch.no_grad():
                model.to(self.device)
                model.eval()
            
            self._log_memory_usage(f"After loading model: {model_name}")
            
            # Don't cache models - use them once and dispose
            return model, processor
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            # Fallback to lightweight model
            if model_name != settings.LIGHTWEIGHT_MODEL:
                return self._load_model(settings.LIGHTWEIGHT_MODEL)
            raise
    
    async def predict(
        self, 
        image_data: bytes, 
        model_name: str = None,
        target_resolution: Optional[int] = None
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Predict depth map from image data with memory management
        
        Args:
            image_data: Raw image bytes
            model_name: Name of the model to use
            target_resolution: Target resolution for processing
            
        Returns:
            Tuple of (depth_map_image, original_image)
        """
        model = None
        processor = None
        
        try:
            self._log_memory_usage("Starting prediction")
            
            # Load and preprocess image
            original_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            # Resize for memory efficiency
            if target_resolution:
                original_image = self._resize_image(original_image, target_resolution)
            else:
                # Default to smaller resolution for memory efficiency
                original_image = self._resize_image(original_image, settings.DEFAULT_RESOLUTION)
            
            model_name = model_name or settings.LIGHTWEIGHT_MODEL
            model, processor = self._load_model(model_name)
            
            self._log_memory_usage("Model loaded, starting inference")
            
            # Handle different model types
            if "depth-anything" in model_name.lower() or "depthpro" in model_name.lower() or "zoedepth" in model_name.lower():
                # DepthAnything, DepthPro, and ZoeDepth using pipeline
                result = model(original_image)
                depth_map = result["depth"]
                depth_array = np.array(depth_map)
            else:
                # DPT models
                inputs = processor(images=original_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_depth = outputs.predicted_depth
                
                # Post-process depth map
                depth_array = predicted_depth.squeeze().cpu().numpy()
                
                # Clear inputs from GPU memory immediately
                del inputs, outputs, predicted_depth
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Normalize and convert to image
            depth_map_image = self._depth_to_image(depth_array)
            
            # Clear depth array after conversion
            del depth_array
            gc.collect()
            
            self._log_memory_usage("Prediction completed")
            
            return depth_map_image, original_image
            
        except Exception as e:
            logger.error(f"Depth prediction failed: {str(e)}")
            raise
        finally:
            # Aggressive cleanup
            self._cleanup_memory(model, processor)
    
    def _resize_image(self, image: Image.Image, target_size: int) -> Image.Image:
        """Resize image while maintaining aspect ratio with memory efficiency"""
        w, h = image.size
        
        # Enforce maximum size for memory efficiency
        max_size = min(target_size, settings.MAX_RESOLUTION)
        
        if w > h:
            new_w, new_h = max_size, int(h * max_size / w)
        else:
            new_w, new_h = int(w * max_size / h), max_size
        
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    def _depth_to_image(self, depth_array: np.ndarray) -> Image.Image:
        """Convert depth array to grayscale image (white=近い, black=遠い)"""
        # Normalize depth values
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        
        if depth_max - depth_min > 0:
            depth_normalized = (depth_array - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_array)
        
        # Convert to grayscale (0-255), where higher values (white) represent 近い objects
        depth_grayscale = (depth_normalized * 255).astype(np.uint8)
        
        # Convert single channel to RGB by replicating the channel
        depth_colored = np.stack([depth_grayscale] * 3, axis=-1)
        
        return Image.fromarray(depth_colored)
    
    def get_depth_array(
        self, 
        image_data: bytes, 
        model_name: str = None
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Get raw depth array for 3D processing with memory management
        
        Returns:
            Tuple of (depth_array, original_image)
        """
        model = None
        processor = None
        
        try:
            self._log_memory_usage("Starting depth array extraction")
            
            original_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            # Resize for memory efficiency
            original_image = self._resize_image(original_image, settings.DEFAULT_RESOLUTION)
            
            model_name = model_name or settings.LIGHTWEIGHT_MODEL
            model, processor = self._load_model(model_name)
            
            if "depth-anything" in model_name.lower() or "depthpro" in model_name.lower() or "zoedepth" in model_name.lower():
                result = model(original_image)
                depth_array = np.array(result["depth"])
            else:
                inputs = processor(images=original_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_depth = outputs.predicted_depth
                
                depth_array = predicted_depth.squeeze().cpu().numpy()
                
                # Clear GPU memory immediately
                del inputs, outputs, predicted_depth
            
            return depth_array, original_image
            
        except Exception as e:
            logger.error(f"Depth array extraction failed: {str(e)}")
            raise
        finally:
            self._cleanup_memory(model, processor)
    
    def clear_cache(self):
        """Clear model cache to free memory"""
        self.models.clear()
        self.processors.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Model cache cleared")
    
    def _cleanup_memory(self, model=None, processor=None):
        """Aggressive memory cleanup after model usage"""
        try:
            if model is not None:
                # Clear model from GPU/CPU
                if hasattr(model, 'cpu'):
                    model.cpu()
                if hasattr(model, 'eval'):
                    model.eval()
                # Delete all references
                del model
            
            if processor is not None:
                del processor
            
            # Clear all cached models
            self.models.clear()
            self.processors.clear()
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear any remaining tensors
            for obj in gc.get_objects():
                if torch.is_tensor(obj):
                    del obj
            
            # Force multiple garbage collection passes
            for _ in range(3):
                gc.collect()
            
            self._log_memory_usage("After cleanup")
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
    
    def _log_memory_usage(self, stage: str):
        """Log current memory usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            logger.info(f"Memory usage at {stage}: {memory_mb:.1f} MB")
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")