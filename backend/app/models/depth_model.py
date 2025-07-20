import torch
import numpy as np
from PIL import Image
import io
import logging
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
        
    def _load_model(self, model_name: str):
        """Load depth estimation model and processor"""
        if model_name in self.models:
            return self.models[model_name], self.processors[model_name]
        
        try:
            logger.info(f"Loading model: {model_name}")
            
            if "dpt" in model_name.lower():
                processor = DPTImageProcessor.from_pretrained(model_name)
                model = DPTForDepthEstimation.from_pretrained(model_name)
            elif "depth-anything" in model_name.lower():
                # Use pipeline for DepthAnything
                pipe = pipeline(
                    task="depth-estimation",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                self.models[model_name] = pipe
                self.processors[model_name] = None
                return pipe, None
            else:
                # Fallback to DPT
                processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
                model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
            
            model.to(self.device)
            model.eval()
            
            self.models[model_name] = model
            self.processors[model_name] = processor
            
            logger.info(f"Successfully loaded model: {model_name}")
            return model, processor
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            # Fallback to default model
            if model_name != settings.DEFAULT_DEPTH_MODEL:
                return self._load_model(settings.DEFAULT_DEPTH_MODEL)
            raise
    
    async def predict(
        self, 
        image_data: bytes, 
        model_name: str = None,
        target_resolution: Optional[int] = None
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Predict depth map from image data
        
        Args:
            image_data: Raw image bytes
            model_name: Name of the model to use
            target_resolution: Target resolution for processing
            
        Returns:
            Tuple of (depth_map_image, original_image)
        """
        try:
            # Load and preprocess image
            original_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            # Resize if target resolution is specified
            if target_resolution:
                original_image = self._resize_image(original_image, target_resolution)
            
            model_name = model_name or settings.DEFAULT_DEPTH_MODEL
            model, processor = self._load_model(model_name)
            
            # Handle different model types
            if "depth-anything" in model_name.lower():
                # DepthAnything using pipeline
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
            
            # Normalize and convert to image
            depth_map_image = self._depth_to_image(depth_array)
            
            return depth_map_image, original_image
            
        except Exception as e:
            logger.error(f"Depth prediction failed: {str(e)}")
            raise
    
    def _resize_image(self, image: Image.Image, target_size: int) -> Image.Image:
        """Resize image while maintaining aspect ratio"""
        w, h = image.size
        if w > h:
            new_w, new_h = target_size, int(h * target_size / w)
        else:
            new_w, new_h = int(w * target_size / h), target_size
        
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    def _depth_to_image(self, depth_array: np.ndarray) -> Image.Image:
        """Convert depth array to colorized image"""
        # Normalize depth values
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        
        if depth_max - depth_min > 0:
            depth_normalized = (depth_array - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth_array)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(
            (depth_normalized * 255).astype(np.uint8), 
            cv2.COLORMAP_VIRIDIS
        )
        
        # Convert BGR to RGB
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(depth_colored)
    
    def get_depth_array(
        self, 
        image_data: bytes, 
        model_name: str = None
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Get raw depth array for 3D processing
        
        Returns:
            Tuple of (depth_array, original_image)
        """
        try:
            original_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            model_name = model_name or settings.DEFAULT_DEPTH_MODEL
            model, processor = self._load_model(model_name)
            
            if "depth-anything" in model_name.lower():
                result = model(original_image)
                depth_array = np.array(result["depth"])
            else:
                inputs = processor(images=original_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_depth = outputs.predicted_depth
                
                depth_array = predicted_depth.squeeze().cpu().numpy()
            
            return depth_array, original_image
            
        except Exception as e:
            logger.error(f"Depth array extraction failed: {str(e)}")
            raise
    
    def clear_cache(self):
        """Clear model cache to free memory"""
        self.models.clear()
        self.processors.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model cache cleared")