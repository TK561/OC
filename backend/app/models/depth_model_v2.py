import torch
import numpy as np
from PIL import Image
import io
import logging
import gc
import psutil
from typing import Tuple, Optional, Dict, Any
from transformers import (
    DPTImageProcessor, 
    DPTForDepthEstimation, 
    pipeline,
    AutoImageProcessor,
    AutoModelForDepthEstimation
)
import cv2
import torch.nn.functional as F

from ..config import settings

logger = logging.getLogger(__name__)

class DepthEstimatorV2:
    """Enhanced depth estimator based on GitHub implementations of DPT, MiDaS, and Depth Anything"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self._log_memory_usage("Initializing DepthEstimatorV2")
        
        # Model-specific configurations based on GitHub research
        self.model_configs = {
            # DPT-Large configuration from isl-org/DPT
            "Intel/dpt-large": {
                "type": "dpt",
                "input_size": 384,
                "align_corners": False,
                "keep_aspect_ratio": True,
                "ensure_multiple_of": 32,
                "resize_method": "bilinear"
            },
            # MiDaS v3.1 configuration from isl-org/MiDaS
            "Intel/dpt-hybrid-midas": {
                "type": "midas",
                "input_size": 384,
                "align_corners": True,
                "keep_aspect_ratio": True,
                "ensure_multiple_of": 32,
                "resize_method": "bilinear"
            },
            # Depth Anything V2 configuration from DepthAnything/Depth-Anything-V2
            "depth-anything/Depth-Anything-V2-Small-hf": {
                "type": "depth_anything_v2",
                "input_size": 518,
                "align_corners": False,
                "keep_aspect_ratio": False,
                "ensure_multiple_of": 14,
                "resize_method": "bilinear"
            },
            "depth-anything/Depth-Anything-V2-Base-hf": {
                "type": "depth_anything_v2",
                "input_size": 518,
                "align_corners": False,
                "keep_aspect_ratio": False,
                "ensure_multiple_of": 14,
                "resize_method": "bilinear"
            },
            "depth-anything/Depth-Anything-V2-Large-hf": {
                "type": "depth_anything_v2",
                "input_size": 518,
                "align_corners": False,
                "keep_aspect_ratio": False,
                "ensure_multiple_of": 14,
                "resize_method": "bilinear"
            }
        }
        
    def _load_model(self, model_name: str) -> Tuple[Any, Any]:
        """Load model with architecture-specific optimizations"""
        self._log_memory_usage(f"Before loading model: {model_name}")
        
        # Clear any existing cache
        self._clear_cache()
        
        try:
            logger.info(f"Loading model: {model_name}")
            config = self.model_configs.get(model_name, {})
            model_type = config.get("type", "unknown")
            
            if model_type == "dpt":
                # DPT-Large implementation based on isl-org/DPT
                processor = DPTImageProcessor.from_pretrained(
                    model_name,
                    do_resize=True,
                    size={"height": config["input_size"], "width": config["input_size"]},
                    keep_aspect_ratio=config["keep_aspect_ratio"],
                    ensure_multiple_of=config["ensure_multiple_of"],
                    do_rescale=True,
                    do_normalize=True
                )
                model = DPTForDepthEstimation.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True
                )
                
            elif model_type == "midas":
                # MiDaS implementation based on isl-org/MiDaS
                processor = DPTImageProcessor.from_pretrained(
                    model_name,
                    do_resize=True,
                    size={"height": config["input_size"], "width": config["input_size"]},
                    keep_aspect_ratio=config["keep_aspect_ratio"],
                    ensure_multiple_of=config["ensure_multiple_of"],
                    do_rescale=True,
                    do_normalize=True
                )
                model = DPTForDepthEstimation.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True
                )
                
            elif model_type == "depth_anything_v2":
                # Depth Anything V2 implementation
                # Use AutoModel for better compatibility
                processor = AutoImageProcessor.from_pretrained(model_name)
                model = AutoModelForDepthEstimation.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True
                )
                
            else:
                # Use pipeline for other models
                pipe = pipeline(
                    task="depth-estimation",
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self._log_memory_usage(f"After loading pipeline: {model_name}")
                return pipe, None
            
            # Move model to device
            model = model.to(self.device)
            model.eval()
            
            self._log_memory_usage(f"After loading model: {model_name}")
            return model, processor
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    async def predict(
        self, 
        image_data: bytes, 
        model_name: str = None,
        target_resolution: Optional[int] = None
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Predict depth map with model-specific processing
        """
        model = None
        processor = None
        
        try:
            self._log_memory_usage("Starting prediction")
            
            # Load image without applying EXIF orientation to prevent unwanted rotation
            original_image = Image.open(io.BytesIO(image_data))
            original_image = original_image.convert("RGB")
            
            # Get model configuration
            model_name = model_name or settings.DEFAULT_DEPTH_MODEL
            config = self.model_configs.get(model_name, {})
            
            # Determine target size
            if target_resolution:
                input_size = min(target_resolution, settings.MAX_RESOLUTION)
            else:
                input_size = config.get("input_size", 384)
            
            # Load model
            model, processor = self._load_model(model_name)
            
            if processor is None:  # Pipeline model
                # Process with pipeline
                result = model(original_image)
                depth_array = np.array(result["depth"])
                
            else:  # Traditional model
                # Prepare image with model-specific preprocessing
                processed_image = self._preprocess_image(
                    original_image, 
                    input_size,
                    config
                )
                
                # Process with model
                inputs = processor(images=processed_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_depth = outputs.predicted_depth
                
                # Post-process based on model type
                depth_array = self._postprocess_depth(
                    predicted_depth,
                    original_image.size,
                    config
                )
                
                # Cleanup
                del inputs, outputs, predicted_depth
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Convert to image (white=near, black=far)
            model_type = config.get("type", "unknown")
            depth_map_image = self._depth_to_image(depth_array, model_type)
            
            # Cleanup
            del depth_array
            gc.collect()
            
            self._log_memory_usage("Prediction completed")
            return depth_map_image, original_image
            
        except Exception as e:
            logger.error(f"Depth prediction failed: {str(e)}")
            raise
        finally:
            self._cleanup_memory(model, processor)
    
    def _preprocess_image(
        self, 
        image: Image.Image, 
        target_size: int,
        config: Dict[str, Any]
    ) -> Image.Image:
        """Model-specific image preprocessing"""
        if config.get("keep_aspect_ratio", True):
            # Resize keeping aspect ratio
            w, h = image.size
            scale = target_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Ensure multiple of N if required
            multiple = config.get("ensure_multiple_of", 1)
            if multiple > 1:
                new_w = int(np.ceil(new_w / multiple) * multiple)
                new_h = int(np.ceil(new_h / multiple) * multiple)
            
            return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            # Direct resize to square
            return image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    def _postprocess_depth(
        self,
        predicted_depth: torch.Tensor,
        original_size: Tuple[int, int],
        config: Dict[str, Any]
    ) -> np.ndarray:
        """Model-specific depth post-processing"""
        # Remove batch dimension
        depth = predicted_depth.squeeze(0)
        
        # Model-specific processing
        model_type = config.get("type", "unknown")
        
        if model_type == "dpt":
            # DPT-Large: Standard interpolation
            depth_resized = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(original_size[1], original_size[0]),  # (H, W)
                mode="bicubic",
                align_corners=False
            )
        elif model_type == "midas":
            # MiDaS: Uses specific interpolation from GitHub
            depth_resized = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(original_size[1], original_size[0]),  # (H, W) 
                mode="bicubic",
                align_corners=False
            )
        else:
            # Default processing
            depth_resized = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(original_size[1], original_size[0]),  # (H, W)
                mode=config.get("resize_method", "bilinear"),
                align_corners=config.get("align_corners", False)
            )
        
        # Convert to numpy
        depth_array = depth_resized.squeeze().cpu().numpy()
        
        return depth_array
    
    def _depth_to_image(self, depth_array: np.ndarray, model_type: str = "unknown") -> Image.Image:
        """Convert depth array to grayscale image (white=near, black=far)"""
        # Normalize depth values
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        
        if depth_max - depth_min > 0:
            # Normalize to 0-1 range
            depth_normalized = (depth_array - depth_min) / (depth_max - depth_min)
            
            # Model-specific processing
            if model_type == "dpt":
                # DPT-Large: Apply gamma correction
                gamma = 0.8
                depth_normalized = np.power(depth_normalized, gamma)
            elif model_type == "midas":
                # MiDaS: Different gamma for MiDaS characteristics
                gamma = 1.0  # Linear for MiDaS
                depth_normalized = np.power(depth_normalized, gamma)
            else:
                # Default processing
                gamma = 0.9
                depth_normalized = np.power(depth_normalized, gamma)
            
            # Model-specific depth interpretation
            # Based on actual output observation:
            # DPT and MiDaS: Both output inverted (near=bright, far=dark) - need inversion
            # Depth Anything: Correct output (near=dark, far=bright) - keep as-is
            logger.info(f"Applying depth normalization for model_type: {model_type}")
            logger.info(f"Before inversion - min: {depth_normalized.min()}, max: {depth_normalized.max()}")
            
            if model_type in ["dpt", "midas"]:
                # Invert for both DPT and MiDaS
                depth_normalized = 1.0 - depth_normalized
                logger.info(f"After inversion - min: {depth_normalized.min()}, max: {depth_normalized.max()}")
        else:
            depth_normalized = np.zeros_like(depth_array)
        
        # Convert to 8-bit grayscale
        depth_grayscale = (depth_normalized * 255).astype(np.uint8)
        
        # Create PIL image for post-processing
        depth_image = Image.fromarray(depth_grayscale, mode='L')
        
        # Model-specific post-processing
        from PIL import ImageFilter, ImageEnhance
        
        if model_type == "dpt":
            # DPT-Large: Strong smoothing
            depth_image = depth_image.filter(ImageFilter.GaussianBlur(radius=2.5))
            enhancer = ImageEnhance.Contrast(depth_image)
            depth_image = enhancer.enhance(1.3)
        elif model_type == "midas":
            # MiDaS: Moderate smoothing
            depth_image = depth_image.filter(ImageFilter.GaussianBlur(radius=1.5))
            enhancer = ImageEnhance.Contrast(depth_image)
            depth_image = enhancer.enhance(1.1)
        else:
            # Default processing
            depth_image = depth_image.filter(ImageFilter.GaussianBlur(radius=2.0))
            enhancer = ImageEnhance.Contrast(depth_image)
            depth_image = enhancer.enhance(1.2)
        
        # Convert back to array and create RGB
        depth_array_final = np.array(depth_image)
        depth_rgb = np.stack([depth_array_final] * 3, axis=-1)
        
        return Image.fromarray(depth_rgb)
    
    def get_depth_array(
        self, 
        image_data: bytes, 
        model_name: str = None
    ) -> Tuple[np.ndarray, Image.Image]:
        """Get raw depth array for 3D processing"""
        model = None
        processor = None
        
        try:
            original_image = Image.open(io.BytesIO(image_data)).convert("RGB")
            
            # Get model configuration
            model_name = model_name or settings.DEFAULT_DEPTH_MODEL
            config = self.model_configs.get(model_name, {})
            input_size = config.get("input_size", 384)
            
            # Load model
            model, processor = self._load_model(model_name)
            
            if processor is None:  # Pipeline model
                result = model(original_image)
                depth_array = np.array(result["depth"])
            else:
                # Preprocess image
                processed_image = self._preprocess_image(
                    original_image,
                    input_size,
                    config
                )
                
                # Process with model
                inputs = processor(images=processed_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    predicted_depth = outputs.predicted_depth
                
                # Post-process
                depth_array = self._postprocess_depth(
                    predicted_depth,
                    original_image.size,
                    config
                )
                
                # Cleanup
                del inputs, outputs, predicted_depth
            
            return depth_array, original_image
            
        except Exception as e:
            logger.error(f"Depth array extraction failed: {str(e)}")
            raise
        finally:
            self._cleanup_memory(model, processor)
    
    def _clear_cache(self):
        """Clear caches"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _cleanup_memory(self, model=None, processor=None):
        """Aggressive memory cleanup"""
        try:
            if model is not None:
                if hasattr(model, 'cpu'):
                    model.cpu()
                del model
            
            if processor is not None:
                del processor
            
            # Clear PyTorch cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
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