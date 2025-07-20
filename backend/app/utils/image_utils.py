import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
from typing import Optional, Tuple
import logging

from ..config import settings

logger = logging.getLogger(__name__)

def validate_image(file) -> bool:
    """Validate uploaded image file"""
    try:
        if not file.content_type in settings.ALLOWED_IMAGE_TYPES:
            return False
        
        if file.size > settings.MAX_FILE_SIZE:
            return False
        
        return True
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        return False

def process_image(image_data: bytes, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """Process and normalize image"""
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        if target_size:
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        raise

def apply_edge_detection(
    image_data: bytes,
    method: str = "canny",
    threshold1: int = 100,
    threshold2: int = 200
) -> Image.Image:
    """Apply edge detection to image"""
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        if method.lower() == "canny":
            edges = cv2.Canny(gray, threshold1, threshold2)
        elif method.lower() == "sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = np.uint8(edges / edges.max() * 255)
        elif method.lower() == "laplacian":
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.uint8(np.absolute(edges))
        else:
            # Default to Canny
            edges = cv2.Canny(gray, threshold1, threshold2)
        
        # Convert back to RGB
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(edges_rgb)
        
    except Exception as e:
        logger.error(f"Edge detection failed: {e}")
        raise

def apply_blur(
    image_data: bytes,
    method: str = "gaussian",
    radius: float = 2.0
) -> Image.Image:
    """Apply blur effect to image"""
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        if method.lower() == "gaussian":
            blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
        elif method.lower() == "box":
            blurred = image.filter(ImageFilter.BoxBlur(radius=radius))
        elif method.lower() == "motion":
            # Convert to OpenCV for motion blur
            image_array = np.array(image)
            kernel_size = int(radius * 2 + 1)
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            blurred_array = cv2.filter2D(image_array, -1, kernel)
            blurred = Image.fromarray(blurred_array)
        else:
            # Default to Gaussian
            blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        return blurred
        
    except Exception as e:
        logger.error(f"Blur processing failed: {e}")
        raise

def apply_color_correction(
    image_data: bytes,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0
) -> Image.Image:
    """Apply color correction to image"""
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Apply brightness
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
        
        # Apply contrast
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
        
        # Apply saturation
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)
        
        return image
        
    except Exception as e:
        logger.error(f"Color correction failed: {e}")
        raise

def apply_histogram_equalization(image_data: bytes) -> Image.Image:
    """Apply histogram equalization to improve contrast"""
    try:
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_array = np.array(image)
        
        # Convert to YUV color space
        yuv = cv2.cvtColor(image_array, cv2.COLOR_RGB2YUV)
        
        # Apply histogram equalization to Y channel
        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
        
        # Convert back to RGB
        rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        
        return Image.fromarray(rgb)
        
    except Exception as e:
        logger.error(f"Histogram equalization failed: {e}")
        raise

def resize_image_smart(
    image: Image.Image,
    target_size: int,
    maintain_aspect: bool = True
) -> Image.Image:
    """Smart image resizing with aspect ratio preservation"""
    try:
        w, h = image.size
        
        if maintain_aspect:
            if w > h:
                new_w, new_h = target_size, int(h * target_size / w)
            else:
                new_w, new_h = int(w * target_size / h), target_size
        else:
            new_w, new_h = target_size, target_size
        
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
    except Exception as e:
        logger.error(f"Image resizing failed: {e}")
        raise

def normalize_depth_map(depth_array: np.ndarray) -> np.ndarray:
    """Normalize depth map values to 0-1 range"""
    try:
        depth_min = depth_array.min()
        depth_max = depth_array.max()
        
        if depth_max - depth_min > 0:
            normalized = (depth_array - depth_min) / (depth_max - depth_min)
        else:
            normalized = np.zeros_like(depth_array)
        
        return normalized
        
    except Exception as e:
        logger.error(f"Depth normalization failed: {e}")
        raise

def apply_colormap(
    depth_array: np.ndarray,
    colormap: str = "viridis"
) -> Image.Image:
    """Apply colormap to depth array"""
    try:
        # Normalize depth
        normalized = normalize_depth_map(depth_array)
        
        # Apply OpenCV colormap
        colormap_dict = {
            "viridis": cv2.COLORMAP_VIRIDIS,
            "plasma": cv2.COLORMAP_PLASMA,
            "hot": cv2.COLORMAP_HOT,
            "cool": cv2.COLORMAP_COOL,
            "jet": cv2.COLORMAP_JET,
            "rainbow": cv2.COLORMAP_RAINBOW
        }
        
        cv_colormap = colormap_dict.get(colormap.lower(), cv2.COLORMAP_VIRIDIS)
        
        # Convert to 8-bit and apply colormap
        depth_8bit = (normalized * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_8bit, cv_colormap)
        
        # Convert BGR to RGB
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(colored_rgb)
        
    except Exception as e:
        logger.error(f"Colormap application failed: {e}")
        raise

def extract_image_metadata(image_data: bytes) -> dict:
    """Extract metadata from image"""
    try:
        image = Image.open(io.BytesIO(image_data))
        
        metadata = {
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.width,
            "height": image.height,
            "has_transparency": image.mode in ("RGBA", "LA") or "transparency" in image.info
        }
        
        # Extract EXIF data if available
        if hasattr(image, '_getexif') and image._getexif():
            metadata["exif"] = dict(image._getexif())
        
        return metadata
        
    except Exception as e:
        logger.error(f"Metadata extraction failed: {e}")
        return {}