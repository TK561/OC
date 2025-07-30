from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import base64
import io
import os
import math
import logging
from typing import Optional
import requests

app = FastAPI(title="DPT/MiDaS/DepthAnything Lightweight API")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pillow-only implementations (Railway compatible) - 8 models to match frontend
MODEL_CONFIGS = {
    # Traditional models
    "Intel/dpt-hybrid-midas": {
        "name": "MiDaS v3.1 (DPT Hybrid)",
        "type": "pillow_midas",
        "size_mb": 0,
        "input_size": 384,
        "description": "MiDaS v3.1 Pillow implementation"
    },
    "Intel/dpt-large": {
        "name": "DPT-Large",
        "type": "pillow_dpt_large", 
        "size_mb": 0,
        "input_size": 384,
        "description": "DPT-Large Pillow implementation"
    },
    "LiheYoung/depth-anything-large-hf": {
        "name": "DepthAnything v1 Large",
        "type": "pillow_depth_anything_v1",
        "size_mb": 0,
        "input_size": 518,
        "description": "DepthAnything v1 Large Pillow implementation"
    },
    # Latest models (2025)
    "depth-anything/Depth-Anything-V2-Small-hf": {
        "name": "DepthAnything V2 Small",
        "type": "pillow_depth_anything_v2_small",
        "size_mb": 0,
        "input_size": 518,
        "description": "DepthAnything V2 Small Pillow implementation"
    },
    "depth-anything/Depth-Anything-V2-Base-hf": {
        "name": "DepthAnything V2 Base",
        "type": "pillow_depth_anything_v2_base",
        "size_mb": 0,
        "input_size": 518,
        "description": "DepthAnything V2 Base Pillow implementation"
    },
    "depth-anything/Depth-Anything-V2-Large-hf": {
        "name": "DepthAnything V2 Large",
        "type": "pillow_depth_anything_v2_large",
        "size_mb": 0,
        "input_size": 518,
        "description": "DepthAnything V2 Large Pillow implementation"
    },
    # Specialized models
    "apple/DepthPro": {
        "name": "Apple DepthPro",
        "type": "pillow_depthpro",
        "size_mb": 0,
        "input_size": 1024,
        "description": "Apple DepthPro Pillow implementation"
    },
    "Intel/zoedepth-nyu-kitti": {
        "name": "ZoeDepth (NYU+KITTI)",
        "type": "pillow_zoedepth",
        "size_mb": 0,
        "input_size": 384,
        "description": "ZoeDepth Pillow implementation"
    }
}

# Model cache
model_cache = {}

def midas_inspired_depth(image: Image.Image):
    """MiDaS-inspired depth estimation using Pillow only"""
    w, h = image.size
    
    # MiDaS-style preprocessing simulation
    resized = image.resize((256, 256), Image.Resampling.LANCZOS)
    
    # Convert to grayscale for analysis
    gray = resized.convert('L')
    
    # MiDaS-like feature extraction
    # Edge detection (simulates gradient features)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    
    # Blur for depth-like features (simulates CNN pooling)
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=3))
    
    # Sharpen for detail preservation
    sharpened = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    # Combine features (simulates MiDaS feature fusion)
    combined = Image.blend(
        Image.blend(edges, blurred, 0.6),
        sharpened, 0.4
    )
    
    # Enhance contrast (simulates final CNN layers)
    enhanced = ImageOps.autocontrast(combined)
    
    # Apply median filter for noise reduction
    filtered = enhanced.filter(ImageFilter.MedianFilter(size=3))
    
    # Resize back to original size
    result = filtered.resize((w, h), Image.Resampling.LANCZOS)
    
    # Final enhancement
    result = ImageOps.autocontrast(result)
    
    return result

def dpt_inspired_depth(image: Image.Image):
    """DPT-inspired depth estimation using advanced Pillow techniques"""
    w, h = image.size
    
    # Multi-scale analysis (DPT-like)
    scales = [1.0, 0.75, 0.5]
    depth_maps = []
    
    for scale in scales:
        if scale != 1.0:
            new_size = (int(w * scale), int(h * scale))
            scaled_img = image.resize(new_size, Image.Resampling.LANCZOS)
        else:
            scaled_img = image
        
        # Advanced edge detection
        gray = scaled_img.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        
        # Texture analysis
        texture = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        # Combine features
        combined = ImageOps.autocontrast(
            Image.blend(edges, texture, 0.5)
        )
        
        if scale != 1.0:
            combined = combined.resize((w, h), Image.Resampling.LANCZOS)
        
        depth_maps.append(combined)
    
    # Fuse multiple scales (DPT-like fusion)
    result = depth_maps[0]
    for depth_map in depth_maps[1:]:
        result = Image.blend(result, depth_map, 0.3)
    
    # Post-processing
    result = result.filter(ImageFilter.GaussianBlur(radius=1))
    result = ImageOps.autocontrast(result)
    
    return result

def depth_anything_inspired(image: Image.Image):
    """DepthAnything-inspired depth estimation"""
    w, h = image.size
    center_x, center_y = w // 2, h // 2
    
    # Convert to different color spaces for rich features
    lab = image.convert('LAB')
    hsv = image.convert('HSV')
    
    # Extract channels
    l_channel = lab.split()[0]  # Lightness
    h_channel = hsv.split()[0]  # Hue
    s_channel = hsv.split()[1]  # Saturation
    
    # Advanced edge detection on multiple channels
    l_edges = l_channel.filter(ImageFilter.FIND_EDGES)
    h_edges = h_channel.filter(ImageFilter.FIND_EDGES)
    
    # Texture analysis
    l_texture = l_channel.filter(ImageFilter.UnsharpMask(radius=1, percent=200, threshold=2))
    
    # Combine features
    depth_img = Image.new('L', (w, h))
    depth_pixels = depth_img.load()
    
    l_edge_pixels = l_edges.load()
    h_edge_pixels = h_edges.load()
    l_texture_pixels = l_texture.load()
    s_pixels = s_channel.load()
    
    max_distance = math.sqrt(center_x**2 + center_y**2)
    
    for y in range(h):
        for x in range(w):
            # Distance from center
            distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            distance_norm = distance / max_distance
            
            # Feature values
            l_edge_val = l_edge_pixels[x, y] / 255.0
            h_edge_val = h_edge_pixels[x, y] / 255.0
            texture_val = l_texture_pixels[x, y] / 255.0
            saturation_val = s_pixels[x, y] / 255.0
            
            # DepthAnything-inspired combination
            depth_value = (
                0.3 * (1 - distance_norm) +      # Center bias
                0.25 * texture_val +             # Texture information
                0.2 * (1 - l_edge_val) +         # Lightness edges
                0.15 * (1 - h_edge_val) +        # Hue edges  
                0.1 * saturation_val             # Color saturation
            )
            
            depth_pixels[x, y] = min(255, max(0, int(depth_value * 255)))
    
    # Multi-step post-processing
    depth_img = depth_img.filter(ImageFilter.MedianFilter(size=3))
    depth_img = depth_img.filter(ImageFilter.GaussianBlur(radius=1.5))
    depth_img = ImageOps.autocontrast(depth_img)
    
    return depth_img

def depth_anything_v2_small(image: Image.Image):
    """DepthAnything V2 Small - optimized for speed"""
    w, h = image.size
    center_x, center_y = w // 2, h // 2
    
    # Convert to grayscale
    gray = image.convert('L')
    
    # Create depth based on distance from center + texture
    depth_img = Image.new('L', (w, h))
    depth_pixels = depth_img.load()
    gray_pixels = gray.load()
    
    max_distance = math.sqrt(center_x**2 + center_y**2)
    
    for y in range(h):
        for x in range(w):
            # Distance from center (simulates depth)
            distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            distance_norm = distance / max_distance
            
            # Brightness as depth cue
            brightness = gray_pixels[x, y] / 255.0
            
            # Combine distance and brightness
            depth_value = (
                0.6 * (1 - distance_norm) +  # Center is closer
                0.4 * brightness              # Brighter areas closer
            )
            
            depth_pixels[x, y] = min(255, max(0, int(depth_value * 255)))
    
    # Enhance contrast
    result = ImageOps.autocontrast(depth_img)
    
    return result

def depth_anything_v2_base(image: Image.Image):
    """DepthAnything V2 Base - balanced performance"""
    w, h = image.size
    center_x, center_y = w // 2, h // 2
    
    # Convert to LAB for better depth cues
    lab = image.convert('LAB')
    l_channel = lab.split()[0]
    
    # Create depth map
    depth_img = Image.new('L', (w, h))
    depth_pixels = depth_img.load()
    l_pixels = l_channel.load()
    
    max_distance = math.sqrt(center_x**2 + center_y**2)
    
    for y in range(h):
        for x in range(w):
            # Distance from center
            distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            distance_norm = distance / max_distance
            
            # Lightness as depth cue
            lightness = l_pixels[x, y] / 255.0
            
            # Vertical position (top = far, bottom = close)
            vertical_norm = y / h
            
            # Combine multiple depth cues
            depth_value = (
                0.4 * (1 - distance_norm) +  # Center bias
                0.3 * lightness +            # Brighter = closer
                0.3 * (1 - vertical_norm)    # Top = farther
            )
            
            depth_pixels[x, y] = min(255, max(0, int(depth_value * 255)))
    
    # Apply smoothing and contrast enhancement
    result = depth_img.filter(ImageFilter.GaussianBlur(radius=1))
    result = ImageOps.autocontrast(result)
    
    return result

def depth_anything_v2_large(image: Image.Image):
    """DepthAnything V2 Large - highest quality"""
    return depth_anything_inspired(image)  # Use the most advanced implementation

def depthpro_inspired(image: Image.Image):
    """Apple DepthPro-inspired metric depth estimation"""
    w, h = image.size
    
    # DepthPro focuses on metric accuracy
    lab = image.convert('LAB')
    l_channel = lab.split()[0]
    a_channel = lab.split()[1]
    b_channel = lab.split()[2]
    
    # Multi-channel analysis for metric depth
    l_edges = l_channel.filter(ImageFilter.FIND_EDGES)
    l_enhanced = l_channel.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    # Color-based depth cues
    a_processed = ImageOps.autocontrast(a_channel)
    b_processed = ImageOps.autocontrast(b_channel)
    
    # Combine channels for depth information
    depth_img = Image.new('L', (w, h))
    depth_pixels = depth_img.load()
    
    l_edge_pixels = l_edges.load()
    l_enh_pixels = l_enhanced.load()
    a_pixels = a_processed.load()
    b_pixels = b_processed.load()
    
    for y in range(h):
        for x in range(w):
            # Metric depth estimation simulation
            edge_val = l_edge_pixels[x, y] / 255.0
            enh_val = l_enh_pixels[x, y] / 255.0
            a_val = a_pixels[x, y] / 255.0
            b_val = b_pixels[x, y] / 255.0
            
            # DepthPro-style combination
            depth_value = (
                0.4 * (1 - edge_val) +       # Edge information
                0.3 * enh_val +              # Enhanced features
                0.15 * a_val +               # Color channel A
                0.15 * b_val                 # Color channel B
            )
            
            depth_pixels[x, y] = min(255, max(0, int(depth_value * 255)))
    
    # Metric depth post-processing
    depth_img = depth_img.filter(ImageFilter.GaussianBlur(radius=2))
    depth_img = ImageOps.autocontrast(depth_img)
    
    return depth_img

def zoedepth_inspired(image: Image.Image):
    """ZoeDepth-inspired absolute depth estimation"""
    w, h = image.size
    
    # ZoeDepth combines relative and absolute depth
    gray = image.convert('L')
    
    # Multi-scale processing for absolute depth
    scales = [1.0, 0.8, 0.6]
    depth_maps = []
    
    for scale in scales:
        if scale != 1.0:
            new_size = (int(w * scale), int(h * scale))
            scaled_img = gray.resize(new_size, Image.Resampling.LANCZOS)
        else:
            scaled_img = gray
        
        # Process at this scale
        edges = scaled_img.filter(ImageFilter.FIND_EDGES)
        enhanced = scaled_img.filter(ImageFilter.EDGE_ENHANCE)
        combined = Image.blend(edges, enhanced, 0.5)
        
        if scale != 1.0:
            combined = combined.resize((w, h), Image.Resampling.LANCZOS)
        
        depth_maps.append(combined)
    
    # Combine scales for absolute depth
    result = depth_maps[0]
    for depth_map in depth_maps[1:]:
        result = Image.blend(result, depth_map, 0.4)
    
    # ZoeDepth-style post-processing
    result = result.filter(ImageFilter.MedianFilter(size=5))
    result = ImageOps.autocontrast(result)
    
    return result

def apply_grayscale_depth_map(depth_image):
    """深度マップをグレースケール表示（白が近い、黒が遠い）"""
    w, h = depth_image.size
    
    # First, ensure good contrast and normalization
    depth_normalized = ImageOps.autocontrast(depth_image)
    
    # Apply histogram equalization for better distribution
    depth_equalized = ImageOps.equalize(depth_normalized)
    
    # Blend original and equalized for better results
    depth_enhanced = Image.blend(depth_normalized, depth_equalized, 0.3)
    
    # Convert to RGB
    colored_img = Image.new('RGB', (w, h))
    depth_pixels = depth_enhanced.load()
    colored_pixels = colored_img.load()
    
    for y in range(h):
        for x in range(w):
            depth_val = depth_pixels[x, y]
            gray_value = depth_val
            color = (gray_value, gray_value, gray_value)
            colored_pixels[x, y] = color
    
    return colored_img

def generate_pointcloud(original_image, depth_image):
    """3Dポイントクラウドデータ生成"""
    w, h = original_image.size
    downsample_factor = 4
    points = []
    colors = []
    
    orig_pixels = original_image.load()
    depth_pixels = depth_image.load()
    
    for y in range(0, h, downsample_factor):
        for x in range(0, w, downsample_factor):
            depth_val = depth_pixels[x, y] / 255.0
            x_norm = (x / w - 0.5) * 1.6
            y_norm = (y / h - 0.5) * 1.6
            z_norm = depth_val * 2 - 1
            
            points.append([x_norm, y_norm, z_norm])
            r, g, b = orig_pixels[x, y]
            colors.append([r/255.0, g/255.0, b/255.0])
    
    return {
        "points": points,
        "colors": colors,
        "count": len(points),
        "downsample_factor": downsample_factor,
        "original_size": {"width": w, "height": h},
        "sampled_size": {"width": w // downsample_factor, "height": h // downsample_factor}
    }

@app.get("/")
async def root():
    return {
        "message": "DPT/MiDaS/DepthAnything Lightweight API", 
        "status": "running",
        "models": [
            {
                "id": k,
                "name": v["name"],
                "size_mb": v["size_mb"],
                "type": v["type"],
                "description": v["description"]
            }
            for k, v in MODEL_CONFIGS.items()
        ],
        "default_model": "depth-anything/Depth-Anything-V2-Base-hf",
        "version": "4.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cached_models": list(model_cache.keys()),
        "total_models": len(MODEL_CONFIGS)
    }

@app.post("/api/predict")
async def predict_depth(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None)
):
    try:
        # Default model selection (match frontend default)
        if model is None or model not in MODEL_CONFIGS:
            model = "depth-anything/Depth-Anything-V2-Base-hf"
        
        logger.info(f"Processing with model: {model}")
        
        # Read and prepare image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Size limitation based on model
        config = MODEL_CONFIGS[model]
        max_size = config["input_size"]
        
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        logger.info(f"Image size: {image.size}")
        
        # Depth estimation based on model type
        model_type = config["type"]
        
        if model_type == "pillow_midas":
            depth_gray = midas_inspired_depth(image)
        elif model_type == "pillow_dpt_large":
            depth_gray = dpt_inspired_depth(image)
        elif model_type == "pillow_depth_anything_v1":
            depth_gray = depth_anything_inspired(image)
        elif model_type == "pillow_depth_anything_v2_small":
            depth_gray = depth_anything_v2_small(image)
        elif model_type == "pillow_depth_anything_v2_base":
            depth_gray = depth_anything_v2_base(image)
        elif model_type == "pillow_depth_anything_v2_large":
            depth_gray = depth_anything_v2_large(image)
        elif model_type == "pillow_depthpro":
            depth_gray = depthpro_inspired(image)
        elif model_type == "pillow_zoedepth":
            depth_gray = zoedepth_inspired(image)
        else:
            # Default fallback
            depth_gray = depth_anything_v2_base(image)
        
        # Apply grayscale colormap
        depth_colored = apply_grayscale_depth_map(depth_gray)
        
        # Generate point cloud
        pointcloud_data = generate_pointcloud(image, depth_gray)
        
        # Convert to base64
        def image_to_base64(img):
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_base64}"
        
        return JSONResponse({
            "success": True,
            "originalUrl": image_to_base64(image),
            "depthMapUrl": image_to_base64(depth_colored),
            "pointcloudData": pointcloud_data,
            "model": model,
            "model_info": MODEL_CONFIGS.get(model, {}),
            "resolution": f"{image.size[0]}x{image.size[1]}",
            "algorithms": ["Edge Detection", "Texture Analysis", "Multi-scale Processing"]
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Depth estimation failed: {str(e)}")

@app.post("/api/clear-cache")
async def clear_cache():
    """Clear model cache to free memory"""
    global model_cache
    model_cache.clear()
    return {"success": True, "message": "Model cache cleared"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)