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

# Pillow-only implementations (Railway compatible)
MODEL_CONFIGS = {
    "midas-inspired": {
        "name": "MiDaS-Inspired",
        "type": "pillow_midas",
        "size_mb": 0,
        "input_size": 256,
        "description": "MiDaS-style depth with Pillow only"
    },
    "dpt-lightweight": {
        "name": "DPT Lightweight",
        "type": "pillow_enhanced", 
        "size_mb": 0,
        "input_size": 384,
        "description": "DPT-inspired Pillow implementation"
    },
    "depth-anything-sim": {
        "name": "DepthAnything Simulator",
        "type": "pillow_advanced",
        "size_mb": 0,
        "input_size": 518,
        "description": "DepthAnything-inspired algorithm"
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

def apply_grayscale_depth_map(depth_image):
    """深度マップをグレースケール表示（白が近い、黒が遠い）"""
    w, h = depth_image.size
    colored_img = Image.new('RGB', (w, h))
    
    depth_pixels = depth_image.load()
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
        "default_model": "depth-anything-sim",
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
        # Default model selection
        if model is None or model not in MODEL_CONFIGS:
            model = "depth-anything-sim"
        
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
        if model == "midas-inspired":
            depth_gray = midas_inspired_depth(image)
        elif model == "dpt-lightweight":
            depth_gray = dpt_inspired_depth(image)
        elif model == "depth-anything-sim":
            depth_gray = depth_anything_inspired(image)
        else:
            depth_gray = depth_anything_inspired(image)
        
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