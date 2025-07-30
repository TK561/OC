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

app = FastAPI(title="Railway Pillow-Based Depth Estimation API")

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

# Model configurations (Pillow-only for Railway compatibility)
MODEL_CONFIGS = {
    "pillow-advanced": {
        "name": "Pillow Advanced CV",
        "type": "pillow",
        "size_mb": 0,
        "input_size": (512, 512),
        "description": "Advanced computer vision using Pillow"
    },
    "pillow-enhanced": {
        "name": "Pillow Enhanced Depth",
        "type": "pillow", 
        "size_mb": 0,
        "input_size": (768, 768),
        "description": "Enhanced depth estimation with multiple algorithms"
    },
    "pillow-fast": {
        "name": "Pillow Fast Mode",
        "type": "pillow",
        "size_mb": 0,
        "input_size": (256, 256),
        "description": "Fast depth estimation for quick processing"
    }
}

def advanced_edge_detection(image):
    """Pillow ベースのエッジ検出"""
    gray = image.convert('L')
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)
    return edges

def texture_analysis(image):
    """テクスチャ分析 (局所分散近似)"""
    gray = image.convert('L')
    w, h = gray.size
    texture_img = Image.new('L', (w, h))
    pixels = gray.load()
    texture_pixels = texture_img.load()
    
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            values = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    values.append(pixels[x + dx, y + dy])
            
            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
            texture_pixels[x, y] = min(255, int(math.sqrt(variance) * 10))
    
    return texture_img

def gradient_magnitude(image):
    """グラデーション強度計算"""
    gray = image.convert('L')
    w, h = gray.size
    gradient_img = Image.new('L', (w, h))
    pixels = gray.load()
    grad_pixels = gradient_img.load()
    
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            gx = (pixels[x+1, y-1] + 2*pixels[x+1, y] + pixels[x+1, y+1] -
                  pixels[x-1, y-1] - 2*pixels[x-1, y] - pixels[x-1, y+1])
            gy = (pixels[x-1, y+1] + 2*pixels[x, y+1] + pixels[x+1, y+1] -
                  pixels[x-1, y-1] - 2*pixels[x, y-1] - pixels[x+1, y-1])
            magnitude = math.sqrt(gx*gx + gy*gy)
            grad_pixels[x, y] = min(255, int(magnitude / 8))
    
    return gradient_img

def pillow_depth_estimation(image, model_type="pillow-advanced"):
    """Pillow のみで高度な深度推定"""
    w, h = image.size
    center_x, center_y = w // 2, h // 2
    
    # Model-specific processing
    if model_type == "pillow-fast":
        # Fast mode - simplified processing
        gray = image.convert('L')
        blurred = gray.filter(ImageFilter.GaussianBlur(radius=2))
        depth_img = ImageOps.autocontrast(blurred)
        return depth_img
    
    elif model_type == "pillow-enhanced":
        # Enhanced mode - more sophisticated processing
        edges = advanced_edge_detection(image)
        texture = texture_analysis(image)  
        gradient = gradient_magnitude(image)
        
        # Additional enhancement
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(1.5)
        edges2 = advanced_edge_detection(enhanced)
        
        edge_pixels = edges.load()
        texture_pixels = texture.load()
        gradient_pixels = gradient.load()
        edge2_pixels = edges2.load()
        
        depth_img = Image.new('L', (w, h))
        depth_pixels = depth_img.load()
        max_distance = math.sqrt(center_x**2 + center_y**2)
        
        for y in range(h):
            for x in range(w):
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                distance_norm = distance / max_distance
                edge_val = edge_pixels[x, y] / 255.0
                texture_val = texture_pixels[x, y] / 255.0 if texture_pixels[x, y] else 0
                gradient_val = gradient_pixels[x, y] / 255.0 if gradient_pixels[x, y] else 0
                edge2_val = edge2_pixels[x, y] / 255.0
                
                depth_value = (
                    0.3 * (1 - distance_norm) +
                    0.2 * texture_val +
                    0.2 * gradient_val +
                    0.15 * (1 - edge_val) +
                    0.15 * (1 - edge2_val)
                )
                
                depth_pixels[x, y] = min(255, max(0, int(depth_value * 255)))
        
        depth_img = depth_img.filter(ImageFilter.GaussianBlur(radius=2))
        
    else:
        # Advanced mode - original implementation
        edges = advanced_edge_detection(image)
        edge_pixels = edges.load()
        texture = texture_analysis(image)
        texture_pixels = texture.load()
        gradient = gradient_magnitude(image)
        gradient_pixels = gradient.load()
        
        depth_img = Image.new('L', (w, h))
        depth_pixels = depth_img.load()
        max_distance = math.sqrt(center_x**2 + center_y**2)
        
        for y in range(h):
            for x in range(w):
                distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                distance_norm = distance / max_distance
                edge_val = edge_pixels[x, y] / 255.0
                texture_val = texture_pixels[x, y] / 255.0 if texture_pixels[x, y] else 0
                gradient_val = gradient_pixels[x, y] / 255.0 if gradient_pixels[x, y] else 0
                
                depth_value = (
                    0.4 * (1 - distance_norm) +
                    0.2 * texture_val +
                    0.2 * gradient_val +
                    0.2 * (1 - edge_val)
                )
                
                depth_pixels[x, y] = min(255, max(0, int(depth_value * 255)))
        
        depth_img = depth_img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
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
    models_available = list(MODEL_CONFIGS.keys())
    return {
        "message": "Railway Pillow-Based Depth Estimation API", 
        "status": "running",
        "pillow_based": True,
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
        "default_model": "pillow-advanced",
        "version": "3.1.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "pillow_based": True,
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
            model = "pillow-advanced"
        
        logger.info(f"Processing with model: {model}")
        
        # Read and prepare image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Size limitation based on model
        config = MODEL_CONFIGS[model]
        max_size = config["input_size"][0]
        
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        logger.info(f"Image size: {image.size}")
        
        # Depth estimation using Pillow
        depth_gray = pillow_depth_estimation(image, model)
        
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
            "pillow_based": True
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Depth estimation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)