from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageFilter, ImageOps
import base64
import io
import os
import math
import logging
from typing import Optional

# PyTorch/Transformers imports with fallback
try:
    import torch
    import numpy as np
    from transformers import pipeline, DPTImageProcessor, DPTForDepthEstimation
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch/Transformers not available, using Pillow-only mode")

app = FastAPI(title="Hybrid Depth Estimation API")

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

# Model configurations
MODEL_CONFIGS = {
    "Intel/dpt-hybrid-midas": {
        "name": "MiDaS v3.1",
        "type": "dpt",
        "size_mb": 470
    },
    "Intel/dpt-large": {
        "name": "DPT-Large", 
        "type": "dpt",
        "size_mb": 1300
    },
    "depth-anything/Depth-Anything-V2-Small-hf": {
        "name": "DepthAnything V2 Small",
        "type": "pipeline",
        "size_mb": 99
    },
    "depth-anything/Depth-Anything-V2-Base-hf": {
        "name": "DepthAnything V2 Base",
        "type": "pipeline", 
        "size_mb": 390
    },
    "depth-anything/Depth-Anything-V2-Large-hf": {
        "name": "DepthAnything V2 Large",
        "type": "pipeline",
        "size_mb": 1300
    },
    "pillow-advanced": {
        "name": "Pillow Advanced CV",
        "type": "pillow",
        "size_mb": 0
    }
}

# Global model cache (simple in-memory)
model_cache = {}

def get_device():
    """Get appropriate device for inference"""
    if not TORCH_AVAILABLE:
        return None
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_model(model_name: str):
    """Load model with caching"""
    if model_name in model_cache:
        logger.info(f"Using cached model: {model_name}")
        return model_cache[model_name]
    
    if not TORCH_AVAILABLE or model_name == "pillow-advanced":
        logger.info("Using Pillow-only mode")
        return None
    
    device = get_device()
    logger.info(f"Loading model {model_name} on {device}")
    
    try:
        config = MODEL_CONFIGS.get(model_name, {})
        
        if config.get("type") == "dpt":
            # Load DPT models
            processor = DPTImageProcessor.from_pretrained(model_name)
            model = DPTForDepthEstimation.from_pretrained(model_name)
            if device != "cpu":
                model = model.to(device)
            model_cache[model_name] = (model, processor, "dpt")
            return model, processor, "dpt"
        else:
            # Load pipeline models (DepthAnything, etc)
            pipe = pipeline("depth-estimation", model=model_name, device=0 if device != "cpu" else -1)
            model_cache[model_name] = (pipe, None, "pipeline")
            return pipe, None, "pipeline"
            
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        # Fallback to Pillow
        return None

def clear_model_cache():
    """Clear model cache to free memory"""
    global model_cache
    model_cache.clear()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model cache cleared")

# Pillow-based functions (kept from original)
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

def pillow_depth_estimation(image):
    """Pillow のみで高度な深度推定"""
    w, h = image.size
    center_x, center_y = w // 2, h // 2
    
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

def pytorch_depth_estimation(image, model_name: str):
    """PyTorch/Transformers based depth estimation"""
    model_info = load_model(model_name)
    if model_info is None:
        # Fallback to Pillow
        return pillow_depth_estimation(image)
    
    if len(model_info) == 3:
        model_or_pipe, processor, model_type = model_info
    else:
        return pillow_depth_estimation(image)
    
    try:
        if model_type == "dpt":
            # DPT model processing
            inputs = processor(images=image, return_tensors="pt")
            device = next(model_or_pipe.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model_or_pipe(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Convert to numpy and normalize
            depth = predicted_depth.squeeze().cpu().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth_img = Image.fromarray((depth * 255).astype(np.uint8))
            
        elif model_type == "pipeline":
            # Pipeline model processing
            result = model_or_pipe(image)
            depth = np.array(result["depth"])
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth_img = Image.fromarray((depth * 255).astype(np.uint8))
        else:
            depth_img = pillow_depth_estimation(image)
            
        return depth_img
        
    except Exception as e:
        logger.error(f"PyTorch inference failed: {e}")
        return pillow_depth_estimation(image)

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
        "message": "Hybrid Depth Estimation API", 
        "status": "running",
        "pytorch_available": TORCH_AVAILABLE,
        "models": models_available,
        "default_model": "depth-anything/Depth-Anything-V2-Base-hf" if TORCH_AVAILABLE else "pillow-advanced",
        "version": "2.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "pytorch_available": TORCH_AVAILABLE,
        "cached_models": list(model_cache.keys())
    }

@app.post("/api/predict")
async def predict_depth(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None)
):
    try:
        # Default model selection
        if model is None or model not in MODEL_CONFIGS:
            model = "depth-anything/Depth-Anything-V2-Base-hf" if TORCH_AVAILABLE else "pillow-advanced"
        
        logger.info(f"Processing with model: {model}")
        
        # Read and prepare image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Size limitation for memory
        max_size = 768 if TORCH_AVAILABLE else 512
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        logger.info(f"Image size: {image.size}")
        
        # Depth estimation
        if TORCH_AVAILABLE and model != "pillow-advanced":
            depth_gray = pytorch_depth_estimation(image, model)
        else:
            depth_gray = pillow_depth_estimation(image)
        
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
            "pytorch_available": TORCH_AVAILABLE
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Depth estimation failed: {str(e)}")

@app.post("/api/clear-cache")
async def clear_cache():
    """Clear model cache to free memory"""
    clear_model_cache()
    return {"success": True, "message": "Model cache cleared"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)