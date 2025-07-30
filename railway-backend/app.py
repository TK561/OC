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
import numpy as np

# ONNX Runtime import with fallback
try:
    import onnxruntime as ort
    import cv2
    ONNX_AVAILABLE = True
    print("✅ ONNX Runtime available")
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️ ONNX Runtime not available, using Pillow-only mode")

app = FastAPI(title="Lightweight Depth Estimation API")

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
    "midas-small": {
        "name": "MiDaS Small",
        "type": "onnx",
        "url": "https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-small.onnx",
        "size_mb": 25,
        "input_size": (256, 256)
    },
    "depth-anything-small": {
        "name": "DepthAnything Small (ONNX)",
        "type": "onnx", 
        "size_mb": 50,
        "input_size": (518, 518)
    },
    "pillow-advanced": {
        "name": "Pillow Advanced CV",
        "type": "pillow",
        "size_mb": 0,
        "input_size": (512, 512)
    }
}

# Simple model cache
model_cache = {}

def download_midas_small():
    """Download MiDaS small model if not exists"""
    model_path = "models/midas_small.onnx"
    os.makedirs("models", exist_ok=True)
    
    if os.path.exists(model_path):
        return model_path
    
    try:
        import urllib.request
        url = "https://github.com/intel-isl/MiDaS/releases/download/v2_1/model-small.onnx"
        logger.info(f"Downloading MiDaS small model from {url}")
        urllib.request.urlretrieve(url, model_path)
        logger.info("✅ MiDaS small model downloaded")
        return model_path
    except Exception as e:
        logger.error(f"❌ Failed to download MiDaS model: {e}")
        return None

def load_onnx_model(model_name: str):
    """Load ONNX model"""
    if model_name in model_cache:
        return model_cache[model_name]
    
    if not ONNX_AVAILABLE:
        return None
    
    try:
        if model_name == "midas-small":
            model_path = download_midas_small()
            if model_path and os.path.exists(model_path):
                session = ort.InferenceSession(model_path)
                model_cache[model_name] = session
                logger.info(f"✅ Loaded ONNX model: {model_name}")
                return session
        
        return None
    except Exception as e:
        logger.error(f"❌ Failed to load ONNX model {model_name}: {e}")
        return None

def preprocess_image_midas(image: Image.Image) -> np.ndarray:
    """Preprocess image for MiDaS model"""
    # Resize to 256x256
    img = image.resize((256, 256), Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Convert RGB to BGR (OpenCV format)
    img_array = img_array[:, :, ::-1]
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Add batch dimension and transpose to NCHW
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def onnx_depth_estimation(image: Image.Image, model_name: str):
    """ONNX-based depth estimation"""
    session = load_onnx_model(model_name)
    if session is None:
        return None
    
    try:
        if model_name == "midas-small":
            # Preprocess
            input_tensor = preprocess_image_midas(image)
            
            # Run inference
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            
            result = session.run([output_name], {input_name: input_tensor})
            depth = result[0][0]  # Remove batch dimension
            
            # Normalize depth
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            
            # Resize back to original size
            depth_resized = cv2.resize(depth, image.size, interpolation=cv2.INTER_LINEAR)
            
            # Convert to PIL Image
            depth_img = Image.fromarray((depth_resized * 255).astype(np.uint8))
            
            return depth_img
        
        return None
        
    except Exception as e:
        logger.error(f"ONNX inference failed: {e}")
        return None

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
        "message": "Lightweight Depth Estimation API", 
        "status": "running",
        "onnx_available": ONNX_AVAILABLE,
        "models": models_available,
        "default_model": "midas-small" if ONNX_AVAILABLE else "pillow-advanced",
        "version": "2.1.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "onnx_available": ONNX_AVAILABLE,
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
            model = "midas-small" if ONNX_AVAILABLE else "pillow-advanced"
        
        logger.info(f"Processing with model: {model}")
        
        # Read and prepare image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Size limitation for memory
        max_size = 768
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        logger.info(f"Image size: {image.size}")
        
        # Depth estimation
        if ONNX_AVAILABLE and model != "pillow-advanced":
            depth_gray = onnx_depth_estimation(image, model)
            if depth_gray is None:
                depth_gray = pillow_depth_estimation(image)
                model = "pillow-advanced"  # Update model name for response
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
            "onnx_available": ONNX_AVAILABLE
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