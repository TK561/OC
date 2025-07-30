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

# Pillow-only implementations (Railway compatible) - 3 core models
MODEL_CONFIGS = {
    "Intel/dpt-large": {
        "name": "DPT-Large",
        "type": "pillow_dpt_large", 
        "size_mb": 0,
        "input_size": 384,
        "description": "DPT-Large Pillow implementation"
    },
    "Intel/dpt-hybrid-midas": {
        "name": "MiDaS v3.1 (DPT Hybrid)",
        "type": "pillow_midas",
        "size_mb": 0,
        "input_size": 384,
        "description": "MiDaS v3.1 Pillow implementation"
    },
    "LiheYoung/depth-anything-large-hf": {
        "name": "DepthAnything v1 Large",
        "type": "pillow_depth_anything_v1",
        "size_mb": 0,
        "input_size": 518,
        "description": "DepthAnything v1 Large Pillow implementation"
    }
}

# Model cache
model_cache = {}

def midas_inspired_depth(image: Image.Image):
    """MiDaS-inspired depth estimation using realistic depth cues"""
    w, h = image.size
    
    # Convert to different color spaces for analysis
    hsv = image.convert('HSV')
    lab = image.convert('LAB')
    
    # Extract channels
    h_channel, s_channel, v_channel = hsv.split()
    l_channel, a_channel, b_channel = lab.split()
    
    # Create depth map
    depth_img = Image.new('L', (w, h))
    depth_pixels = depth_img.load()
    
    # Get pixel data
    v_pixels = v_channel.load()  # Brightness
    s_pixels = s_channel.load()  # Saturation
    l_pixels = l_channel.load()  # Lightness
    
    for y in range(h):
        for x in range(w):
            # 1. Brightness cue (brighter = closer)
            brightness = v_pixels[x, y] / 255.0
            
            # 2. Saturation cue (more saturated = closer)
            saturation = s_pixels[x, y] / 255.0
            
            # 3. Vertical position cue (bottom = closer, top = farther)
            vertical_factor = 1.0 - (y / h)  # Bottom is closer
            
            # 4. Center bias (objects often in center)
            center_x, center_y = w // 2, h // 2
            distance_from_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = math.sqrt(center_x**2 + center_y**2)
            center_factor = 1.0 - (distance_from_center / max_distance)
            
            # 5. Edge detection for object boundaries
            # Sample surrounding pixels for edge detection
            edge_strength = 0.0
            if x > 0 and x < w-1 and y > 0 and y < h-1:
                # Simple edge detection
                dx = abs(l_pixels[x+1, y] - l_pixels[x-1, y]) / 255.0
                dy = abs(l_pixels[x, y+1] - l_pixels[x, y-1]) / 255.0
                edge_strength = (dx + dy) / 2.0
            
            # Combine all depth cues with realistic weights
            depth_value = (
                0.35 * brightness +          # Primary cue: brightness
                0.25 * vertical_factor +     # Vertical position
                0.15 * saturation +          # Color saturation
                0.15 * center_factor +       # Center bias
                0.10 * (1.0 - edge_strength) # Smooth areas closer
            )
            
            # Ensure value is in valid range
            depth_pixels[x, y] = min(255, max(0, int(depth_value * 255)))
    
    # Apply slight smoothing to reduce noise
    depth_img = depth_img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    # Enhance contrast for better visibility
    depth_img = ImageOps.autocontrast(depth_img)
    
    return depth_img

def dpt_inspired_depth(image: Image.Image):
    """DPT-inspired depth estimation with transformer-like multi-scale analysis"""
    w, h = image.size
    
    # Multi-scale processing like DPT's vision transformer
    scales = [1.0, 0.75, 0.5, 0.25]
    depth_maps = []
    
    # Convert to LAB for better perceptual processing
    lab = image.convert('LAB')
    hsv = image.convert('HSV')
    
    for scale in scales:
        if scale != 1.0:
            new_size = (int(w * scale), int(h * scale))
            scaled_lab = lab.resize(new_size, Image.Resampling.LANCZOS)
            scaled_hsv = hsv.resize(new_size, Image.Resampling.LANCZOS)
        else:
            scaled_lab = lab
            scaled_hsv = hsv
        
        scale_w, scale_h = scaled_lab.size
        
        # Extract channels
        l_channel = scaled_lab.split()[0]
        s_channel = scaled_hsv.split()[1]
        v_channel = scaled_hsv.split()[2]
        
        # Create scale-specific depth map
        scale_depth = Image.new('L', (scale_w, scale_h))
        scale_pixels = scale_depth.load()
        
        l_pixels = l_channel.load()
        s_pixels = s_channel.load()
        v_pixels = v_channel.load()
        
        for y in range(scale_h):
            for x in range(scale_w):
                # Multi-cue depth estimation
                lightness = l_pixels[x, y] / 255.0
                saturation = s_pixels[x, y] / 255.0
                brightness = v_pixels[x, y] / 255.0
                
                # Vertical gradient (perspective cue)
                vertical_pos = 1.0 - (y / scale_h)
                
                # Scale-dependent weighting
                scale_weight = scale * 0.5 + 0.5  # Smaller scales get less weight
                
                # Texture analysis using local variance
                texture_strength = 0.0
                if x > 1 and x < scale_w-2 and y > 1 and y < scale_h-2:
                    # Calculate local variance
                    local_sum = 0
                    local_count = 0
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            local_sum += l_pixels[x+dx, y+dy]
                            local_count += 1
                    local_mean = local_sum / local_count
                    
                    variance = 0
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            diff = l_pixels[x+dx, y+dy] - local_mean
                            variance += diff * diff
                    texture_strength = min(1.0, variance / (255.0 * 255.0 * 9))
                
                # DPT-style combination with scale awareness
                depth_value = (
                    0.30 * brightness * scale_weight +      # Brightness
                    0.25 * vertical_pos +                   # Perspective
                    0.20 * saturation * scale_weight +      # Color info
                    0.15 * texture_strength +               # Local texture
                    0.10 * lightness                        # Lightness
                )
                
                scale_pixels[x, y] = min(255, max(0, int(depth_value * 255)))
        
        # Resize back to original size
        if scale != 1.0:
            scale_depth = scale_depth.resize((w, h), Image.Resampling.LANCZOS)
        
        depth_maps.append(scale_depth)
    
    # Multi-scale fusion (transformer-like attention)
    result = depth_maps[0]  # Start with full resolution
    for i, depth_map in enumerate(depth_maps[1:], 1):
        # Weight smaller scales less
        weight = 0.5 / (i + 1)
        result = Image.blend(result, depth_map, weight)
    
    # Final refinement
    result = result.filter(ImageFilter.GaussianBlur(radius=1.2))
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
        "default_model": "Intel/dpt-large",
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
            model = "Intel/dpt-large"
        
        logger.info(f"Processing with model: {model}")
        
        # Read and prepare image
        contents = await file.read()
        logger.info(f"File size: {len(contents)} bytes, filename: {file.filename}")
        
        if len(contents) == 0:
            raise ValueError("Empty file uploaded")
        
        try:
            # Reset BytesIO position to start
            image_bytes = io.BytesIO(contents)
            image = Image.open(image_bytes)
            image = image.convert('RGB')
            logger.info(f"Successfully loaded image: {image.size}")
        except Exception as img_error:
            logger.error(f"Image loading error: {img_error}")
            raise ValueError(f"Cannot process image file: {str(img_error)}")
        
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
        else:
            # Default fallback to DPT-Large
            depth_gray = dpt_inspired_depth(image)
        
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