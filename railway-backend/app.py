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
    """MiDaS風深度推定 - OpenCVスタイルのマルチスケール処理"""
    w, h = image.size
    original_image = image.copy()
    
    # MiDaS風の前処理: 384x384にリサイズ（アスペクト比維持）
    if w > h:
        new_w, new_h = 384, int(384 * h / w)
    else:
        new_w, new_h = int(384 * w / h), 384
    
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # RGB値を[-1, 1]に正規化（MiDaS/DPTスタイル）
    import numpy as np
    img_array = np.array(resized, dtype=np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5  # [-1, 1]正規化
    
    # グレースケール変換
    gray_array = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    
    # 疑似CNN処理: 複数フィルタでの特徴抽出
    depth_features = []
    
    # Feature 1: Sobel edge detection (疑似gradient computation)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    grad_x = scipy_like_filter2d(gray_array, sobel_x)
    grad_y = scipy_like_filter2d(gray_array, sobel_y)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    depth_features.append(gradient_mag)
    
    # Feature 2: Laplacian of Gaussian (疑似multi-scale processing)
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    log_response = scipy_like_filter2d(gray_array, laplacian)
    depth_features.append(np.abs(log_response))
    
    # Feature 3: Gaussian blur differences (疑似scale-space)
    blur1 = gaussian_blur_array(gray_array, radius=1)
    blur2 = gaussian_blur_array(gray_array, radius=3)
    scale_diff = np.abs(gray_array - blur1) + np.abs(blur1 - blur2)
    depth_features.append(scale_diff)
    
    # Feature 4: Local variance (疑似texture analysis)
    local_var = compute_local_variance_fast(gray_array, window=5)
    depth_features.append(local_var)
    
    # 疑似CNN風の特徴結合
    combined_features = np.zeros_like(gray_array)
    weights = [0.3, 0.25, 0.25, 0.2]  # 特徴ごとの重み
    
    for i, feature in enumerate(depth_features):
        # 正規化
        if feature.max() > feature.min():
            normalized = (feature - feature.min()) / (feature.max() - feature.min())
        else:
            normalized = feature
        combined_features += weights[i] * normalized
    
    # 垂直位置バイアス（遠近法）
    height_bias = np.linspace(1.0, 0.3, new_h).reshape(-1, 1)
    height_bias = np.tile(height_bias, (1, new_w))
    
    # 最終的な深度マップ
    pseudo_depth = combined_features * height_bias
    
    # MiDaSスタイルの逆深度変換
    pseudo_depth = 1.0 / (pseudo_depth + 0.1)  # 逆深度
    pseudo_depth = (pseudo_depth - pseudo_depth.min()) / (pseudo_depth.max() - pseudo_depth.min())
    
    # [0, 255]にスケール
    depth_map = (pseudo_depth * 255).astype(np.uint8)
    
    # PIL Imageに変換
    depth_pil = Image.fromarray(depth_map, mode='L')
    
    # 元のサイズに戻す（バイキュービック補間）
    depth_final = depth_pil.resize((w, h), Image.Resampling.BICUBIC)
    
    # 後処理: 軽いぼかしとコントラスト調整
    depth_final = depth_final.filter(ImageFilter.GaussianBlur(radius=1.0))
    depth_final = ImageOps.autocontrast(depth_final, cutoff=1)
    
    return depth_final

def scipy_like_filter2d(image_array, kernel):
    """SciPyのfilter2D風の実装（Pillowのみ使用）"""
    h, w = image_array.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    
    # パディング
    padded = np.pad(image_array, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    result = np.zeros_like(image_array)
    
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(region * kernel)
    
    return result

def gaussian_blur_array(image_array, radius):
    """NumPy配列でのガウシアンブラー近似"""
    # 簡易ガウシアンカーネル
    size = max(3, int(radius * 2) + 1)
    if size % 2 == 0:
        size += 1
    
    center = size // 2
    kernel = np.zeros((size, size))
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x*x + y*y) / (2 * radius * radius))
    
    kernel /= kernel.sum()
    
    return scipy_like_filter2d(image_array, kernel)

def compute_local_variance_fast(image_array, window):
    """高速ローカル分散計算"""
    h, w = image_array.shape
    half_window = window // 2
    variance_map = np.zeros_like(image_array)
    
    for i in range(half_window, h - half_window):
        for j in range(half_window, w - half_window):
            region = image_array[i-half_window:i+half_window+1, 
                               j-half_window:j+half_window+1]
            variance_map[i, j] = np.var(region)
    
    return variance_map

def compute_texture_variance(gray_img, w, h):
    """Compute local texture variance using sliding window"""
    variance_img = Image.new('L', (w, h))
    variance_pixels = variance_img.load()
    gray_pixels = gray_img.load()
    
    window_size = 5
    half_window = window_size // 2
    
    for y in range(h):
        for x in range(w):
            # Calculate local variance in window
            total_sum = 0
            count = 0
            
            # First pass: calculate mean
            for dy in range(-half_window, half_window + 1):
                for dx in range(-half_window, half_window + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        total_sum += gray_pixels[nx, ny]
                        count += 1
            
            if count > 0:
                mean_val = total_sum / count
                
                # Second pass: calculate variance
                variance_sum = 0
                for dy in range(-half_window, half_window + 1):
                    for dx in range(-half_window, half_window + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            diff = gray_pixels[nx, ny] - mean_val
                            variance_sum += diff * diff
                
                variance = variance_sum / count
                variance_pixels[x, y] = min(255, int(math.sqrt(variance)))
            else:
                variance_pixels[x, y] = 0
    
    return variance_img

def dpt_inspired_depth(image: Image.Image):
    """DPT-inspired depth estimation using dense prediction principles"""
    w, h = image.size
    
    # Multi-scale analysis inspired by DPT's transformer patches
    scales = [1.0, 0.75, 0.5]  # Reduced scales for efficiency
    depth_maps = []
    
    # Convert to multiple color spaces for rich feature extraction
    lab = image.convert('LAB')
    rgb = image
    gray = image.convert('L')
    
    for scale in scales:
        if scale != 1.0:
            new_size = (int(w * scale), int(h * scale))
            scaled_lab = lab.resize(new_size, Image.Resampling.LANCZOS)
            scaled_rgb = rgb.resize(new_size, Image.Resampling.LANCZOS)
            scaled_gray = gray.resize(new_size, Image.Resampling.LANCZOS)
        else:
            scaled_lab = lab
            scaled_rgb = rgb
            scaled_gray = gray
        
        scale_w, scale_h = scaled_lab.size
        
        # Extract channels
        l_channel, a_channel, b_channel = scaled_lab.split()
        r_channel, g_channel, b_rgb_channel = scaled_rgb.split()
        
        # Create scale-specific depth map using advanced depth cues
        scale_depth = Image.new('L', (scale_w, scale_h))
        scale_pixels = scale_depth.load()
        
        # Load pixel data
        l_pixels = l_channel.load()
        a_pixels = a_channel.load()
        r_pixels = r_channel.load()
        g_pixels = g_channel.load()
        b_pixels = b_rgb_channel.load()
        gray_pixels = scaled_gray.load()
        
        for y in range(scale_h):
            for x in range(scale_w):
                # 1. **Linear perspective cue** - stronger at smaller scales
                perspective_strength = (scale_h - y) / scale_h
                perspective_weight = 1.0 / scale  # Stronger at smaller scales
                perspective_cue = perspective_strength * perspective_weight
                
                # 2. **Shape-from-shading** using LAB lightness
                lightness = l_pixels[x, y] / 255.0
                
                # 3. **Color constancy and aerial perspective**
                # Blue shift indicates distance (atmospheric scattering)
                blue_shift = b_pixels[x, y] / (r_pixels[x, y] + 1)  # Avoid division by zero
                blue_shift = min(2.0, blue_shift)  # Clamp
                atmospheric_cue = 1.0 - (blue_shift - 0.5) * 0.5  # Blue = far, Red = near
                
                # 4. **Texture gradient** - fine texture = near, coarse = far
                texture_detail = 0.0
                if 2 <= x < scale_w-2 and 2 <= y < scale_h-2:
                    # Laplacian of Gaussian for texture analysis
                    laplacian = (
                        -1*gray_pixels[x-1,y-1] - 1*gray_pixels[x,y-1] - 1*gray_pixels[x+1,y-1] +
                        -1*gray_pixels[x-1,y] + 8*gray_pixels[x,y] - 1*gray_pixels[x+1,y] +
                        -1*gray_pixels[x-1,y+1] - 1*gray_pixels[x,y+1] - 1*gray_pixels[x+1,y+1]
                    )
                    texture_detail = min(1.0, abs(laplacian) / (255.0 * 4))
                
                # 5. **Occlusion boundaries** using edge information
                edge_strength = 0.0
                if 1 <= x < scale_w-1 and 1 <= y < scale_h-1:
                    # Gradient magnitude
                    gx = gray_pixels[x+1,y] - gray_pixels[x-1,y]
                    gy = gray_pixels[x,y+1] - gray_pixels[x,y-1]
                    edge_strength = math.sqrt(gx*gx + gy*gy) / (255.0 * 2)
                    edge_strength = min(1.0, edge_strength)
                
                # Objects with strong edges are often closer
                occlusion_cue = edge_strength
                
                # 6. **Scale-aware combination** (transformer-like attention)
                scale_attention = scale  # Full scale gets most attention
                
                # Combine depth cues with scale-dependent weights
                depth_value = (
                    0.25 * perspective_cue * scale_attention +     # Perspective (scale-dependent)
                    0.20 * lightness +                            # Shape-from-shading
                    0.20 * texture_detail +                       # Texture gradient
                    0.15 * atmospheric_cue +                      # Atmospheric perspective
                    0.10 * occlusion_cue +                        # Occlusion boundaries
                    0.10 * (1.0 - y/scale_h)                     # Simple height cue
                )
                
                # Apply non-linear depth transformation
                depth_value = depth_value ** 0.8  # Gamma correction for better perception
                
                scale_pixels[x, y] = min(255, max(0, int(depth_value * 255)))
        
        # Resize back to original size with high-quality interpolation
        if scale != 1.0:
            scale_depth = scale_depth.resize((w, h), Image.Resampling.LANCZOS)
        
        depth_maps.append(scale_depth)
    
    # Dense prediction fusion (inspired by DPT's dense prediction head)
    result = depth_maps[0]  # Start with full resolution
    
    # Progressive fusion with attention-like weighting
    for i, depth_map in enumerate(depth_maps[1:], 1):
        fusion_weight = 0.3 / i  # Decrease weight for smaller scales
        result = Image.blend(result, depth_map, fusion_weight)
    
    # Post-processing for realistic depth perception
    # 1. Edge-preserving smoothing
    result = result.filter(ImageFilter.MedianFilter(size=3))
    
    # 2. Contrast enhancement with gamma correction
    result = ImageOps.autocontrast(result, cutoff=1)
    
    # 3. Final depth normalization
    result_pixels = result.load()
    for y in range(h):
        for x in range(w):
            # Apply subtle depth curve for more realistic falloff
            normalized_depth = result_pixels[x, y] / 255.0
            enhanced_depth = 1.0 - math.exp(-2.5 * normalized_depth)
            result_pixels[x, y] = int(enhanced_depth * 255)
    
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