from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import base64
import io
import os
import math
import logging
import time
from typing import Optional
import requests
import numpy as np

app = FastAPI(title="DPT/MiDaS/DepthAnything Lightweight API")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Changed to False for COEP compatibility
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Add Cross-Origin Resource Policy for COEP compatibility
@app.middleware("http")
async def add_corp_header(request, call_next):
    response = await call_next(request)
    response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
    return response

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

def midas_inspired_depth(image: Image.Image, original_size=None):
    """MiDaS風深度推定 - OpenCVスタイルのマルチスケール処理"""
    w, h = image.size
    logger.info(f"MiDaS function - Input size: {image.size} (w={w}, h={h})")
    logger.info(f"MiDaS function - Original size param: {original_size}")
    original_image = image.copy()
    
    # MiDaS風の前処理: リサイズを無効化して元のサイズを保持
    # ユーザーリクエスト: リサイズしない
    new_w = w
    new_h = h
    logger.info(f"MiDaS - Original: {w}x{h} -> No resize, keeping original size: {new_w}x{new_h}")
    
    resized = image
    logger.info(f"MiDaS - After resize: {resized.size}")
    
    # RGB値を[-1, 1]に正規化（MiDaS/DPTスタイル）
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
    pseudo_inverse_depth = 1.0 / (pseudo_depth + 0.1)  # 逆深度
    
    # 正規化
    if pseudo_inverse_depth.max() > pseudo_inverse_depth.min():
        normalized = (pseudo_inverse_depth - pseudo_inverse_depth.min()) / (pseudo_inverse_depth.max() - pseudo_inverse_depth.min())
    else:
        normalized = pseudo_inverse_depth
    
    # MiDaSは逆深度を出力するため、大きな値=近い
    # 白=近い、黒=遠いにするため、値を反転する
    normalized = 1.0 - normalized  # MiDaSも反転が必要
    
    # [0, 255]にスケール
    depth_map = (normalized * 255).astype(np.uint8)
    
    # PIL Imageに変換
    logger.info(f"MiDaS - Before fromarray: depth_map.shape = {depth_map.shape}")
    
    # TEMPORARILY DISABLE transpose to test if this is causing the 90-degree rotation
    logger.info(f"MiDaS - Target image size: {new_w}x{new_h} (WxH)")
    logger.info(f"MiDaS - Depth array shape: {depth_map.shape} (should be {new_h}x{new_w})")
    logger.info("MiDaS - TRANSPOSE DISABLED FOR TESTING")
    
    # # Simple fix: Always ensure depth_map.shape == (new_h, new_w)
    # if depth_map.shape != (new_h, new_w):
    #     if depth_map.shape == (new_w, new_h):
    #         logger.info("MiDaS - Transposing depth_map to match image dimensions")
    #         depth_map = depth_map.T
    #     else:
    #         logger.warning(f"MiDaS - Unexpected depth_map shape: {depth_map.shape}, expected: ({new_h}, {new_w})")
    
    logger.info(f"MiDaS - Final depth_map shape: {depth_map.shape}")
    
    depth_pil = Image.fromarray(depth_map, mode='L')
    logger.info(f"MiDaS - After fromarray: depth_pil.size = {depth_pil.size}")
    
    # 元のサイズに戻す（バイキュービック補間）
    target_size = original_size if original_size else (w, h)
    logger.info(f"MiDaS - Resizing depth map from {depth_pil.size} to target_size: {target_size}")
    logger.info(f"MiDaS - Original image was: (w={w}, h={h}), target_size is: {target_size}")
    depth_final = depth_pil.resize(target_size, Image.Resampling.BICUBIC)
    logger.info(f"MiDaS - Final depth map size: {depth_final.size}")
    
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

def dpt_inspired_depth(image: Image.Image, original_size=None):
    """DPT風深度推定 - GitHub調査に基づく逆深度処理"""
    w, h = image.size
    logger.info(f"DPT function - Input size: {image.size} (w={w}, h={h})")
    logger.info(f"DPT function - Original size param: {original_size}")
    
    # DPT風前処理: リサイズを無効化して元のサイズを保持
    # ユーザーリクエスト: リサイズしない
    new_w = w
    new_h = h
    logger.info(f"DPT - Original: {w}x{h} -> No resize, keeping original size: {new_w}x{new_h}")
    
    resized = image
    logger.info(f"DPT - After resize: {resized.size}")
    
    # RGB値を[0,1]に正規化
    img_array = np.array(resized, dtype=np.float32) / 255.0
    
    # グレースケール変換
    gray_array = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    
    # 疑似トランスフォーマー処理: マルチスケール特徴抽出
    scales = [1.0]  # Single scale to reduce memory usage
    scale_features = []
    
    for scale in scales:
        if scale != 1.0:
            scale_h, scale_w = int(new_h * scale), int(new_w * scale)
            scaled_gray = np.array(Image.fromarray((gray_array * 255).astype(np.uint8)).resize((scale_w, scale_h), Image.Resampling.LANCZOS)) / 255.0
        else:
            scaled_gray = gray_array
            scale_h, scale_w = new_h, new_w
        
        # 密な予測のための特徴抽出
        # 1. エッジ特徴（Transformer attentionを模擬）
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        
        grad_x = scipy_like_filter2d(scaled_gray, sobel_x)
        grad_y = scipy_like_filter2d(scaled_gray, sobel_y)
        edge_strength = np.sqrt(grad_x**2 + grad_y**2)
        
        # 2. テクスチャ特徴
        laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
        texture_response = np.abs(scipy_like_filter2d(scaled_gray, laplacian))
        
        # 特徴を元サイズにリサイズ
        if scale != 1.0:
            edge_strength = np.array(Image.fromarray((edge_strength * 255).astype(np.uint8)).resize((new_w, new_h), Image.Resampling.LANCZOS)) / 255.0
            texture_response = np.array(Image.fromarray((texture_response * 255).astype(np.uint8)).resize((new_w, new_h), Image.Resampling.LANCZOS)) / 255.0
        
        # スケール重み付け
        scale_weight = scale
        combined_feature = (edge_strength * 0.6 + texture_response * 0.4) * scale_weight
        scale_features.append(combined_feature)
    
    # Single scale processing for memory efficiency
    fused_features = scale_features[0]
    
    # 垂直バイアス（透視）
    height_bias = np.linspace(1.0, 0.4, new_h).reshape(-1, 1)
    height_bias = np.tile(height_bias, (1, new_w))
    
    # 最終的な逆深度マップ（DPT風）
    pseudo_inverse_depth = fused_features * height_bias
    
    # DPTスタイルの正規化
    if pseudo_inverse_depth.max() > pseudo_inverse_depth.min():
        normalized = (pseudo_inverse_depth - pseudo_inverse_depth.min()) / (pseudo_inverse_depth.max() - pseudo_inverse_depth.min())
    else:
        normalized = pseudo_inverse_depth
    
    # DPTは逆深度を出力するため、大きな値=近い
    # 白=近い、黒=遠いにするため、値をそのまま使用（MiDaSと同じ処理）
    inverted_depth = normalized  # DPTも反転不要
    
    # [0, 255]にスケール
    depth_map = (inverted_depth * 255).astype(np.uint8)
    
    # PIL Imageに変換
    logger.info(f"DPT - Before fromarray: depth_map.shape = {depth_map.shape}")
    
    # Ensure depth map shape exactly matches image dimensions (h, w)
    logger.info(f"DPT - Target image size: {new_w}x{new_h} (WxH)")
    logger.info(f"DPT - Depth array shape: {depth_map.shape} (should be {new_h}x{new_w})")
    
    # TEMPORARILY DISABLE transpose to test if this is causing the 90-degree rotation
    logger.info("DPT - TRANSPOSE DISABLED FOR TESTING")
    
    # # Simple fix: Always ensure depth_map.shape == (new_h, new_w)
    # if depth_map.shape != (new_h, new_w):
    #     if depth_map.shape == (new_w, new_h):
    #         logger.info("DPT - Transposing depth_map to match image dimensions")
    #         depth_map = depth_map.T
    #     else:
    #         logger.warning(f"DPT - Unexpected depth_map shape: {depth_map.shape}, expected: ({new_h}, {new_w})")
    
    logger.info(f"DPT - Final depth_map shape: {depth_map.shape}")
    
    depth_pil = Image.fromarray(depth_map, mode='L')
    logger.info(f"DPT - After fromarray: depth_pil.size = {depth_pil.size}")
    
    # 元のサイズに戻す（バイキュービック補間）
    target_size = original_size if original_size else (w, h)
    logger.info(f"DPT - Resizing depth map from {depth_pil.size} to target_size: {target_size}")
    logger.info(f"DPT - Original image was: (w={w}, h={h}), target_size is: {target_size}")
    depth_final = depth_pil.resize(target_size, Image.Resampling.BICUBIC)
    logger.info(f"DPT - Final depth map size: {depth_final.size}")
    
    # 後処理
    depth_final = depth_final.filter(ImageFilter.GaussianBlur(radius=1.0))
    depth_final = ImageOps.autocontrast(depth_final, cutoff=1)
    
    return depth_final

def depth_anything_inspired(image: Image.Image, original_size=None):
    """DepthAnything風深度推定 - GitHub調査に基づく通常深度処理"""
    w, h = image.size
    logger.info(f"DepthAnything function - Input size: {image.size} (w={w}, h={h})")
    logger.info(f"DepthAnything function - Original size param: {original_size}")
    
    # DepthAnything風前処理: リサイズを無効化して元のサイズを保持
    # ユーザーリクエスト: リサイズしない
    new_w = w
    new_h = h
    logger.info(f"DepthAnything - Original: {w}x{h} -> No resize, keeping original size: {new_w}x{new_h}")
    
    resized = image
    logger.info(f"DepthAnything - After resize: {resized.size}")
    
    # RGB値を[0,1]に正規化（ImageNet統計使用）
    img_array = np.array(resized, dtype=np.float32) / 255.0
    # ImageNet正規化: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    img_array[:,:,0] = (img_array[:,:,0] - 0.485) / 0.229  # R
    img_array[:,:,1] = (img_array[:,:,1] - 0.456) / 0.224  # G  
    img_array[:,:,2] = (img_array[:,:,2] - 0.406) / 0.225  # B
    
    # グレースケール変換
    gray_array = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    # 正規化の影響を調整
    gray_array = (gray_array + 2.0) / 4.0  # [-2, 2] -> [0, 1]の概算
    
    # DepthAnything風特徴抽出
    # 1. エッジ検出
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    grad_x = scipy_like_filter2d(gray_array, sobel_x)
    grad_y = scipy_like_filter2d(gray_array, sobel_y)
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 2. ローカル分散（テクスチャ）
    local_variance = compute_local_variance_fast(gray_array, window=5)
    
    # 3. 明度情報
    brightness = np.clip(gray_array, 0, 1)
    
    # 垂直位置（透視手がかり）
    height_factor = np.linspace(0.2, 1.0, new_h).reshape(-1, 1)  # 上=遠い、下=近い
    height_factor = np.tile(height_factor, (1, new_w))
    
    # DepthAnything風特徴結合
    depth_features = (
        0.3 * brightness +                    # 明度
        0.25 * (1.0 - edge_magnitude) +       # 滑らかな領域=近い
        0.25 * local_variance +               # テクスチャ
        0.2 * height_factor                   # 垂直位置
    )
    
    # 正規化
    if depth_features.max() > depth_features.min():
        normalized_depth = (depth_features - depth_features.min()) / (depth_features.max() - depth_features.min())
    else:
        normalized_depth = depth_features
    
    # DepthAnythingは通常深度を出力（小さな値=近い）
    # 白=近い、黒=遠いにするため、MiDaSと同じように反転処理を適用
    inverted_depth = 1.0 - normalized_depth  # Depth Anythingも反転処理が必要
    
    # [0, 255]にスケール
    depth_map = (inverted_depth * 255).astype(np.uint8)
    
    # PIL Imageに変換
    logger.info(f"DepthAnything - Before fromarray: depth_map.shape = {depth_map.shape}")
    
    # Ensure depth map shape exactly matches image dimensions (h, w)
    logger.info(f"DepthAnything - Target image size: {new_w}x{new_h} (WxH)")
    logger.info(f"DepthAnything - Depth array shape: {depth_map.shape} (should be {new_h}x{new_w})")
    
    # TEMPORARILY DISABLE transpose to test if this is causing the 90-degree rotation
    logger.info("DepthAnything - TRANSPOSE DISABLED FOR TESTING")
    
    # # Simple fix: Always ensure depth_map.shape == (new_h, new_w)
    # if depth_map.shape != (new_h, new_w):
    #     if depth_map.shape == (new_w, new_h):
    #         logger.info("DepthAnything - Transposing depth_map to match image dimensions")
    #         depth_map = depth_map.T
    #     else:
    #         logger.warning(f"DepthAnything - Unexpected depth_map shape: {depth_map.shape}, expected: ({new_h}, {new_w})")
    
    logger.info(f"DepthAnything - Final depth_map shape: {depth_map.shape}")
    
    depth_pil = Image.fromarray(depth_map, mode='L')
    logger.info(f"DepthAnything - After fromarray: depth_pil.size = {depth_pil.size}")
    
    # 元のサイズに戻す（バイキュービック補間）
    target_size = original_size if original_size else (w, h)
    logger.info(f"DepthAnything - Resizing depth map from {depth_pil.size} to target_size: {target_size}")
    logger.info(f"DepthAnything - Original image was: (w={w}, h={h}), target_size is: {target_size}")
    depth_final = depth_pil.resize(target_size, Image.Resampling.BICUBIC)
    logger.info(f"DepthAnything - Final depth map size: {depth_final.size}")
    
    # 後処理
    depth_final = depth_final.filter(ImageFilter.GaussianBlur(radius=1.5))
    depth_final = ImageOps.autocontrast(depth_final)
    
    return depth_final

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
    scales = [1.0]  # Single scale to reduce memory usage
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
    """3Dポイントクラウドデータ生成 - アスペクト比を考慮した座標変換"""
    # EXIF処理はすでにメイン処理で適用済みなのでここでは適用しない
    
    w, h = original_image.size
    downsample_factor = 12
    points = []
    colors = []
    
    orig_pixels = original_image.load()
    depth_pixels = depth_image.load()
    
    # 元画像の縦横比をそのまま保持するスケーリング計算
    aspect_ratio = w / h
    base_scale = 1.6
    
    # アスペクト比を正確に反映
    if aspect_ratio > 1.0:  # 横長画像
        scale_x = base_scale
        scale_y = base_scale / aspect_ratio
    else:  # 縦長画像または正方形
        scale_x = base_scale * aspect_ratio
        scale_y = base_scale
    
    logger.info(f"Point cloud generation: image size {w}x{h}, aspect_ratio={aspect_ratio:.3f}, scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")
    
    for y in range(0, h, downsample_factor):
        for x in range(0, w, downsample_factor):
            # Ensure we don't go out of bounds
            if x < w and y < h:
                # PIL load() uses (x, y) coordinate system
                depth_val = depth_pixels[x, y] / 255.0
                
                # 元画像の縦横比を保持した座標計算
                x_norm = (x / w - 0.5) * scale_x
                # Y軸を反転して3D座標系に合わせる（上向きが正）
                y_norm = -(y / h - 0.5) * scale_y
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
        "sampled_size": {"width": w // downsample_factor, "height": h // downsample_factor},
        "aspect_ratio": aspect_ratio,
        "scaling": {"x": scale_x, "y": scale_y}
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
            # Reset BytesIO position to start and read image
            image_bytes = io.BytesIO(contents)
            image_bytes.seek(0)  # Ensure we're at the beginning
            image = Image.open(image_bytes)
            
            # Handle different image formats
            if image.format not in ['JPEG', 'PNG', 'BMP', 'TIFF', 'WEBP']:
                logger.warning(f"Unusual image format: {image.format}")
            
            # Apply EXIF orientation correction in backend as well
            # This ensures proper orientation regardless of frontend processing
            logger.info(f"Image received: {image.size}, mode: {image.mode}, format: {image.format}")
            
            # フロントエンドで既にEXIF処理済みのため、バックエンドでは処理しない
            logger.info(f"Image received from frontend (EXIF already processed): {image.size}")
            
            image = image.convert('RGB')
            logger.info(f"After RGB conversion: {image.size}")
            
            # Skip original image copy for now - use resized image for everything
            # This saves memory by not keeping two copies
            logger.info(f"Skipping original image copy to save memory")
            
        except Exception as img_error:
            logger.error(f"Image loading error: {img_error}")
            raise ValueError(f"Cannot process image file: {str(img_error)}")
        
        # Balanced size limitation for Railway memory constraints
        max_pixels = 250_000  # About 500x500 or 640x390, balanced quality/memory
        current_pixels = image.size[0] * image.size[1]
        
        if current_pixels > max_pixels:
            # Calculate scale to fit within pixel budget while preserving aspect ratio
            scale = (max_pixels / current_pixels) ** 0.5
            new_width = int(image.size[0] * scale)
            new_height = int(image.size[1] * scale)
            logger.info(f"Railway memory optimization: Resizing from {image.size[0]}x{image.size[1]} to {new_width}x{new_height} (scale: {scale:.3f})")
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Processing with resized image: {image.size}")
        else:
            logger.info(f"Image size {image.size[0]}x{image.size[1]} is within Railway memory limits")
        
        # Store original image size before any processing
        original_size = image.size
        logger.info(f"Stored original_size for depth functions: {original_size}")
        
        # Get model configuration
        config = MODEL_CONFIGS[model]
        logger.info(f"Using model config: {config}")
        logger.info("Railway deployment refresh - portrait image fix active")
        
        # Depth estimation based on model type
        model_type = config["type"]
        logger.info(f"Processing with model_type: {model_type}")
        
        try:
            if model_type == "pillow_midas":
                depth_gray = midas_inspired_depth(image, original_size)
            elif model_type == "pillow_dpt_large":
                depth_gray = dpt_inspired_depth(image, original_size)
            elif model_type == "pillow_depth_anything_v1":
                depth_gray = depth_anything_inspired(image, original_size)
            else:
                # Default fallback to DPT-Large
                logger.info(f"Unknown model_type {model_type}, using DPT fallback")
                depth_gray = dpt_inspired_depth(image, original_size)
            
            logger.info(f"Depth estimation completed successfully. Result size: {depth_gray.size}")
        except Exception as depth_error:
            logger.error(f"Depth estimation failed: {depth_error}")
            raise HTTPException(status_code=500, detail=f"Depth estimation failed: {str(depth_error)}")
        
        # Apply grayscale colormap
        try:
            depth_colored = apply_grayscale_depth_map(depth_gray)
            logger.info(f"Colormap applied successfully. Size: {depth_colored.size}")
        except Exception as colormap_error:
            logger.error(f"Colormap application failed: {colormap_error}")
            raise HTTPException(status_code=500, detail=f"Colormap processing failed: {str(colormap_error)}")
        
        # Generate point cloud using resized image
        try:
            pointcloud_data = generate_pointcloud(image, depth_gray)
            logger.info(f"Point cloud generated successfully. Points: {pointcloud_data['count']}")
        except Exception as pointcloud_error:
            logger.error(f"Point cloud generation failed: {pointcloud_error}")
            raise HTTPException(status_code=500, detail=f"Point cloud generation failed: {str(pointcloud_error)}")
        
        # Convert to base64
        def image_to_base64(img):
            try:
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_base64 = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/png;base64,{img_base64}"
            except Exception as b64_error:
                logger.error(f"Base64 conversion failed for image {img.size}: {b64_error}")
                raise
        
        try:
            logger.info("Generating response...")
            response_data = {
                "success": True,
                "originalUrl": image_to_base64(image),
                "depthMapUrl": image_to_base64(depth_colored),
                "pointcloudData": pointcloud_data,
                "model": model,
                "model_info": MODEL_CONFIGS.get(model, {}),
                "resolution": f"{image.size[0]}x{image.size[1]}",
                "algorithms": ["Edge Detection", "Texture Analysis", "Multi-scale Processing"]
            }
            logger.info("Response generated successfully")
            return JSONResponse(response_data)
        except Exception as response_error:
            logger.error(f"Response generation failed: {response_error}")
            raise HTTPException(status_code=500, detail=f"Response generation failed: {str(response_error)}")
        
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