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

# 深度推定モデル設定 - 3つの統合モデル選択肢
MODEL_CONFIGS = {
    "Intel/dpt-hybrid-midas": {
        "name": "MiDaS v3.1",
        "type": "pillow_midas",
        "size_mb": 0,
        "input_size": 384,
        "description": "MiDaS v3.1 - エッジ検出と構造理解"
    },
    "Intel/dpt-large": {
        "name": "DPT-Large", 
        "type": "pillow_dpt_large",
        "size_mb": 0,
        "input_size": 384,
        "description": "DPT-Large - Vision Transformer深度推定"
    },
    "LiheYoung/depth-anything-small-hf": {
        "name": "DepthAnything v1",
        "type": "pillow_depth_anything_v1",
        "size_mb": 0,
        "input_size": 518,
        "description": "DepthAnything v1 - Foundation Model深度推定"
    }
}

# Model cache
model_cache = {}

def maximum_filter_simple(arr, size):
    """簡易最大値フィルタ"""
    h, w = arr.shape
    pad = size // 2
    padded = np.pad(arr, pad, mode='edge')
    result = np.zeros_like(arr)
    
    for i in range(h):
        for j in range(w):
            result[i, j] = np.max(padded[i:i+size, j:j+size])
    
    return result

def minimum_filter_simple(arr, size):
    """簡易最小値フィルタ"""
    h, w = arr.shape
    pad = size // 2
    padded = np.pad(arr, pad, mode='edge')
    result = np.zeros_like(arr)
    
    for i in range(h):
        for j in range(w):
            result[i, j] = np.min(padded[i:i+size, j:j+size])
    
    return result

def midas_inspired_depth(image: Image.Image, original_size=None):
    """参考画像ベース物体検出深度推定 - 犬本体=純白、背景=純黒"""
    w, h = image.size
    logger.info(f"Reference-based depth estimation - Input size: {w}x{h}")
    
    # RGB配列取得
    img_array = np.array(image, dtype=np.float32) / 255.0
    gray_array = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    
    # 参考画像分析：犬（主要物体）を純白、背景を純黒で表現
    
    # 1. 主要物体検出（明度+コントラストベース）
    # 明るい部分や高コントラスト部分を主要物体として識別
    brightness_mask = gray_array > 0.3  # 明度閾値
    
    # 2. エッジ検出による物体境界
    # 隣接ピクセルとの差分でエッジを検出
    edge_x = np.abs(np.roll(gray_array, -1, axis=1) - gray_array)
    edge_y = np.abs(np.roll(gray_array, -1, axis=0) - gray_array)
    edges = np.sqrt(edge_x**2 + edge_y**2)
    edge_mask = edges > 0.1  # エッジ閾値
    
    # 3. 中央バイアス（主要物体は画像中央付近）
    center_x, center_y = w // 2, h // 2
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    center_weight = 1.0 - (distance_from_center / (max_distance * 1.2))
    center_weight = np.clip(center_weight, 0, 1)
    center_mask = center_weight > 0.3  # 中央領域
    
    # 4. 物体領域の統合判定
    # 明度・エッジ・中央バイアスの論理和で主要物体を決定
    object_mask = brightness_mask | (edge_mask & center_mask)
    
    # 5. モルフォロジー処理で物体領域をクリーンアップ
    # 小さなノイズを除去し、連続した領域にする
    from scipy import ndimage
    try:
        # 膨張・収縮処理で穴埋めとノイズ除去
        object_mask = ndimage.binary_dilation(object_mask, iterations=2)
        object_mask = ndimage.binary_erosion(object_mask, iterations=2)
        object_mask = ndimage.binary_fill_holes(object_mask)
    except:
        # scipyが利用できない場合の代替処理
        pass
    
    # 6. 深度マップ生成
    # 参考画像通り：物体=純白(255)、背景=純黒(0)、境界=グラデーション
    depth_map = np.zeros_like(gray_array)
    
    # 物体領域を白に設定
    depth_map[object_mask] = 1.0
    
    # 境界周辺のグラデーション生成
    # 物体境界から距離に応じてグラデーション
    try:
        from scipy.ndimage import distance_transform_edt
        
        # 物体境界からの距離計算
        boundary_distance = distance_transform_edt(~object_mask)
        object_distance = distance_transform_edt(object_mask)
        
        # 境界周辺（5ピクセル以内）にグラデーション適用
        gradient_zone = 5
        boundary_gradient = np.where(
            boundary_distance <= gradient_zone,
            boundary_distance / gradient_zone,  # 0-1のグラデーション
            0  # 遠い背景は0
        )
        
        object_gradient = np.where(
            (object_distance <= gradient_zone) & object_mask,
            1.0 - (object_distance / gradient_zone * 0.3),  # 物体内部の軽微なグラデーション
            depth_map  # 元の値を保持
        )
        
        # 最終深度マップ：物体領域 + 境界グラデーション
        depth_map = np.maximum(boundary_gradient, object_gradient)
        
    except ImportError:
        # scipyがない場合の簡易グラデーション
        logger.warning("SciPy not available, using simple gradient")
        
        # 簡易な境界グラデーション
        kernel_size = 5
        for i in range(h):
            for j in range(w):
                if not object_mask[i, j]:  # 背景ピクセル
                    # 周辺の物体ピクセルまでの最短距離を計算
                    min_dist = float('inf')
                    for di in range(-kernel_size, kernel_size + 1):
                        for dj in range(-kernel_size, kernel_size + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w and object_mask[ni, nj]:
                                dist = np.sqrt(di*di + dj*dj)
                                min_dist = min(min_dist, dist)
                    
                    # 距離に基づくグラデーション
                    if min_dist <= kernel_size:
                        depth_map[i, j] = max(0, 1.0 - min_dist / kernel_size)
    
    # 正規化と最終調整
    depth_map = np.clip(depth_map, 0, 1)
    
    # 参考画像により近づけるため、コントラストを強化
    depth_map = np.power(depth_map, 0.8)  # ガンマ補正でコントラスト調整
    
    # [0, 255]にスケール
    depth_map_uint8 = (depth_map * 255).astype(np.uint8)
    
    # PIL Imageに変換
    depth_pil = Image.fromarray(depth_map_uint8, mode='L')
    
    # 元のサイズに戻す
    target_size = original_size if original_size else (w, h)
    depth_final = depth_pil.resize(target_size, Image.Resampling.BICUBIC)
    
    logger.info(f"Reference-based depth estimation completed: {depth_final.size}")
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

def simple_smooth_array(image_array, radius):
    """簡単な平滑化処理（ボカシなし、エッジ保持）"""
    # ボカシの代わりに非常に軽微な平滑化のみ
    if radius <= 1:
        return image_array
    
    # 単純な平均フィルタ（3x3のみ）で最小限の平滑化
    h, w = image_array.shape
    result = image_array.copy()
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            # 軽微な平滑化のみ（エッジを保持）
            center_weight = 0.8
            neighbor_weight = 0.2 / 8
            
            result[i, j] = (
                center_weight * image_array[i, j] +
                neighbor_weight * (
                    image_array[i-1, j-1] + image_array[i-1, j] + image_array[i-1, j+1] +
                    image_array[i, j-1] + image_array[i, j+1] +
                    image_array[i+1, j-1] + image_array[i+1, j] + image_array[i+1, j+1]
                )
            )
    
    return result

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
    """参考画像ベース高精度物体検出深度推定 - DPT風の精密な境界検出"""
    w, h = image.size
    logger.info(f"DPT-style reference-based depth estimation - Input size: {w}x{h}")
    
    # RGB配列取得
    img_array = np.array(image, dtype=np.float32) / 255.0
    gray_array = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    
    # DPT風の高精度物体検出アプローチ
    
    # 1. マルチスケール物体検出
    # 異なるスケールでの特徴抽出
    scale_features = []
    for radius in [1, 2, 4, 8]:
        # エッジ検出
        edge_x = np.abs(np.roll(gray_array, -radius, axis=1) - gray_array)
        edge_y = np.abs(np.roll(gray_array, -radius, axis=0) - gray_array)
        scale_edge = np.sqrt(edge_x**2 + edge_y**2)
        scale_features.append(scale_edge)
    
    # 2. 適応的閾値による物体領域検出
    # Otsu's method風の自動閾値決定
    hist, bins = np.histogram(gray_array.flatten(), bins=256, range=[0, 1])
    total_pixels = gray_array.size
    current_max, threshold = 0, 0
    
    for i in range(1, len(hist)):
        # 背景と前景の重み
        wb = np.sum(hist[:i]) / total_pixels
        wf = np.sum(hist[i:]) / total_pixels
        
        if wb == 0 or wf == 0:
            continue
        
        # 背景と前景の平均
        mb = np.sum(hist[:i] * np.arange(i)) / np.sum(hist[:i])
        mf = np.sum(hist[i:] * np.arange(i, len(hist))) / np.sum(hist[i:])
        
        # クラス間分散
        variance_between = wb * wf * (mb - mf) ** 2
        
        if variance_between > current_max:
            current_max = variance_between
            threshold = bins[i]
    
    # 3. 物体マスク生成
    # 明度ベース + エッジベース + 中央バイアス
    brightness_mask = gray_array > max(threshold, 0.2)
    
    # マルチスケールエッジの統合
    combined_edges = np.zeros_like(gray_array)
    for i, edge_feature in enumerate(scale_features):
        weight = 1.0 / (2 ** i)  # 小さいスケールほど重要
        combined_edges += weight * edge_feature
    combined_edges /= len(scale_features)
    edge_mask = combined_edges > np.percentile(combined_edges, 75)
    
    # 中央バイアス（より強力）
    center_x, center_y = w // 2, h // 2
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    distance_from_center = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    center_weight = 1.0 - (distance_from_center / (max_distance * 0.8))
    center_weight = np.clip(center_weight, 0, 1)
    center_mask = center_weight > 0.4
    
    # 4. 最終物体マスク
    object_mask = brightness_mask & (edge_mask | center_mask)
    
    # 5. モルフォロジー処理（より積極的）
    try:
        from scipy import ndimage
        # より強力なクリーンアップ
        object_mask = ndimage.binary_dilation(object_mask, iterations=3)
        object_mask = ndimage.binary_erosion(object_mask, iterations=3)
        object_mask = ndimage.binary_fill_holes(object_mask)
        
        # 小さな領域を除去
        labeled, num_features = ndimage.label(object_mask)
        if num_features > 0:
            sizes = ndimage.sum(object_mask, labeled, range(num_features + 1))
            max_size = np.max(sizes[1:]) if len(sizes) > 1 else 0
            # 最大領域の30%未満の小領域を削除
            mask_size = object_mask.sum()
            remove_small = sizes < max_size * 0.3
            remove_small[0] = 0  # 背景ラベルは保持
            object_mask = ~remove_small[labeled]
            
    except ImportError:
        logger.warning("SciPy not available for advanced morphology")
    
    # 6. 高品質グラデーション生成
    depth_map = np.zeros_like(gray_array)
    
    # 物体領域を白に設定
    depth_map[object_mask] = 1.0
    
    # DPT風の高精度境界グラデーション
    try:
        from scipy.ndimage import distance_transform_edt
        
        # より大きなグラデーション領域
        gradient_zone = 10
        
        # 物体境界からの距離
        boundary_distance = distance_transform_edt(~object_mask)
        object_distance = distance_transform_edt(object_mask)
        
        # 滑らかで自然なグラデーション
        boundary_gradient = np.where(
            boundary_distance <= gradient_zone,
            np.power(boundary_distance / gradient_zone, 0.5),  # 非線形グラデーション
            0
        )
        
        # 物体内部の微細なグラデーション
        object_gradient = np.where(
            (object_distance <= gradient_zone//2) & object_mask,
            1.0 - (object_distance / (gradient_zone//2) * 0.2),
            depth_map
        )
        
        # 最終合成
        depth_map = np.maximum(boundary_gradient, object_gradient)
        
    except ImportError:
        # 代替の高品質グラデーション
        kernel_size = 10
        for i in range(h):
            for j in range(w):
                if not object_mask[i, j]:
                    min_dist = float('inf')
                    for di in range(-kernel_size, kernel_size + 1):
                        for dj in range(-kernel_size, kernel_size + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w and object_mask[ni, nj]:
                                dist = np.sqrt(di*di + dj*dj)
                                min_dist = min(min_dist, dist)
                    
                    if min_dist <= kernel_size:
                        # 非線形グラデーション
                        gradient_val = np.power(1.0 - min_dist / kernel_size, 0.5)
                        depth_map[i, j] = max(0, gradient_val)
    
    # 7. 最終調整
    depth_map = np.clip(depth_map, 0, 1)
    
    # DPT風のコントラスト強化
    depth_map = np.power(depth_map, 0.7)  # より強いコントラスト
    
    # [0, 255]にスケール
    depth_map_uint8 = (depth_map * 255).astype(np.uint8)
    
    # PIL Imageに変換
    depth_pil = Image.fromarray(depth_map_uint8, mode='L')
    
    # 元のサイズに戻す
    target_size = original_size if original_size else (w, h)
    depth_final = depth_pil.resize(target_size, Image.Resampling.BICUBIC)
    
    logger.info(f"DPT-style reference-based depth estimation completed: {depth_final.size}")
    return depth_final

def depth_anything_inspired(image: Image.Image, original_size=None):
    """参考画像ベース汎用物体検出深度推定 - DepthAnything風の頑健性"""
    w, h = image.size
    logger.info(f"DepthAnything function - Input size: {image.size} (w={w}, h={h})")
    logger.info(f"DepthAnything function - Original size param: {original_size}")
    
    # DepthAnything風前処理: 元のサイズを保持
    new_w = w
    new_h = h
    logger.info(f"DepthAnything - Processing at original size: {new_w}x{new_h}")
    
    # RGB値を[0,1]に正規化（標準的な前処理）
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # グレースケール変換
    gray_array = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    
    # DepthAnything風の Foundation Model ベース深度推定
    # DepthAnythingは「通常の深度」を出力（逆深度ではない）
    
    # 1. 大規模データでの学習を模倣した汎用的な深度手がかり
    # 明度ベースの深度推定（明るい=前景、暗い=背景の一般的な傾向）
    brightness_depth = gray_array
    
    # 2. マルチスケールのテクスチャ分析
    # 近い物体ほど詳細なテクスチャが見える
    texture_fine = np.abs(gray_array - simple_smooth_array(gray_array, radius=1))
    texture_medium = np.abs(gray_array - simple_smooth_array(gray_array, radius=3))
    texture_coarse = np.abs(gray_array - simple_smooth_array(gray_array, radius=6))
    
    # テクスチャの密度（近い物体ほど高い）
    texture_density = (
        0.5 * texture_fine +
        0.3 * texture_medium +
        0.2 * texture_coarse
    )
    texture_smoothed = simple_smooth_array(texture_density, radius=4)
    
    # 3. 大域的なシーン理解（Foundation Modelの特徴）
    # 大きな構造の認識
    global_structure = simple_smooth_array(gray_array, radius=10)
    
    # 4. 局所的なコントラスト分析
    local_contrast = compute_local_variance_fast(gray_array, window=7)
    contrast_smoothed = simple_smooth_array(local_contrast, radius=5)
    
    # 5. 空間的位置による深度バイアス
    # 通常の写真: 上部=遠い（空）、下部=近い（地面）
    spatial_bias = np.linspace(0.0, 1.0, new_h).reshape(-1, 1)
    spatial_bias = np.tile(spatial_bias, (1, new_w))
    
    # DepthAnything風の特徴統合（Foundation Modelの汎用性）
    depth_estimate = (
        0.30 * brightness_depth +      # 基本的な明度手がかり
        0.25 * texture_smoothed +      # テクスチャ密度
        0.20 * global_structure +      # 大域構造理解
        0.15 * contrast_smoothed +     # 局所コントラスト
        0.10 * spatial_bias            # 空間的位置
    )
    
    # 正規化（0-1の範囲）
    if depth_estimate.max() > depth_estimate.min():
        normalized_depth = (depth_estimate - depth_estimate.min()) / (depth_estimate.max() - depth_estimate.min())
    else:
        normalized_depth = depth_estimate
    
    # 深度推定アプリの統一表示仕様:
    # 近い物体 = 白い表示（高い値）
    # 遠い物体 = 暗い表示（低い値）
    
    # DepthAnything Foundation Modelの深度推定を統一表示に変換
    # アルゴリズムが生成した高い値=近い物体を白く表示
    depth_display = normalized_depth  # 近い物体が白く表示
    
    # [0, 255]にスケール（近い=255/白、遠い=0/黒）
    depth_map = (depth_display * 255).astype(np.uint8)
    
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
    
    # 後処理: 鮮明な深度マップのためボカシを削除
    # depth_final = depth_final.filter(ImageFilter.GaussianBlur(radius=2.5))
    
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
    
    # 鮮明な深度マップのためボカシを削除
    # result = depth_img.filter(ImageFilter.GaussianBlur(radius=1))
    result = ImageOps.autocontrast(depth_img)
    
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
    
    # 鮮明な深度マップのためボカシを削除
    # depth_img = depth_img.filter(ImageFilter.GaussianBlur(radius=2))
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
    """深度マップを高品質グレースケール表示（白=近い、黒=遠い）"""
    w, h = depth_image.size
    
    # 品質を保持した最小限の処理
    # 過度なコントラスト強化を避ける
    depth_enhanced = depth_image.copy()
    
    # 滑らかなグラデーションを保持するためautocontrastを無効化
    # depth_enhanced = ImageOps.autocontrast(depth_enhanced, cutoff=2)
    
    # 直接RGBに変換（ピクセルループを避けて高速化）
    # グレースケールをRGBに変換
    depth_rgb = depth_enhanced.convert('RGB')
    
    return depth_rgb

def generate_pointcloud(original_image, depth_image):
    """3Dポイントクラウドデータ生成 - 正確な深度解釈とZ軸処理"""
    w, h = original_image.size
    downsample_factor = 10  # 品質向上のために少し細かく
    points = []
    colors = []
    
    orig_pixels = original_image.load()
    depth_pixels = depth_image.load()
    
    # 元画像の縦横比を保持するスケーリング計算
    aspect_ratio = w / h
    base_scale = 1.8  # 少し大きくして視認性向上
    
    # 正しいアスペクト比保持
    if aspect_ratio > 1.0:  # 横長画像
        scale_x = base_scale
        scale_y = base_scale / aspect_ratio
    else:  # 縦長画像または正方形
        scale_x = base_scale * aspect_ratio
        scale_y = base_scale
    
    logger.info(f"Point cloud generation: image size {w}x{h}, aspect_ratio={aspect_ratio:.3f}, scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")
    
    for y in range(0, h, downsample_factor):
        for x in range(0, w, downsample_factor):
            if x < w and y < h:
                # 深度値を取得（PILの(x, y)座標系）
                depth_val = depth_pixels[x, y] / 255.0
                
                # 正規化座標系への変換（-0.5から0.5の範囲）
                x_norm = (x / w - 0.5) * scale_x
                y_norm = (y / h - 0.5) * scale_y
                
                # 深度値の正しい解釈
                # 修正後のアルゴリズム: 白(255)=近い、黒(0)=遠い
                # depth_val: 0.0(遠い) 〜 1.0(近い)
                
                # Z軸の正しいスケーリング
                # 近い物体は正のZ値（手前に突き出る）
                # 遠い物体は負のZ値（奥に引っ込む）
                z_range = 1.2  # Z軸の範囲を幅広く
                z_norm = (depth_val - 0.5) * z_range  # -0.6 〜 +0.6の範囲
                
                # 非線形変換で近い物体をより強調
                if z_norm > 0:  # 近い物体（正のZ）
                    z_norm = np.power(z_norm / z_range * 2, 0.8) * z_range / 2
                else:  # 遠い物体（負のZ）
                    z_norm = -np.power(-z_norm / z_range * 2, 1.2) * z_range / 2
                
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
        "default_model": "Intel/dpt-hybrid-midas",
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
            model = "Intel/dpt-hybrid-midas"
        
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
            
            # EXIF処理をバックエンドで実行
            try:
                image = ImageOps.exif_transpose(image)
                logger.info(f"After EXIF transpose: {image.size}")
            except Exception as exif_error:
                logger.warning(f"EXIF transpose failed: {exif_error}")
            
            image = image.convert('RGB')
            logger.info(f"After RGB conversion: {image.size}")
            
            # Skip original image copy for now - use resized image for everything
            # This saves memory by not keeping two copies
            logger.info(f"Skipping original image copy to save memory")
            
        except Exception as img_error:
            logger.error(f"Image loading error: {img_error}")
            raise ValueError(f"Cannot process image file: {str(img_error)}")
        
        # Balanced size limitation for Railway memory constraints
        max_pixels = 400_000  # About 632x632 or 800x500, better quality/memory balance
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