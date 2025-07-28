from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import filters, feature, measure
import base64
import io
import os

app = FastAPI(title="Lightweight AI Depth Estimation API")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def advanced_depth_estimation(image):
    """高度な画像処理ベースの深度推定"""
    # RGB → グレースケール変換
    gray = np.array(image.convert('L'))
    h, w = gray.shape
    
    # 1. エッジ検出 (Canny)
    edges = feature.canny(gray, sigma=2.0)
    
    # 2. テクスチャ分析 (局所標準偏差)
    texture = ndimage.generic_filter(gray.astype(float), np.std, size=5)
    
    # 3. グラデーション解析
    grad_y, grad_x = np.gradient(gray.astype(float))
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # 4. 距離変換 (中心からの距離)
    center_x, center_y = w // 2, h // 2
    y, x = np.ogrid[:h, :w]
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    distance_norm = distance_from_center / max_distance
    
    # 5. 重み付き深度マップ合成
    depth_map = (
        0.3 * (1 - distance_norm) +          # 中心ほど近い
        0.2 * (texture / texture.max()) +   # テクスチャが豊富な部分は近い
        0.2 * (gradient_magnitude / gradient_magnitude.max()) +  # エッジ部分は近い
        0.3 * (1 - edges.astype(float))     # エッジではない部分は遠い
    )
    
    # 6. ガウシアンスムージング
    depth_map = ndimage.gaussian_filter(depth_map, sigma=1.5)
    
    # 7. 正規化
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    
    return depth_map

def apply_viridis_colormap(depth_array):
    """Viridis風カラーマップ適用"""
    # Viridis色定義 (簡略版)
    viridis_colors = np.array([
        [68, 1, 84],      # 濃い紫
        [59, 82, 139],    # 青紫
        [33, 144, 140],   # 青緑
        [93, 201, 99],    # 緑
        [253, 231, 37]    # 黄色
    ])
    
    h, w = depth_array.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 深度値を0-4の範囲にマップ
    indices = (depth_array * 4).astype(int)
    indices = np.clip(indices, 0, 3)
    
    # 線形補間でカラーマップ適用
    for i in range(h):
        for j in range(w):
            idx = indices[i, j]
            alpha = (depth_array[i, j] * 4) - idx
            
            if idx < 4:
                color = (1 - alpha) * viridis_colors[idx] + alpha * viridis_colors[idx + 1]
            else:
                color = viridis_colors[4]
            
            colored[i, j] = color.astype(np.uint8)
    
    return colored

@app.get("/")
async def root():
    return {
        "message": "Lightweight AI Depth Estimation API", 
        "status": "running",
        "model": "Advanced-Computer-Vision",
        "algorithms": ["Canny Edge", "Texture Analysis", "Gradient", "Distance Transform"],
        "note": "Real computer vision algorithms without heavy AI models"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "algorithms": "computer_vision_ready"
    }

@app.post("/api/predict")
async def predict_depth(file: UploadFile = File(...)):
    try:
        # 画像読み込み
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # サイズ制限
        max_size = 512
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        print(f"Processing image with advanced CV: {image.size}")
        
        # 高度な深度推定
        depth_map = advanced_depth_estimation(image)
        
        # カラーマップ適用
        depth_colored = apply_viridis_colormap(depth_map)
        depth_image = Image.fromarray(depth_colored)
        
        print(f"✅ Advanced depth estimation completed")
        
        # Base64エンコード
        def image_to_base64(img):
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_base64}"
        
        return JSONResponse({
            "success": True,
            "originalUrl": image_to_base64(image),
            "depthMapUrl": image_to_base64(depth_image),
            "model": "Advanced-Computer-Vision",
            "resolution": f"{image.size[0]}x{image.size[1]}",
            "note": "Real computer vision depth estimation using edge detection, texture analysis, and gradient computation",
            "algorithms": ["Canny Edge Detection", "Texture Analysis", "Gradient Computation", "Distance Transform"],
            "depth_range": f"{depth_map.min():.3f} - {depth_map.max():.3f}"
        })
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Depth estimation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)