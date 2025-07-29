from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageFilter, ImageOps
import base64
import io
import os
import math

app = FastAPI(title="Advanced Computer Vision Depth Estimation API")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def advanced_edge_detection(image):
    """Pillow ベースのエッジ検出"""
    gray = image.convert('L')
    
    # エッジ検出 (FIND_EDGES)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    
    # より強いエッジ検出のためにコントラスト強化
    edges = ImageOps.autocontrast(edges)
    
    return edges

def texture_analysis(image):
    """テクスチャ分析 (局所分散近似)"""
    gray = image.convert('L')
    w, h = gray.size
    
    # 新しい画像を作成
    texture_img = Image.new('L', (w, h))
    pixels = gray.load()
    texture_pixels = texture_img.load()
    
    # 3x3 ウィンドウでの局所分散計算
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            # 3x3 近傍の値を取得
            values = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    values.append(pixels[x + dx, y + dy])
            
            # 分散計算
            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
            texture_pixels[x, y] = min(255, int(math.sqrt(variance) * 10))
    
    return texture_img

def gradient_magnitude(image):
    """グラデーション強度計算"""
    gray = image.convert('L')
    w, h = gray.size
    
    # Sobel フィルタ近似
    gradient_img = Image.new('L', (w, h))
    pixels = gray.load()
    grad_pixels = gradient_img.load()
    
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            # Sobel X
            gx = (pixels[x+1, y-1] + 2*pixels[x+1, y] + pixels[x+1, y+1] -
                  pixels[x-1, y-1] - 2*pixels[x-1, y] - pixels[x-1, y+1])
            
            # Sobel Y  
            gy = (pixels[x-1, y+1] + 2*pixels[x, y+1] + pixels[x+1, y+1] -
                  pixels[x-1, y-1] - 2*pixels[x, y-1] - pixels[x+1, y-1])
            
            # マグニチュード
            magnitude = math.sqrt(gx*gx + gy*gy)
            grad_pixels[x, y] = min(255, int(magnitude / 8))
    
    return gradient_img

def advanced_depth_estimation(image):
    """Pillow のみで高度な深度推定"""
    w, h = image.size
    center_x, center_y = w // 2, h // 2
    
    # 1. エッジ検出
    edges = advanced_edge_detection(image)
    edge_pixels = edges.load()
    
    # 2. テクスチャ分析
    texture = texture_analysis(image)
    texture_pixels = texture.load()
    
    # 3. グラデーション分析
    gradient = gradient_magnitude(image)
    gradient_pixels = gradient.load()
    
    # 4. 最終深度マップ作成
    depth_img = Image.new('L', (w, h))
    depth_pixels = depth_img.load()
    
    max_distance = math.sqrt(center_x**2 + center_y**2)
    
    for y in range(h):
        for x in range(w):
            # 中心からの距離
            distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            distance_norm = distance / max_distance
            
            # 各特徴量を正規化
            edge_val = edge_pixels[x, y] / 255.0
            texture_val = texture_pixels[x, y] / 255.0 if texture_pixels[x, y] else 0
            gradient_val = gradient_pixels[x, y] / 255.0 if gradient_pixels[x, y] else 0
            
            # 重み付き合成
            depth_value = (
                0.4 * (1 - distance_norm) +     # 中心ほど近い
                0.2 * texture_val +             # テクスチャが豊富 = 近い
                0.2 * gradient_val +            # グラデーション強い = 近い  
                0.2 * (1 - edge_val)           # エッジでない = 遠い
            )
            
            depth_pixels[x, y] = min(255, max(0, int(depth_value * 255)))
    
    # 5. スムージング
    depth_img = depth_img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    return depth_img

def apply_viridis_colormap_pillow(depth_image):
    """Pillow でViridis風カラーマップ"""
    w, h = depth_image.size
    colored_img = Image.new('RGB', (w, h))
    
    depth_pixels = depth_image.load()
    colored_pixels = colored_img.load()
    
    # Viridis 色定義
    viridis_colors = [
        (68, 1, 84),      # 濃い紫
        (59, 82, 139),    # 青紫  
        (33, 144, 140),   # 青緑
        (93, 201, 99),    # 緑
        (253, 231, 37)    # 黄色
    ]
    
    for y in range(h):
        for x in range(w):
            depth_val = depth_pixels[x, y] / 255.0
            
            # 色インデックス計算
            color_idx = depth_val * 4
            idx = int(color_idx)
            alpha = color_idx - idx
            
            if idx >= 4:
                color = viridis_colors[4]
            else:
                color1 = viridis_colors[idx]
                color2 = viridis_colors[min(idx + 1, 4)]
                
                # 線形補間
                color = (
                    int(color1[0] * (1 - alpha) + color2[0] * alpha),
                    int(color1[1] * (1 - alpha) + color2[1] * alpha),
                    int(color1[2] * (1 - alpha) + color2[2] * alpha)
                )
            
            colored_pixels[x, y] = color
    
    return colored_img

def generate_pointcloud(original_image, depth_image):
    """3Dポイントクラウドデータ生成"""
    w, h = original_image.size
    
    # ダウンサンプリング（3D表示用に軽量化）
    downsample_factor = 4
    points = []
    colors = []
    
    orig_pixels = original_image.load()
    depth_pixels = depth_image.load()
    
    for y in range(0, h, downsample_factor):
        for x in range(0, w, downsample_factor):
            # 深度値を取得（0-255 → 0-1 → 実際の深度）
            depth_val = depth_pixels[x, y] / 255.0
            
            # 3D座標計算 - 正しい画像座標系変換（backendと統一）
            # X,Y: 画像座標を正規化
            x_norm = (x / w - 0.5) * 1.6  # -0.8 to 0.8 やや圧縮
            y_norm = -((y / h - 0.5) * 1.6)  # -0.8 to 0.8（Y軸反転で画像上部が3D上部に）
            
            # Z: 深度値（深い = 遠い）- 初期設定に復元
            z_norm = (1.0 - depth_val) * 2 - 1  # -1 to 1（初期設定）
            
            # ポイント追加
            points.append([x_norm, y_norm, z_norm])
            
            # 色情報取得
            r, g, b = orig_pixels[x, y]
            colors.append([r/255.0, g/255.0, b/255.0])
    
    return {
        "points": points,
        "colors": colors,
        "count": len(points),
        "downsample_factor": downsample_factor
    }

@app.get("/")
async def root():
    return {
        "message": "Advanced Computer Vision Depth Estimation API", 
        "status": "running",
        "model": "Pillow-Advanced-CV",
        "algorithms": ["Edge Detection", "Texture Analysis", "Gradient Magnitude", "Distance Transform"],
        "note": "Real computer vision algorithms using only Pillow - no NumPy dependencies",
        "version": "1.1.0",
        "features": ["2D Depth Map", "3D Point Cloud Generation"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "algorithms": "pillow_cv_ready"
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
        
        print(f"Processing with Pillow-based CV: {image.size}")
        
        # Pillow ベース深度推定
        depth_gray = advanced_depth_estimation(image)
        
        # カラーマップ適用
        depth_colored = apply_viridis_colormap_pillow(depth_gray)
        
        print(f"✅ Pillow-based depth estimation completed")
        
        # Base64エンコード
        def image_to_base64(img):
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_base64}"
        
        # 3Dポイントクラウドデータ生成
        pointcloud_data = generate_pointcloud(image, depth_gray)
        
        return JSONResponse({
            "success": True,
            "originalUrl": image_to_base64(image),
            "depthMapUrl": image_to_base64(depth_colored),
            "pointcloudData": pointcloud_data,
            "model": "Pillow-Advanced-CV",
            "resolution": f"{image.size[0]}x{image.size[1]}",
            "note": "Real computer vision depth estimation using Pillow-based edge detection, texture analysis, and gradient computation",
            "algorithms": ["Pillow Edge Detection", "Texture Variance Analysis", "Sobel Gradient", "Distance Transform"],
            "implementation": "Pure Pillow - No NumPy",
            "features": ["2D Depth Map", "3D Point Cloud"]
        })
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Depth estimation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)