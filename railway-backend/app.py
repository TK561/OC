from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageFilter
import base64
import io
import os
import math

app = FastAPI(title="Depth Estimation API - Ultra Lightweight")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_mock_depth_map(image):
    """NumPy不要の軽量深度マップ生成"""
    # グレースケール変換
    gray = image.convert('L')
    w, h = gray.size
    center_x, center_y = w // 2, h // 2
    
    # 新しい画像を作成
    depth_image = Image.new('L', (w, h))
    pixels = depth_image.load()
    
    # 最大距離計算
    max_distance = math.sqrt(center_x**2 + center_y**2)
    
    # ピクセルごとに深度値計算
    for y in range(h):
        for x in range(w):
            # 中心からの距離
            distance = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            # 深度値（中央が明るく、端が暗く）
            depth_value = int(255 - (distance / max_distance * 255))
            pixels[x, y] = max(0, min(255, depth_value))
    
    return depth_image

def create_colored_depth(depth_gray):
    """グレースケールからカラー深度マップ"""
    w, h = depth_gray.size
    colored = Image.new('RGB', (w, h))
    pixels_colored = colored.load()
    pixels_gray = depth_gray.load()
    
    for y in range(h):
        for x in range(w):
            gray_val = pixels_gray[x, y]
            # シンプルなカラーマップ (青→緑→赤)
            if gray_val < 85:
                r, g, b = 0, gray_val * 3, 255
            elif gray_val < 170:
                r, g, b = 0, 255, 255 - (gray_val - 85) * 3
            else:
                r, g, b = (gray_val - 170) * 3, 255 - (gray_val - 170) * 3, 0
            
            pixels_colored[x, y] = (r, g, b)
    
    return colored

@app.get("/")
async def root():
    return {
        "message": "Ultra Lightweight Depth Estimation API", 
        "status": "running",
        "model": "pure-python-mock",
        "note": "Pure Python implementation without NumPy for Railway deployment"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/api/predict")
async def predict_depth(file: UploadFile = File(...)):
    try:
        # 画像読み込み
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # サイズ制限
        max_size = 256  # さらに小さく
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # モック深度マップ生成
        depth_gray = create_mock_depth_map(image)
        depth_colored = create_colored_depth(depth_gray)
        
        # Base64エンコード
        def image_to_base64(img):
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_base64}"
        
        return JSONResponse({
            "success": True,
            "originalUrl": image_to_base64(image),
            "depthMapUrl": image_to_base64(depth_colored),
            "model": "Pure-Python-Mock",
            "resolution": f"{image.size[0]}x{image.size[1]}",
            "note": "Pure Python implementation without heavy dependencies"
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)