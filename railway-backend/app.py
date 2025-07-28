from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import base64
import io
import os
import requests

app = FastAPI(title="Depth Estimation API - Lightweight")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def create_mock_depth_map(image):
    """軽量なモック深度マップ生成"""
    # グレースケール変換
    gray = image.convert('L')
    gray_array = np.array(gray)
    
    # 簡単な深度風エフェクト（中央が近く、端が遠い）
    h, w = gray_array.shape
    center_x, center_y = w // 2, h // 2
    
    # 距離マップ作成
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    # 正規化して深度マップ作成
    depth_map = 255 - (distance / max_distance * 255)
    depth_map = depth_map.astype(np.uint8)
    
    return Image.fromarray(depth_map, mode='L')

@app.get("/")
async def root():
    return {
        "message": "Lightweight Depth Estimation API", 
        "status": "running",
        "model": "mock-gradient",
        "note": "Using lightweight mock implementation for Railway deployment"
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
        max_size = 512
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # モック深度マップ生成
        depth_gray = create_mock_depth_map(image)
        
        # カラーマップ適用（手動でviridis風）
        depth_array = np.array(depth_gray)
        colored_depth = np.zeros((depth_array.shape[0], depth_array.shape[1], 3), dtype=np.uint8)
        
        # 簡単なカラーマップ
        colored_depth[:, :, 0] = depth_array // 4  # R
        colored_depth[:, :, 1] = depth_array // 2  # G  
        colored_depth[:, :, 2] = depth_array       # B
        
        depth_image = Image.fromarray(colored_depth)
        
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
            "model": "Mock-Gradient",
            "resolution": f"{image.size[0]}x{image.size[1]}",
            "note": "Lightweight implementation for demo purposes"
        })
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)