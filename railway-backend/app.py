from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation
import numpy as np
from PIL import Image
import cv2
import base64
import io
import os

app = FastAPI(title="AI Depth Estimation API")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル変数
processor = None
model = None
device = "cpu"

def load_model():
    """軽量DPTモデルを読み込み"""
    global processor, model
    if processor is None or model is None:
        print("Loading DPT-Small model...")
        model_name = "Intel/dpt-hybrid-midas"
        try:
            processor = DPTImageProcessor.from_pretrained(model_name)
            model = DPTForDepthEstimation.from_pretrained(model_name)
            model.to(device)
            model.eval()
            print(f"✅ DPT model loaded successfully on {device}")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise e

# 起動時にモデル読み込み
@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        print(f"Startup failed: {e}")

@app.get("/")
async def root():
    return {
        "message": "AI Depth Estimation API", 
        "status": "running",
        "model": "Intel/dpt-hybrid-midas",
        "device": device,
        "note": "Real AI depth estimation using DPT model"
    }

@app.get("/health")
async def health_check():
    model_status = model is not None and processor is not None
    return {
        "status": "healthy" if model_status else "loading",
        "model_loaded": model_status,
        "device": device
    }

@app.post("/api/predict")
async def predict_depth(file: UploadFile = File(...)):
    try:
        # モデル確認
        if model is None or processor is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
        
        # 画像読み込み
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # サイズ制限（Railway メモリ制限対応）
        max_size = 384
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        print(f"Processing image: {image.size}")
        
        # AI深度推定
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # 深度マップ後処理
        depth = predicted_depth.squeeze().cpu().numpy()
        depth_min = depth.min()
        depth_max = depth.max()
        
        # 正規化
        if depth_max > depth_min:
            depth_norm = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_norm = depth
        
        # 0-255スケールに変換
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        
        # カラーマップ適用
        depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        depth_image = Image.fromarray(depth_colored)
        
        print(f"✅ Depth estimation completed")
        
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
            "model": "Intel/dpt-hybrid-midas",
            "resolution": f"{image.size[0]}x{image.size[1]}",
            "note": "Real AI depth estimation using DPT model",
            "depth_range": f"{depth_min:.3f} - {depth_max:.3f}"
        })
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Depth estimation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)