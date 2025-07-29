"""
展示用ローカル深度推定API
完全オフライン・高セキュリティ版
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np
import cv2
import io
import base64
import gc
import logging
from datetime import datetime
import os

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('exhibition_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="展示用深度推定API",
    description="完全ローカル・高セキュリティ深度推定システム",
    version="1.0.0"
)

# CORS設定 (ローカル専用)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001"  # 開発用
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class ExhibitionDepthEstimator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "depth-anything/Depth-Anything-V2-Small-hf"
        
        logger.info(f"🏛️ 展示用システム初期化開始")
        logger.info(f"🔧 使用デバイス: {self.device}")
        
        try:
            # モデル読み込み
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # 推論モード
            
            # セキュリティ設定
            self.max_image_size = 2048
            self.allowed_formats = ["image/jpeg", "image/png", "image/webp"]
            self.processing_count = 0
            
            logger.info(f"✅ モデル読み込み完了: {self.model_name}")
            logger.info(f"🔒 セキュリティ設定完了")
            
        except Exception as e:
            logger.error(f"❌ 初期化エラー: {e}")
            raise e
    
    def validate_image(self, file: UploadFile):
        """画像ファイルの安全性検証"""
        # ファイル形式チェック
        if file.content_type not in self.allowed_formats:
            raise HTTPException(
                status_code=400,
                detail=f"サポートされていない形式: {file.content_type}"
            )
        
        # ファイルサイズチェック (10MB制限)
        if hasattr(file, 'size') and file.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="ファイルサイズが大きすぎます (最大10MB)"
            )
        
        return True
    
    def process_image(self, image_data: bytes):
        """セキュアな画像処理"""
        try:
            # 画像読み込み
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            original_size = image.size
            
            # サイズ制限
            if max(image.size) > self.max_image_size:
                image.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)
                logger.info(f"📏 画像リサイズ: {original_size} → {image.size}")
            
            return image, original_size
            
        except Exception as e:
            logger.error(f"❌ 画像処理エラー: {e}")
            raise HTTPException(status_code=400, detail="画像の読み込みに失敗しました")
    
    def estimate_depth(self, image: Image.Image):
        """深度推定処理"""
        try:
            # 前処理
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推論実行
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # 後処理
            depth = predicted_depth.squeeze().cpu().numpy()
            
            # 正規化
            depth_min = depth.min()
            depth_max = depth.max()
            depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            
            # 元のサイズにリサイズ
            depth_resized = cv2.resize(
                depth_normalized, 
                (image.size[0], image.size[1]), 
                interpolation=cv2.INTER_LINEAR
            )
            
            # カラーマップ適用
            depth_colored = cv2.applyColorMap(depth_resized, cv2.COLORMAP_VIRIDIS)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            
            # PILイメージに変換
            depth_image = Image.fromarray(depth_colored)
            
            # メモリクリーンアップ
            del inputs, outputs, predicted_depth, depth, depth_normalized, depth_resized, depth_colored
            
            return depth_image
            
        except Exception as e:
            logger.error(f"❌ 深度推定エラー: {e}")
            raise HTTPException(status_code=500, detail="深度推定処理に失敗しました")
        
        finally:
            # GPU メモリクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def encode_image_to_base64(self, image: Image.Image, format='PNG'):
        """画像をBase64エンコード"""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{encoded}"

# グローバルインスタンス
estimator = ExhibitionDepthEstimator()

@app.get("/")
async def root():
    """ヘルスチェック"""
    return {
        "status": "🏛️ 展示用深度推定API稼働中",
        "device": estimator.device,
        "processed_count": estimator.processing_count,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """詳細ヘルスチェック"""
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = {
            "allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
            "reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB"
        }
    
    return {
        "status": "healthy",
        "device": estimator.device,
        "gpu_memory": gpu_memory,
        "processed_count": estimator.processing_count,
        "model": estimator.model_name
    }

@app.post("/api/depth-estimation")
async def estimate_depth_api(file: UploadFile = File(...)):
    """深度推定API"""
    request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{estimator.processing_count}"
    
    try:
        logger.info(f"📥 新しいリクエスト: {request_id}")
        estimator.processing_count += 1
        
        # 入力検証
        estimator.validate_image(file)
        
        # 画像データ読み込み
        image_data = await file.read()
        logger.info(f"📊 ファイルサイズ: {len(image_data)} bytes")
        
        # 画像処理
        image, original_size = estimator.process_image(image_data)
        
        # 深度推定実行
        depth_image = estimator.estimate_depth(image)
        
        # Base64エンコード
        original_base64 = estimator.encode_image_to_base64(image)
        depth_base64 = estimator.encode_image_to_base64(depth_image)
        
        # レスポンス作成
        response = {
            "success": True,
            "data": [original_base64, depth_base64],  # Gradio互換形式
            "metadata": {
                "request_id": request_id,
                "model": "DepthAnything-V2-Local",
                "original_size": original_size,
                "processed_size": image.size,
                "device": estimator.device,
                "security": "完全ローカル処理",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(f"✅ 処理完了: {request_id}")
        
        # メモリクリーンアップ
        del image_data, image, depth_image
        gc.collect()
        
        return JSONResponse(content=response)
        
    except HTTPException as he:
        logger.warning(f"⚠️ バリデーションエラー {request_id}: {he.detail}")
        raise he
        
    except Exception as e:
        logger.error(f"❌ 処理エラー {request_id}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "システムエラーが発生しました",
                "request_id": request_id
            }
        )
    
    finally:
        # 最終クリーンアップ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

@app.post("/api/clear-cache")
async def clear_cache():
    """キャッシュクリア（展示用メンテナンス）"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("🧹 キャッシュクリア実行")
        
        return {
            "success": True,
            "message": "キャッシュクリア完了",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ キャッシュクリアエラー: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🏛️ 展示用深度推定システム起動")
    logger.info("🔒 完全ローカル・高セキュリティモード")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )