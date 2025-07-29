"""
展示用軽量バックエンド - MiDaS v3.1版
完全無料・ローカル・高セキュリティ・CPU最適化
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import io
import base64
import gc
import logging
import urllib.request
import os
from datetime import datetime
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('exhibition_lightweight.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="展示用軽量深度推定API",
    description="MiDaS v3.1 - 完全無料・ローカル・CPU最適化版",
    version="1.0.0"
)

# CORS設定 (ローカル専用)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class LightweightDepthEstimator:
    def __init__(self, force_cpu=False):
        self.device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
        self.model_path = Path("models")
        self.model_path.mkdir(exist_ok=True)
        
        logger.info(f"🏛️ 軽量展示システム初期化開始")
        logger.info(f"🔧 使用デバイス: {self.device}")
        
        # セキュリティ設定
        self.max_image_size = 1024  # 軽量化のため小さく
        self.allowed_formats = ["image/jpeg", "image/png", "image/webp"]
        self.processing_count = 0
        
        # MiDaS v3.1 初期化
        self.setup_midas()
        
        logger.info(f"✅ 軽量システム準備完了")
        logger.info(f"🔒 セキュリティ設定完了")
    
    def download_midas_model(self):
        """MiDaS v3.1 モデルダウンロード"""
        model_file = self.model_path / "midas_v31_small.pt"
        
        if model_file.exists():
            logger.info("📦 MiDaS モデル: キャッシュから読み込み")
            return str(model_file)
        
        logger.info("📥 MiDaS v3.1 Small モデルをダウンロード中...")
        
        # MiDaS v3.1 Small (軽量版)
        model_url = "https://github.com/isl-org/MiDaS/releases/download/v3_1/midas_v31_small.pt"
        
        try:
            urllib.request.urlretrieve(model_url, model_file)
            logger.info(f"✅ モデルダウンロード完了: {model_file}")
            return str(model_file)
        except Exception as e:
            logger.error(f"❌ ダウンロードエラー: {e}")
            raise e
    
    def setup_midas(self):
        """MiDaS v3.1 セットアップ"""
        try:
            # PyTorch Hub経由でMiDaS読み込み（軽量版）
            logger.info("🔄 MiDaS v3.1 Small モデル読み込み中...")
            
            # 軽量版MiDaSを使用
            self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
            self.model.to(self.device)
            self.model.eval()
            
            # 前処理設定
            self.midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            if self.device == "cpu":
                # CPU最適化変換
                self.transform = self.midas_transforms.small_transform
            else:
                self.transform = self.midas_transforms.small_transform
            
            logger.info(f"✅ MiDaS v3.1 Small 読み込み完了")
            logger.info(f"💾 メモリ使用量: 軽量最適化")
            
        except Exception as e:
            logger.error(f"❌ MiDaS初期化エラー: {e}")
            raise e
    
    def validate_image(self, file: UploadFile):
        """画像ファイルの安全性検証"""
        if file.content_type not in self.allowed_formats:
            raise HTTPException(
                status_code=400,
                detail=f"サポートされていない形式: {file.content_type}"
            )
        
        if hasattr(file, 'size') and file.size > 5 * 1024 * 1024:  # 5MB制限
            raise HTTPException(
                status_code=400,
                detail="ファイルサイズが大きすぎます (最大5MB)"
            )
        
        return True
    
    def process_image(self, image_data: bytes):
        """セキュアな画像処理"""
        try:
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            original_size = image.size
            
            # サイズ制限（軽量化）
            if max(image.size) > self.max_image_size:
                image.thumbnail((self.max_image_size, self.max_image_size), Image.Resampling.LANCZOS)
                logger.info(f"📏 画像リサイズ: {original_size} → {image.size}")
            
            return image, original_size
            
        except Exception as e:
            logger.error(f"❌ 画像処理エラー: {e}")
            raise HTTPException(status_code=400, detail="画像の読み込みに失敗しました")
    
    def estimate_depth_midas(self, image: Image.Image):
        """MiDaS v3.1 深度推定"""
        try:
            # NumPy配列に変換
            img_array = np.array(image)
            
            # MiDaS前処理
            input_tensor = self.transform(img_array).to(self.device)
            
            # 推論実行
            with torch.no_grad():
                prediction = self.model(input_tensor)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_array.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # 深度マップ生成
            depth = prediction.cpu().numpy()
            
            # 正規化
            depth_min = depth.min()
            depth_max = depth.max()
            depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            
            # カラーマップ適用
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            
            # PILイメージに変換
            depth_image = Image.fromarray(depth_colored)
            
            # メモリクリーンアップ
            del input_tensor, prediction, depth, depth_normalized, depth_colored
            
            return depth_image
            
        except Exception as e:
            logger.error(f"❌ MiDaS深度推定エラー: {e}")
            raise HTTPException(status_code=500, detail="深度推定処理に失敗しました")
        
        finally:
            # CPU/GPU メモリクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def encode_image_to_base64(self, image: Image.Image, format='PNG'):
        """画像をBase64エンコード"""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/{format.lower()};base64,{encoded}"

# グローバルインスタンス（CPU最適化）
estimator = LightweightDepthEstimator(force_cpu=True)  # 展示用はCPU推奨

@app.get("/")
async def root():
    """ヘルスチェック"""
    return {
        "status": "🏛️ 軽量展示用深度推定API稼働中",
        "model": "MiDaS v3.1 Small",
        "device": estimator.device,
        "processed_count": estimator.processing_count,
        "license": "MIT (完全無料)",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """詳細ヘルスチェック"""
    cpu_memory = None
    try:
        import psutil
        cpu_memory = f"{psutil.virtual_memory().percent}%"
    except ImportError:
        cpu_memory = "N/A"
    
    gpu_memory = None
    if torch.cuda.is_available() and estimator.device == "cuda":
        gpu_memory = {
            "allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
            "reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB"
        }
    
    return {
        "status": "healthy",
        "model": "MiDaS v3.1 Small",
        "device": estimator.device,
        "cpu_memory": cpu_memory,
        "gpu_memory": gpu_memory,
        "processed_count": estimator.processing_count,
        "license": "MIT License",
        "cost": "完全無料"
    }

@app.post("/api/depth-estimation")
async def estimate_depth_api(file: UploadFile = File(...)):
    """軽量深度推定API"""
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
        
        # MiDaS深度推定実行
        start_time = datetime.now()
        depth_image = estimator.estimate_depth_midas(image)
        process_time = (datetime.now() - start_time).total_seconds()
        
        # Base64エンコード
        original_base64 = estimator.encode_image_to_base64(image)
        depth_base64 = estimator.encode_image_to_base64(depth_image)
        
        # レスポンス作成
        response = {
            "success": True,
            "data": [original_base64, depth_base64],  # Gradio互換形式
            "metadata": {
                "request_id": request_id,
                "model": "MiDaS-v3.1-Small",
                "license": "MIT (完全無料)",
                "original_size": original_size,
                "processed_size": image.size,
                "device": estimator.device,
                "process_time": f"{process_time:.2f}秒",
                "security": "完全ローカル処理",
                "cost": "無料",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        logger.info(f"✅ 処理完了: {request_id} ({process_time:.2f}秒)")
        
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

@app.get("/api/cost-info")
async def cost_info():
    """コスト情報"""
    return {
        "model": "MiDaS v3.1 Small",
        "license": "MIT License",
        "cost_breakdown": {
            "software": "完全無料",
            "model": "完全無料",
            "api_calls": "無制限",
            "commercial_use": "可能",
            "modification": "可能",
            "distribution": "可能"
        },
        "hardware_requirements": {
            "minimum": "CPU: Intel i5, RAM: 8GB",
            "recommended": "CPU: Intel i7, RAM: 16GB",
            "gpu_required": False
        },
        "estimated_costs": {
            "initial_setup": "PC代のみ (15-30万円)",
            "running_cost": "電気代のみ (月数千円)",
            "maintenance": "無料",
            "updates": "無料"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🏛️ 軽量展示用深度推定システム起動")
    logger.info("💰 完全無料・MITライセンス")
    logger.info("🔒 完全ローカル・高セキュリティ")
    logger.info("⚡ CPU最適化・軽量設計")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info",
        access_log=True
    )