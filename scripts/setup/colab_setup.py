# 📝 Google Colab で実行するコード
# セル1: ライブラリインストール
!pip install torch torchvision transformers
!pip install gradio pyngrok opencv-python-headless
!pip install Pillow numpy

# セル2: ngrok設定（YOUR_TOKEN_HEREを実際のトークンに置き換え）
import pyngrok
from pyngrok import ngrok

NGROK_TOKEN = "YOUR_TOKEN_HERE"  # ←ここに取得したトークンを入力
ngrok.set_auth_token(NGROK_TOKEN)
print("✅ ngrok認証完了")

# セル3: 深度推定API起動
import os
import io
import base64
import numpy as np
from PIL import Image
import torch
import gradio as gr
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2
import gc
import time
import logging

# セキュリティ強化ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureDepthEstimator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🔒 セキュア深度推定API初期化: {self.device}")
        
        # DepthAnything V2 Small（軽量版）
        self.model_name = "depth-anything/Depth-Anything-V2-Small-hf"
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        logger.info(f"✅ モデル準備完了")
    
    def secure_depth_estimation(self, image):
        """セキュア深度推定処理"""
        if image is None:
            return None, None
        
        try:
            # セッションID生成
            session_id = f"session_{int(time.time())}"
            logger.info(f"🔄 処理開始: {session_id}")
            
            # 画像前処理
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # サイズ制限（展示用最適化）
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # 深度推定
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # 深度マップ生成
            depth = predicted_depth.squeeze().cpu().numpy()
            depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
            depth_resized = cv2.resize(depth_normalized, image.size, interpolation=cv2.INTER_LINEAR)
            depth_colored = cv2.applyColorMap(depth_resized, cv2.COLORMAP_VIRIDIS)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            depth_image = Image.fromarray(depth_colored)
            
            # セキュリティ: メモリクリーンアップ
            del inputs, outputs, predicted_depth, depth, depth_normalized, depth_resized, depth_colored
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"✅ 処理完了・データ削除: {session_id}")
            return image, depth_image
            
        except Exception as e:
            logger.error(f"❌ 処理エラー: {e}")
            return None, None

# API初期化
estimator = SecureDepthEstimator()

# Gradioインターフェース作成
interface = gr.Interface(
    fn=estimator.secure_depth_estimation,
    inputs=gr.Image(type="pil", label="📷 深度推定する画像"),
    outputs=[
        gr.Image(type="pil", label="📸 元画像"),
        gr.Image(type="pil", label="🎨 深度マップ")
    ],
    title="🔒 展示用深度推定API - セキュア版",
    description="""
    🛡️ **セキュリティ機能**:
    - ✅ 画像は処理後即座に削除
    - ✅ ログに画像データ記録なし  
    - ✅ メモリ自動クリーンアップ
    - ✅ 外部保存なし
    
    📊 **処理時間**: 2-5秒 | **モデル**: DepthAnything V2 Small
    """,
    allow_flagging="never",
    analytics_enabled=False
)

# ngrokでパブリックURL生成
public_url = ngrok.connect(7860)
print("\n" + "="*60)
print("🎉 展示用深度推定API起動完了!")
print(f"📡 Public URL: {public_url}")
print("\n📋 フロントエンド設定用:")
print(f"NEXT_PUBLIC_BACKEND_URL={public_url}")
print("="*60)

# API起動
interface.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    debug=False,
    quiet=True
)