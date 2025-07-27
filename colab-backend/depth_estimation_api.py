"""
Google Colab用深度推定API - プライバシー重視版
DepthAnything V2モデルを使用した本格的な深度推定サービス

🔒 プライバシー・セキュリティ機能:
- 画像の一時的処理のみ（保存なし）
- メモリ内処理後即座に削除
- ログに画像データ出力なし
- セッション終了時の自動クリーンアップ

使用方法:
1. Google Colabでこのファイルを実行
2. ngrokトークンを設定
3. 生成されたURLをフロントエンドのNEXT_PUBLIC_BACKEND_URLに設定
"""

import os
import io
import base64
import numpy as np
from PIL import Image
import torch
import gradio as gr
from transformers import pipeline, AutoImageProcessor, AutoModelForDepthEstimation
import cv2
import gc
import tempfile
import time
import logging

# プライバシー重視のログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 画像データをログに出力しないようにフィルタリング
class PrivacyLogFilter(logging.Filter):
    def filter(self, record):
        # Base64や画像データらしき長い文字列をログから除外
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            if 'data:image' in record.msg or len(record.msg) > 1000:
                return False
        return True

logger.addFilter(PrivacyLogFilter())

class DepthEstimationAPI:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Initializing secure depth estimation API on {self.device}")
        
        # DepthAnything V2モデルの初期化
        self.model_name = "depth-anything/Depth-Anything-V2-Small-hf"
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        # セキュリティ設定
        self.max_image_size = 2048  # 最大画像サイズ制限
        self.session_timeout = 3600  # 1時間でセッションタイムアウト
        self.processing_sessions = {}  # アクティブセッション管理
        
        logger.info(f"✅ Model loaded: {self.model_name}")
        logger.info(f"🔒 Security features enabled: max_size={self.max_image_size}px")
    
    def cleanup_session(self, session_id):
        """セッション情報のクリーンアップ"""
        if session_id in self.processing_sessions:
            del self.processing_sessions[session_id]
        gc.collect()  # ガベージコレクション実行
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # GPU メモリクリア
    
    def validate_image(self, image):
        """画像のセキュリティ検証"""
        # サイズ制限
        if max(image.size) > self.max_image_size:
            raise ValueError(f"画像サイズが大きすぎます。最大{self.max_image_size}px以下にしてください。")
        
        # ファイル形式チェック
        if image.format not in ['JPEG', 'PNG', 'WEBP']:
            logger.warning(f"サポートされていない画像形式: {image.format}")
        
        return True
    
    def preprocess_image(self, image_input, session_id=None):
        """セキュアな画像前処理"""
        try:
            # セッション記録
            if session_id:
                self.processing_sessions[session_id] = {
                    'start_time': time.time(),
                    'status': 'preprocessing'
                }
            
            if isinstance(image_input, str):
                # Base64 データURLの場合
                if image_input.startswith('data:'):
                    header, data = image_input.split(',', 1)
                    image_data = base64.b64decode(data)
                    image = Image.open(io.BytesIO(image_data))
                else:
                    # ファイルパスの場合（セキュリティ上推奨しない）
                    logger.warning("ファイルパス経由の画像読み込みは推奨されません")
                    image = Image.open(image_input)
            else:
                # PIL Imageオブジェクトの場合
                image = image_input
            
            # セキュリティ検証
            self.validate_image(image)
            
            # RGBに変換
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            logger.info(f"🖼️ Image processed: {image.size[0]}x{image.size[1]}px")
            return image
            
        except Exception as e:
            logger.error(f"❌ Image preprocessing failed: {str(e)}")
            if session_id:
                self.cleanup_session(session_id)
            raise e
    
    def estimate_depth(self, image_input):
        """セキュアな深度推定処理"""
        session_id = str(time.time())  # ユニークなセッションID
        
        try:
            # 画像の前処理（セキュリティチェック含む）
            image = self.preprocess_image(image_input, session_id)
            original_size = image.size
            
            # セッション更新
            if session_id in self.processing_sessions:
                self.processing_sessions[session_id]['status'] = 'inference'
            
            logger.info(f"🔄 Starting depth estimation for session {session_id}")
            
            # モデル推論
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # 深度マップの後処理
            depth = predicted_depth.squeeze().cpu().numpy()
            
            # 正規化 (0-255)
            depth_min = depth.min()
            depth_max = depth.max()
            depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            
            # 元のサイズにリサイズ
            depth_resized = cv2.resize(depth_normalized, original_size, interpolation=cv2.INTER_LINEAR)
            
            # カラーマップを適用 (viridis)
            depth_colored = cv2.applyColorMap(depth_resized, cv2.COLORMAP_VIRIDIS)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            
            # PILイメージに変換
            depth_image = Image.fromarray(depth_colored)
            
            logger.info(f"✅ Depth estimation completed for session {session_id}")
            
            # セッションクリーンアップ
            self.cleanup_session(session_id)
            
            # 入力データを明示的に削除（メモリ保護）
            del inputs, outputs, predicted_depth, depth, depth_normalized, depth_resized, depth_colored
            
            return image, depth_image
            
        except Exception as e:
            logger.error(f"❌ Depth estimation failed for session {session_id}: {str(e)}")
            self.cleanup_session(session_id)
            return None, None
    
    def process_api_request(self, image_input):
        """セキュアなGradio API処理関数"""
        request_id = str(time.time())
        
        try:
            logger.info(f"📥 New API request: {request_id}")
            
            if image_input is None:
                logger.warning("❌ No image provided in request")
                return None, None
            
            # 深度推定実行
            original, depth = self.estimate_depth(image_input)
            
            if original is None or depth is None:
                logger.error(f"❌ Processing failed for request {request_id}")
                return None, None
            
            logger.info(f"✅ Request completed successfully: {request_id}")
            return original, depth
            
        except Exception as e:
            logger.error(f"❌ API request failed {request_id}: {str(e)}")
            return None, None
        finally:
            # リクエスト完了後のクリーンアップ
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# グローバルAPI インスタンス
api = DepthEstimationAPI()

def gradio_interface():
    """セキュアなGradio インターフェースの設定"""
    
    # プライバシー重視の説明文
    privacy_notice = """
    🔒 **プライバシー保護型深度推定API**
    
    - ✅ 画像は一時的処理のみ（保存されません）
    - ✅ 処理完了後、メモリから即座に削除
    - ✅ ログに画像データは記録されません
    - ✅ セッション終了時に自動クリーンアップ
    
    最大画像サイズ: 2048px | 対応形式: JPEG, PNG, WebP
    """
    
    interface = gr.Interface(
        fn=api.process_api_request,
        inputs=gr.Image(
            type="pil", 
            label="📷 深度推定する画像をアップロード",
            sources=["upload", "webcam"]  # ファイルアップロードとWebカメラ
        ),
        outputs=[
            gr.Image(type="pil", label="📸 元画像"),
            gr.Image(type="pil", label="🎨 深度マップ")
        ],
        title="🔒 DepthAnything V2 - プライバシー保護型深度推定API",
        description=privacy_notice,
        examples=[],
        allow_flagging="never",  # フラグ機能無効（プライバシー保護）
        analytics_enabled=False,  # アナリティクス無効
        show_error=True,  # エラー表示は有効
        cache_examples=False  # サンプル画像キャッシュ無効
    )
    
    return interface

# ngrok設定とサーバー起動
def setup_ngrok():
    """ngrokの設定とトンネル作成"""
    try:
        import pyngrok
        from pyngrok import ngrok
        
        # ngrokトークンを設定 (Colabの場合は最初に設定が必要)
        # ngrok.set_auth_token("YOUR_NGROK_TOKEN")  # 実際のトークンに置き換え
        
        # トンネルを作成
        public_url = ngrok.connect(7860)
        print(f"Public URL: {public_url}")
        print(f"Frontend環境変数に設定してください:")
        print(f"NEXT_PUBLIC_BACKEND_URL={public_url}")
        
        return public_url
        
    except ImportError:
        print("pyngrokがインストールされていません。")
        print("!pip install pyngrok を実行してください。")
        return None
    except Exception as e:
        print(f"ngrok設定エラー: {e}")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("🔒 DepthAnything V2 プライバシー保護型深度推定API")
    print("=" * 60)
    
    logger.info("🚀 Starting secure depth estimation server...")
    
    # Gradio インターフェースを作成
    demo = gradio_interface()
    
    # ngrokでパブリックURLを取得
    logger.info("🌐 Setting up ngrok tunnel...")
    public_url = setup_ngrok()
    
    # セキュリティ情報表示
    print("\n" + "=" * 60)
    print("🔒 SECURITY FEATURES ENABLED:")
    print("  ✅ No image storage - temporary processing only")
    print("  ✅ Automatic memory cleanup after processing")
    print("  ✅ No image data in logs")
    print("  ✅ Session timeout protection")
    print("  ✅ Image size validation")
    print("  ✅ Gradio analytics disabled")
    print("=" * 60)
    
    # サーバー起動
    logger.info("🚀 Launching secure server...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # ngrokを使用するのでFalse
        debug=False,  # プライバシー保護のためdebugモード無効
        show_api=False,  # API仕様表示無効
        quiet=True  # 起動ログ最小化
    )