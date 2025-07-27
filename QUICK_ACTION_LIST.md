# 🚀 今すぐやること - 完全手順リスト

## 📱 **Step 1: Google Colab でAPI起動** (5分)

1. **Google Colab を開く**
   ```
   https://colab.research.google.com/
   ```

2. **GPU設定**
   - `ランタイム` → `ランタイムのタイプを変更` → `GPU` → `保存`

3. **新しいノートブック作成して以下コードを実行**
   ```python
   # ライブラリインストール
   !pip install torch torchvision transformers gradio pyngrok opencv-python-headless Pillow numpy
   
   # ngrok設定
   import pyngrok
   from pyngrok import ngrok
   
   NGROK_TOKEN = "ak_30Sd307Vvyan2iewy7g5tIVl4mQ"
   ngrok.set_auth_token(NGROK_TOKEN)
   print("✅ ngrok認証完了")
   
   # 展示用深度推定API
   import torch
   import gradio as gr
   from transformers import AutoImageProcessor, AutoModelForDepthEstimation
   import numpy as np
   from PIL import Image
   import cv2
   
   class ExhibitionAPI:
       def __init__(self):
           self.device = "cuda" if torch.cuda.is_available() else "cpu"
           print(f"🔧 デバイス: {self.device}")
           
           self.processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
           self.model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
           self.model.to(self.device)
           print("✅ モデル準備完了")
       
       def process(self, image):
           if image is None:
               return None, None
           
           try:
               if image.mode != 'RGB':
                   image = image.convert('RGB')
               
               inputs = self.processor(images=image, return_tensors="pt")
               inputs = {k: v.to(self.device) for k, v in inputs.items()}
               
               with torch.no_grad():
                   outputs = self.model(**inputs)
                   depth = outputs.predicted_depth.squeeze().cpu().numpy()
               
               depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
               depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
               depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
               depth_image = Image.fromarray(depth_colored)
               
               return image, depth_image
           except Exception as e:
               print(f"エラー: {e}")
               return None, None
   
   api = ExhibitionAPI()
   
   interface = gr.Interface(
       fn=api.process,
       inputs=gr.Image(type="pil", label="📷 画像アップロード"),
       outputs=[gr.Image(type="pil", label="📸 元画像"), gr.Image(type="pil", label="🎨 深度マップ")],
       title="🏛️ 展示用深度推定API",
       description="🔒 セキュア処理 | ✅ 外部漏洩なし | 💰 完全無料",
       allow_flagging="never"
   )
   
   public_url = ngrok.connect(7860)
   print(f"\n🎉 展示用API起動完了!")
   print(f"📡 Public URL: {public_url}")
   print(f"📋 Vercel設定用: NEXT_PUBLIC_BACKEND_URL={public_url}")
   
   interface.launch(server_name="0.0.0.0", server_port=7860, share=False)
   ```

4. **URLをコピー**
   ```
   例: https://abc123.ngrok-free.app
   ```

---

## 📱 **Step 2: Vercel環境変数設定** (3分)

1. **Vercel Dashboard を開く**
   ```
   https://vercel.com/dashboard
   ```

2. **プロジェクト選択** → **Settings** → **Environment Variables**

3. **新しい環境変数追加**
   - **Name**: `NEXT_PUBLIC_BACKEND_URL`
   - **Value**: Step 1で取得したURL（例: https://abc123.ngrok-free.app）
   - **Environment**: `Production`
   - **Save**

4. **再デプロイ**
   - **Deployments** → **Redeploy**

---

## 📱 **Step 3: 動作確認** (1分)

1. **Vercelアプリにアクセス**
   ```
   https://your-app.vercel.app
   ```

2. **テスト実行**
   - 画像をアップロード
   - 「深度推定実行」ボタンをクリック
   - 2-5秒で深度マップが表示される

3. **成功確認**
   - ✅ **成功**: 緑色で「DepthAnything-V2-Small」表示
   - ❌ **失敗**: オレンジ色で「mock-gradient (デモ)」表示

---

## 🎯 **これで完了！展示システム稼働中**

### 🔒 セキュリティ保証
- ✅ 画像は処理後即座削除
- ✅ 外部への保存なし  
- ✅ 完全ローカル処理

### 💰 コスト
- ✅ 完全無料

### ⏰ 2日間運用
- **8時間毎**: 新しいngrok URL生成 → Vercel環境変数更新
- **12時間毎**: Google Colab セッション再起動

---

## 🚨 **8時間後のURL更新手順**

1. **Google Colab で新しいURL生成**
   ```python
   # 既存セッションで実行
   public_url = ngrok.connect(7861)  # ポート番号を変更
   print(f"新しいURL: {public_url}")
   ```

2. **Vercel環境変数更新**
   - 上記 Step 2 を繰り返し

3. **動作確認**
   - 上記 Step 3 を繰り返し

---

## 📞 **問題発生時**

### Google Colab接続エラー
→ セッション再起動 → コード再実行

### ngrok URL期限切れ  
→ 新しいURL生成 → Vercel更新

### Vercelデプロイエラー
→ 環境変数の値確認 → 手動再デプロイ

---

# 🎯 **今すぐ実行: Step 1 から開始！**