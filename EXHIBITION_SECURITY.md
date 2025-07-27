# 🏛️ 展示用深度推定システム - セキュリティ仕様書

## 🎯 展示用途向けセキュリティ対策

### 🔒 推奨構成: ローカル実行環境

```
┌─────────────────────────────────────────────────────────────┐
│                    展示会場内ネットワーク                      │
│  ┌─────────────────┐           ┌─────────────────┐          │
│  │   展示用PC      │   LAN     │  GPU搭載PC      │          │
│  │  (フロントエンド) │◄─────────►│  (バックエンド)   │          │
│  │  - Next.js      │           │  - ローカルAPI   │          │
│  │  - タッチ操作    │           │  - DepthAnything │          │
│  └─────────────────┘           └─────────────────┘          │
│                                                              │
│  🚫 外部インターネット接続なし                               │
└─────────────────────────────────────────────────────────────┘
```

## 🛡️ 展示用セキュリティレベル

### レベル1: 基本展示 (現在の構成)
- ✅ Google Colab + Vercel
- ⚠️ 外部サービス依存
- 🎯 **学術展示・デモ向け**

### レベル2: 中級展示 (推奨)
- ✅ ローカルGPU + Vercel
- 🔒 処理は完全内部
- 🎯 **企業展示・商用デモ向け**

### レベル3: 高セキュリティ展示 (最安全)
- ✅ 完全ローカル環境
- 🔒 外部通信完全遮断
- 🎯 **機密性重視・政府機関向け**

## 🏆 推奨: レベル3 完全ローカル構成

### 必要機材
```
展示用PC (推奨スペック):
- CPU: Intel i7 / AMD Ryzen 7 以上
- GPU: NVIDIA RTX 3060 以上 (VRAM 8GB+)
- RAM: 16GB以上
- Storage: SSD 500GB以上
- OS: Windows 11 / Ubuntu 20.04+
```

### ソフトウェア構成
```
1. フロントエンド: Next.js (localhost:3000)
2. バックエンド: FastAPI + DepthAnything V2 (localhost:8000)
3. ネットワーク: 内部LAN のみ
4. GPU: ローカルCUDA推論
```

## 🔧 ローカル環境セットアップ

### 1. GPU環境構築
```bash
# NVIDIA ドライバーインストール
# CUDA Toolkit 11.8+
# cuDNN インストール

# Python環境
conda create -n depth_estimation python=3.9
conda activate depth_estimation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. ローカルバックエンド作成
```python
# local_backend/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image
import numpy as np
import cv2
import io
import base64

app = FastAPI(title="展示用深度推定API")

# CORS設定 (ローカル専用)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # フロントエンドのみ
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

class LocalDepthEstimator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "depth-anything/Depth-Anything-V2-Small-hf"
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        print(f"🔒 展示用ローカルAPI起動: {self.device}")

estimator = LocalDepthEstimator()

@app.post("/api/depth-estimation")
async def estimate_depth(file: UploadFile = File(...)):
    try:
        # セキュリティチェック
        if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
            raise HTTPException(400, "サポートされていない画像形式")
        
        # 画像読み込み
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # サイズ制限
        if max(image.size) > 2048:
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        
        # 深度推定
        inputs = estimator.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(estimator.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = estimator.model(**inputs)
            depth = outputs.predicted_depth.squeeze().cpu().numpy()
        
        # 後処理
        depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        
        # Base64変換
        depth_image = Image.fromarray(depth_colored)
        depth_buffer = io.BytesIO()
        depth_image.save(depth_buffer, format='PNG')
        depth_base64 = base64.b64encode(depth_buffer.getvalue()).decode()
        
        # 元画像
        original_buffer = io.BytesIO()
        image.save(original_buffer, format='PNG')
        original_base64 = base64.b64encode(original_buffer.getvalue()).decode()
        
        # メモリクリーンアップ
        del inputs, outputs, depth, image_data
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            "success": True,
            "original": f"data:image/png;base64,{original_base64}",
            "depth_map": f"data:image/png;base64,{depth_base64}",
            "model": "DepthAnything-V2-Local",
            "security": "完全ローカル処理"
        }
        
    except Exception as e:
        print(f"エラー: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. フロントエンド修正
```javascript
// frontend/pages/index.tsx での API呼び出し変更
const handleDepthEstimation = async () => {
  if (!uploadedImage) return;
  
  setIsProcessing(true);
  try {
    const formData = new FormData();
    
    // Blob URLを File オブジェクトに変換
    const response = await fetch(uploadedImage);
    const blob = await response.blob();
    const file = new File([blob], 'upload.jpg', { type: blob.type });
    formData.append('file', file);
    
    // ローカルAPI呼び出し
    const apiResponse = await fetch('http://localhost:8000/api/depth-estimation', {
      method: 'POST',
      body: formData,
    });
    
    const result = await apiResponse.json();
    
    if (result.success) {
      setDepthResult({
        depthMapUrl: result.depth_map,
        originalUrl: result.original,
        success: true,
        modelUsed: result.model,
        resolution: 'original'
      });
      setActiveTab('depth');
      console.log('🔒 ローカル深度推定完了');
    } else {
      throw new Error(result.error);
    }
  } catch (error) {
    console.error('深度推定エラー:', error);
    alert('深度推定に失敗しました');
  } finally {
    setIsProcessing(false);
  }
};
```

## 🏛️ 展示会場での運用

### セットアップ手順
1. **事前準備**
   ```bash
   # モデルダウンロード (インターネット接続時)
   python -c "
   from transformers import AutoImageProcessor, AutoModelForDepthEstimation
   AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')
   AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')
   "
   ```

2. **会場設置**
   ```bash
   # バックエンド起動
   cd local_backend/
   python main.py
   
   # フロントエンド起動
   cd frontend/
   npm run build
   npm start
   ```

3. **動作確認**
   - ローカルネットワークのみでテスト
   - 外部インターネット接続を遮断
   - GPU動作確認

### 展示用UI改善
```javascript
// 展示用の大きなボタンとタッチ操作対応
<button 
  className="btn-primary text-4xl py-8 px-16 rounded-2xl"
  onClick={handleDepthEstimation}
>
  🎨 深度推定を実行
</button>

// 処理中アニメーション
{isProcessing && (
  <div className="flex flex-col items-center space-y-4">
    <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600"></div>
    <p className="text-2xl">AI処理中...</p>
  </div>
)}
```

## 📋 展示用セキュリティチェックリスト

### ✅ 設置前確認
- [ ] 外部インターネット接続遮断
- [ ] ローカルファイアウォール設定
- [ ] GPUドライバー最新版
- [ ] 全機能のローカル動作確認
- [ ] 非常時のシステム停止手順確認

### ✅ 展示中監視
- [ ] システムリソース監視
- [ ] 処理時間の記録
- [ ] エラーログの確認
- [ ] 来場者データの非保存確認

### ✅ 撤去時確認
- [ ] 一時ファイルの完全削除
- [ ] キャッシュクリア
- [ ] ログファイル削除
- [ ] ハードウェア初期化

## 🎖️ 展示用免責事項

```
【展示システム利用について】

本システムは研究・展示目的で提供されています。

✅ 保証事項:
- 画像データは処理後即座に削除されます
- ローカル環境での完全処理
- 外部への画像送信はありません

⚠️ 利用者責任:
- アップロードする画像の内容責任
- 機密情報を含む画像の使用禁止
- システム障害時の責任免除

🔒 技術仕様:
- AI: DepthAnything V2 モデル
- 処理: 完全ローカル実行
- データ: 非永続化
```

## 💰 導入コスト概算

### 初期費用
- **GPU搭載PC**: 30-50万円
- **展示用モニター**: 10-20万円
- **ソフトウェア**: 無料 (オープンソース)
- **設営費**: 5-10万円

### 運用費用
- **電気代**: 1日約500-1000円
- **保守**: なし (自己完結)
- **ライセンス**: なし

**総評**: 完全ローカル環境なら最高レベルのセキュリティを実現できます！