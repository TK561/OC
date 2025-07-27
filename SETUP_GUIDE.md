# 🚀 深度推定・3D可視化システム - 完全セットアップガイド

## 📋 システム概要

このシステムは以下の構成で動作します：

```
┌─────────────────┐    HTTPS    ┌─────────────────┐    内部通信    ┌─────────────────┐
│   Vercel        │◄──────────►│  Google Colab   │◄─────────────►│ DepthAnything   │
│   (Frontend)    │             │   + ngrok       │                │   V2 Model      │
│                 │             │   (Backend)     │                │   (AI Engine)   │
│ - Next.js       │             │ - Gradio API    │                │ - GPU推論       │
│ - 画像アップロード │             │ - セキュリティ    │                │ - 深度マップ生成 │
│ - 3D可視化      │             │ - プライバシー保護│                │                 │
└─────────────────┘             └─────────────────┘                └─────────────────┘
```

## 🎯 ステップ1: Google Colab バックエンド設定

### 1.1 Google Colab準備
1. **Google Colabにアクセス**
   ```
   https://colab.research.google.com/
   ```

2. **GPUランタイム設定**
   - `ランタイム` → `ランタイムのタイプを変更`
   - `ハードウェア アクセラレータ`: **GPU (T4)**
   - `保存`をクリック

### 1.2 ngrokアカウント設定
1. **ngrokアカウント作成**
   ```
   https://ngrok.com/signup
   ```

2. **認証トークン取得**
   - Dashboard → `Your Authtoken`
   - トークンをコピー（後で使用）

### 1.3 Colabでセットアップ実行

**方法A: ノートブックアップロード**
1. `colab-backend/setup_colab.ipynb` をダウンロード
2. Google Colabにアップロード
3. セルを順番に実行

**方法B: 手動コード実行**
```python
# セル1: ライブラリインストール
!pip install torch torchvision transformers
!pip install gradio pyngrok opencv-python-headless
!pip install Pillow numpy

# セル2: ngrok設定
import pyngrok
from pyngrok import ngrok
NGROK_TOKEN = "YOUR_TOKEN_HERE"  # 取得したトークンを入力
ngrok.set_auth_token(NGROK_TOKEN)

# セル3: APIサーバー起動
# (depth_estimation_api.py のコードをコピペ)
```

### 1.4 URL確認
実行成功時に以下が表示されます：
```
🚀 API Server is running!
Public URL: https://abc123.ngrok-free.app
📋 Frontend環境変数に設定してください:
NEXT_PUBLIC_BACKEND_URL=https://abc123.ngrok-free.app
```

**このURLをコピー！**

## 🎯 ステップ2: Vercel フロントエンド設定

### 2.1 Vercelアカウント設定
1. **Vercelアカウント作成**
   ```
   https://vercel.com/signup
   ```
   - GitHub連携推奨

### 2.2 プロジェクト準備
1. **フロントエンドコードの準備**
   ```bash
   cd frontend/
   ```

2. **環境変数設定**
   ```bash
   # .env.local ファイルを作成
   echo "NEXT_PUBLIC_BACKEND_URL=https://abc123.ngrok-free.app" > .env.local
   ```
   ⚠️ `abc123.ngrok-free.app` の部分を**ステップ1.4で取得したURL**に置き換え

### 2.3 Vercelデプロイ
1. **Vercel CLI インストール**
   ```bash
   npm install -g vercel
   ```

2. **ログイン**
   ```bash
   vercel login
   ```
   - GitHub認証を選択

3. **デプロイ実行**
   ```bash
   cd frontend/
   vercel --prod --yes
   ```

4. **デプロイ完了確認**
   ```
   ✅ Production: https://your-app.vercel.app
   ```

## 🎯 ステップ3: システム動作確認

### 3.1 基本動作テスト
1. **Vercelアプリにアクセス**
   - ブラウザで `https://your-app.vercel.app` を開く

2. **画像アップロード**
   - 「画像アップロード」で写真を選択
   - 「深度推定実行」をクリック

3. **結果確認**
   - ✅ 実AI: `DepthAnything-V2-Small` 表示（緑色）
   - ⚠️ モック: `mock-gradient (デモ)` 表示（オレンジ色）

### 3.2 セキュリティ確認
1. **プライバシー保護**
   - 画像は処理後自動削除
   - ブラウザの開発者ツールでネットワークタブ確認

2. **API応答確認**
   ```javascript
   // コンソールで確認
   console.log('✅ Real AI depth estimation successful!')  // 成功時
   console.log('⚠️ Using mock depth estimation (real API unavailable)')  // フォールバック時
   ```

## 🔧 ステップ4: トラブルシューティング

### 4.1 よくある問題と解決策

#### ❌ Problem: Google Colab接続エラー
```
ERROR: ngrok接続エラー
```
**解決策:**
1. ngrokトークンを再確認
2. Colabランタイムを再起動
3. セルを最初から再実行

#### ❌ Problem: Vercelビルドエラー
```
ERROR: Build failed
```
**解決策:**
```bash
cd frontend/
npm install
npm run build  # ローカルでテスト
vercel --prod --yes  # 再デプロイ
```

#### ❌ Problem: API接続失敗
```
⚠️ Using mock depth estimation
```
**解決策:**
1. ngrok URLの有効性確認
2. .env.local の URL更新
3. Vercel環境変数の確認

#### ❌ Problem: 画像処理エラー
```
画像サイズが大きすぎます
```
**解決策:**
- 画像を2048px以下にリサイズ
- JPEG/PNG/WebP形式を使用

### 4.2 デバッグ方法

#### Google Colab側
```python
# GPU確認
!nvidia-smi

# メモリ確認
import torch
print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")

# ngrok接続確認
import requests
response = requests.get(f"{public_url}/")
print(f"Status: {response.status_code}")
```

#### フロントエンド側
```javascript
// ブラウザ開発者ツールのコンソールで確認
console.log('Backend URL:', process.env.NEXT_PUBLIC_BACKEND_URL)

// ネットワークタブでAPI通信確認
// XHRリクエストの status code を確認
```

## 📊 ステップ5: システム監視・メンテナンス

### 5.1 定期チェック項目

#### 日次チェック
- [ ] ngrok URL有効性（8時間制限）
- [ ] Google Colab セッション状態（12時間制限）
- [ ] Vercel デプロイ状態

#### 週次チェック
- [ ] システム全体の動作確認
- [ ] セキュリティログの確認
- [ ] パフォーマンス測定

### 5.2 制限事項と対策

#### 時間制限
| サービス | 制限時間 | 対策 |
|----------|----------|------|
| Google Colab セッション | 12時間 | 定期的な再実行 |
| ngrok URL (無料版) | 8時間 | URL更新・再デプロイ |
| アイドル状態 | 90分 | 定期アクセス |

#### コスト最適化
```bash
# ngrok Pro版（推奨）
# - 永続URL: $8/月
# - 複数同時接続
# - カスタムドメイン

# Google Colab Pro版（オプション）
# - 長時間セッション: $9.99/月
# - 高性能GPU
# - 優先アクセス
```

## 🎓 ステップ6: 応用・カスタマイズ

### 6.1 高度な設定

#### セキュリティ強化
```bash
# 追加セキュリティヘッダー
# vercel.json に追加
{
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "X-Content-Type-Options",
          "value": "nosniff"
        },
        {
          "key": "X-Frame-Options",
          "value": "DENY"
        }
      ]
    }
  ]
}
```

#### パフォーマンス最適化
```javascript
// 画像最適化
import Image from 'next/image'

// レスポンシブ画像
<Image 
  src={uploadedImage} 
  alt="Uploaded" 
  width={800} 
  height={600}
  priority
/>
```

### 6.2 機能拡張例

#### 複数モデル対応
```python
# Google Colab側
models = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf", 
    "large": "depth-anything/Depth-Anything-V2-Large-hf"
}
```

#### バッチ処理対応
```javascript
// フロントエンド側
const handleBatchUpload = async (files) => {
  for (const file of files) {
    await processDepthEstimation(file)
  }
}
```

## 🚀 クイックスタート（5分版）

**最短設定手順:**

1. **Google Colab**
   ```
   1. colab.research.google.com でGPUランタイム設定
   2. ngrok.com でトークン取得
   3. setup_colab.ipynb実行
   4. 生成されたURLをコピー
   ```

2. **Vercel**
   ```bash
   cd frontend/
   echo "NEXT_PUBLIC_BACKEND_URL=YOUR_NGROK_URL" > .env.local
   vercel login
   vercel --prod --yes
   ```

3. **テスト**
   ```
   Vercelアプリで画像アップロード → 深度推定実行
   ```

**以上で完了！** 🎉

---

## 📞 サポート・ヘルプ

### エラー時の連絡事項
1. エラーメッセージのスクリーンショット
2. ブラウザの開発者ツール（Console, Network）
3. 使用環境（OS, ブラウザ）
4. 実行したステップ

### 参考ドキュメント
- [Vercel Documentation](https://vercel.com/docs)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [ngrok Documentation](https://ngrok.com/docs)
- [DepthAnything V2 Paper](https://arxiv.org/abs/2406.09414)