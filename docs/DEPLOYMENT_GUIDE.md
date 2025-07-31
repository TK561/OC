# デプロイメントガイド

## 🚀 Railway デプロイ

### 前提条件
- Railwayアカウント
- GitHubリポジトリとの連携

### デプロイ手順

1. **Railwayプロジェクトの作成**
   - Railway.appにログイン
   - "New Project" → "Deploy from GitHub repo"を選択
   - リポジトリを選択

2. **環境変数の設定**
   ```env
   PYTHON_VERSION=3.10
   ENVIRONMENT=production
   MODEL_CACHE_DIR=/tmp/models
   TEMP_DIR=/tmp/temp
   DEFAULT_DEPTH_MODEL=depth-anything/Depth-Anything-V2-Small-hf
   ```

3. **デプロイ設定**
   - Root Directory: `railway-backend`（既存の軽量版を使用する場合）
   - Root Directory: `.`（新しいフル機能版を使用する場合）

4. **ビルド設定（フル機能版の場合）**
   - `railway.json`がルートディレクトリにあることを確認
   - CPU版のPyTorchを使用するため、`requirements_railway.txt`を使用

### メモリ最適化のヒント
- 軽量モデル（Depth-Anything-V2-Small）を使用
- CPU版PyTorchでメモリ使用量を削減
- リクエストごとにモデルをロード/アンロード

## 🔷 Vercel デプロイ（フロントエンド）

### 前提条件
- Vercelアカウント
- Next.jsプロジェクト

### デプロイ手順

1. **Vercelプロジェクトの作成**
   - Vercelダッシュボードで"New Project"
   - GitHubリポジトリをインポート

2. **環境変数の設定**
   ```env
   NEXT_PUBLIC_BACKEND_URL=https://your-railway-app.up.railway.app
   ```

3. **ビルド設定**
   - Framework: Next.js
   - Root Directory: `frontend`
   - Build Command: `npm run build`
   - Output Directory: `.next`

4. **vercel.json設定**
   - APIリダイレクトの設定
   - CORSヘッダーの設定

### デプロイ後の確認

1. **Railway（バックエンド）**
   - ヘルスチェック: `https://your-app.up.railway.app/health`
   - API確認: `https://your-app.up.railway.app/`

2. **Vercel（フロントエンド）**
   - アプリケーション: `https://your-app.vercel.app`
   - APIプロキシ確認

## 📝 トラブルシューティング

### Railway
- **メモリ不足エラー**
  - より小さいモデルを使用
  - `requirements_railway.txt`でCPU版PyTorchを使用
  
- **ビルド失敗**
  - Python versionを3.10に設定
  - ビルドコマンドを確認

### Vercel
- **CORS エラー**
  - バックエンドのCORS設定を確認
  - vercel.jsonのrewritesを確認

- **API接続エラー**
  - 環境変数のURLを確認
  - HTTPSを使用していることを確認