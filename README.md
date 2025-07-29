# 深度推定・3D可視化 API

高度な深度推定技術を使用した3D可視化Webアプリケーションです。

## ✨ 主要機能
- **深度推定**: 複数のモデル対応（DPT-Large、MiDaS、DepthAnything、Pillow-CV）
- **3D可視化**: インタラクティブな3Dポイントクラウド表示
- **画像処理**: エッジ検出、ブラー、色調補正
- **エクスポート**: PLY/OBJ形式での3Dデータ出力
- **操作性**: ドラッグ回転・ホイールズーム対応

## 📁 プロジェクト構成

### 🎯 メイン実装
- **`/frontend/`** - Next.js + Three.js フロントエンド（Vercel）
- **`/backend/`** - FastAPI フル機能バックエンド（Render/Railway）
- **`/railway-backend/`** - Railway専用軽量版（Pillow-CV）

### 🧪 実験・開発
- **`/experiments/`** - 開発・実験用コード
  - `/colab/` - Google Colab用
  - `/local/` - ローカル開発用

### 🚀 デプロイメント
- **`/deployments/`** - 各プラットフォーム用
  - `/huggingface/` - HF Spaces用
  - `/api-variants/` - 各種API版
  - `/demos/` - デモ・展示用

### 📚 ドキュメント・ユーティリティ
- **`/docs/`** - プロジェクトドキュメント
- **`/scripts/`** - 実行・デプロイスクリプト
- **`/tests/`** - テストファイル群
- **`/config/`** - 設定ファイル
- **`/security/`** - セキュリティ設定

## 技術スタック

### フロントエンド
- Next.js 14
- TypeScript
- Three.js + React Three Fiber
- Tailwind CSS
- Vercel (デプロイ)

### バックエンド
- FastAPI
- PyTorch + Hugging Face Transformers
- OpenCV
- Railway/Render (デプロイ)

## 開発環境セットアップ

### 前提条件
- Node.js 18+
- Python 3.9+
- Git

### フロントエンド

```bash
cd frontend
npm install
npm run dev
```

### バックエンド

```bash
cd backend

# 仮想環境作成（推奨）
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# 依存関係インストール
pip install -r requirements.txt

# 開発サーバー起動
uvicorn app.main:app --reload
```

## 環境変数

### フロントエンド (.env.local)
```
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

### バックエンド (.env)
```
ENVIRONMENT=development
HUGGINGFACE_TOKEN=your_token_here
MODEL_CACHE_DIR=./models
TEMP_DIR=./temp
```

## デプロイ

### Vercel (フロントエンド)
1. Vercelアカウントでリポジトリを接続
2. `frontend`フォルダをルートディレクトリに設定
3. 環境変数`NEXT_PUBLIC_BACKEND_URL`を設定

### Railway/Render (バックエンド)
1. `backend`フォルダをデプロイ
2. 必要な環境変数を設定
3. Dockerfileを使用して自動デプロイ

## 使用方法

1. 画像をアップロード
2. 深度推定モデルを選択
3. 深度マップと3D可視化を生成
4. 結果をエクスポート

## 開発ステータス

- [x] プロジェクト基盤構築
- [x] フロントエンド環境設定
- [x] バックエンド環境設定
- [x] 深度推定モデル統合
- [x] 3D可視化実装
- [x] UI/UX実装
- [x] 基本機能完成
- [ ] デプロイ・最適化
- [ ] 本番環境テスト

## クイックスタート

```bash
# 開発環境を一括起動
./start_dev.sh

# または手動で起動
cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt && uvicorn app.main:app --reload &
cd frontend && npm install && npm run dev
```

## テスト

```bash
# バックエンドのテスト実行
python test_backend.py
```

## デプロイ

詳細な手順は [DEPLOYMENT.md](./DEPLOYMENT.md) を参照してください。

## ライセンス

MIT License