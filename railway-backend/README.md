# 深度推定 API - Railway版

## 概要
DepthAnything V2を使用した深度推定APIのRailway版です。

## 機能
- FastAPI + DepthAnything V2モデル
- 画像アップロードによる深度推定
- Base64形式でのレスポンス
- CORS対応

## エンドポイント
- `GET /` - ヘルスチェック
- `GET /health` - モデル状態確認
- `POST /api/predict` - 深度推定実行

## デプロイ手順

### 1. Railwayアカウント作成
https://railway.app/

### 2. GitHubリポジトリ接続
- 新しいプロジェクト作成
- GitHubリポジトリを選択
- `railway-backend`フォルダを指定

### 3. 環境変数設定（任意）
- `PORT`: 自動設定
- `PYTHONPATH`: 自動設定

### 4. デプロイ
- 自動的にビルド・デプロイが開始
- 数分でAPI利用可能

## 使用方法
```bash
# ヘルスチェック
curl https://your-app.railway.app/health

# 深度推定
curl -X POST https://your-app.railway.app/api/predict \
  -F "file=@image.jpg"
```

## 技術スタック
- FastAPI
- DepthAnything V2
- PyTorch (CPU)
- OpenCV
- Railway (デプロイ)