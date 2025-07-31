# 深度推定 API - Railway版

## 概要
Pillowベースの高度なコンピュータビジョン深度推定APIのRailway版です。

## 機能
- FastAPI + Pillow Advanced CV
- エッジ検出、テクスチャ解析、グラデーション解析
- 3Dポイントクラウド生成
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
- Pillow (Advanced Computer Vision)
- 数学的アルゴリズム（Sobel、テクスチャ分散）
- Railway (デプロイ)

## 特徴
- NumPy不要の軽量実装
- 高速処理
- 3D可視化対応