# Render CLI の代替手段

## 現状の問題
Render CLIは現在非推奨（deprecated）となっており、直接的なCLI操作はサポートされていません。

## 代替手段

### 1. Infrastructure as Code (render.yaml)

プロジェクトルートに `render.yaml` を配置することで、Renderの設定をコードで管理できます。

```yaml
services:
  - type: web
    name: depth-estimation-backend
    env: python
    repo: https://github.com/kanalia7355/OC_display.git
    branch: master
    rootDir: backend
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
    plan: starter
    autoDeploy: true
```

### 2. Render Dashboard での手動デプロイ

#### Step 1: サービス作成
1. **https://render.com/dashboard** にアクセス
2. **New +** → **Web Service**
3. **Build and deploy from a Git repository**
4. GitHub リポジトリを選択

#### Step 2: 設定
```
Name: depth-estimation-backend
Environment: Python 3
Region: Frankfurt (EU Central)
Branch: master
Root Directory: backend
Build Command: pip install -r requirements.txt
Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

#### Step 3: 環境変数
```
ENVIRONMENT=production
MODEL_CACHE_DIR=/opt/render/project/src/models
TEMP_DIR=/opt/render/project/src/temp
PYTHONPATH=/opt/render/project/src
```

### 3. GitHub Actions での自動化

```yaml
# .github/workflows/deploy.yml
name: Deploy to Render

on:
  push:
    branches: [ master ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Trigger Render Deploy
      run: |
        echo "Render will auto-deploy on push to master"
```

### 4. Render API を使った操作

#### API トークン取得
1. **Render Dashboard** → **Account Settings** → **API Keys**
2. **Create API Key** で新規トークン作成

#### サービス一覧取得
```bash
curl -H "Authorization: Bearer rnd_YOUR_API_KEY" \
     https://api.render.com/v1/services
```

#### デプロイ実行
```bash
curl -X POST \
     -H "Authorization: Bearer rnd_YOUR_API_KEY" \
     https://api.render.com/v1/services/SERVICE_ID/deploys
```

#### ログ取得
```bash
curl -H "Authorization: Bearer rnd_YOUR_API_KEY" \
     https://api.render.com/v1/services/SERVICE_ID/events
```

### 5. curl を使った簡易CLI作成

```bash
#!/bin/bash
# render-helper.sh

RENDER_API_KEY="rnd_YOUR_API_KEY"
SERVICE_ID="your-service-id"

case "$1" in
    "deploy")
        echo "🚀 Deploying to Render..."
        curl -X POST \
             -H "Authorization: Bearer $RENDER_API_KEY" \
             https://api.render.com/v1/services/$SERVICE_ID/deploys
        ;;
    "logs")
        echo "📋 Fetching logs..."
        curl -H "Authorization: Bearer $RENDER_API_KEY" \
             https://api.render.com/v1/services/$SERVICE_ID/events | jq .
        ;;
    "status")
        echo "📊 Service status..."
        curl -H "Authorization: Bearer $RENDER_API_KEY" \
             https://api.render.com/v1/services/$SERVICE_ID | jq .
        ;;
    *)
        echo "Usage: $0 {deploy|logs|status}"
        ;;
esac
```

## 推奨ワークフロー

### 開発時
1. **ローカル開発**: `./start_dev.sh`
2. **テスト**: `python test_backend.py`
3. **Git push**: 自動でCI/CDが動作

### デプロイ時
1. **初回**: Render Dashboardで手動セットアップ
2. **継続**: Git pushで自動デプロイ
3. **監視**: Render Dashboardでログ確認

### 緊急時
1. **API直接呼び出し**でデプロイ実行
2. **Dashboard**でサービス再起動
3. **環境変数**をDashboard経由で変更

## 実際のデプロイ手順

### 1. render.yaml でサービス定義
```bash
git add render.yaml
git commit -m "Add Render configuration"
git push
```

### 2. Render Dashboard でサービス作成
- render.yaml を認識して自動設定
- 手動調整が必要な項目のみ変更

### 3. 環境変数設定
```bash
# Dashboard の Environment Variables で設定
ENVIRONMENT=production
MODEL_CACHE_DIR=/opt/render/project/src/models
TEMP_DIR=/opt/render/project/src/temp
```

### 4. 自動デプロイ確認
- master ブランチへの push で自動デプロイ
- Dashboard でログ確認

これにより、CLI無しでもRenderを効率的に操作できます。