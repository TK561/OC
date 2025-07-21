# Render ポートデバッグガイド

## 現在の修正内容

### 1. カスタム起動スクリプト (run_server.py)
```python
# 詳細なデバッグ情報を出力
print(f"🔍 PORT environment variable: {port}")
print(f"🔍 Python path: {sys.path}")
print(f"🔍 Working directory: {os.getcwd()}")

# フォールバック処理
if port is None:
    port = "10000"  # Render のデフォルト
```

### 2. render.yaml の変更
```yaml
# uvicorn コマンドから Python スクリプトに変更
startCommand: python run_server.py
```

### 3. デバッグエンドポイント
```bash
# ルートエンドポイントで環境情報確認
curl https://your-app.onrender.com/

# ヘルスチェックでポート情報確認
curl https://your-app.onrender.com/health
```

## Render Dashboard での確認手順

### 1. ログの確認
**Logs** タブで以下の出力を確認：
```
🚀 Starting Depth Estimation API
🔍 PORT environment variable: 10000
✅ Using port: 10000
✅ Successfully imported app
INFO:     Uvicorn running on http://0.0.0.0:10000
```

### 2. 環境変数の確認
**Environment** タブで設定されている変数：
```
ENVIRONMENT=production
MODEL_CACHE_DIR=/opt/render/project/src/models
TEMP_DIR=/opt/render/project/src/temp
PYTHONPATH=/opt/render/project/src
```

**注意**: `PORT` は手動設定せず、Render が自動設定

### 3. サービス設定の確認
**Settings** タブで以下を確認：
```
Build Command: pip install --upgrade pip && pip install -r requirements.txt
Start Command: python run_server.py
Health Check Path: /health
```

## まだエラーが出る場合の対処法

### Option 1: Dockerfile使用
```dockerfile
# backend/Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

EXPOSE $PORT
CMD ["python", "run_server.py"]
```

**render.yaml** を更新:
```yaml
services:
  - type: web
    dockerfilePath: ./Dockerfile
    envVars: # 同じ環境変数
```

### Option 2: gunicorn使用
```bash
# requirements.txt に追加
gunicorn>=20.1.0

# startCommand を変更
startCommand: gunicorn app.main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT
```

### Option 3: 直接uvicorn（最後の手段）
```yaml
startCommand: |
  python -c "
  import os
  print('PORT:', os.getenv('PORT', 'NOT_SET'))
  " && uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## デバッグコマンド

### Renderのログを詳細確認
```bash
# ログでエラーを検索
# Dashboard > Logs で以下をキーワード検索:
# - "PORT"
# - "timeout"
# - "failed to detect"
# - "Uvicorn running"
```

### 手動でのポート確認
```bash
# ローカルテスト
cd backend
PORT=10000 python run_server.py

# 別ターミナルで確認
curl http://localhost:10000/health
```

## よくある問題と解決策

### 1. モジュールインポートエラー
```
ModuleNotFoundError: No module named 'app'
```
**解決策**: `PYTHONPATH` 環境変数の設定確認

### 2. requirements.txt の問題
```
ERROR: Could not find a version that satisfies the requirement
```
**解決策**: 
```txt
# 軽量版を使用
opencv-python-headless>=4.8.0
torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
```

### 3. メモリ不足
```
Process killed (OOM)
```
**解決策**: Standard プラン ($7/month) にアップグレード

## 緊急時の対処

### 1. 最小構成での起動テスト
```python
# minimal_server.py
import os
import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World", "Port": os.getenv("PORT")}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
```

### 2. startCommand を一時変更
```yaml
startCommand: python minimal_server.py
```

### 3. 段階的な機能追加
1. 最小サーバーで起動確認
2. FastAPI基本機能追加
3. 深度推定機能追加

これらの修正でポート問題が解決されるはずです。