# Render ポートエラーの修正方法

## エラー内容
```
Timed out
Port scan timeout reached, failed to detect open port 10000 from PORT environment variable.
Bind your service to port 10000 or update the PORT environment variable to the correct port.
```

## 原因
RenderはWebサービスに対して動的にポートを割り当てますが、アプリケーションが正しくそのポートにバインドされていない。

## 修正済み内容

### 1. main.py の修正
```python
# 修正前
uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

# 修正後  
port = int(os.getenv("PORT", 8000))
uvicorn.run("app.main:app", host="0.0.0.0", port=port)
print(f"Server starting on port {port}")
```

### 2. render.yaml の修正
```yaml
# PORTの環境変数設定を削除（Renderが自動設定）
envVars:
  - key: ENVIRONMENT
    value: production
  # PORT環境変数は削除（Renderが自動で設定）
```

### 3. startCommand の確認
```yaml
startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Render Dashboard での手動修正

### 1. Environment Variables
以下の環境変数のみを設定（PORTは設定しない）:
```
ENVIRONMENT=production
MODEL_CACHE_DIR=/opt/render/project/src/models
TEMP_DIR=/opt/render/project/src/temp
PYTHONPATH=/opt/render/project/src
```

### 2. Start Command
```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### 3. Health Check
```
Health Check Path: /health
```

## トラブルシューティング

### 1. ログでポート確認
Render Dashboardの**Logs**タブで以下を確認:
```
INFO:     Uvicorn running on http://0.0.0.0:10000
```

### 2. 環境変数デバッグ
一時的にmain.pyに追加:
```python
import os
print(f"PORT environment variable: {os.getenv('PORT', 'Not set')}")
```

### 3. 手動でのポート強制指定
緊急時の対処法:
```python
# main.py (一時的な修正)
port = int(os.getenv("PORT", 10000))  # デフォルトを10000に
```

## ベストプラクティス

### 1. ポート設定の優先順位
1. Renderが設定するPORT環境変数
2. アプリケーションのデフォルトポート

### 2. ログ確認
```python
import logging
logger = logging.getLogger(__name__)

port = int(os.getenv("PORT", 8000))
logger.info(f"Starting server on port {port}")
```

### 3. ヘルスチェック強化
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "port": os.getenv("PORT", "unknown"),
        "timestamp": datetime.now().isoformat()
    }
```

## 再デプロイ手順

### 1. 変更をプッシュ
```bash
git add .
git commit -m "fix: Render port configuration"
git push
```

### 2. Render で再デプロイ
- Dashboard → Services → depth-estimation-backend
- **Manual Deploy** ボタンをクリック

### 3. ログ確認
- **Logs** タブでサーバー起動ログを確認
- `Uvicorn running on http://0.0.0.0:XXXX` が表示されれば成功

### 4. ヘルスチェック
```bash
curl https://your-app-name.onrender.com/health
```

## よくある間違い

### ❌ 間違い
```yaml
envVars:
  - key: PORT
    value: "10000"  # これは設定しない
```

### ✅ 正解
```yaml
# PORTはRenderが自動設定するため、envVarsには含めない
startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

この修正により、Renderのポートタイムアウトエラーが解決されます。