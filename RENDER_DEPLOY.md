# Render デプロイガイド

## Render でのバックエンドデプロイ

### 1. Render アカウント作成
1. https://render.com でアカウント作成
2. GitHub アカウントで連携推奨

### 2. Web Service の作成

#### Step 1: 新規サービス作成
1. Render ダッシュボード → "New +"
2. "Web Service" を選択
3. GitHub リポジトリを接続

#### Step 2: サービス設定
```
Name: depth-estimation-backend
Environment: Python 3
Region: Frankfurt (EU Central) または Oregon (US West)
Branch: master
Root Directory: backend
```

#### Step 3: ビルド設定
```
Build Command: pip install -r requirements.txt
Start Command: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### 3. 環境変数の設定

```
ENVIRONMENT=production
MODEL_CACHE_DIR=/opt/render/project/src/models
TEMP_DIR=/opt/render/project/src/temp
PYTHONPATH=/opt/render/project/src
PORT=10000
```

### 4. Dockerfile を使用する場合

#### backend/Dockerfile の更新
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# システム依存関係
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python依存関係
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコード
COPY . .

# ディレクトリ作成
RUN mkdir -p models temp

EXPOSE $PORT

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

#### Render 設定（Dockerfile使用時）
```
Build Command: docker build -t depth-app .
Start Command: docker run -p $PORT:$PORT depth-app
```

### 5. 高度な設定

#### render.yaml（インフラストラクチャ as Code）
```yaml
services:
  - type: web
    name: depth-estimation-backend
    env: python
    repo: https://github.com/kanalia7355/OC_display.git
    buildCommand: pip install -r backend/requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: ENVIRONMENT
        value: production
      - key: MODEL_CACHE_DIR
        value: /opt/render/project/src/models
      - key: TEMP_DIR
        value: /opt/render/project/src/temp
    plan: starter
```

## トラブルシューティング

### 1. メモリ不足エラー

#### 解決策1: プランアップグレード
- Starter Plan (512MB) → Standard Plan (2GB)

#### 解決策2: モデル最適化
```python
# app/models/depth_model.py
import torch

# モデル軽量化
model = model.half()  # FP16使用
torch.backends.cudnn.benchmark = False
```

#### 解決策3: 遅延ロード
```python
# モデルを必要時のみロード
def get_model(model_name):
    if model_name not in loaded_models:
        loaded_models[model_name] = load_model(model_name)
    return loaded_models[model_name]
```

### 2. タイムアウトエラー

#### Render 設定調整
```
Health Check Path: /health
Health Check Grace Period: 300 seconds
```

#### アプリケーション側対応
```python
# app/main.py
import asyncio

@app.middleware("http")
async def timeout_middleware(request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=25.0)
    except asyncio.TimeoutError:
        return {"error": "Request timeout"}
```

### 3. 依存関係エラー

#### requirements.txt の最適化
```
# ヘッドレス版を使用してサイズ削減
opencv-python-headless>=4.8.0

# 軽量版を選択
torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu
```

### 4. モデルダウンロードエラー

#### Hugging Face トークンの設定
```
Environment Variables:
HUGGINGFACE_TOKEN=your_token_here
```

#### オフラインモデル使用
```python
# モデルを事前にダウンロードしてリポジトリに含める
# または起動時に一度だけダウンロード
```

## パフォーマンス最適化

### 1. レスポンス時間短縮
```python
# キャッシュ戦略
from functools import lru_cache

@lru_cache(maxsize=10)
def get_cached_result(image_hash, model_name):
    return process_image(image_hash, model_name)
```

### 2. 並列処理
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)

async def process_async(image_data):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, process_image, image_data)
```

### 3. モニタリング設定
```python
# ヘルスチェック強化
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "memory_usage": get_memory_usage(),
        "model_status": check_models(),
        "timestamp": datetime.now().isoformat()
    }
```

## ログとモニタリング

### 1. ログ設定
```python
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
```

### 2. エラー追跡
```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FastApiIntegration()]
)
```

## セキュリティ

### 1. CORS設定
```python
# app/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-frontend-domain.vercel.app",
        "https://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 2. レート制限
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/depth/estimate")
@limiter.limit("10/minute")
async def estimate_depth(request: Request, ...):
    # 処理
```