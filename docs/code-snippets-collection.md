# 便利なコードスニペット集

## 1. デバッグ・テスト用

### Railway API テスト (HTML)
```html
<!DOCTYPE html>
<html>
<head>
    <title>Railway API Test</title>
</head>
<body>
    <h1>API Test</h1>
    <input type="file" id="fileInput" accept="image/*">
    <button onclick="testAPI()">Test</button>
    <pre id="result"></pre>
    
    <script>
    async function testAPI() {
        const input = document.getElementById('fileInput');
        const file = input.files[0];
        if (!file) return;
        
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('https://web-production-a0df.up.railway.app/api/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            document.getElementById('result').textContent = JSON.stringify(data, null, 2);
        } catch (error) {
            document.getElementById('result').textContent = 'Error: ' + error.message;
        }
    }
    </script>
</body>
</html>
```

### cURL テスト
```bash
# ヘルスチェック
curl https://web-production-a0df.up.railway.app/health

# 画像アップロードテスト
curl -X POST https://web-production-a0df.up.railway.app/api/predict \
  -F "file=@test.jpg" \
  -H "Accept: application/json" | python -m json.tool

# レスポンスタイム測定
time curl -X POST https://web-production-a0df.up.railway.app/api/predict \
  -F "file=@test.jpg" -o /dev/null -s -w "%{time_total}\n"
```

### Python テストスクリプト
```python
import requests
import time
from PIL import Image
import io

def test_api_performance(iterations=10):
    # テスト画像生成
    img = Image.new('RGB', (512, 512), color='red')
    
    times = []
    for i in range(iterations):
        # 画像をバイト列に変換
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # API呼び出し
        start = time.time()
        response = requests.post(
            'https://web-production-a0df.up.railway.app/api/predict',
            files={'file': ('test.jpg', img_bytes, 'image/jpeg')}
        )
        elapsed = time.time() - start
        
        times.append(elapsed)
        print(f"Test {i+1}: {elapsed:.2f}s - Status: {response.status_code}")
    
    print(f"\n平均応答時間: {sum(times)/len(times):.2f}秒")
    print(f"最小: {min(times):.2f}秒, 最大: {max(times):.2f}秒")

if __name__ == "__main__":
    test_api_performance()
```

## 2. エラーハンドリング

### フロントエンド エラー境界
```typescript
// components/ErrorBoundary.tsx
import React, { Component, ErrorInfo, ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false
  }

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error:', error, errorInfo)
  }

  public render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="error-fallback">
          <h2>エラーが発生しました</h2>
          <details style={{ whiteSpace: 'pre-wrap' }}>
            {this.state.error?.toString()}
          </details>
        </div>
      )
    }

    return this.props.children
  }
}
```

### API エラーハンドリング
```typescript
// lib/api-client.ts
class APIError extends Error {
  constructor(
    message: string,
    public status: number,
    public data?: any
  ) {
    super(message)
    this.name = 'APIError'
  }
}

export async function apiCall<T>(
  url: string,
  options?: RequestInit
): Promise<T> {
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        ...options?.headers,
      }
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => null)
      throw new APIError(
        errorData?.detail || `HTTP ${response.status}`,
        response.status,
        errorData
      )
    }

    return await response.json()
  } catch (error) {
    if (error instanceof APIError) throw error
    
    // ネットワークエラー
    throw new APIError(
      'ネットワークエラーが発生しました',
      0,
      { originalError: error }
    )
  }
}
```

## 3. パフォーマンス最適化

### 画像の遅延読み込み
```typescript
// hooks/useIntersectionObserver.ts
import { useEffect, useRef, useState } from 'react'

export function useIntersectionObserver(options?: IntersectionObserverInit) {
  const [isIntersecting, setIsIntersecting] = useState(false)
  const targetRef = useRef<HTMLElement>(null)

  useEffect(() => {
    const target = targetRef.current
    if (!target) return

    const observer = new IntersectionObserver(([entry]) => {
      setIsIntersecting(entry.isIntersecting)
    }, options)

    observer.observe(target)

    return () => observer.disconnect()
  }, [options])

  return { targetRef, isIntersecting }
}

// 使用例
function LazyImage({ src, alt }: { src: string; alt: string }) {
  const { targetRef, isIntersecting } = useIntersectionObserver({
    threshold: 0.1
  })

  return (
    <div ref={targetRef}>
      {isIntersecting ? (
        <img src={src} alt={alt} />
      ) : (
        <div className="placeholder" />
      )}
    </div>
  )
}
```

### メモ化されたCanvas描画
```typescript
// hooks/useCanvas.ts
import { useRef, useEffect, useCallback } from 'react'

type DrawFunction = (ctx: CanvasRenderingContext2D) => void

export function useCanvas(
  draw: DrawFunction,
  deps: React.DependencyList = []
) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  
  const render = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    
    // キャンバスクリア
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    // 描画関数実行
    draw(ctx)
  }, deps)
  
  useEffect(() => {
    render()
  }, [render])
  
  return canvasRef
}
```

## 4. Railway デプロイ用

### サイズチェックスクリプト
```bash
#!/bin/bash
# check-size.sh

echo "=== Docker Image Size Check ==="

# 仮想環境作成
python -m venv test_env
source test_env/bin/activate

# 依存関係インストール
pip install -r requirements.txt

# サイズ計算
SIZE=$(du -sh test_env | cut -f1)
echo "Virtual environment size: $SIZE"

# 各パッケージのサイズ
pip list --format=freeze | while IFS='==' read -r package version; do
    SIZE=$(pip show "$package" | grep -E "^Location:" | xargs du -sh 2>/dev/null | cut -f1)
    echo "$package: $SIZE"
done

# クリーンアップ
deactivate
rm -rf test_env
```

### 最小限のDockerfile
```dockerfile
FROM python:3.11-slim

# 必要最小限のシステムパッケージ
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存関係のみ先にコピー（キャッシュ活用）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコピー
COPY . .

# 非rootユーザーで実行
RUN useradd -m -u 1000 appuser
USER appuser

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 5. 3D表示ユーティリティ

### ポイントクラウドエクスポート
```typescript
// utils/pointcloud-export.ts
export function exportToPLY(
  points: number[][],
  colors: number[][]
): string {
  let ply = 'ply\n'
  ply += 'format ascii 1.0\n'
  ply += `element vertex ${points.length}\n`
  ply += 'property float x\n'
  ply += 'property float y\n'
  ply += 'property float z\n'
  ply += 'property uchar red\n'
  ply += 'property uchar green\n'
  ply += 'property uchar blue\n'
  ply += 'end_header\n'
  
  for (let i = 0; i < points.length; i++) {
    const [x, y, z] = points[i]
    const [r, g, b] = colors[i]
    ply += `${x} ${y} ${z} ${Math.floor(r*255)} ${Math.floor(g*255)} ${Math.floor(b*255)}\n`
  }
  
  return ply
}

export function downloadPLY(
  points: number[][],
  colors: number[][],
  filename = 'pointcloud.ply'
) {
  const plyContent = exportToPLY(points, colors)
  const blob = new Blob([plyContent], { type: 'text/plain' })
  const url = URL.createObjectURL(blob)
  
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  
  URL.revokeObjectURL(url)
}
```

### 簡易3Dコントロール
```typescript
// components/Simple3DControls.tsx
import { useState, useEffect } from 'react'

interface Controls3D {
  rotationX: number
  rotationY: number
  zoom: number
}

export function use3DControls(initial: Controls3D = {
  rotationX: 0,
  rotationY: 0,
  zoom: 1
}) {
  const [controls, setControls] = useState(initial)
  
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      switch(e.key) {
        case 'ArrowUp':
          setControls(c => ({ ...c, rotationX: c.rotationX - 0.1 }))
          break
        case 'ArrowDown':
          setControls(c => ({ ...c, rotationX: c.rotationX + 0.1 }))
          break
        case 'ArrowLeft':
          setControls(c => ({ ...c, rotationY: c.rotationY - 0.1 }))
          break
        case 'ArrowRight':
          setControls(c => ({ ...c, rotationY: c.rotationY + 0.1 }))
          break
        case '+':
          setControls(c => ({ ...c, zoom: Math.min(c.zoom * 1.1, 5) }))
          break
        case '-':
          setControls(c => ({ ...c, zoom: Math.max(c.zoom * 0.9, 0.1) }))
          break
      }
    }
    
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])
  
  return controls
}
```

## 6. 開発環境セットアップ

### VSCode 設定
```json
// .vscode/settings.json
{
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "[python]": {
    "editor.defaultFormatter": "ms-python.python"
  },
  "[typescript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[typescriptreact]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

### Git フック
```bash
# .git/hooks/pre-commit
#!/bin/sh

# TypeScriptの型チェック
cd frontend
npm run type-check
if [ $? -ne 0 ]; then
  echo "TypeScript type check failed"
  exit 1
fi

# Pythonのフォーマットチェック
cd ../railway-backend
black --check .
if [ $? -ne 0 ]; then
  echo "Python formatting check failed"
  echo "Run 'black .' to fix"
  exit 1
fi

echo "Pre-commit checks passed!"
```

## 7. モニタリング・ログ

### 簡易パフォーマンスモニター
```typescript
// utils/performance-monitor.ts
class PerformanceMonitor {
  private metrics: Map<string, number[]> = new Map()
  
  startMeasure(name: string): () => void {
    const start = performance.now()
    
    return () => {
      const duration = performance.now() - start
      
      if (!this.metrics.has(name)) {
        this.metrics.set(name, [])
      }
      
      this.metrics.get(name)!.push(duration)
      
      // 最新10件のみ保持
      const values = this.metrics.get(name)!
      if (values.length > 10) {
        values.shift()
      }
    }
  }
  
  getStats(name: string) {
    const values = this.metrics.get(name) || []
    if (values.length === 0) return null
    
    const avg = values.reduce((a, b) => a + b, 0) / values.length
    const min = Math.min(...values)
    const max = Math.max(...values)
    
    return { avg, min, max, count: values.length }
  }
  
  logAll() {
    console.table(
      Array.from(this.metrics.entries()).map(([name, values]) => ({
        name,
        ...this.getStats(name)
      }))
    )
  }
}

export const perfMonitor = new PerformanceMonitor()

// 使用例
const endMeasure = perfMonitor.startMeasure('api-call')
// ... 処理 ...
endMeasure()
perfMonitor.logAll()
```

### 構造化ログユーティリティ
```python
# utils/logger.py
import json
import time
from functools import wraps

class StructuredLogger:
    def __init__(self, name):
        self.name = name
    
    def log(self, level, message, **kwargs):
        log_entry = {
            'timestamp': time.time(),
            'logger': self.name,
            'level': level,
            'message': message,
            **kwargs
        }
        print(json.dumps(log_entry))
    
    def info(self, message, **kwargs):
        self.log('INFO', message, **kwargs)
    
    def error(self, message, **kwargs):
        self.log('ERROR', message, **kwargs)
    
    def measure_time(self, operation_name):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start
                    self.info(f'{operation_name} completed', 
                             duration=duration, 
                             status='success')
                    return result
                except Exception as e:
                    duration = time.time() - start
                    self.error(f'{operation_name} failed', 
                              duration=duration, 
                              error=str(e),
                              status='error')
                    raise
            return wrapper
        return decorator

# 使用例
logger = StructuredLogger('depth-estimation')

@logger.measure_time('process_image')
def process_image(image):
    # 処理
    pass
```