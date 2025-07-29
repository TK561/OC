# API リファレンス

## Base URL
```
https://web-production-a0df.up.railway.app
```

## エンドポイント

### 1. GET /
ステータス確認エンドポイント

**レスポンス例:**
```json
{
  "message": "Advanced Computer Vision Depth Estimation API",
  "status": "running",
  "model": "Pillow-Advanced-CV",
  "algorithms": [
    "Edge Detection",
    "Texture Analysis",
    "Gradient Magnitude",
    "Distance Transform"
  ],
  "note": "Real computer vision algorithms using only Pillow - no NumPy dependencies",
  "version": "1.1.0",
  "features": ["2D Depth Map", "3D Point Cloud Generation"]
}
```

### 2. GET /health
ヘルスチェックエンドポイント

**レスポンス例:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "algorithms": "pillow_cv_ready"
}
```

### 3. POST /api/predict
深度推定実行エンドポイント

**リクエスト:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: 
  - `file`: 画像ファイル (JPEG/PNG)

**レスポンス例:**
```json
{
  "success": true,
  "originalUrl": "data:image/png;base64,...",
  "depthMapUrl": "data:image/png;base64,...",
  "pointcloudData": {
    "points": [
      [-0.5, -0.5, 0.2],
      [-0.48, -0.5, 0.15],
      ...
    ],
    "colors": [
      [0.8, 0.2, 0.1],
      [0.7, 0.3, 0.2],
      ...
    ],
    "count": 1600,
    "downsample_factor": 4
  },
  "model": "Pillow-Advanced-CV",
  "resolution": "512x512",
  "note": "Real computer vision depth estimation...",
  "algorithms": [
    "Pillow Edge Detection",
    "Texture Variance Analysis",
    "Sobel Gradient",
    "Distance Transform"
  ],
  "implementation": "Pure Pillow - No NumPy",
  "features": ["2D Depth Map", "3D Point Cloud"]
}
```

## エラーレスポンス

### 500 Internal Server Error
```json
{
  "detail": "Depth estimation failed: [error message]"
}
```

## 使用例

### cURL
```bash
curl -X POST https://web-production-a0df.up.railway.app/api/predict \
  -F "file=@image.jpg" \
  -H "Accept: application/json"
```

### JavaScript (Fetch API)
```javascript
const formData = new FormData();
formData.append('file', imageFile);

const response = await fetch('https://web-production-a0df.up.railway.app/api/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
```

### Python (requests)
```python
import requests

url = "https://web-production-a0df.up.railway.app/api/predict"
files = {'file': open('image.jpg', 'rb')}

response = requests.post(url, files=files)
result = response.json()
```

## 制限事項

1. **画像サイズ**: 最大512x512にリサイズ
2. **ファイル形式**: JPEG, PNG対応
3. **レスポンスサイズ**: Base64エンコードのため大きい
4. **処理時間**: 約1-3秒（画像サイズによる）

## 3Dデータ形式

### points配列
- 各要素: `[x, y, z]`
- 範囲: -1.0 ～ 1.0
- 座標系: 右手系

### colors配列
- 各要素: `[r, g, b]`
- 範囲: 0.0 ～ 1.0
- 形式: 正規化RGB

### downsample_factor
- デフォルト: 4
- 意味: 元画像の1/4解像度でサンプリング