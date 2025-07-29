# 実装詳細

## 1. 深度推定アルゴリズム

### 純Pillowベース実装
重いAIモデルの代わりに、コンピュータビジョンアルゴリズムを実装：

#### エッジ検出
```python
def advanced_edge_detection(image):
    gray = image.convert('L')
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)
    return edges
```

#### テクスチャ分析
```python
def texture_analysis(image):
    # 3x3ウィンドウで局所分散計算
    # 分散が大きい = テクスチャが豊富 = 近い
```

#### Sobelグラデーション
```python
def gradient_magnitude(image):
    # Sobel X/Y フィルタで勾配計算
    # グラデーション強度から深度推定
```

### 深度マップ生成
1. 中心からの距離（遠近法）
2. エッジ情報（輪郭は近い）
3. テクスチャ密度（詳細は近い）
4. グラデーション強度

## 2. 3Dポイントクラウド生成

### 座標変換
```python
def generate_pointcloud(original_image, depth_image):
    # 画像座標 → 3D空間座標
    x_norm = (x / w - 0.5) * 2  # -1 to 1
    y_norm = (y / h - 0.5) * 2  # -1 to 1
    z_norm = (1.0 - depth_val) * 2 - 1  # 深度
```

### データ構造
```json
{
  "points": [[x, y, z], ...],
  "colors": [[r, g, b], ...],
  "count": 1600,
  "downsample_factor": 4
}
```

### パフォーマンス最適化
- ダウンサンプリング（1/4）
- 効率的なループ処理
- メモリ使用量削減

## 3. フロントエンド3Dビュー

### Canvas基盤レンダリング
Three.jsの代わりに2D Canvasで3D投影：

```typescript
// 3D → 2D透視投影
const perspective = 2
const projectedX = centerX + (rotatedX * scale) / (perspective - finalZ)
const projectedY = centerY + (rotatedY * scale) / (perspective - finalZ)
```

### インタラクション
- マウスドラッグで回転
- X軸・Y軸回転対応
- リアルタイムレンダリング

## 4. API設計

### エンドポイント
- `GET /`: ステータス確認
- `GET /health`: ヘルスチェック
- `POST /api/predict`: 深度推定実行

### レスポンス形式
```typescript
interface DepthEstimationResponse {
  depthMapUrl: string      // Base64深度マップ
  originalUrl: string      // Base64元画像
  success: boolean
  model: string           // 使用モデル名
  resolution: string      // 処理解像度
  pointcloudData?: {      // 3Dデータ
    points: number[][]
    colors: number[][]
    count: number
    downsample_factor: number
  }
}
```

## 5. 最適化戦略

### サイズ削減
1. NumPy不使用（純Python実装）
2. PyTorch/Transformers削除
3. 最小限の依存関係

### パフォーマンス
1. 画像リサイズ（最大512px）
2. ダウンサンプリング
3. 効率的なアルゴリズム

### エラーハンドリング
1. タイムアウト設定
2. フォールバック処理
3. 詳細なエラーログ