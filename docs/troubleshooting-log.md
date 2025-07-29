# トラブルシューティングログ

## 1. Railway 4GBサイズ制限問題

### 問題
```
Error: Image size exceeded 10 GB limit
```

### 試行錯誤
1. **PyTorch + Transformers** (失敗: 4GB超過)
   - transformers==4.35.2
   - torch==2.1.0
   - サイズ: 約6GB

2. **Intel DPT-Hybrid-MiDaS** (失敗: 4GB超過)
   - timm==0.9.10
   - サイズ: 約5GB

3. **NumPy依存削除** (成功)
   - 純Pillow実装
   - サイズ: 約200MB

### 解決策
```txt
# requirements.txt (最終版)
fastapi==0.104.1
uvicorn==0.24.0
python-multipart==0.0.6
Pillow==10.0.1
```

## 2. TypeScript型定義エラー

### 問題
```
Property 'model' does not exist on type 'DepthEstimationResponse'
```

### 原因
1. APIレスポンスフィールド名の不一致
2. tsconfig.jsonのパス設定ミス

### 解決
```typescript
// 修正前
modelUsed: string

// 修正後
model: string
```

```json
// tsconfig.json
"@/shared/*": ["./shared/*"]  // ../shared/*から修正
```

## 3. 3Dデータ未表示問題

### 問題
「3Dデータが生成されていません」エラー

### デバッグ手順
1. コンソールログ追加
2. APIレスポンス確認
3. pointcloudData未送信発見

### 解決
```typescript
// index.tsx修正
setDepthResult({
  ...
  pointcloudData: result.pointcloudData  // 追加
})
```

## 4. 画像反転問題

### 問題
3Dビューで画像が上下逆

### 原因
Y軸座標計算の誤り

### 解決
```python
# 修正前
y_norm = (0.5 - y / h) * 2  # 反転

# 修正後  
y_norm = (y / h - 0.5) * 2  # 正常
```

## 5. 無限ループログ問題

### 問題
ThreeSceneでログが無限出力

### 原因
レンダリング関数外でconsole.log実行

### 解決
デバッグログを削除

## 6. Vercelビルドエラー

### エラー一覧
1. **型定義不一致**: model vs modelUsed
2. **オプショナルフィールド**: note, features追加
3. **パス解決**: tsconfig.json修正

### 最終的な型定義
```typescript
export interface DepthEstimationResponse {
  depthMapUrl: string
  originalUrl: string
  success: boolean
  model: string
  resolution: string
  note?: string
  algorithms?: string[]
  implementation?: string
  features?: string[]
  pointcloudData?: {
    points: number[][]
    colors: number[][]
    count: number
    downsample_factor: number
  }
}
```

## 7. Railway再デプロイ問題

### 症状
コード更新後も古いAPIレスポンス

### 対策
1. バージョン番号追加でキャッシュクリア
2. 手動での再デプロイ
3. Root Directory設定確認

### 確認コマンド
```bash
curl https://web-production-a0df.up.railway.app/
```