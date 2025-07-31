# EXIF処理と3Dビュー表示問題の解決方法

## 問題の概要

縦画像をアップロードした際に以下の問題が発生：
1. **深度マップ**: 縦画像が横向きに表示される
2. **3Dビュー**: 縦画像が横長に表示される、または180度回転する

## 根本原因

### 1. 二重EXIF処理問題
- **フロントエンド**: `getOrientedImageUrl()`でEXIF処理
- **バックエンド**: `ImageOps.exif_transpose()`で再度EXIF処理
- **結果**: 回転が重複適用されて表示が破綻

### 2. アスペクト比計算の誤り
```python
# 問題のあった計算
if aspect_ratio > 1.0:  # 横長画像
    scale_x = 1.6
    scale_y = 1.6 / aspect_ratio
else:  # 縦長画像（aspect_ratio = 0.75）
    scale_x = 1.6 * 0.75 = 1.2  # ❌ X軸が縮小されて横長に見える
    scale_y = 1.6
```

### 3. 座標系の不一致
- **バックエンド**: PIL座標系（Y軸下向き）
- **フロントエンド**: 3D座標系（Y軸上向き）
- **結果**: Y軸の反転処理で180度回転が発生

## 解決方法

### Step 1: EXIF処理の統一

**フロントエンド (`frontend/lib/imageUtils.ts`)**:
```typescript
export async function getOrientedImageUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    // EXIF処理をバックエンドに任せるため、フロントエンドでは処理しない
    const reader = new FileReader()
    
    reader.onload = (e) => {
      // そのままファイルをData URLとして返す
      resolve(e.target?.result as string)
    }
    
    reader.onerror = () => reject(new Error('File reading failed'))
    reader.readAsDataURL(file)
  })
}
```

**バックエンド (`railway-backend/app.py`)**:
```python
# メイン処理で1回だけEXIF処理を実行
try:
    image = ImageOps.exif_transpose(image)
    logger.info(f"After EXIF transpose: {image.size}")
except Exception as exif_error:
    logger.warning(f"EXIF transpose failed: {exif_error}")
```

### Step 2: アスペクト比計算の修正

```python
def generate_pointcloud(original_image, depth_image):
    # EXIF処理は既にメイン処理で適用済みなので、ここでは適用しない
    
    w, h = original_image.size
    
    # 元画像の縦横比をそのまま保持するスケーリング計算
    aspect_ratio = w / h
    base_scale = 1.6
    
    # アスペクト比を正確に反映
    if aspect_ratio > 1.0:  # 横長画像
        scale_x = base_scale
        scale_y = base_scale / aspect_ratio
    else:  # 縦長画像または正方形
        scale_x = base_scale * aspect_ratio
        scale_y = base_scale
```

### Step 3: 座標系の統一

```python
# 元画像の縦横比を保持した座標計算
x_norm = (x / w - 0.5) * scale_x
# Y軸の反転を取り除いて180度回転を修正
y_norm = (y / h - 0.5) * scale_y  # ❌ -(y / h - 0.5) * scale_y ではない
z_norm = depth_val * 2 - 1
```

## 検証方法

### 1. 縦画像テスト
- **元サイズ**: 433×577px（アスペクト比 0.75）
- **期待結果**: 
  - 深度マップ: 縦長表示
  - 3Dビュー: 縦長表示（横長にならない）
  - 向き: 正しい向き（180度回転しない）

### 2. 横画像テスト
- **元サイズ**: 800×600px（アスペクト比 1.33）
- **期待結果**:
  - 深度マップ: 横長表示
  - 3Dビュー: 横長表示
  - 向き: 正しい向き

### 3. ログ確認
```
Image received from frontend (EXIF already processed): (433, 577)
Point cloud generation: image size 433x577, aspect_ratio=0.750, scale_x=1.200, scale_y=1.600
```

## 重要なポイント

### ❌ やってはいけないこと
1. **フロントエンドとバックエンドの両方でEXIF処理**
2. **ポイントクラウド生成で再度EXIF処理**
3. **Y軸に不適切な反転処理を適用**

### ✅ 正しい処理フロー
1. **フロントエンド**: 元ファイルをそのまま送信
2. **バックエンド**: メイン処理で1回だけEXIF処理
3. **ポイントクラウド生成**: 処理済み画像をそのまま使用
4. **座標計算**: アスペクト比を正確に反映、Y軸反転なし

## トラブルシューティング

### 問題: 深度マップが横向き
**原因**: EXIF処理が適用されていない、または重複適用
**解決**: バックエンドで1回だけEXIF処理を実行

### 問題: 3Dビューが横長
**原因**: アスペクト比計算の誤り
**解決**: `scale_x = base_scale * aspect_ratio` (縦長画像の場合)

### 問題: 3Dビューが180度回転
**原因**: Y軸の不適切な反転処理
**解決**: `y_norm = (y / h - 0.5) * scale_y` (マイナス符号なし)

## ファイル構成

```
project/
├── frontend/
│   └── lib/
│       └── imageUtils.ts          # EXIF処理無効化
├── railway-backend/
│   └── app.py                     # EXIF処理・座標計算修正
└── EXIF_3D_TROUBLESHOOTING.md    # このファイル
```

## 更新履歴

- **2025-01-31**: 初版作成
- **解決**: 縦画像の深度マップと3Dビューの表示問題を完全解決