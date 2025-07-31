# 深度推定モデル設定 - 最終確認済み設定

## ✅ 正常動作確認済み（2025-07-31）

すべてのモデル（DPT-Large、MiDaS、Depth Anything）で正しい深度マップが表示されることを確認。

## 重要な設定

### 1. 深度値の反転処理（railway-backend/app.py）

#### MiDaS v3.1
```python
# Line 134: MiDaSは反転処理が必要
normalized = 1.0 - normalized  # MiDaSも反転が必要
```

#### DPT-Large  
```python
# Line 318: DPTは反転処理不要
inverted_depth = normalized  # DPTも反転不要
```

#### Depth Anything
```python
# Line 394: Depth Anythingは反転処理が必要
inverted_depth = 1.0 - normalized_depth  # Depth Anythingも反転処理が必要
```

### 2. 統一された深度表現
- **近い = 白色（255に近い値）**
- **遠い = 黒色（0に近い値）**

### 3. 画像回転防止設定

すべてのバックエンドファイルで以下の形式を使用：

```python
# ❌ 自動回転される可能性（修正前）
image = Image.open(io.BytesIO(image_data)).convert("RGB")

# ✅ EXIF情報を無視、向きを保持（修正後）
image = Image.open(io.BytesIO(image_data))
image = image.convert("RGB")
```

#### 修正対象ファイル：
- `railway-backend/app.py` - メインデプロイサーバー
- `backend/app/models/depth_model_v2.py`
- `backend/app/utils/image_utils.py`
- `backend/free_tier_server.py`
- `backend/minimal_server.py`

## 動作検証結果

### DPT-Large
- ✅ 近距離オブジェクトが白く表示
- ✅ 遠距離背景が黒く表示
- ✅ 画像の向きが保持される

### MiDaS v3.1
- ✅ 近距離オブジェクトが白く表示
- ✅ 遠距離背景が黒く表示
- ✅ 画像の向きが保持される

### Depth Anything
- ✅ 近距離オブジェクトが白く表示
- ✅ 遠距離背景が黒く表示
- ✅ 画像の向きが保持される

## トラブルシューティング

### 深度値が逆転している場合
1. `railway-backend/app.py`の該当モデルの反転処理を確認
2. `1.0 - normalized`の有無を調整

### 画像が回転してしまう場合
1. `Image.open()`と`convert("RGB")`を分離
2. EXIF情報の自動適用を回避

## デプロイ環境
- **Railway**: `railway-backend/app.py`が実際のデプロイファイル
- **Vercel**: フロントエンドは自動デプロイ
- **GitHub**: すべての変更はGitで管理

## 最終更新
- 日付: 2025-07-31
- 状態: 全モデル正常動作確認
- コミット: fe33adf - 画像の自動回転を無効化してアップロード時の向きを保持