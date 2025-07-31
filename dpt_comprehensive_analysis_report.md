# DPT-Large (isl-org/DPT) 深度値出力方向 詳細調査レポート

## 調査概要

DPT-Large (Dense Prediction Transformer) の深度値出力方向について、GitHub公式リポジトリ、学術論文、HuggingFace実装、実際のコード例を通じて詳細に調査しました。

**調査日**: 2025年7月30日  
**対象モデル**: DPT-Large (Intel/isl-org)  
**調査範囲**: 公式実装、学術論文、HuggingFace実装、実用例

---

## 主要な結論

### **DPTは逆深度（Inverse Depth）を出力する**

- **出力値**: `1/depth` (スケールとシフトを含む)
- **高い値**: 近い物体（小さい実際の深度）
- **低い値**: 遠い物体（大きい実際の深度）
- **可視化**: **白=近い、黒=遠い**

---

## 1. GitHub公式リポジトリ (isl-org/DPT) の分析

### 1.1 アーキテクチャの特徴

**リポジトリ**: https://github.com/isl-org/DPT

```python
# models.pyから抜粋した重要なコード
class DPTDepthModel(DPT):
    def forward(self, x):
        inv_depth = super().forward(x).squeeze(dim=1)
        
        if self.invert:
            # 逆深度から通常深度への変換
            depth = self.scale * inv_depth + self.shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth  # 逆数を取る
            return depth
        else:
            return inv_depth  # 逆深度のまま返す
```

### 1.2 深度値処理の流れ

1. **モデル出力**: 逆深度値（inverse depth）
2. **スケーリング**: データセット固有のスケーリング適用
   - KITTI: 256倍
   - NYU: 1000倍
3. **保存**: PFMまたはPNG形式で保存
4. **正規化**: 0-255または0-65535の範囲に正規化

### 1.3 重要な実装詳細

```python
# run_monodepth.pyから抜粋
def write_depth(path, depth, bits=1, absolute_depth=False):
    """深度マップの保存"""
    if absolute_depth:
        # 絶対深度値として保存
        out = depth.astype("uint16")
    else:
        # 相対深度として正規化
        depth_min = depth.min()
        depth_max = depth.max()
        
        max_val = (2**(8*bits))-1
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
```

---

## 2. 学術論文「Vision Transformers for Dense Prediction」での深度値定義

### 2.1 論文の基本情報

- **タイトル**: Vision Transformers for Dense Prediction
- **発表**: ICCV 2021
- **著者**: Intel Labs
- **論文リンク**: https://arxiv.org/abs/2103.13413

### 2.2 深度推定の理論的基盤

#### 逆深度の採用理由

1. **数値的安定性**: 遠距離の表現に適している
2. **視差ベース学習**: ステレオデータセットとの親和性
3. **スケール・シフト不変**: 多様なデータセットでの学習に適している

#### 数学的関係

```
逆深度 = 1 / 実際の深度

例:
- 0.5m → 2.0 (逆深度)
- 1.0m → 1.0 (逆深度)
- 2.0m → 0.5 (逆深度)
- 10.0m → 0.1 (逆深度)
```

### 2.3 損失関数の特徴

- **アフィン不変損失**: スケールとシフトの不明性を無視
- **中央値による平行移動**: 平行移動の正規化
- **平均絶対偏差によるスケール**: スケールの正規化

---

## 3. HuggingFace Transformers実装の分析

### 3.1 モデル構造

```python
# HuggingFace実装の使用例
from transformers import AutoImageProcessor, DPTForDepthEstimation

processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

# 推論実行
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth
```

### 3.2 実装の特徴

1. **バックボーン**: Vision Transformer (ViT)
2. **ネック**: DPTNeck（特徴量融合）
3. **ヘッド**: DPTDepthEstimationHead（深度推定）
4. **後処理**: `post_process_depth_estimation`による自動リサイズ

### 3.3 HuggingFace vs 公式実装の違い

| 項目 | HuggingFace | 公式実装 |
|------|-------------|----------|
| 逆深度変換 | 自動的に処理 | 手動でinvertフラグ制御 |
| 後処理 | `image_processor`で自動 | 手動でスケーリング |
| モデル種類 | Intel/dpt-large等 | 複数のチェックポイント |
| 使いやすさ | 高い | 低い（設定が必要） |

---

## 4. 実装コードでの出力方向確認

### 4.1 逆深度変換の理論

```python
# 深度から逆深度への変換例
depth_values = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]  # メートル
inverse_depth = [1/d for d in depth_values]

# 結果: [2.0, 1.0, 0.5, 0.2, 0.1, 0.02]
# 近い物体ほど高い逆深度値
```

### 4.2 可視化での確認

```python
# MiDaSの逆深度出力を通常深度に変換
# (多くの実装例で確認)
formatted = (255 - output * 255 / np.max(output)).astype("uint8")
```

この処理により：
- **MiDaSの高い値（近い物体）** → **暗い色**
- **MiDaSの低い値（遠い物体）** → **明るい色**

---

## 5. サンプル画像での実際の出力値傾向

### 5.1 実用例からの確認

FiftyOneチュートリアルの分析より：

1. **前処理**: `AutoImageProcessor`で画像を準備
2. **推論**: DPTForDepthEstimationで逆深度を出力
3. **後処理**: 
   - 元画像サイズに補間
   - MiDaSの逆深度マップを通常深度に変換
   - 0-255範囲に正規化

### 5.2 可視化での出力特性

```python
# 典型的な可視化処理
def visualize_depth(depth_map):
    # 深度マップの正規化 (逆深度対応)
    normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    # 0-255スケールに変換
    return (normalized * 255).astype(np.uint8)
```

**結果の解釈**:
- **明るいピクセル**: 近い物体（高い逆深度値）
- **暗いピクセル**: 遠い物体（低い逆深度値）

---

## 6. 公式実装とHuggingFace実装の違い

### 6.1 深度値の扱い

| 側面 | 公式実装 | HuggingFace実装 |
|------|----------|-----------------|
| **出力** | 生の逆深度値 | 処理済み深度値 |
| **正規化** | 手動実装が必要 | 自動処理 |
| **可視化** | util/io.pyで処理 | post_process_*で処理 |
| **データセット対応** | KITTI/NYU固有設定 | 汎用的な処理 |

### 6.2 使用場面での選択

- **研究用途**: 公式実装（細かい制御が可能）
- **実用アプリ**: HuggingFace実装（簡単で安定）

---

## 7. 論文での深度値記述の詳細

### 7.1 MiDaSとの関係

DPTはMiDaSフレームワークをベースにしており：

- **MiDaS v2**: 逆深度を推定
- **DPT**: MiDaSの後継として逆深度推定を継承
- **学習データ**: 異種データセットでのスケール・シフト不変学習

### 7.2 相対深度vs絶対深度

```
DPT出力 = 正規化された逆深度
数式: V_norm = (1/d - 1/d_max) / (1/d_min - 1/d_max)

ここで:
- d: 実際の深度
- d_min, d_max: 既知の最小・最大深度
- V_norm: 0-1正規化された逆深度
```

---

## 8. 実装コードでの深度マップ生成ロジック

### 8.1 公式実装での流れ

```python
# 1. モデル推論
with torch.no_grad():
    prediction = model(input_tensor)

# 2. 後処理
if invert:
    # 逆深度から深度への変換
    depth = 1.0 / (scale * prediction + shift)
else:
    # 逆深度のまま使用
    depth = prediction

# 3. 可視化用正規化
depth_vis = (depth - depth.min()) / (depth.max() - depth.min())
```

### 8.2 HuggingFace実装での流れ

```python
# 1. 前処理
inputs = processor(images=image, return_tensors="pt")

# 2. 推論
outputs = model(**inputs)

# 3. 後処理（自動）
processed_outputs = processor.post_process_depth_estimation(
    outputs, target_sizes=[(image.height, image.width)]
)
```

---

## 9. 重要な結論と推奨事項

### 9.1 重要な結論

1. **DPT-Largeは逆深度（1/depth）を出力する**
2. **高い出力値 = 近い物体、低い出力値 = 遠い物体**
3. **可視化時は「白=近い、黒=遠い」が標準**
4. **公式実装とHuggingFace実装で後処理が異なる**

### 9.2 実用上の推奨事項

#### モデル選択
- **研究・カスタマイズ**: 公式実装使用
- **プロダクション**: HuggingFace実装使用

#### 可視化時の注意
```python
# 正しい可視化（逆深度対応）
depth_vis = 255 - (depth_normalized * 255)  # 反転して近い=明るく

# または
depth_vis = (1.0 / depth_values)  # 逆深度で可視化
```

#### 数値処理時の注意
- 深度比較時は逆深度であることを考慮
- ゼロ除算エラーに注意（最小閾値設定）
- スケール・シフト処理の必要性を確認

---

## 10. 参考資料

### 10.1 主要なソース

1. **GitHub公式リポジトリ**: https://github.com/isl-org/DPT
2. **学術論文**: https://arxiv.org/abs/2103.13413
3. **HuggingFace実装**: https://huggingface.co/docs/transformers/en/model_doc/dpt
4. **FiftyOneチュートリアル**: https://docs.voxel51.com/tutorials/monocular_depth_estimation.html

### 10.2 関連技術

- **MiDaS**: DPTの基盤技術
- **Vision Transformer (ViT)**: バックボーンアーキテクチャ
- **Dense Prediction**: 画素レベルの予測タスク

---

## 付録: コード実装例

### A.1 公式実装風の深度推定

```python
import torch
from dpt.models import DPTDepthModel

# モデル読み込み
model = DPTDepthModel(
    path="weights/dpt_large-midas-2f21e586.pt",
    backbone="vitl16_384",
    non_negative=True,
    enable_attention_hooks=False,
)

# 推論
with torch.no_grad():
    prediction = model.forward(input_tensor)
    
    # 逆深度から深度への変換（必要に応じて）
    if invert:
        depth = 1.0 / (scale * prediction + shift)
        depth[depth < 1e-8] = 1e-8
```

### A.2 HuggingFace実装での深度推定

```python
from transformers import AutoImageProcessor, DPTForDepthEstimation
import torch

# モデルとプロセッサー読み込み
processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

# 推論実行
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

# 後処理
processed_outputs = processor.post_process_depth_estimation(
    outputs, target_sizes=[(image.height, image.width)]
)
```

---

**調査完了日**: 2025年7月30日  
**調査者**: Claude Code Analysis System  
**レポート形式**: Markdown形式の詳細技術レポート