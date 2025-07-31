# DPT、MiDaS、Depth Anythingの深度推定結果の違いに関する調査結果

## 主な違いの原因

### 1. アーキテクチャの違い
- **DPT-Large**: 純粋なVision Transformer → 細かいディテールを保持
- **MiDaS v3.1**: CNN+Transformerハイブリッド → バランスの取れた結果
- **Depth Anything**: 大規模データで学習したViT → 汎化性能が高い

### 2. 学習データの違い
- **DPT**: 高品質な屋内データセット中心（MIX-6）
- **MiDaS**: 複数データセットの混合（MIX-6 + ReDWeb）
- **Depth Anything**: 6200万枚の大規模野生データ

### 3. 出力特性の違い

#### 深度値の範囲
- **DPT**: 最も広い値域、細かい深度差を表現
- **MiDaS**: 中間的な値域、滑らかな遷移
- **Depth Anything**: 場面により大きく変動

#### エッジ処理
- **DPT**: シャープで精密なエッジ
- **MiDaS**: 適度にスムーズ化されたエッジ
- **Depth Anything**: 自然なエッジ保持

#### ノイズ特性
- **DPT**: 低ノイズだが、暗部で不安定
- **MiDaS**: 最も安定、ノイズが少ない
- **Depth Anything**: シーンにより変動

## 実用上の影響

### 3D変換時の違い
1. **点群の密度**: DPTが最も高密度
2. **表面の滑らかさ**: MiDaSが最も滑らか
3. **全体的な形状**: Depth Anythingが最も自然

### 用途別の推奨

#### 精密な計測が必要な場合
→ **DPT-Large**を使用
- 建築物の3Dスキャン
- 産業用検査
- 研究目的

#### バランスの良い結果が欲しい場合
→ **MiDaS v3.1**を使用
- 一般的な3D効果
- 写真編集
- リアルタイム処理

#### 多様なシーンに対応したい場合
→ **Depth Anything**を使用
- 野外撮影
- 未知のシーン
- ロバスト性重視

## 技術的な対策

### 統一的な後処理
```python
def normalize_depth_map(depth_map, model_type):
    if model_type == 'dpt':
        # DPTは値域が広いので圧縮
        return np.log1p(depth_map) / np.log1p(depth_map.max())
    elif model_type == 'midas':
        # MiDaSは既に適度に正規化されている
        return depth_map / depth_map.max()
    elif model_type == 'depth_anything':
        # Depth Anythingは外れ値を除去
        percentile_99 = np.percentile(depth_map, 99)
        return np.clip(depth_map, 0, percentile_99) / percentile_99
```

### モデル間の互換性確保
1. 共通の深度スケールに変換
2. カメラパラメータの統一
3. 座標系の整合性確保

## 結論

3つのモデルの違いは、それぞれの設計思想と最適化目標の違いから生じています：

- **DPT**: 精度と詳細性を重視
- **MiDaS**: 実用性とバランスを重視  
- **Depth Anything**: 汎用性とロバスト性を重視

これらの違いを理解し、用途に応じて適切なモデルを選択することが重要です。また、複数のモデルの結果を組み合わせることで、より良い結果を得られる可能性もあります。