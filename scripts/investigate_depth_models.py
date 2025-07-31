"""
各モデルの実際の出力を調査するスクリプト
"""

# DPT-Largeの公式説明から：
# https://huggingface.co/Intel/dpt-large
# "DPT uses the Vision Transformer (ViT) as backbone and adds a neck + head on top for monocular depth estimation."
# "The model outputs relative depth predictions"

# MiDaSの公式説明から：
# https://github.com/isl-org/MiDaS
# "MiDaS computes relative inverse depth from a single image"
# "The output is an inverse depth map"

# Depth Anythingの公式説明から：
# https://github.com/LiheYoung/Depth-Anything
# "Our foundation model for robust monocular depth estimation"
# "The model outputs metric depth"

"""
重要な発見：
1. DPT-Large: relative depth を出力（小さい値 = 近い）
2. MiDaS: relative inverse depth を出力（大きい値 = 近い）
3. Depth Anything: metric depth を出力（大きい値 = 遠い）

つまり：
- DPTは実際には「近い = 小さい値」なので、反転は不要
- MiDaSは「近い = 大きい値」なので、反転が必要
- Depth Anythingは「遠い = 大きい値」なので、反転は不要
"""

def correct_depth_normalization(depth_array, model_name):
    """
    正しい深度値の正規化
    
    統一ルール: 近い = 暗い（0）、遠い = 明るい（1）
    """
    import numpy as np
    
    model_lower = model_name.lower()
    
    # まず0-1に正規化
    min_val = np.min(depth_array)
    max_val = np.max(depth_array)
    
    if max_val - min_val > 0:
        normalized = (depth_array - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(depth_array)
    
    # モデルごとの処理
    if 'dpt-large' in model_lower:
        # DPT-Large: relative depth (小さい値 = 近い)
        # そのまま使用（反転不要）
        return normalized
    
    elif 'midas' in model_lower or 'dpt-hybrid-midas' in model_lower:
        # MiDaS: inverse depth (大きい値 = 近い)
        # 反転が必要
        return 1.0 - normalized
    
    elif 'depth' in model_lower and 'anything' in model_lower:
        # Depth Anything: metric depth (大きい値 = 遠い)
        # そのまま使用（反転不要）
        return normalized
    
    else:
        # デフォルト
        return normalized