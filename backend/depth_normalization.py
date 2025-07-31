"""
深度値の正規化と統一処理
各モデルの出力を統一的な深度表現に変換する
"""
import numpy as np

def normalize_depth_output(depth_array, model_name):
    """
    モデルごとの深度値を統一的な表現に正規化
    
    統一ルール: 近い = 暗い（0に近い）、遠い = 明るい（1に近い）
    これはDepth Anythingのデフォルト表現と一致
    
    Args:
        depth_array: モデルから出力された深度配列
        model_name: 使用したモデル名
    
    Returns:
        正規化された深度配列（0-1の範囲）
    """
    
    # モデル名を正規化
    model_lower = model_name.lower()
    
    # DPTとMiDaSは「近い＝大きい値」なので反転が必要
    if 'dpt' in model_lower or 'midas' in model_lower:
        # 値を反転（1 - normalized_value）
        # まず0-1に正規化
        min_val = np.min(depth_array)
        max_val = np.max(depth_array)
        
        if max_val - min_val > 0:
            normalized = (depth_array - min_val) / (max_val - min_val)
        else:
            normalized = np.zeros_like(depth_array)
        
        # 反転して返す
        return 1.0 - normalized
    
    # Depth Anythingは「遠い＝大きい値」なのでそのまま正規化
    elif 'depth' in model_lower and 'anything' in model_lower:
        min_val = np.min(depth_array)
        max_val = np.max(depth_array)
        
        if max_val - min_val > 0:
            return (depth_array - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(depth_array)
    
    # その他のモデルはデフォルトで正規化のみ
    else:
        min_val = np.min(depth_array)
        max_val = np.max(depth_array)
        
        if max_val - min_val > 0:
            return (depth_array - min_val) / (max_val - min_val)
        else:
            return np.zeros_like(depth_array)

def apply_colormap(depth_normalized, colormap='viridis'):
    """
    正規化された深度マップにカラーマップを適用
    
    Args:
        depth_normalized: 0-1に正規化された深度配列
        colormap: 使用するカラーマップ名
    
    Returns:
        RGB画像配列（0-255）
    """
    import matplotlib.pyplot as plt
    
    # カラーマップを取得
    cmap = plt.get_cmap(colormap)
    
    # カラーマップを適用（RGBAが返される）
    colored = cmap(depth_normalized)
    
    # RGBのみ取得して255スケールに変換
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return rgb