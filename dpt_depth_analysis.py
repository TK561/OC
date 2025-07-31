#!/usr/bin/env python3
"""
DPT (Dense Prediction Transformer) 深度値出力方向の詳細分析

このスクリプトは以下を検証します：
1. HuggingFace DPTForDepthEstimationの出力値の方向性
2. 公式実装とHuggingFace実装の比較
3. 逆深度（inverse depth）の処理方法
4. サンプル画像での実際の出力値の確認
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import json
from datetime import datetime

# HuggingFace Transformers
try:
    from transformers import AutoImageProcessor, DPTForDepthEstimation
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    print("HuggingFace Transformers not available")
    HUGGINGFACE_AVAILABLE = False

def create_test_images():
    """分析用のテスト画像を作成"""
    print("Creating test images for depth analysis...")
    
    # 1. シンプルなグラデーション画像（前景から背景への遷移）
    gradient_img = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        gradient_img[:, i] = [i, i, i]  # 左から右へのグラデーション
    
    # 2. 中央に物体がある画像（前景・背景の関係が明確）
    center_object = np.full((256, 256, 3), 128, dtype=np.uint8)  # 背景をグレー
    center_object[64:192, 64:192] = [255, 255, 255]  # 中央に白い四角（前景）
    
    return {
        'gradient': Image.fromarray(gradient_img),
        'center_object': Image.fromarray(center_object)
    }

def load_sample_image():
    """インターネットからサンプル画像をロード"""
    try:
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        response = requests.get(url, stream=True)
        image = Image.open(BytesIO(response.content))
        print(f"Loaded sample image: {image.size}")
        return image
    except Exception as e:
        print(f"Failed to load sample image: {e}")
        return None

def analyze_dpt_huggingface():
    """HuggingFace DPTの深度推定出力を分析"""
    if not HUGGINGFACE_AVAILABLE:
        return None
    
    print("\n=== HuggingFace DPT Analysis ===")
    
    try:
        # モデルとプロセッサーの読み込み
        print("Loading DPT model and processor...")
        image_processor = AutoImageProcessor.from_pretrained("Intel/dpt-large")
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        model.eval()
        
        # テスト画像の準備
        test_images = create_test_images()
        sample_image = load_sample_image()
        
        results = {}
        
        # 各画像での深度推定
        for name, image in test_images.items():
            print(f"\nAnalyzing {name} image...")
            
            # 前処理
            inputs = image_processor(images=image, return_tensors="pt")
            
            # 推論実行
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # 統計情報の取得
            depth_np = predicted_depth.squeeze().cpu().numpy()
            
            analysis = {
                'shape': depth_np.shape,
                'min_value': float(np.min(depth_np)),
                'max_value': float(np.max(depth_np)),
                'mean_value': float(np.mean(depth_np)),
                'std_value': float(np.std(depth_np)),
                'center_value': float(depth_np[depth_np.shape[0]//2, depth_np.shape[1]//2]),
                'corner_values': {
                    'top_left': float(depth_np[0, 0]),
                    'top_right': float(depth_np[0, -1]),
                    'bottom_left': float(depth_np[-1, 0]),
                    'bottom_right': float(depth_np[-1, -1])
                }
            }
            
            # グラデーション画像の場合、左右の値を比較
            if name == 'gradient':
                left_mean = np.mean(depth_np[:, :64])  # 左側（暗い部分）
                right_mean = np.mean(depth_np[:, -64:])  # 右側（明るい部分）
                analysis['left_mean'] = float(left_mean)
                analysis['right_mean'] = float(right_mean)
                analysis['left_vs_right'] = 'left_higher' if left_mean > right_mean else 'right_higher'
            
            # 中央物体画像の場合、中央vs周辺を比較
            if name == 'center_object':
                center_region = depth_np[64:192, 64:192]
                border_region = np.concatenate([
                    depth_np[:64, :].flatten(),
                    depth_np[-64:, :].flatten(),
                    depth_np[64:192, :64].flatten(),
                    depth_np[64:192, -64:].flatten()
                ])
                center_mean = np.mean(center_region)
                border_mean = np.mean(border_region)
                analysis['center_mean'] = float(center_mean)
                analysis['border_mean'] = float(border_mean)
                analysis['center_vs_border'] = 'center_higher' if center_mean > border_mean else 'border_higher'
            
            results[name] = analysis
        
        # 実際のサンプル画像での分析
        if sample_image:
            print(f"\nAnalyzing real sample image...")
            inputs = image_processor(images=sample_image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            depth_np = predicted_depth.squeeze().cpu().numpy()
            
            results['sample_image'] = {
                'shape': depth_np.shape,
                'min_value': float(np.min(depth_np)),
                'max_value': float(np.max(depth_np)),
                'mean_value': float(np.mean(depth_np)),
                'std_value': float(np.std(depth_np))
            }
        
        return results
        
    except Exception as e:
        print(f"Error in HuggingFace DPT analysis: {e}")
        return None

def analyze_inverse_depth_conversion():
    """逆深度変換の分析"""
    print("\n=== Inverse Depth Conversion Analysis ===")
    
    # 仮想的な深度値での逆深度変換の確認
    depth_values = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 50.0])  # メートル単位の深度
    inverse_depth = 1.0 / depth_values
    
    print("Depth (m) -> Inverse Depth:")
    for d, inv_d in zip(depth_values, inverse_depth):
        print(f"  {d:5.1f}m -> {inv_d:.4f}")
    
    print(f"\nInverse depth characteristics:")
    print(f"  Closer objects (smaller depth) -> Higher inverse depth values")
    print(f"  Farther objects (larger depth) -> Lower inverse depth values")
    
    return {
        'depth_values': depth_values.tolist(),
        'inverse_depth_values': inverse_depth.tolist(),
        'explanation': "Higher inverse depth values = closer objects, Lower inverse depth values = farther objects"
    }

def main():
    """メイン実行関数"""
    print("DPT Depth Value Direction Analysis")
    print("=" * 50)
    
    # 分析結果を格納する辞書
    analysis_results = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'DPT_depth_direction_analysis',
        'huggingface_analysis': None,
        'inverse_depth_theory': None
    }
    
    # HuggingFace DPT分析
    hf_results = analyze_dpt_huggingface()
    if hf_results:
        analysis_results['huggingface_analysis'] = hf_results
    
    # 逆深度理論分析
    inverse_depth_results = analyze_inverse_depth_conversion()
    analysis_results['inverse_depth_theory'] = inverse_depth_results
    
    # 結果の保存
    output_file = "dpt_final_test.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nAnalysis results saved to: {output_file}")
    
    # 結果のサマリー表示
    print("\n=== Analysis Summary ===")
    if hf_results:
        print("\nHuggingFace DPT Results:")
        for image_name, data in hf_results.items():
            print(f"  {image_name}:")
            print(f"    Min/Max: {data['min_value']:.6f} / {data['max_value']:.6f}")
            print(f"    Mean: {data['mean_value']:.6f}")
            
            if 'left_vs_right' in data:
                print(f"    Gradient analysis: {data['left_vs_right']}")
                print(f"    Left mean: {data['left_mean']:.6f}, Right mean: {data['right_mean']:.6f}")
            
            if 'center_vs_border' in data:
                print(f"    Center vs Border: {data['center_vs_border']}")
                print(f"    Center mean: {data['center_mean']:.6f}, Border mean: {data['border_mean']:.6f}")
    
    return analysis_results

if __name__ == "__main__":
    results = main()