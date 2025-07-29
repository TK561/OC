#!/usr/bin/env python3
"""
展示用ローカル深度推定システム - セットアップスクリプト
"""

import subprocess
import sys
import torch
import os
from pathlib import Path

def check_gpu():
    """GPU使用可能性チェック"""
    print("🔍 GPU環境チェック中...")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU検出: {gpu_name}")
        print(f"✅ GPU メモリ: {gpu_memory:.1f}GB")
        return True
    else:
        print("⚠️ GPU未検出 - CPU実行になります")
        print("   展示用にはGPU推奨です")
        return False

def download_model():
    """モデル事前ダウンロード"""
    print("📥 DepthAnything V2 モデルダウンロード中...")
    
    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        
        model_name = "depth-anything/Depth-Anything-V2-Small-hf"
        
        print("   プロセッサーダウンロード...")
        processor = AutoImageProcessor.from_pretrained(model_name)
        
        print("   モデルダウンロード...")
        model = AutoModelForDepthEstimation.from_pretrained(model_name)
        
        print("✅ モデルダウンロード完了")
        print(f"   保存場所: ~/.cache/huggingface/")
        
        return True
        
    except Exception as e:
        print(f"❌ モデルダウンロードエラー: {e}")
        return False

def create_startup_script():
    """起動スクリプト作成"""
    print("📝 起動スクリプト作成中...")
    
    # Windows バッチファイル
    batch_content = """@echo off
echo 🏛️ 展示用深度推定システム起動中...
echo.

REM 仮想環境アクティベート（必要に応じて）
REM call conda activate depth_estimation

REM API サーバー起動
python main.py

pause
"""
    
    with open("start_exhibition.bat", "w", encoding="utf-8") as f:
        f.write(batch_content)
    
    # Linux/Mac シェルスクリプト
    shell_content = """#!/bin/bash
echo "🏛️ 展示用深度推定システム起動中..."
echo

# 仮想環境アクティベート（必要に応じて）
# source activate depth_estimation

# API サーバー起動
python main.py
"""
    
    with open("start_exhibition.sh", "w") as f:
        f.write(shell_content)
    
    # 実行権限付与
    if os.name != 'nt':
        os.chmod("start_exhibition.sh", 0o755)
    
    print("✅ 起動スクリプト作成完了")
    print("   Windows: start_exhibition.bat")
    print("   Linux/Mac: start_exhibition.sh")

def create_config():
    """設定ファイル作成"""
    print("⚙️ 設定ファイル作成中...")
    
    config_content = """# 展示用設定ファイル
{
    "system": {
        "name": "展示用深度推定システム",
        "version": "1.0.0",
        "mode": "exhibition"
    },
    "security": {
        "local_only": true,
        "max_image_size": 2048,
        "allowed_formats": ["image/jpeg", "image/png", "image/webp"],
        "auto_cleanup": true
    },
    "performance": {
        "gpu_memory_fraction": 0.8,
        "batch_size": 1,
        "precision": "fp16"
    },
    "exhibition": {
        "touch_interface": true,
        "auto_demo": false,
        "display_processing_time": true,
        "log_visitor_count": true
    }
}
"""
    
    with open("config.json", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print("✅ 設定ファイル作成完了: config.json")

def setup_logging():
    """ログディレクトリ作成"""
    print("📁 ログディレクトリ作成中...")
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    print("✅ ログディレクトリ作成完了: logs/")

def main():
    """メインセットアップ処理"""
    print("🏛️" + "="*50)
    print("   展示用深度推定システム - セットアップ")
    print("="*53)
    print()
    
    # 1. GPU チェック
    gpu_available = check_gpu()
    print()
    
    # 2. モデルダウンロード
    if not download_model():
        print("❌ セットアップ失敗: モデルダウンロードに失敗しました")
        return False
    print()
    
    # 3. 起動スクリプト作成
    create_startup_script()
    print()
    
    # 4. 設定ファイル作成
    create_config()
    print()
    
    # 5. ログディレクトリ作成
    setup_logging()
    print()
    
    # セットアップ完了
    print("🎉 セットアップ完了!")
    print()
    print("📋 次のステップ:")
    print("   1. フロントエンドのバックエンドURLを変更:")
    print("      NEXT_PUBLIC_BACKEND_URL=http://localhost:8000")
    print()
    print("   2. システム起動:")
    if os.name == 'nt':
        print("      start_exhibition.bat をダブルクリック")
    else:
        print("      ./start_exhibition.sh")
    print()
    print("   3. ブラウザで確認:")
    print("      http://localhost:8000/ (API)")
    print("      http://localhost:3000/ (フロントエンド)")
    print()
    
    if not gpu_available:
        print("⚠️ 注意: GPU未検出")
        print("   展示用途では処理時間が長くなる可能性があります")
        print("   GPUドライバーとCUDAのインストールを推奨します")
    
    print("🔒 セキュリティ: 完全ローカル実行で最高レベルの安全性")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ セットアップが中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ セットアップエラー: {e}")
        sys.exit(1)