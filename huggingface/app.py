import gradio as gr
import torch
import numpy as np
from PIL import Image
import io
from transformers import DPTImageProcessor, DPTForDepthEstimation
import cv2

# グローバル変数でモデルを保持
processor = None
model = None

def load_model():
    """モデルを一度だけ読み込む"""
    global processor, model
    if processor is None or model is None:
        print("Loading depth estimation model...")
        processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"Model loaded on {device}")

def estimate_depth(image):
    """深度推定を実行"""
    try:
        # モデル読み込み
        load_model()
        
        # 画像の前処理
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # RGB変換
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # サイズ制限（メモリ効率のため）
        max_size = 512
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # 推論実行
        inputs = processor(images=image, return_tensors="pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # 深度マップの後処理
        depth = predicted_depth.squeeze().cpu().numpy()
        depth_min = depth.min()
        depth_max = depth.max()
        
        if depth_max - depth_min > 0:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth_normalized = np.zeros_like(depth)
        
        # カラーマップ適用
        depth_colored = cv2.applyColorMap(
            (depth_normalized * 255).astype(np.uint8), 
            cv2.COLORMAP_VIRIDIS
        )
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(depth_colored), image
        
    except Exception as e:
        print(f"Error in depth estimation: {e}")
        # エラー時は元画像をそのまま返す
        return image, image

def process_image(image):
    """Gradio用の処理関数"""
    if image is None:
        return None, None
    
    depth_map, original = estimate_depth(image)
    return original, depth_map

# Gradio インターフェース作成
with gr.Blocks(title="深度推定 API", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌊 深度推定・3D可視化 API")
    gr.Markdown("画像をアップロードして深度マップを生成します")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="入力画像", 
                type="pil",
                height=400
            )
            submit_btn = gr.Button("深度推定実行", variant="primary", size="lg")
        
        with gr.Column():
            with gr.Tab("元画像"):
                output_original = gr.Image(label="元画像", height=400)
            with gr.Tab("深度マップ"):
                output_depth = gr.Image(label="深度マップ", height=400)
    
    with gr.Row():
        gr.Markdown("""
        ### 📝 使い方
        1. 画像をアップロードまたはドラッグ&ドロップ
        2. 「深度推定実行」ボタンをクリック
        3. 深度マップが生成されます（紫=近い、黄=遠い）
        
        ### ⚡ 技術情報
        - モデル: Intel DPT-Hybrid-MiDaS
        - 処理時間: 数秒〜数十秒
        - 最大解像度: 512px（メモリ効率のため）
        """)
    
    # イベントハンドラー
    submit_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[output_original, output_depth]
    )
    
    # サンプル画像も処理可能
    input_image.change(
        fn=process_image,
        inputs=[input_image],
        outputs=[output_original, output_depth]
    )

# アプリケーション起動
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )