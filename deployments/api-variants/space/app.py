import gradio as gr

def depth_estimation(image):
    """最もシンプルな深度推定テスト"""
    if image is None:
        return None, None
    
    # まずは画像をそのまま返すテスト
    return image, image

# 最小限のGradio Interface
demo = gr.Interface(
    fn=depth_estimation,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(label="元画像"),
        gr.Image(label="深度マップ")
    ],
    title="深度推定 API",
    description="テスト中"
)

demo.launch()