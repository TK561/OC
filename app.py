import gradio as gr
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np
from PIL import Image
import cv2

def process_image(image):
    """Process uploaded image and return depth map"""
    if image is None:
        return None, None
    
    try:
        # Initialize model (lightweight approach)
        device = "cpu"
        model_name = "depth-anything/Depth-Anything-V2-Small-hf"
        
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForDepthEstimation.from_pretrained(model_name)
        model.to(device)
        model.eval()
        
        # Resize for faster processing
        max_size = 384
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Process
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            depth = outputs.predicted_depth.squeeze().cpu().numpy()
        
        # Normalize and colorize
        depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_VIRIDIS)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        depth_image = Image.fromarray(depth_colored)
        
        return image, depth_image
        
    except Exception as e:
        print(f"Error: {e}")
        return image, None

# Create interface
with gr.Blocks(title="Depth Estimation") as demo:
    gr.Markdown("# Depth Estimation API")
    gr.Markdown("Upload an image to generate a depth map using DepthAnything V2")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            process_btn = gr.Button("Generate Depth Map", variant="primary")
        
        with gr.Column():
            output_original = gr.Image(label="Original")
            output_depth = gr.Image(label="Depth Map")
    
    process_btn.click(
        fn=process_image,
        inputs=input_image,
        outputs=[output_original, output_depth]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)