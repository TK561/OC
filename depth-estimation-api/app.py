import gradio as gr
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np
from PIL import Image
import cv2

device = "cpu"
model_name = "depth-anything/Depth-Anything-V2-Small-hf"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForDepthEstimation.from_pretrained(model_name)
model.to(device)
model.eval()

def predict_depth(image):
    if image is None:
        return None, None
    
    try:
        image = image.convert('RGB')
        
        # サイズ調整
        max_size = 256
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            depth = outputs.predicted_depth.squeeze().cpu().numpy()
        
        depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_VIRIDIS)
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        depth_image = Image.fromarray(depth_colored)
        
        return image, depth_image
        
    except Exception as e:
        return image, None

demo = gr.Interface(
    fn=predict_depth,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil", label="Original"),
        gr.Image(type="pil", label="Depth Map")
    ],
    title="Depth Estimation API"
)

if __name__ == "__main__":
    demo.launch()