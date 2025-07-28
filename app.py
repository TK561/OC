import gradio as gr
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np
from PIL import Image
import cv2
import base64
import io

class DepthEstimationAPI:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        model_name = "depth-anything/Depth-Anything-V2-Small-hf"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully")
    
    def predict(self, image_input):
        """Process image and return depth map"""
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # Base64 encoded image
                if image_input.startswith('data:image'):
                    header, encoded = image_input.split(',', 1)
                    image_bytes = base64.b64decode(encoded)
                    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                else:
                    # File path
                    image = Image.open(image_input).convert('RGB')
            else:
                # PIL Image
                image = image_input.convert('RGB') if hasattr(image_input, 'convert') else image_input
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                depth = outputs.predicted_depth.squeeze().cpu().numpy()
            
            # Create depth visualization
            depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_VIRIDIS)
            depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
            depth_image = Image.fromarray(depth_colored)
            
            # Clean up
            del inputs, outputs, depth, depth_normalized, depth_colored
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return [image, depth_image]
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return [None, None]

# Initialize API
api = DepthEstimationAPI()

# Create Gradio interface with API support
with gr.Blocks() as demo:
    gr.Markdown("# Depth Estimation API")
    gr.Markdown("AI-powered depth estimation using DepthAnything V2")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Generate Depth Map", variant="primary")
        
        with gr.Column():
            output_original = gr.Image(type="pil", label="Original Image")
            output_depth = gr.Image(type="pil", label="Depth Map")
    
    # Define the API endpoint
    submit_btn.click(
        fn=api.predict,
        inputs=input_image,
        outputs=[output_original, output_depth],
        api_name="predict"  # This creates the /api/predict endpoint
    )

# Launch with proper settings for Hugging Face Spaces
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)