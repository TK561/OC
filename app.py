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
        self.device = "cpu"  # Force CPU for Hugging Face free tier
        print(f"Using device: {self.device}")
        
        # Use the small model for better CPU performance
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
            if image_input is None:
                return None, None
                
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
            
            # Resize image for faster CPU processing
            max_size = 512
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
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
            
            # Resize depth map back to original size
            depth_image = Image.fromarray(depth_colored)
            depth_image = depth_image.resize(image.size, Image.Resampling.LANCZOS)
            
            # Clean up
            del inputs, outputs, depth, depth_normalized, depth_colored
            
            return image, depth_image
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, None

# Initialize API
api = DepthEstimationAPI()

# Create simple interface for better compatibility
demo = gr.Interface(
    fn=api.predict,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=[
        gr.Image(type="pil", label="Original Image"),
        gr.Image(type="pil", label="Depth Map")
    ],
    title="Depth Estimation API",
    description="AI-powered depth estimation using DepthAnything V2 (CPU optimized)",
    allow_flagging="never",
    cache_examples=False
)

# Launch for Hugging Face Spaces
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)