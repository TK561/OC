import streamlit as st
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np
from PIL import Image
import cv2
import io

@st.cache_resource
def load_model():
    """Load and cache the depth estimation model"""
    device = "cpu"
    model_name = "depth-anything/Depth-Anything-V2-Small-hf"
    
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    return processor, model, device

def process_image(image, processor, model, device):
    """Process uploaded image and return depth map"""
    if image is None:
        return None, None
    
    try:
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
        st.error(f"Error processing image: {e}")
        return image, None

def main():
    st.title("Depth Estimation API")
    st.markdown("Upload an image to generate a depth map using DepthAnything V2")
    
    # Load model
    processor, model, device = load_model()
    st.success("Model loaded successfully!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        # Process image
        with st.spinner("Generating depth map..."):
            original, depth_map = process_image(image, processor, model, device)
        
        if depth_map is not None:
            with col2:
                st.subheader("Depth Map")
                st.image(depth_map, use_column_width=True)
        else:
            st.error("Failed to generate depth map")

if __name__ == "__main__":
    main()