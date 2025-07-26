#!/usr/bin/env bash
# Build script for Render deployment

echo "🔨 Starting Render build process..."

# Upgrade pip
pip install --upgrade pip

# Install CPU-only PyTorch to reduce size
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p models temp

# Download models during build (optional - can be done at runtime)
# python -c "from transformers import DPTForDepthEstimation, DPTImageProcessor; DPTImageProcessor.from_pretrained('Intel/dpt-hybrid-midas'); DPTForDepthEstimation.from_pretrained('Intel/dpt-hybrid-midas')"

echo "✅ Build completed!"