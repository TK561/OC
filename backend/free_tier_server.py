#!/usr/bin/env python3
"""
Render Free Tier Optimized Server
Memory-efficient depth estimation with model loading optimization
"""
import os
import sys
import gc
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Depth Estimation API (Free Tier)",
    description="Memory-optimized API for depth estimation",
    version="1.0.0"
)

# CORS middleware - Allow all origins for free tier simplicity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for free tier
    allow_credentials=False,  # Must be False when allow_origins is "*"
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Global variables for lazy loading
depth_estimator = None
available_models = {
    "Intel/dpt-hybrid-midas": "MiDaS (軽量)",
    "Intel/dpt-large": "DPT-Large (高精度・重い)",
    "LiheYoung/depth-anything-large-hf": "DepthAnything (汎用・重い)"
}

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0  # psutil not available

def load_depth_estimator():
    """Lazy load depth estimator only when needed"""
    global depth_estimator
    
    if depth_estimator is None:
        try:
            logger.info(f"💾 Loading depth estimator (Memory: {get_memory_usage():.1f}MB)")
            
            # Import here to avoid loading on startup
            from transformers import DPTImageProcessor, DPTForDepthEstimation
            from PIL import Image
            import torch
            import numpy as np
            
            # Use the lightest model for free tier
            model_name = "Intel/dpt-hybrid-midas"  # Smaller than dpt-large
            
            # Load with CPU-only to save memory
            processor = DPTImageProcessor.from_pretrained(model_name)
            model = DPTForDepthEstimation.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True
            )
            
            depth_estimator = {
                'processor': processor,
                'model': model,
                'model_name': model_name
            }
            
            logger.info(f"✅ Model loaded (Memory: {get_memory_usage():.1f}MB)")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    
    return depth_estimator

@app.get("/")
async def root():
    memory_mb = get_memory_usage()
    return {
        "message": "Depth Estimation API (Free Tier)",
        "version": "1.0.0",
        "status": "running",
        "memory_usage_mb": round(memory_mb, 1),
        "port": os.getenv("PORT", "not set"),
        "environment": "production"
    }

@app.get("/health")
async def health_check():
    memory_mb = get_memory_usage()
    return {
        "status": "healthy",
        "memory_usage_mb": round(memory_mb, 1),
        "port": os.getenv("PORT", "not set"),
        "host": "0.0.0.0"
    }

@app.get("/cors-test")
async def cors_test():
    """CORS test endpoint"""
    return {
        "message": "CORS test successful",
        "timestamp": "2025-07-25T19:30:00Z",
        "cors_enabled": True
    }

@app.get("/api/depth/models")
async def get_available_models():
    """Get list of available models"""
    return {
        "models": list(available_models.keys()),
        "default": "Intel/dpt-hybrid-midas",
        "descriptions": available_models
    }

@app.post("/api/depth/estimate")
async def estimate_depth(
    file: UploadFile = File(...),
    model_name: str = "Intel/dpt-hybrid-midas"
):
    """
    Estimate depth from uploaded image (Free tier optimized)
    """
    try:
        logger.info(f"🔍 Starting depth estimation (Memory: {get_memory_usage():.1f}MB)")
        
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Read image data
        image_data = await file.read()
        
        # Load model lazily
        estimator = load_depth_estimator()
        
        # Import processing libraries
        from PIL import Image
        import torch
        import numpy as np
        import io
        import base64
        
        # Process image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Resize for free tier memory constraints
        max_size = 512  # Reduce from default 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"📏 Resized image to {new_size}")
        
        # Prepare inputs
        inputs = estimator['processor'](images=image, return_tensors="pt")
        
        logger.info(f"🧠 Running model inference (Memory: {get_memory_usage():.1f}MB)")
        
        # Run inference
        with torch.no_grad():
            outputs = estimator['model'](**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Post-process
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        
        # Create depth map image
        depth_image = Image.fromarray(formatted, mode='L')
        
        # Convert to base64 for response
        img_buffer = io.BytesIO()
        depth_image.save(img_buffer, format='PNG')
        depth_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        # Convert original to base64
        orig_buffer = io.BytesIO()
        image.save(orig_buffer, format='PNG')
        orig_b64 = base64.b64encode(orig_buffer.getvalue()).decode()
        
        # Clean up memory
        del inputs, outputs, predicted_depth, prediction
        gc.collect()
        
        logger.info(f"✅ Depth estimation completed (Memory: {get_memory_usage():.1f}MB)")
        
        return JSONResponse({
            "success": True,
            "depth_map_url": f"data:image/png;base64,{depth_b64}",
            "original_url": f"data:image/png;base64,{orig_b64}",
            "model_used": estimator['model_name'],
            "resolution": f"{image.width}x{image.height}",
            "memory_usage_mb": round(get_memory_usage(), 1)
        })
        
    except Exception as e:
        logger.error(f"❌ Depth estimation error: {str(e)}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        # Clean up on error
        gc.collect()
        
        # Return more detailed error information
        error_detail = {
            "error": str(e),
            "error_type": type(e).__name__,
            "memory_usage_mb": round(get_memory_usage(), 1),
            "suggestion": "画像サイズを小さくするか、しばらく待ってから再試行してください"
        }
        raise HTTPException(status_code=500, detail=error_detail)

def main():
    port = int(os.getenv("PORT", 10000))
    logger.info(f"🚀 Starting Free Tier Depth Estimation API on port {port}")
    logger.info(f"💾 Initial memory usage: {get_memory_usage():.1f}MB")
    logger.info(f"🔍 Environment: {os.getenv('ENVIRONMENT', 'not set')}")
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        raise

if __name__ == "__main__":
    main()