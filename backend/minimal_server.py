#!/usr/bin/env python3
"""
Minimal server for Render free tier - uses mock depth estimation
"""
import os
import sys
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Minimal Depth Estimation API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create temp directory
os.makedirs("./temp", exist_ok=True)

# CORS headers for static files
@app.middleware("http")
async def add_cors_to_static(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/temp/"):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Cache-Control"] = "public, max-age=3600"
    return response

# Mount static files
app.mount("/temp", StaticFiles(directory="./temp"), name="temp")

@app.get("/")
async def root():
    return {"message": "Minimal Depth Estimation API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.options("/api/depth/estimate")
async def estimate_options():
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@app.post("/api/depth/estimate")
async def estimate_depth(file: UploadFile = File(...)):
    """
    Mock depth estimation for free tier - creates gradient depth map
    """
    try:
        # Read and validate image
        image_data = await file.read()
        if len(image_data) > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(status_code=400, detail="File too large")
        
        # Open image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Resize to small size
        max_size = 256
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Create simple gradient depth map
        width, height = image.size
        depth_array = np.zeros((height, width), dtype=np.float32)
        
        # Simple gradient from top to bottom
        for y in range(height):
            depth_array[y, :] = y / height
        
        # Add some noise for realism
        noise = np.random.normal(0, 0.05, (height, width))
        depth_array = np.clip(depth_array + noise, 0, 1)
        
        # Convert to colorized depth map using a simple colormap
        depth_normalized = (depth_array * 255).astype(np.uint8)
        
        # Create a simple color gradient (blue to red)
        depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
        depth_colored[:, :, 0] = depth_normalized  # Red channel
        depth_colored[:, :, 1] = 128  # Green channel (constant)
        depth_colored[:, :, 2] = 255 - depth_normalized  # Blue channel (inverted)
        
        depth_image = Image.fromarray(depth_colored, 'RGB')
        
        # Save images
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir='./temp') as tmp_depth:
            depth_image.save(tmp_depth.name)
            depth_name = os.path.basename(tmp_depth.name)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir='./temp') as tmp_orig:
            image.save(tmp_orig.name)
            orig_name = os.path.basename(tmp_orig.name)
        
        return JSONResponse(
            content={
                "success": True,
                "depthMapUrl": f"/temp/{depth_name}",
                "originalUrl": f"/temp/{orig_name}",
                "modelUsed": "gradient-mock",
                "resolution": f"{width}x{height}",
                "note": "Mock depth map (free tier)"
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )

@app.get("/api/depth/models")
async def get_models():
    return {
        "models": ["gradient-mock"],
        "default": "gradient-mock"
    }

# Handle OPTIONS for static files
@app.options("/temp/{path:path}")
async def static_options(path: str):
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Cross-Origin-Resource-Policy": "cross-origin"
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)