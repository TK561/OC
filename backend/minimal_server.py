#!/usr/bin/env python3
"""
最小構成のテストサーバー（CORS問題解決用）
"""
import os
import uvicorn
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Minimal Depth API Test")

# CORS設定 - 最も許可的な設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Important: False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/")
def read_root():
    return {
        "message": "Minimal server is running!",
        "port": os.getenv("PORT", "not set"),
        "environment": os.getenv("ENVIRONMENT", "not set"),
        "python_path": os.getenv("PYTHONPATH", "not set")
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "port": os.getenv("PORT", "not set"),
        "pid": os.getpid()
    }

@app.get("/test")
def test_endpoint():
    return {"test": "success", "port": os.getenv("PORT")}

@app.get("/cors-test")
async def cors_test():
    """CORS test endpoint"""
    return {
        "message": "CORS working correctly",
        "cors_enabled": True,
        "allow_origin": "*",
        "timestamp": "2025-07-25"
    }

@app.options("/api/depth/estimate")
async def options_estimate():
    """Handle preflight requests explicitly"""
    return JSONResponse(
        content={"message": "CORS preflight OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "3600"
        }
    )

@app.post("/api/depth/estimate")
async def estimate_depth_mock(file: UploadFile = File(...)):
    """
    Mock depth estimation endpoint to test CORS functionality
    """
    try:
        logger.info("🔍 Received depth estimation request")
        
        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Read image data
        image_data = await file.read()
        file_size = len(image_data)
        
        logger.info(f"📁 Processing image: {file_size / 1024:.1f}KB")
        
        # Return mock successful response
        import base64
        from PIL import Image
        import numpy as np
        import io
        
        # Create a more visible mock depth map (256x256 gradient)
        width, height = 256, 256
        depth_array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create a gradient pattern to simulate depth
        for y in range(height):
            for x in range(width):
                # Create circular gradient pattern
                center_x, center_y = width // 2, height // 2
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                
                # Normalize distance to 0-255 range
                intensity = int(255 * (1 - distance / max_distance))
                
                # Create depth color map (blue = near, red = far)
                if intensity > 200:
                    depth_array[y, x] = [0, 0, 255]  # Blue (near)
                elif intensity > 150:
                    depth_array[y, x] = [0, 255, 255]  # Cyan
                elif intensity > 100:
                    depth_array[y, x] = [0, 255, 0]  # Green
                elif intensity > 50:
                    depth_array[y, x] = [255, 255, 0]  # Yellow
                else:
                    depth_array[y, x] = [255, 0, 0]  # Red (far)
        
        # Convert to PIL Image and then to base64
        depth_img = Image.fromarray(depth_array)
        depth_buffer = io.BytesIO()
        depth_img.save(depth_buffer, format='PNG')
        depth_b64 = base64.b64encode(depth_buffer.getvalue()).decode()
        
        # Create mock response with original image data
        orig_b64 = base64.b64encode(image_data).decode()
        
        mock_response = {
            "success": True,
            "depthMapUrl": f"data:image/png;base64,{depth_b64}",
            "originalUrl": f"data:image/{file.content_type.split('/')[-1]};base64,{orig_b64}",
            "modelUsed": "mock-model-for-cors-testing",
            "resolution": f"{width}x{height}",
            "note": "Mock data - CORS testing successful!"
        }
        
        logger.info("✅ Mock depth estimation completed")
        
        response = JSONResponse(mock_response)
        
        # Explicit CORS headers
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "*"
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in mock depth estimation: {str(e)}")
        
        error_response = JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "note": "Mock API error - debugging in progress"
            }
        )
        
        # Explicit CORS headers even for errors
        error_response.headers["Access-Control-Allow-Origin"] = "*"
        error_response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS" 
        error_response.headers["Access-Control-Allow-Headers"] = "*"
        
        # Security headers
        error_response.headers["X-Content-Type-Options"] = "nosniff"
        error_response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        error_response.headers["Content-Type"] = "application/json; charset=utf-8"
        
        return error_response

if __name__ == "__main__":
    port_env = os.getenv("PORT")
    print(f"🚀 Starting minimal server")
    print(f"🔍 PORT environment variable: {port_env}")
    print(f"🔍 Working directory: {os.getcwd()}")
    print(f"🔍 Environment: {os.getenv('ENVIRONMENT', 'not set')}")
    
    if port_env is None:
        print("❌ PORT not set, using fallback 10000")
        port = 10000
    else:
        port = int(port_env)
        print(f"✅ Using PORT: {port}")
    
    # サーバー起動
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )