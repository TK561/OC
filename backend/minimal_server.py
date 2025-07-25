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
        
        # Create simple mock depth data
        mock_response = {
            "success": True,
            "depth_map_url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==",
            "original_url": f"data:image/{file.content_type.split('/')[-1]};base64,{base64.b64encode(image_data[:100]).decode()}",
            "model_used": "mock-model-for-cors-testing",
            "resolution": "1x1",
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