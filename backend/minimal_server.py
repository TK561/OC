#!/usr/bin/env python3
"""
最小構成のテストサーバー（デバッグ用）
"""
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Minimal Depth API Test")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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