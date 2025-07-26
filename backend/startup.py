#!/usr/bin/env python3
"""
Simple startup script for Render deployment
"""
import os
import sys

# Ensure we can import the app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Import and run uvicorn directly
    import uvicorn
    from app.main import app
    
    port = int(os.getenv("PORT", 10000))
    
    print(f"Starting server on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )