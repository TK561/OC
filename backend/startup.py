#!/usr/bin/env python3
"""
Simple startup script for Render deployment
"""
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure we can import the app
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def main():
    try:
        import uvicorn
        from app.main import app
        
        port = int(os.getenv("PORT", 10000))
        
        logger.info(f"🚀 Starting Depth Estimation API on port {port}")
        logger.info(f"📂 Working directory: {os.getcwd()}")
        logger.info(f"🐍 Python path: {sys.path[:3]}...")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        raise

if __name__ == "__main__":
    main()