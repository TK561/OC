#!/usr/bin/env python3
"""
Render-optimized server startup script
"""
import os
import uvicorn
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Get port from environment - Render sets this automatically
    port = os.getenv("PORT", "10000")
    
    logger.info("=" * 50)
    logger.info("🚀 Starting Depth Estimation API for Render")
    logger.info(f"📍 PORT: {port}")
    logger.info(f"📍 Working directory: {os.getcwd()}")
    logger.info(f"📍 Python executable: {sys.executable}")
    logger.info("=" * 50)
    
    try:
        port_int = int(port)
    except ValueError:
        logger.error(f"Invalid port value: {port}, using default 10000")
        port_int = 10000
    
    # Import the app with better error handling
    try:
        # Try direct import first
        from app.main import app
        logger.info("✅ Successfully imported app directly")
    except ImportError as e:
        logger.warning(f"Direct import failed: {e}")
        try:
            # Add current directory to Python path
            current_dir = os.path.dirname(os.path.abspath(__file__))
            sys.path.insert(0, current_dir)
            from app.main import app
            logger.info("✅ Successfully imported app with modified path")
        except ImportError as e2:
            logger.error(f"❌ Failed to import app: {e2}")
            logger.error(f"Python path: {sys.path}")
            logger.error(f"Directory contents: {os.listdir('.')}")
            sys.exit(1)
    
    # Start the server with proper configuration
    logger.info(f"🌐 Starting server on 0.0.0.0:{port_int}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port_int,
        log_level="info",
        access_log=True,
        use_colors=True
    )

if __name__ == "__main__":
    main()