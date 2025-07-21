#!/usr/bin/env python3
"""
Render-optimized server startup script
"""
import os
import uvicorn
import sys

# Ensure the app module is in Python path
sys.path.insert(0, '/opt/render/project/src')

def main():
    # Get port from environment
    port = os.getenv("PORT")
    
    print(f"🚀 Starting Depth Estimation API")
    print(f"🔍 PORT environment variable: {port}")
    print(f"🔍 Python path: {sys.path}")
    print(f"🔍 Working directory: {os.getcwd()}")
    print(f"🔍 Files in current directory: {os.listdir('.')}")
    
    if port is None:
        print("❌ PORT environment variable not set!")
        port = "10000"  # Fallback for Render
    
    try:
        port_int = int(port)
        print(f"✅ Using port: {port_int}")
    except ValueError:
        print(f"❌ Invalid port value: {port}")
        port_int = 10000
    
    # Import the app
    try:
        from app.main import app
        print("✅ Successfully imported app")
    except ImportError as e:
        print(f"❌ Failed to import app: {e}")
        print(f"🔍 Trying alternative import...")
        try:
            import sys
            sys.path.append('/opt/render/project/src')
            from app.main import app
            print("✅ Successfully imported app with modified path")
        except ImportError as e2:
            print(f"❌ Failed alternative import: {e2}")
            sys.exit(1)
    
    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port_int,
        log_level="info"
    )

if __name__ == "__main__":
    main()