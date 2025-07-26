from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import logging
from dotenv import load_dotenv

from .routers import depth, processing
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# Log startup information
logger.info("=" * 50)
logger.info("🚀 Initializing Depth Estimation API")
logger.info(f"📍 Environment: {settings.ENVIRONMENT}")
logger.info(f"📍 PORT: {os.getenv('PORT', 'not set')}")
logger.info("=" * 50)

app = FastAPI(
    title="Depth Estimation API",
    description="API for depth estimation and 3D visualization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Add explicit CORS headers for all responses
@app.middleware("http")
async def add_cors_headers(request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

app.include_router(depth.router, prefix="/api/depth", tags=["depth"])
app.include_router(processing.router, prefix="/api/processing", tags=["processing"])

# Add static file serving for temporary files
if os.path.exists(settings.TEMP_DIR):
    app.mount("/temp", StaticFiles(directory=settings.TEMP_DIR), name="temp")

@app.get("/")
async def root():
    return {
        "message": "Depth Estimation API",
        "version": "1.0.0",
        "status": "running",
        "port": os.getenv("PORT", "not set"),
        "environment": settings.ENVIRONMENT
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "port": os.getenv("PORT", "not set"),
        "host": "0.0.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development"
    )
    print(f"Server starting on port {port}")