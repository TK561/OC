from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
import tempfile
import os
import logging

from ..models.depth_model import DepthEstimator
from ..utils.image_utils import validate_image, process_image
from ..utils.pointcloud import generate_pointcloud, save_pointcloud
from ..config import settings

# Create necessary directories
import os
os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

router = APIRouter()

depth_estimator = DepthEstimator()

@router.post("/estimate")
async def estimate_depth(
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
    resolution: Optional[int] = None
):
    """
    Estimate depth from uploaded image
    """
    try:
        if not validate_image(file):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        image_data = await file.read()
        
        if model_name and model_name not in settings.AVAILABLE_MODELS:
            model_name = settings.DEFAULT_DEPTH_MODEL
        
        depth_map, original_image = await depth_estimator.predict(
            image_data, 
            model_name=model_name or settings.DEFAULT_DEPTH_MODEL,
            target_resolution=resolution
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_depth:
            depth_map.save(tmp_depth.name)
            depth_path = tmp_depth.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_orig:
            original_image.save(tmp_orig.name)
            orig_path = tmp_orig.name
        
        return JSONResponse({
            "success": True,
            "depth_map_url": f"/temp/{os.path.basename(depth_path)}",
            "original_url": f"/temp/{os.path.basename(orig_path)}",
            "model_used": model_name or settings.DEFAULT_DEPTH_MODEL,
            "resolution": f"{original_image.width}x{original_image.height}"
        })
        
    except Exception as e:
        logger.error(f"Depth estimation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/generate-3d")
async def generate_3d(
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
    point_density: Optional[float] = 1.0,
    export_format: Optional[str] = "ply"
):
    """
    Generate 3D point cloud from image
    """
    try:
        if not validate_image(file):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        image_data = await file.read()
        
        depth_map, original_image = await depth_estimator.predict(
            image_data,
            model_name=model_name or settings.DEFAULT_DEPTH_MODEL
        )
        
        pointcloud = generate_pointcloud(
            original_image,
            depth_map,
            density=point_density
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{export_format}') as tmp_3d:
            save_pointcloud(pointcloud, tmp_3d.name, format=export_format)
            cloud_path = tmp_3d.name
        
        return FileResponse(
            cloud_path,
            media_type='application/octet-stream',
            filename=f"pointcloud.{export_format}"
        )
        
    except Exception as e:
        logger.error(f"3D generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"3D generation failed: {str(e)}")

@router.get("/models")
async def get_available_models():
    """
    Get list of available depth estimation models
    """
    return {
        "models": settings.AVAILABLE_MODELS,
        "default": settings.DEFAULT_DEPTH_MODEL
    }