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

# No global depth_estimator - create per request for memory efficiency

@router.options("/estimate")
async def estimate_depth_options():
    """Handle preflight requests"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

@router.post("/estimate")
async def estimate_depth(
    file: UploadFile = File(...),
    model_name: Optional[str] = None,
    resolution: Optional[int] = None
):
    """
    Estimate depth from uploaded image
    """
    depth_estimator = None
    try:
        if not validate_image(file):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        image_data = await file.read()
        
        if model_name and model_name not in settings.AVAILABLE_MODELS:
            model_name = settings.LIGHTWEIGHT_MODEL
        
        # Create estimator per request for memory efficiency
        depth_estimator = DepthEstimator()
        
        depth_map, original_image = await depth_estimator.predict(
            image_data, 
            model_name=model_name or settings.LIGHTWEIGHT_MODEL,
            target_resolution=resolution
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_depth:
            depth_map.save(tmp_depth.name)
            depth_path = tmp_depth.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_orig:
            original_image.save(tmp_orig.name)
            orig_path = tmp_orig.name
        
        # Move files to temp directory with proper names
        temp_depth_name = f"depth_{os.path.basename(depth_path)}"
        temp_orig_name = f"orig_{os.path.basename(orig_path)}"
        
        temp_depth_final = os.path.join(settings.TEMP_DIR, temp_depth_name)
        temp_orig_final = os.path.join(settings.TEMP_DIR, temp_orig_name)
        
        import shutil
        shutil.move(depth_path, temp_depth_final)
        shutil.move(orig_path, temp_orig_final)
        
        return JSONResponse(
            content={
                "success": True,
                "depthMapUrl": f"/temp/{temp_depth_name}",
                "originalUrl": f"/temp/{temp_orig_name}",
                "modelUsed": model_name or settings.LIGHTWEIGHT_MODEL,
                "resolution": f"{original_image.width}x{original_image.height}",
                "note": "深度マップが正常に生成されました"
            },
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"Depth estimation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    finally:
        # Clean up depth estimator
        if depth_estimator:
            depth_estimator.clear_cache()
            del depth_estimator

@router.options("/generate-3d")
async def generate_3d_options():
    """Handle preflight requests"""
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*"
        }
    )

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
    depth_estimator = None
    try:
        if not validate_image(file):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        image_data = await file.read()
        
        # Create estimator per request for memory efficiency
        depth_estimator = DepthEstimator()
        
        depth_map, original_image = await depth_estimator.predict(
            image_data,
            model_name=model_name or settings.LIGHTWEIGHT_MODEL
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
    finally:
        # Clean up depth estimator
        if depth_estimator:
            depth_estimator.clear_cache()
            del depth_estimator

@router.get("/models")
async def get_available_models():
    """
    Get list of available depth estimation models
    """
    return {
        "models": settings.AVAILABLE_MODELS,
        "default": settings.DEFAULT_DEPTH_MODEL
    }

@router.post("/test")
async def test_upload(file: UploadFile = File(...)):
    """
    Test file upload without processing (for debugging)
    """
    try:
        if not validate_image(file):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        image_data = await file.read()
        file_size = len(image_data)
        
        return JSONResponse({
            "success": True,
            "message": "File upload successful",
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": file_size,
            "size_kb": round(file_size / 1024, 2)
        })
        
    except Exception as e:
        logger.error(f"Test upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload test failed: {str(e)}")