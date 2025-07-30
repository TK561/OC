from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from typing import Optional
import tempfile
import os
import logging
import shutil

from ..models.depth_model_v2 import DepthEstimatorV2
from ..utils.image_utils import validate_image
from ..utils.pointcloud import generate_pointcloud, save_pointcloud
from ..config import settings

# Create necessary directories
os.makedirs(settings.MODEL_CACHE_DIR, exist_ok=True)
os.makedirs(settings.TEMP_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

router = APIRouter()

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
    model_name: Optional[str] = Query(None, description="Model to use for depth estimation"),
    resolution: Optional[int] = Query(None, description="Target resolution for processing")
):
    """
    Estimate depth from uploaded image using enhanced models
    
    Supported models:
    - Intel/dpt-large: DPT-Large model
    - Intel/dpt-hybrid-midas: MiDaS v3.1 model
    - depth-anything/Depth-Anything-V2-Small-hf: Depth Anything V2 Small
    - depth-anything/Depth-Anything-V2-Base-hf: Depth Anything V2 Base
    - depth-anything/Depth-Anything-V2-Large-hf: Depth Anything V2 Large
    """
    depth_estimator = None
    try:
        # Validate image
        if not validate_image(file):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        
        # Read image data
        image_data = await file.read()
        
        # Validate model name
        if model_name and model_name not in settings.AVAILABLE_MODELS:
            logger.warning(f"Requested model {model_name} not available, using default")
            model_name = settings.DEFAULT_DEPTH_MODEL
        
        # Create enhanced depth estimator
        depth_estimator = DepthEstimatorV2()
        
        try:
            # Process image
            depth_map, original_image = await depth_estimator.predict(
                image_data, 
                model_name=model_name or settings.DEFAULT_DEPTH_MODEL,
                target_resolution=resolution
            )
            
            # Log successful processing
            logger.info(f"Successfully processed image with model: {model_name or settings.DEFAULT_DEPTH_MODEL}")
            
        except Exception as pred_error:
            logger.error(f"Prediction error: {str(pred_error)}")
            return JSONResponse(
                status_code=503,
                content={
                    "success": False,
                    "error": "Service temporarily unavailable",
                    "detail": str(pred_error),
                    "suggestion": "Try using a smaller model or reducing resolution"
                },
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "*"
                }
            )
        
        # Save temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_depth:
            depth_map.save(tmp_depth.name)
            depth_path = tmp_depth.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_orig:
            original_image.save(tmp_orig.name)
            orig_path = tmp_orig.name
        
        # Move files to temp directory
        temp_depth_name = f"depth_{os.path.basename(depth_path)}"
        temp_orig_name = f"orig_{os.path.basename(orig_path)}"
        
        temp_depth_final = os.path.join(settings.TEMP_DIR, temp_depth_name)
        temp_orig_final = os.path.join(settings.TEMP_DIR, temp_orig_name)
        
        shutil.move(depth_path, temp_depth_final)
        shutil.move(orig_path, temp_orig_final)
        
        return JSONResponse(
            content={
                "success": True,
                "depthMapUrl": f"/temp/{temp_depth_name}",
                "originalUrl": f"/temp/{temp_orig_name}",
                "modelUsed": model_name or settings.DEFAULT_DEPTH_MODEL,
                "resolution": f"{original_image.width}x{original_image.height}",
                "processingNote": "Enhanced depth estimation with GitHub-based implementation",
                "depthRange": {
                    "convention": "white=near, black=far",
                    "type": "relative depth"
                }
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
        # Clean up
        if depth_estimator:
            del depth_estimator

@router.post("/generate-3d")
async def generate_3d(
    file: UploadFile = File(...),
    model_name: Optional[str] = Query(None, description="Model to use"),
    point_density: Optional[float] = Query(1.0, description="Point cloud density"),
    export_format: Optional[str] = Query("ply", description="Export format (ply, obj)")
):
    """
    Generate 3D point cloud from image using enhanced depth estimation
    """
    depth_estimator = None
    try:
        if not validate_image(file):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        image_data = await file.read()
        
        # Create enhanced depth estimator
        depth_estimator = DepthEstimatorV2()
        
        # Get depth array and original image
        depth_array, original_image = depth_estimator.get_depth_array(
            image_data,
            model_name=model_name or settings.DEFAULT_DEPTH_MODEL
        )
        
        # Generate point cloud
        pointcloud = generate_pointcloud(
            original_image,
            depth_array,
            density=point_density
        )
        
        # Save point cloud
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{export_format}') as tmp_3d:
            save_pointcloud(pointcloud, tmp_3d.name, format=export_format)
            cloud_path = tmp_3d.name
        
        return FileResponse(
            cloud_path,
            media_type='application/octet-stream',
            filename=f"pointcloud.{export_format}",
            headers={
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        logger.error(f"3D generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"3D generation failed: {str(e)}")
    finally:
        if depth_estimator:
            del depth_estimator

@router.get("/models/info")
async def get_model_info():
    """
    Get detailed information about available models
    """
    model_info = {
        "Intel/dpt-large": {
            "name": "DPT-Large",
            "description": "High accuracy depth estimation from Intel",
            "architecture": "Dense Prediction Transformer",
            "input_size": 384,
            "features": ["High precision", "Good for detailed scenes", "Vision Transformer backbone"]
        },
        "Intel/dpt-hybrid-midas": {
            "name": "MiDaS v3.1",
            "description": "Fast and robust depth estimation",
            "architecture": "Hybrid CNN-Transformer",
            "input_size": 384,
            "features": ["Fast processing", "Good balance", "Multi-dataset training"]
        },
        "depth-anything/Depth-Anything-V2-Small-hf": {
            "name": "Depth Anything V2 Small",
            "description": "Lightweight model for real-time processing",
            "architecture": "Vision Transformer (ViT-S)",
            "input_size": 518,
            "features": ["Very fast", "Low memory", "Mobile-friendly"]
        },
        "depth-anything/Depth-Anything-V2-Base-hf": {
            "name": "Depth Anything V2 Base",
            "description": "Balanced model for general use",
            "architecture": "Vision Transformer (ViT-B)",
            "input_size": 518,
            "features": ["Better accuracy", "Moderate speed", "Good generalization"]
        },
        "depth-anything/Depth-Anything-V2-Large-hf": {
            "name": "Depth Anything V2 Large",
            "description": "Highest quality depth estimation",
            "architecture": "Vision Transformer (ViT-L)",
            "input_size": 518,
            "features": ["Best accuracy", "State-of-the-art", "NeurIPS 2024"]
        }
    }
    
    return {
        "models": model_info,
        "default": settings.DEFAULT_DEPTH_MODEL,
        "implementation": "Enhanced GitHub-based implementation",
        "depth_convention": "white=near, black=far"
    }