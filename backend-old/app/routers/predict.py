from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import tempfile
import os
import logging
from PIL import Image
import io
import base64

from ..models.depth_model_v2 import DepthEstimatorV2
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/")
async def predict(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default="Intel/dpt-large")
):
    """
    Main prediction endpoint used by frontend
    Compatible with /api/predict
    """
    try:
        logger.info(f"=== Predict API called with model: {model} ===")
        
        # Validate image
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Read image data
        image_data = await file.read()
        logger.info(f"Received image: {file.filename}, size: {len(image_data)} bytes")
        
        # Create depth estimator
        depth_estimator = DepthEstimatorV2()
        
        try:
            # Process with specified model
            depth_map, original_image = await depth_estimator.predict(
                image_data, 
                model_name=model
            )
            
            logger.info(f"Successfully processed with model: {model}")
            
        except Exception as pred_error:
            logger.error(f"Prediction error: {str(pred_error)}")
            raise HTTPException(status_code=500, detail=f"Depth estimation failed: {str(pred_error)}")
        
        # Save temporary files
        os.makedirs(settings.TEMP_DIR, exist_ok=True)
        
        # Save depth map
        temp_depth_path = os.path.join(settings.TEMP_DIR, f"depth_{os.urandom(8).hex()}.png")
        depth_map.save(temp_depth_path)
        temp_depth_name = os.path.basename(temp_depth_path)
        
        # Save original
        temp_orig_path = os.path.join(settings.TEMP_DIR, f"orig_{os.urandom(8).hex()}.png")
        original_image.save(temp_orig_path)
        temp_orig_name = os.path.basename(temp_orig_path)
        
        # Get model info
        model_config = depth_estimator.model_configs.get(model, {})
        model_type = model_config.get("type", "unknown")
        
        # Generate point cloud data
        try:
            # Convert depth map to numpy array
            import numpy as np
            depth_array = np.array(depth_map.convert('L'))
            depth_normalized = depth_array / 255.0
            
            # Create point cloud data
            height, width = depth_array.shape
            points = []
            colors = []
            
            # Sample points (reduce density for performance)
            step = 4
            for y in range(0, height, step):
                for x in range(0, width, step):
                    z = depth_normalized[y, x]
                    # Normalize coordinates
                    nx = (x / width - 0.5) * 2
                    ny = (y / height - 0.5) * 2
                    nz = z * 2 - 1
                    
                    points.append([nx, ny, nz])
                    
                    # Get color from original image
                    if original_image.mode == 'RGB':
                        r, g, b = original_image.getpixel((x, y))
                        colors.append([r/255, g/255, b/255])
                    else:
                        gray = original_image.getpixel((x, y))
                        colors.append([gray/255, gray/255, gray/255])
            
            pointcloud_data = {
                "points": points,
                "colors": colors
            }
        except Exception as e:
            logger.error(f"Point cloud generation error: {e}")
            pointcloud_data = None
        
        # Prepare response
        response_data = {
            "success": True,
            "depthMapUrl": f"/temp/{temp_depth_name}",
            "originalUrl": f"/temp/{temp_orig_name}",
            "model": model,
            "modelType": model_type,
            "resolution": f"{original_image.width}x{original_image.height}",
            "note": f"Processed with {model}",
            "algorithms": model_config.get("algorithms", []),
            "implementation": model_config.get("implementation", ""),
            "features": model_config.get("features", []),
            "pointcloudData": pointcloud_data
        }
        
        logger.info(f"=== Predict API response: success, model_type: {model_type} ===")
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Predict API error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")