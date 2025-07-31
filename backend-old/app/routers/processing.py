from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import tempfile
import os
import logging

from ..utils.image_utils import (
    validate_image, 
    apply_edge_detection,
    apply_blur,
    apply_color_correction
)
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/edge-detection")
async def edge_detection(
    file: UploadFile = File(...),
    method: Optional[str] = "canny",
    threshold1: Optional[int] = 100,
    threshold2: Optional[int] = 200
):
    """
    Apply edge detection to uploaded image
    """
    try:
        if not validate_image(file):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        image_data = await file.read()
        
        processed_image = apply_edge_detection(
            image_data,
            method=method,
            threshold1=threshold1,
            threshold2=threshold2
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            processed_image.save(tmp_file.name)
            result_path = tmp_file.name
        
        return JSONResponse({
            "success": True,
            "processed_url": f"/temp/{os.path.basename(result_path)}",
            "method": method,
            "parameters": {
                "threshold1": threshold1,
                "threshold2": threshold2
            }
        })
        
    except Exception as e:
        logger.error(f"Edge detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/blur")
async def blur_image(
    file: UploadFile = File(...),
    method: Optional[str] = "gaussian",
    radius: Optional[float] = 2.0
):
    """
    Apply blur effect to uploaded image
    """
    try:
        if not validate_image(file):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        image_data = await file.read()
        
        processed_image = apply_blur(
            image_data,
            method=method,
            radius=radius
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            processed_image.save(tmp_file.name)
            result_path = tmp_file.name
        
        return JSONResponse({
            "success": True,
            "processed_url": f"/temp/{os.path.basename(result_path)}",
            "method": method,
            "parameters": {
                "radius": radius
            }
        })
        
    except Exception as e:
        logger.error(f"Blur processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/color-correction")
async def color_correction(
    file: UploadFile = File(...),
    brightness: Optional[float] = 1.0,
    contrast: Optional[float] = 1.0,
    saturation: Optional[float] = 1.0
):
    """
    Apply color correction to uploaded image
    """
    try:
        if not validate_image(file):
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        image_data = await file.read()
        
        processed_image = apply_color_correction(
            image_data,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation
        )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            processed_image.save(tmp_file.name)
            result_path = tmp_file.name
        
        return JSONResponse({
            "success": True,
            "processed_url": f"/temp/{os.path.basename(result_path)}",
            "parameters": {
                "brightness": brightness,
                "contrast": contrast,
                "saturation": saturation
            }
        })
        
    except Exception as e:
        logger.error(f"Color correction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")