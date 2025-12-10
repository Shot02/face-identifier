"""
Utility functions for face processing.
"""

import base64
import numpy as np
from io import BytesIO
from PIL import Image
from typing import Tuple, Optional


def base64_to_image(image_data: str) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Convert base64 image data to PIL Image.
    
    Args:
        image_data: Base64 encoded image data (may include data URL prefix)
        
    Returns:
        tuple: (PIL Image, error_message)
    """
    try:
        # Handle data URL format
        if ',' in image_data:
            header, data = image_data.split(',', 1)
        else:
            data = image_data
        
        # Decode base64
        img_bytes = base64.b64decode(data)
        image = Image.open(BytesIO(img_bytes)).convert('RGB')
        return image, None
        
    except Exception as e:
        return None, f"Image conversion error: {str(e)}"


def image_to_base64(image: Image.Image, format: str = 'JPEG') -> str:
    """
    Convert PIL Image to base64 string.
    
    Args:
        image: PIL Image
        format: Output format (JPEG, PNG, etc.)
        
    Returns:
        str: Base64 encoded image data
    """
    buffer = BytesIO()
    image.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def crop_face_with_padding(
    image: Image.Image,
    box: list,
    padding_percent: float = 0.2
) -> Image.Image:
    """
    Crop face from image with padding.
    
    Args:
        image: Original PIL Image
        box: [x1, y1, x2, y2] coordinates
        padding_percent: Padding percentage relative to face size
        
    Returns:
        Image: Cropped face image with padding
    """
    x1, y1, x2, y2 = box
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Calculate padding
    pad_x = int(face_width * padding_percent)
    pad_y = int(face_height * padding_percent)
    
    # Apply padding with bounds checking
    x1_padded = max(0, x1 - pad_x)
    y1_padded = max(0, y1 - pad_y)
    x2_padded = min(image.width, x2 + pad_x)
    y2_padded = min(image.height, y2 + pad_y)
    
    return image.crop((x1_padded, y1_padded, x2_padded, y2_padded))


def resize_image(image: Image.Image, max_size: Tuple[int, int] = (800, 600)) -> Image.Image:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: PIL Image
        max_size: Maximum (width, height)
        
    Returns:
        Image: Resized image
    """
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image