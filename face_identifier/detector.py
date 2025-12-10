"""
Face detection module using MTCNN.
"""

import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN


class FaceDetector:
    """Detect faces in images using MTCNN."""
    
    def __init__(self, device='cpu', min_confidence=0.9):
        """
        Initialize the face detector.
        
        Args:
            device (str): 'cpu' or 'cuda'
            min_confidence (float): Minimum confidence for face detection
        """
        self.device = device
        self.min_confidence = min_confidence
        self.mtcnn = MTCNN(keep_all=True, device=device)
        
    def detect(self, image):
        """
        Detect faces in an image.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            list: List of dictionaries containing face information:
                - 'box': [x1, y1, x2, y2]
                - 'confidence': detection confidence
                - 'landmarks': facial landmarks
                - 'face_image': cropped face as PIL Image
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = Image.fromarray(image)
                else:
                    raise ValueError("Input must be RGB image")
            
            # Detect faces
            boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
            
            if boxes is None:
                return []
            
            # Filter by confidence and extract face information
            faces = []
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                if prob >= self.min_confidence:
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    
                    # Ensure coordinates are within image bounds
                    if (x2 > x1 and y2 > y1 and 
                        x1 >= 0 and y1 >= 0 and 
                        x2 <= image.width and y2 <= image.height):
                        
                        # Crop face
                        face_crop = image.crop((x1, y1, x2, y2))
                        
                        face_info = {
                            'box': [x1, y1, x2, y2],
                            'confidence': float(prob),
                            'landmarks': landmarks[i].tolist() if landmarks is not None else None,
                            'face_image': face_crop,
                            'face_id': len(faces)
                        }
                        faces.append(face_info)
            
            return faces
            
        except Exception as e:
            raise ValueError(f"Face detection error: {str(e)}")
    
    def detect_single(self, image):
        """
        Detect a single face in an image.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            dict or None: Face information or None if no face detected
        """
        faces = self.detect(image)
        return faces[0] if faces else None