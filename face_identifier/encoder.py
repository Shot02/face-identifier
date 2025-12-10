"""
Face encoding module using FaceNet.
"""

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1


class FaceEncoder:
    """Encode faces into embeddings using FaceNet."""
    
    def __init__(self, device='cpu'):
        """
        Initialize the face encoder.
        
        Args:
            device (str): 'cpu' or 'cuda'
        """
        self.device = device
        self.resnet = InceptionResnetV1(
            pretrained='vggface2',
            classify=False,
            device=device
        ).eval()
    
    def encode(self, face_image):
        """
        Encode a face image into a 512-dimensional embedding.
        
        Args:
            face_image: PIL Image of a face
            
        Returns:
            np.ndarray: Normalized 512-dimensional face embedding
        """
        try:
            # Resize to FaceNet input size
            face_resized = face_image.resize((160, 160))
            
            # Convert to tensor and normalize
            face_tensor = torch.tensor(np.array(face_resized))
            face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
            face_tensor = face_tensor.to(self.device)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.resnet(face_tensor).cpu().numpy().flatten()
            
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            raise ValueError(f"Face encoding error: {str(e)}")
    
    def encode_detected_faces(self, image, detected_faces):
        """
        Encode multiple detected faces.
        
        Args:
            image: Original PIL Image
            detected_faces: List of face information from detector
            
        Returns:
            list: List of dictionaries with face info and encoding
        """
        encoded_faces = []
        
        for face_info in detected_faces:
            try:
                face_image = face_info['face_image']
                encoding = self.encode(face_image)
                
                face_data = {
                    **face_info,
                    'encoding': encoding
                }
                encoded_faces.append(face_data)
                
            except Exception as e:
                continue
        
        return encoded_faces
    
    def generate_hash(self, encoding, hash_bits=64):
        """
        Generate binary hash from face encoding.
        
        Args:
            encoding (np.ndarray): Face encoding
            hash_bits (int): Number of bits in hash
            
        Returns:
            tuple: (binary_hash, hash_bucket)
        """
        if encoding is None:
            return None, None
        
        # Use median value as threshold for binary conversion
        if len(encoding) < hash_bits:
            padded_encoding = np.pad(
                encoding, 
                (0, max(0, hash_bits - len(encoding)))
            )
        else:
            padded_encoding = encoding[:hash_bits]
        
        median_val = np.median(padded_encoding)
        binary_hash = ''.join(['1' if x >= median_val else '0' for x in padded_encoding])
        
        # First 8 bits as bucket for potential indexing
        hash_bucket = binary_hash[:8]
        
        return binary_hash, hash_bucket