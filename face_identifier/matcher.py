"""
Face matching and comparison module.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class FaceMatcher:
    """Match and compare face embeddings."""
    
    @staticmethod
    def cosine_similarity(encoding1: np.ndarray, encoding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            encoding1: First face encoding
            encoding2: Second face encoding
            
        Returns:
            float: Cosine similarity between 0 and 1
        """
        try:
            if encoding1 is None or encoding2 is None:
                return 0.0
                
            # Cosine similarity
            dot_product = np.dot(encoding1, encoding2)
            norm1 = np.linalg.norm(encoding1)
            norm2 = np.linalg.norm(encoding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            return 0.0
    
    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> int:
        """
        Calculate Hamming distance between two binary hashes.
        
        Args:
            hash1: First binary hash
            hash2: Second binary hash
            
        Returns:
            int: Hamming distance
        """
        if len(hash1) != len(hash2):
            # Use the shorter length
            min_len = min(len(hash1), len(hash2))
            hash1 = hash1[:min_len]
            hash2 = hash2[:min_len]
        
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def match(
        self,
        new_embedding: np.ndarray,
        stored_embeddings: List[np.ndarray],
        threshold: float = 0.6
    ) -> Dict:
        """
        Match a new embedding against stored embeddings.
        
        Args:
            new_embedding: New face embedding to match
            stored_embeddings: List of stored embeddings to compare against
            threshold: Minimum similarity threshold for a match
            
        Returns:
            dict: Match results containing:
                - 'match_found': bool
                - 'best_match_index': int or None
                - 'similarity': float (0-1)
                - 'above_threshold': bool
        """
        if not stored_embeddings:
            return {
                'match_found': False,
                'best_match_index': None,
                'similarity': 0.0,
                'above_threshold': False
            }
        
        best_similarity = 0.0
        best_index = -1
        
        # Find best match
        for i, stored_embedding in enumerate(stored_embeddings):
            similarity = self.cosine_similarity(new_embedding, stored_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = i
        
        above_threshold = best_similarity >= threshold
        
        return {
            'match_found': above_threshold,
            'best_match_index': best_index if above_threshold else None,
            'similarity': best_similarity,
            'above_threshold': above_threshold
        }
    
    def match_multiple(
        self,
        new_embeddings: List[np.ndarray],
        stored_embeddings: List[np.ndarray],
        threshold: float = 0.6
    ) -> List[Dict]:
        """
        Match multiple new embeddings against stored embeddings.
        
        Args:
            new_embeddings: List of new face embeddings
            stored_embeddings: List of stored embeddings
            threshold: Minimum similarity threshold
            
        Returns:
            list: List of match results for each new embedding
        """
        results = []
        for embedding in new_embeddings:
            result = self.match(embedding, stored_embeddings, threshold)
            results.append(result)
        return results
    
    def find_duplicate(
        self,
        new_embedding: np.ndarray,
        stored_embeddings: List[np.ndarray],
        duplicate_threshold: float = 0.75
    ) -> Optional[int]:
        """
        Check if new embedding is a duplicate of any stored embedding.
        
        Args:
            new_embedding: New face embedding
            stored_embeddings: List of stored embeddings
            duplicate_threshold: Threshold for duplicate detection
            
        Returns:
            int or None: Index of duplicate or None if no duplicate found
        """
        for i, stored_embedding in enumerate(stored_embeddings):
            similarity = self.cosine_similarity(new_embedding, stored_embedding)
            if similarity >= duplicate_threshold:
                return i
        return None