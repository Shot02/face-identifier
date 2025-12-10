# Simple test of the library
from face_identifier import FaceDetector, FaceEncoder, FaceMatcher
import numpy as np

print("Simple Library Test")
print("=" * 50)

# 1. Create instances
detector = FaceDetector()
encoder = FaceEncoder()
matcher = FaceMatcher()

print("✓ Library imported and initialized")

# 2. Create dummy data
dummy_embedding = np.random.randn(512)
dummy_embedding = dummy_embedding / np.linalg.norm(dummy_embedding)

stored_embeddings = [dummy_embedding]

# 3. Test matching
result = matcher.match(dummy_embedding, stored_embeddings, threshold=0.5)

print(f"\nMatch test:")
print(f"  Similarity: {result['similarity']:.3f}")
print(f"  Match found: {result['match_found']}")

print("\n✓ Library is working!")