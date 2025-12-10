"""
Face Identifier - Quick Start Guide
A complete example showing how to use the library.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pickle
import os

# Import the library
from face_identifier import FaceDetector, FaceEncoder, FaceMatcher
from face_identifier.utils import base64_to_image, crop_face_with_padding

print("=" * 60)
print("FACE IDENTIFIER - QUICK START")
print("=" * 60)


# ============================================================================
# EXAMPLE 1: Basic Usage
# ============================================================================
print("\n1. BASIC USAGE")
print("-" * 40)

# Initialize the components
detector = FaceDetector()
encoder = FaceEncoder()
matcher = FaceMatcher()

print("✓ Components initialized")

# Create some dummy stored embeddings for testing
print("\nCreating sample data...")
stored_embeddings = []
stored_names = []

# Create 3 sample embeddings
for i in range(3):
    embedding = np.random.randn(512)
    embedding = embedding / np.linalg.norm(embedding)  # Normalize
    stored_embeddings.append(embedding)
    stored_names.append(f"Person {i+1}")

print(f"Created {len(stored_embeddings)} sample embeddings")

# ============================================================================
# EXAMPLE 2: Face Detection
# ============================================================================
print("\n\n2. FACE DETECTION")
print("-" * 40)

# Try to load a real image, or create a dummy one
image_path = "test_face.jpg"
if os.path.exists(image_path):
    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
else:
    print("No test image found. Creating a dummy image...")
    # Create a dummy image with colored squares
    image = Image.new('RGB', (800, 600), color='lightblue')
    draw = ImageDraw.Draw(image)
    # Draw some face-like shapes
    draw.ellipse([200, 150, 400, 350], fill='beige', outline='black', width=2)
    draw.ellipse([250, 200, 280, 230], fill='blue')  # left eye
    draw.ellipse([320, 200, 350, 230], fill='blue')  # right eye
    draw.arc([280, 250, 340, 290], 0, 180, fill='red', width=3)  # mouth
    
print(f"Image size: {image.size}")

# Detect faces
print("\nDetecting faces...")
faces = detector.detect(image)

if faces:
    print(f"✓ Found {len(faces)} face(s)")
    for i, face in enumerate(faces):
        print(f"\n  Face {i+1}:")
        print(f"    Box: {face['box']}")
        print(f"    Confidence: {face['confidence']:.3f}")
        
        # Save face crop for display
        face['face_image'].save(f"face_{i+1}.jpg", "JPEG")
        print(f"    Face saved as: face_{i+1}.jpg")
else:
    print("✗ No faces detected")
    # Create a dummy face for demonstration
    print("Using dummy data for demonstration...")
    dummy_face = {
        'face_image': image.crop((0, 0, 100, 100)),
        'box': [50, 50, 150, 150],
        'confidence': 0.95
    }
    faces = [dummy_face]

# ============================================================================
# EXAMPLE 3: Face Encoding
# ============================================================================
print("\n\n3. FACE ENCODING")
print("-" * 40)

if faces:
    face = faces[0]
    print("Encoding first detected face...")
    
    try:
        # Encode the face
        embedding = encoder.encode(face['face_image'])
        print(f"✓ Encoding successful")
        print(f"  Shape: {embedding.shape}")
        print(f"  Norm: {np.linalg.norm(embedding):.6f}")
        
        # Generate hash
        binary_hash, hash_bucket = encoder.generate_hash(embedding, hash_bits=64)
        print(f"  Binary hash (first 32 bits): {binary_hash[:32]}...")
        print(f"  Hash bucket: {hash_bucket}")
        
    except Exception as e:
        print(f"✗ Encoding failed: {e}")
        # Create dummy embedding for testing
        embedding = np.random.randn(512)
        embedding = embedding / np.linalg.norm(embedding)
        print("Using dummy embedding for demonstration")

# ============================================================================
# EXAMPLE 4: Face Matching
# ============================================================================
print("\n\n4. FACE MATCHING")
print("-" * 40)

print("Matching against stored embeddings...")
if 'embedding' in locals():
    # Match with different thresholds
    thresholds = [0.5, 0.6, 0.7]
    
    for threshold in thresholds:
        result = matcher.match(
            new_embedding=embedding,
            stored_embeddings=stored_embeddings,
            threshold=threshold
        )
        
        print(f"\nThreshold: {threshold}")
        print(f"  Match found: {result['match_found']}")
        print(f"  Similarity: {result['similarity']:.3f}")
        if result['match_found']:
            print(f"  Matched with: {stored_names[result['best_match_index']]}")
        else:
            print(f"  Best similarity: {result['similarity']:.3f}")

# ============================================================================
# EXAMPLE 5: Real-world Usage Pattern
# ============================================================================
print("\n\n5. REAL-WORLD USAGE PATTERN")
print("-" * 40)

class SimpleFaceRecognitionSystem:
    """Example of how to use the library in a real application"""
    
    def __init__(self):
        self.detector = FaceDetector()
        self.encoder = FaceEncoder()
        self.matcher = FaceMatcher()
        
        # In a real app, you'd store these in a database
        self.embeddings_db = []  # List of embeddings
        self.users_db = []       # List of user info
        
    def register_user(self, name, image_path):
        """Register a new user"""
        print(f"\nRegistering user: {name}")
        
        # Load image
        image = Image.open(image_path) if os.path.exists(image_path) else Image.new('RGB', (800, 600), 'gray')
        
        # Detect face
        faces = self.detector.detect(image)
        if not faces:
            return {"success": False, "error": "No face detected"}
        
        # Encode face
        embedding = self.encoder.encode(faces[0]['face_image'])
        
        # Check for duplicates
        duplicate_idx = self.matcher.find_duplicate(
            embedding, 
            self.embeddings_db,
            duplicate_threshold=0.75
        )
        
        if duplicate_idx is not None:
            return {"success": False, "error": f"Similar face already registered"}
        
        # Store
        self.embeddings_db.append(embedding)
        self.users_db.append({
            "name": name,
            "id": len(self.users_db) + 1,
            "registration_date": "2024-01-01"
        })
        
        return {
            "success": True,
            "user_id": len(self.users_db),
            "face_box": faces[0]['box'],
            "confidence": faces[0]['confidence']
        }
    
    def recognize(self, image_path):
        """Recognize faces in an image"""
        print(f"\nRecognizing faces in: {image_path}")
        
        # Load image
        image = Image.open(image_path) if os.path.exists(image_path) else Image.new('RGB', (800, 600), 'gray')
        
        # Detect all faces
        faces = self.detector.detect(image)
        
        results = []
        for face in faces:
            # Encode face
            embedding = self.encoder.encode(face['face_image'])
            
            # Match
            result = self.matcher.match(
                embedding,
                self.embeddings_db,
                threshold=0.6
            )
            
            if result['match_found']:
                user_info = self.users_db[result['best_match_index']]
                status = "known"
            else:
                user_info = {"name": "Unknown", "id": None}
                status = "unknown"
            
            results.append({
                "box": face['box'],
                "confidence": face['confidence'],
                "similarity": result['similarity'],
                "status": status,
                "user": user_info
            })
        
        return results

# Demonstrate the system
print("Creating a simple face recognition system...")
system = SimpleFaceRecognitionSystem()

# Simulate registering users
print("\nSimulating user registration:")
for i in range(2):
    result = system.register_user(f"Test User {i+1}", f"user_{i+1}.jpg")
    print(f"  User {i+1}: {result['success'] if 'success' in result else 'Failed'}")

print(f"\nTotal registered users: {len(system.users_db)}")

# ============================================================================
# EXAMPLE 6: Working with Base64 Images (like from web)
# ============================================================================
print("\n\n6. WORKING WITH BASE64 IMAGES")
print("-" * 40)

# Create a base64 image from our current image
from face_identifier.utils import image_to_base64
import base64

# Convert image to base64
base64_data = image_to_base64(image, format='JPEG')
print(f"Image converted to base64")
print(f"Base64 length: {len(base64_data)} characters")
print(f"First 50 chars: {base64_data[:50]}...")

# Convert back to image
converted_image, error = base64_to_image(f"data:image/jpeg;base64,{base64_data}")
if error:
    print(f"Error: {error}")
else:
    print(f"✓ Base64 conversion successful")
    print(f"  Converted image size: {converted_image.size}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 60)
print("QUICK START COMPLETE!")
print("=" * 60)
print("\nWhat you can do now:")
print("1. Take a photo with your phone/computer")
print("2. Save it as 'test_face.jpg' in this folder")
print("3. Run this script again to see real face detection")
print("\nLibrary components available:")
print("  - FaceDetector(): Detect faces in images")
print("  - FaceEncoder(): Convert faces to embeddings")
print("  - FaceMatcher(): Compare and match embeddings")
print("\nRemember: This library doesn't store data!")
print("You need to handle storage in your application.")
print("=" * 60)

# Save the sample data for later use
print("\nSaving sample data to 'sample_data.pkl'...")
with open('sample_data.pkl', 'wb') as f:
    pickle.dump({
        'embeddings': stored_embeddings,
        'names': stored_names
    }, f)
print("✓ Sample data saved")