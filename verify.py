import os
import sys

print("Verifying Face Identifier Library Installation")
print("=" * 60)

# Check required files
required_files = [
    "face_identifier/__init__.py",
    "face_identifier/detector.py", 
    "face_identifier/encoder.py",
    "face_identifier/matcher.py",
    "face_identifier/utils.py",
    "face_identifier/exceptions.py",
    "setup.py",
    "LICENSE",
    "README.md"
]

print("\n1. Checking required files...")
missing_files = []
for file in required_files:
    if os.path.exists(file):
        print(f"  ✓ {file}")
    else:
        print(f"  ✗ {file} (MISSING)")
        missing_files.append(file)

if missing_files:
    print(f"\n⚠ Missing {len(missing_files)} file(s)")
    for file in missing_files:
        print(f"  - {file}")
else:
    print(f"\n✓ All required files present")

# Test imports
print("\n2. Testing imports...")
try:
    from face_identifier import FaceDetector, FaceEncoder, FaceMatcher
    print("  ✓ Main imports successful")
    
    # Test creating instances
    detector = FaceDetector()
    encoder = FaceEncoder() 
    matcher = FaceMatcher()
    print("  ✓ Component instances created")
    
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# Check dependencies
print("\n3. Checking dependencies...")
try:
    import numpy as np
    print(f"  ✓ NumPy {np.__version__}")
except ImportError:
    print("  ✗ NumPy not installed")

try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
except ImportError:
    print("  ✗ PyTorch not installed")

try:
    from PIL import Image
    print("  ✓ PIL/Pillow installed")
except ImportError:
    print("  ✗ PIL/Pillow not installed")

try:
    import facenet_pytorch
    print("  ✓ facenet-pytorch installed")
except ImportError:
    print("  ✗ facenet-pytorch not installed")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)

if missing_files:
    print("\nACTION REQUIRED:")
    print("Create missing files with the code provided earlier.")
else:
    print("\n✅ Library is properly set up!")
    print("\nNext steps:")
    print("1. Run: python test_library.py")
    print("2. Run: python quick_start.py")
    print("3. Integrate with your Django project")