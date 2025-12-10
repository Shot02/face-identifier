"""
Face Identifier - A clean, reusable face detection and recognition library.
"""

from .detector import FaceDetector
from .encoder import FaceEncoder
from .matcher import FaceMatcher

__version__ = "1.0.0"
__all__ = ["FaceDetector", "FaceEncoder", "FaceMatcher"]