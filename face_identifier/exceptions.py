"""
Custom exceptions for the face identifier library.
"""


class FaceIdentifierError(Exception):
    """Base exception for face identifier errors."""
    pass


class FaceDetectionError(FaceIdentifierError):
    """Exception raised when face detection fails."""
    pass


class FaceEncodingError(FaceIdentifierError):
    """Exception raised when face encoding fails."""
    pass


class ImageProcessingError(FaceIdentifierError):
    """Exception raised when image processing fails."""
    pass


class NoFaceDetectedError(FaceDetectionError):
    """Exception raised when no face is detected in an image."""
    pass


class LowConfidenceError(FaceDetectionError):
    """Exception raised when face detection confidence is too low."""
    pass