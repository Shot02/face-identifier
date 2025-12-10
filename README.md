# Face Identifier

A clean, reusable Python library for face detection and recognition, extracted from a Django project into a standalone package.

## Features

- **Face Detection**: Uses MTCNN for accurate face detection
- **Face Encoding**: Generates 512-dimensional embeddings using FaceNet
- **Face Matching**: Cosine similarity-based matching with configurable thresholds
- **No Dependencies**: No Django, no database, no web framework dependencies
- **Clean API**: Simple, intuitive interface

## Installation

```bash
pip install face-identifier

git clone https://github.com/yourusername/face-identifier.git
cd face-identifier
pip install -e .