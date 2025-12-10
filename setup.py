from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="face-identifier",
    version="1.0.0",
    author="Otto Shalom",
    author_email="shalomottodamilare@gmail.com",
    description="A clean, reusable face detection and recognition library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Shot02/face-identifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pillow>=8.0.0",
        "torch>=1.7.0",
        "facenet-pytorch>=2.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    license="MIT",
)
