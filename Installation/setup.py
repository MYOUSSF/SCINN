"""Setup script for SCINN package."""

from setuptools import setup, find_packages
import os

# Read README from Documentation folder
readme_path = os.path.join(os.path.dirname(__file__), '..', 'Documentation', 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "SCINN: A lightweight Physics-Informed Neural Networks framework in PyTorch"

# Change to parent directory to find packages
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

setup(
    name="scinn",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A lightweight Physics-Informed Neural Networks framework in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/scinn",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "matplotlib>=3.3.0",
            "scipy>=1.6.0",
        ],
        "sampling": [
            "scikit-optimize>=0.9.0",
        ],
    },
)
