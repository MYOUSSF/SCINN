"""
SCINN: A lightweight Physics-Informed Neural Networks framework in PyTorch.

This package provides a clean API for defining and solving PDEs using PINNs.

Main components:
    - geometry: Geometric domains and operations
    - nn: Neural network architectures and gradient utilities
    - icbc: Initial and boundary conditions
    - Data: Problem setup container
    - Solver: PINN training solver
"""

__version__ = "0.1.0"

from . import geometry
from . import nn
from . import icbc
from .data import Data
from .solver import Solver

__all__ = [
    "geometry",
    "nn", 
    "icbc",
    "Data",
    "Solver",
]
