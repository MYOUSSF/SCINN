"""Geometry module for defining computational domains."""

from .geometry import Geometry, Interval, Rectangle, Disk, Circle, Ellipse, TimeDomain
from .ops import Union, Intersection, Difference, CrossProduct
from .sampler import sample

__all__ = [
    "Geometry",
    "Interval",
    "Rectangle", 
    "Disk",
    "Circle",
    "Ellipse",
    "TimeDomain",
    "Union",
    "Intersection", 
    "Difference",
    "CrossProduct",
    "sample"
]
