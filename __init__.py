__all__ = [
    "data",
    "geometry",
    "grad",
    "icbc",
    "nn",
    "utils",
    "Model",
    "Variable",
    "zcs",
]


from . import data
from . import geometry
from . import gradients as grad
from . import icbc


# Backward compatibility
from .icbc import (
    DirichletBC,
    Interface2DBC,
    NeumannBC,
    OperatorBC,
    PeriodicBC,
    RobinBC,
    PointSetBC,
    PointSetOperatorBC,
    IC,
)

maps = nn