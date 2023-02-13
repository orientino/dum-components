from typing import Literal
from ._base import NormalizingFlow
from .maf import MaskedAutoregressiveFlow
from .radial import RadialFlow
from .residual import ResidualFlow
from .neural_spline import NeuralSplineFlow

FlowType = Literal["radial", "maf", "residual"]
"""
A reference to a flow type that can be used with :class:`NaturalPosteriorNetwork`:

- `radial`: A :class:`~natpn.nn.flow.RadialFlow`.
- `maf`: A :class:`~natpn.nn.flow.MaskedAutoregressiveFlow`.
"""

__all__ = [
    "ResidualFlow", 
    "NeuralSplineFlow", 
    "MaskedAutoregressiveFlow",
    "NormalizingFlow", 
    "RadialFlow",
    "FlowType",
]
