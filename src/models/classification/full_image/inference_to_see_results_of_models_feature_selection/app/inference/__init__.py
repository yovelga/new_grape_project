"""Inference module

Provides probability map generation and inference utilities.
"""

from .prob_map import (
    build_prob_map,
    ProbabilityMapGenerator,
)

# Re-export PreprocessConfig from centralized location for convenience
from ..config.types import PreprocessConfig

__all__ = [
    "build_prob_map",
    "PreprocessConfig",
    "ProbabilityMapGenerator",
]
