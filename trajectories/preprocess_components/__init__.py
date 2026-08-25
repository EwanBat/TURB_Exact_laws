"""
Preprocessing components for satellite trajectories.

Package split for easier reading and editing:
  - trajectories.py : trajectory path definitions and parameter generators
  - geometry.py     : satellite offsets, tangents, arc length, extraction, combine
  - preprocessor.py : TrajectoryPreprocessor class, param_to_txt, setup_logging
"""
from trajectories.preprocess_components.preprocessor import (
    TrajectoryPreprocessor,
    param_to_txt,
    setup_logging,
)

__all__ = ["TrajectoryPreprocessor", "param_to_txt", "setup_logging"]