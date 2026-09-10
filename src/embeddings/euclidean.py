"""
Euclidean Embedding
===================

Baseline embedding: treat 256-bit states as coordinates in ℝ²⁵⁶.
Distance = Euclidean (equivalently, sqrt of Hamming distance on bits).

This is a chosen representation. Compare identical constructions on independent
reference datasets before interpreting observed structure.

Author: Bee Davis
"""

import numpy as np
from .base import BaseEmbedding, DistortionBounds


class EuclideanEmbedding(BaseEmbedding):
    """
    Euclidean embedding: bits as coordinates in ℝ²⁵⁶.
    
    Properties:
    - Metric is flat (zero curvature everywhere)
    - Distance is Euclidean, equivalent to sqrt(Hamming) for binary states
    - No distortion - Euclidean distance IS the intrinsic metric
    - Serves as null hypothesis baseline
    """
    
    def __init__(self, scale: float = 1.0):
        """
        Args:
            scale: Scaling factor for coordinates (default 1.0)
        """
        self.scale = scale
    
    @property
    def dimension(self) -> int:
        return 256
    
    def embed(self, state: np.ndarray) -> np.ndarray:
        """
        Embed state as coordinates in ℝ²⁵⁶.
        
        Args:
            state: shape (256,) bit vector or (8,) uint32 working vars
        
        Returns:
            shape (256,) float64 coordinates
        """
        normalized = self.normalize_state(state)
        return normalized.astype(np.float64) * self.scale
    
    def distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Euclidean distance between points.
        
        For binary-valued coordinates, this equals sqrt(Hamming distance).
        """
        return np.linalg.norm(p1 - p2)
    
    def local_metric(self, p: np.ndarray) -> np.ndarray:
        """
        Metric tensor is identity (flat space).
        """
        return np.eye(256) * (self.scale ** 2)
    
    def hamming_distance(self, p1: np.ndarray, p2: np.ndarray) -> int:
        """
        Hamming distance (number of differing bits).
        
        For unscaled binary states, Euclidean² = Hamming.
        """
        return int(np.sum(np.abs(p1 - p2) > 0.5))
    
    def distance_matrix(self, points: np.ndarray) -> np.ndarray:
        """
        Efficient pairwise Euclidean distances using broadcasting.
        """
        # ||a - b||² = ||a||² + ||b||² - 2 a·b
        sq_norms = np.sum(points ** 2, axis=1)
        D_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * (points @ points.T)
        D_sq = np.maximum(D_sq, 0)  # Numerical stability
        return np.sqrt(D_sq)
