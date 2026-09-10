"""
Base Embedding Interface
========================

Abstract base class defining the embedding interface.
All embeddings must implement: embed, distance, local_metric.

Author: Bee Davis
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass


@dataclass
class DistortionBounds:
    """
    Bounds on geodesic-Euclidean distortion for a path family.
    
    From Davis manifold theory:
    - ε(L) is the maximum distortion for paths of length ≤ L
    - For well-regularized embeddings: ε(L) ≤ K(λ) * L
    """
    path_length: float           # Maximum path length L
    distortion_bound: float      # ε(L)
    regularization_strength: float  # λ controlling smoothness
    
    def is_valid_regime(self, actual_path_length: float) -> bool:
        """Check if path length is within bounded-distortion regime."""
        return actual_path_length <= self.path_length


class BaseEmbedding(ABC):
    """
    Abstract base class for manifold embeddings.
    
    Embeddings map discrete 256-bit SHA-256 states to continuous manifolds
    where geometric analysis (heat kernels, curvature, etc.) can be performed.
    
    Key properties:
    - embed(): Map discrete state to manifold coordinates
    - distance(): Compute geodesic distance between points
    - local_metric(): Return metric tensor at a point
    """
    
    @property
    def dimension(self) -> int:
        """Manifold dimension (typically 256 for SHA-256 states)."""
        return 256
    
    @abstractmethod
    def embed(self, state: np.ndarray) -> np.ndarray:
        """
        Map 256-bit state to manifold coordinates.
        
        Args:
            state: shape (256,) uint8 array of bits, or (8,) uint32 array
        
        Returns:
            shape (d,) float64 array of manifold coordinates
        """
        pass
    
    @abstractmethod
    def distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Compute geodesic distance between two embedded points.
        
        Args:
            p1: First point on manifold
            p2: Second point on manifold
        
        Returns:
            Geodesic distance d(p1, p2)
        """
        pass
    
    @abstractmethod
    def local_metric(self, p: np.ndarray) -> np.ndarray:
        """
        Return metric tensor at point p.
        
        Args:
            p: Point on manifold
        
        Returns:
            shape (d, d) metric tensor g_ij
        """
        pass
    
    def embed_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Embed multiple states.
        
        Args:
            states: shape (n, 256) or (n, 8) array of states
        
        Returns:
            shape (n, d) array of embedded points
        """
        return np.array([self.embed(s) for s in states])
    
    def distance_matrix(self, points: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance matrix.
        
        Args:
            points: shape (n, d) array of embedded points
        
        Returns:
            shape (n, n) distance matrix
        """
        n = points.shape[0]
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self.distance(points[i], points[j])
                D[i, j] = d
                D[j, i] = d
        return D
    
    def distortion_bounds(self, path_length: float) -> Optional[DistortionBounds]:
        """
        Get distortion bounds for paths up to given length.
        
        Returns None if bounds are not available (e.g., for Euclidean).
        Override in subclasses that provide bounded distortion.
        """
        return None
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state to consistent format (256-bit vector).
        
        Handles both (256,) bit vectors and (8,) uint32 working variables.
        """
        if state.shape == (8,):
            # Expand uint32 working variables to bits
            bits = np.zeros(256, dtype=np.uint8)
            for i, var in enumerate(state):
                var_int = int(var) & 0xFFFFFFFF
                for bit_pos in range(32):
                    bits[i * 32 + bit_pos] = (var_int >> (31 - bit_pos)) & 1
            return bits
        elif state.shape == (256,):
            return state.astype(np.uint8)
        else:
            raise ValueError(f"Unexpected state shape: {state.shape}")
