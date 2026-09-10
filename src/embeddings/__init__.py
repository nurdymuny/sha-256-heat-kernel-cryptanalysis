"""
Manifold Embeddings for SHA-256 State Space
============================================

This module provides embeddings from discrete 256-bit SHA-256 states
to continuous manifolds for geometric analysis.

Embeddings:
    - EuclideanEmbedding: Baseline ℝ²⁵⁶ (null-structure control)
    - HyperbolicEmbedding: Poincaré ball H²⁵⁶ (tree-like structure)
    - DavisManifoldEmbedding: Geometry-first with distortion bounds
"""

from .base import BaseEmbedding
from .euclidean import EuclideanEmbedding
from .hyperbolic import HyperbolicEmbedding
from .davis import DavisManifoldEmbedding

__all__ = [
    'BaseEmbedding',
    'EuclideanEmbedding',
    'HyperbolicEmbedding',
    'DavisManifoldEmbedding',
]


def get_embedding(name: str, **kwargs) -> BaseEmbedding:
    """
    Factory function to get embedding by name.
    
    Args:
        name: 'euclidean', 'hyperbolic', or 'davis'
        **kwargs: Additional arguments for specific embeddings
    
    Returns:
        BaseEmbedding instance
    """
    embeddings = {
        'euclidean': EuclideanEmbedding,
        'hyperbolic': HyperbolicEmbedding,
        'davis': DavisManifoldEmbedding,
    }
    
    if name.lower() not in embeddings:
        raise ValueError(f"Unknown embedding: {name}. Available: {list(embeddings.keys())}")
    
    return embeddings[name.lower()](**kwargs)
