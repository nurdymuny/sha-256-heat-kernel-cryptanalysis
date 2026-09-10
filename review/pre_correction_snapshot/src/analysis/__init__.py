"""
Geometric Analysis Module
=========================

Tools for analyzing the geometry of SHA-256 state space:
- LaplacianConstructor: Build graph Laplacians from embedded states
- HeatKernelSolver: Solve heat equation and compute spectral invariants
- AnomalyDetector: Compare against null hypothesis

Based on theory from heat kernel.tex and manifold.tex.
"""

from .laplacian import LaplacianConstructor
from .heat_kernel import HeatKernelSolver, HeatKernelResult
from .anomaly import AnomalyDetector, AnomalyReport

__all__ = [
    'LaplacianConstructor',
    'HeatKernelSolver',
    'HeatKernelResult',
    'AnomalyDetector',
    'AnomalyReport',
]
