"""
SHA-256 Heat Kernel Geometric Analysis
======================================

This package implements spectral geometry analysis of SHA-256's internal 
state space using heat kernel methods and the Davis manifold framework.

Modules:
    sha256: Instrumented SHA-256 implementation with trajectory capture
    embeddings: Manifold embeddings (Euclidean, Hyperbolic, Davis)
    analysis: Laplacian construction, heat kernel solving, anomaly detection
    experiments: Experiment protocols from spec
    utils: Shared utilities

Author: Bee Davis
"""

from . import sha256
from . import embeddings
from . import analysis
from . import experiments
from . import utils

__version__ = "0.1.0"
__author__ = "Bee Davis"
