"""
Experiment Protocols
====================

Implementation of the three main experiments from the spec:
1. Baseline Diffusion Profile
2. Avalanche Geometry  
3. Davis Manifold Comparison
"""

from .baseline import experiment_1_baseline
from .avalanche import experiment_2_avalanche
from .comparison import experiment_3_davis_comparison

__all__ = [
    'experiment_1_baseline',
    'experiment_2_avalanche',
    'experiment_3_davis_comparison',
]
