"""
Experiment 1: Baseline Diffusion Profile
========================================

Objective: Characterize heat kernel behavior on SHA-256 state space.

Protocol:
1. Generate random inputs
2. Collect SHA-256 trajectories
3. For each sampled round, build Laplacian and analyze
4. Compare against null distribution
5. Track structure decay across rounds

Success Criteria:
- If p_value < 0.01 for any round, we have evidence of structure
- If structure persists past round 32, cryptographically interesting
- If structure is present at round 0 but gone by round 16, SHA-256 working as designed

Author: Bee Davis
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..sha256.core import InstrumentedSHA256
from ..sha256.input_generator import InputGenerator
from ..embeddings import get_embedding, BaseEmbedding
from ..analysis.laplacian import LaplacianConstructor
from ..analysis.heat_kernel import HeatKernelSolver, HeatKernelResult
from ..analysis.anomaly import AnomalyDetector, AnomalyReport, NullDistribution


@dataclass
class BaselineConfig:
    """Configuration for baseline diffusion experiment."""
    n_samples: int = 10000              # Number of random inputs
    sample_rounds: List[int] = None     # Which rounds to analyze
    embedding: str = 'euclidean'        # 'euclidean', 'hyperbolic', 'davis'
    k_neighbors: int = 50               # k for k-NN graph
    n_eigenvalues: int = 200            # Number of eigenvalues to compute
    null_samples: int = 100             # Samples for null distribution
    seed: Optional[int] = None          # Random seed
    
    def __post_init__(self):
        if self.sample_rounds is None:
            self.sample_rounds = [0, 8, 16, 24, 32, 40, 48, 56, 64]


@dataclass
class BaselineResult:
    """Results from baseline diffusion experiment."""
    config: BaselineConfig
    heat_kernel_results: Dict[int, HeatKernelResult]
    anomaly_reports: Dict[int, AnomalyReport]
    decay_analysis: Dict[str, Any]
    null_distribution: NullDistribution
    
    # Summary
    structure_detected_rounds: List[int]
    max_significance: float  # Highest confidence of structure
    structure_persistence: int  # Last round with structure


def experiment_1_baseline(config: Dict) -> Dict:
    """
    Run baseline diffusion profile experiment.
    
    Args:
        config: Dictionary with configuration (converted to BaselineConfig)
    
    Returns:
        Dictionary with all results
    """
    # Parse config
    cfg = BaselineConfig(
        n_samples=config.get('n_samples', 10000),
        sample_rounds=config.get('sample_rounds', None),
        embedding=config.get('embedding', 'euclidean'),
        k_neighbors=config.get('k_neighbors', 50),
        n_eigenvalues=config.get('n_eigenvalues', 200),
        null_samples=config.get('null_samples', 100),
        seed=config.get('seed', None)
    )
    
    print(f"Running Experiment 1: Baseline Diffusion Profile")
    print(f"  Samples: {cfg.n_samples}")
    print(f"  Embedding: {cfg.embedding}")
    print(f"  Rounds: {cfg.sample_rounds}")
    
    # 1. Generate inputs
    gen = InputGenerator(seed=cfg.seed)
    inputs = gen.random_batch(cfg.n_samples)
    print(f"  Generated {len(inputs)} random inputs")
    
    # 2. Collect trajectories
    sha = InstrumentedSHA256(sample_rounds=cfg.sample_rounds)
    trajectories = sha.hash_batch(inputs)
    print(f"  Collected {len(trajectories)} trajectories")
    
    # 3. Setup analysis components
    embedding = get_embedding(cfg.embedding)
    constructor = LaplacianConstructor(embedding, cfg.k_neighbors)
    solver = HeatKernelSolver(cfg.n_eigenvalues)
    detector = AnomalyDetector(null_samples=cfg.null_samples)
    
    # 4. Build null distribution (do once for efficiency)
    print("  Building null distribution...")
    null_dist = detector.build_null_distribution(
        n_points=cfg.n_samples,
        embedding=embedding,
        k_neighbors=cfg.k_neighbors
    )
    print(f"    Generated {null_dist.n_samples} null samples")
    
    # 5. Analyze each round
    heat_kernel_results = {}
    anomaly_reports = {}
    
    for round_num in cfg.sample_rounds:
        print(f"  Analyzing round {round_num}...")
        
        try:
            # Build Laplacian
            lap_result = constructor.build_from_trajectories(trajectories, round_num)
            
            # Solve heat kernel
            hk_result = solver.solve(lap_result.laplacian)
            heat_kernel_results[round_num] = hk_result
            
            # Compare to null
            report = detector.analyze(hk_result, null_dist)
            anomaly_reports[round_num] = report
            
            print(f"    Spectral gap: {hk_result.spectral_gap:.4f}")
            print(f"    Structure detected: {report.structure_detected} (p={report.eigenvalue_p_value:.4f})")
            
        except Exception as e:
            print(f"    Error at round {round_num}: {e}")
            continue
    
    # 6. Decay analysis
    decay_analysis = detector.compare_rounds(heat_kernel_results)
    
    # 7. Summary
    structure_rounds = [
        r for r, report in anomaly_reports.items() 
        if report.structure_detected
    ]
    
    max_significance = max(
        (report.confidence for report in anomaly_reports.values()),
        default=0.0
    )
    
    structure_persistence = max(structure_rounds) if structure_rounds else -1
    
    print(f"\n=== Experiment 1 Results ===")
    print(f"  Structure detected at rounds: {structure_rounds}")
    print(f"  Maximum significance: {max_significance:.4f}")
    print(f"  Structure persists until round: {structure_persistence}")
    
    return {
        'config': cfg.__dict__,
        'heat_kernel_results': heat_kernel_results,
        'anomaly_reports': {r: report.__dict__ for r, report in anomaly_reports.items()},
        'decay_analysis': decay_analysis,
        'structure_detected_rounds': structure_rounds,
        'max_significance': max_significance,
        'structure_persistence': structure_persistence
    }


def run_baseline_quick(
    n_samples: int = 1000,
    embedding: str = 'euclidean',
    rounds: List[int] = None
) -> Dict:
    """
    Quick baseline run for testing.
    
    Args:
        n_samples: Number of samples (default 1000)
        embedding: Which embedding
        rounds: Which rounds (default [0, 32, 64])
    
    Returns:
        Results dictionary
    """
    if rounds is None:
        rounds = [0, 32, 64]
    
    config = {
        'n_samples': n_samples,
        'sample_rounds': rounds,
        'embedding': embedding,
        'k_neighbors': 30,
        'n_eigenvalues': 50,
        'null_samples': 20,
        'seed': 42
    }
    
    return experiment_1_baseline(config)


if __name__ == "__main__":
    # Quick test
    results = run_baseline_quick(n_samples=500, rounds=[0, 16, 32, 48, 64])
    print("\nDone!")
