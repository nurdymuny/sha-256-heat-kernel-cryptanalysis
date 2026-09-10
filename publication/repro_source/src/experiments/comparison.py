"""
Experiment 3: Davis Manifold Comparison
=======================================

Objective: Test if Davis manifold reveals structure invisible to standard embeddings.

Protocol:
1. Run experiments 1 and 2 with all three embeddings (Euclidean, Hyperbolic, Davis)
2. Compare results across embeddings
3. Identify where Davis sees structure that others don't

Success Criteria:
- If Davis shows statistically significant structure where Euclidean doesn't,
  the manifold construction is revealing hidden geometry
- If all three agree (no structure), SHA-256 is robust to geometric analysis
- If hyperbolic shows structure but Euclidean doesn't, there's tree-like structure

Author: Bee Davis
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .baseline import experiment_1_baseline
from .avalanche import experiment_2_avalanche


@dataclass
class ComparisonConfig:
    """Configuration for comparison experiment."""
    n_samples: int = 10000
    n_pairs: int = 5000
    sample_rounds: List[int] = None
    k_neighbors: int = 50
    n_eigenvalues: int = 200
    null_samples: int = 100
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.sample_rounds is None:
            self.sample_rounds = [0, 16, 32, 48, 64]


def compute_davis_advantage(comparison: Dict) -> Dict:
    """
    Compute the advantage of Davis manifold over other embeddings.
    
    A positive advantage means Davis detected more structure.
    """
    advantages = {}
    
    if 'euclidean' in comparison and 'davis' in comparison:
        euc = comparison['euclidean']
        dav = comparison['davis']
        
        # Structure detection advantage
        euc_detected = len(euc.get('structure_detected_rounds', []))
        dav_detected = len(dav.get('structure_detected_rounds', []))
        advantages['structure_detection'] = dav_detected - euc_detected
        
        # Significance advantage
        advantages['significance'] = (
            dav.get('max_significance', 0) - 
            euc.get('max_significance', 0)
        )
        
        # Persistence advantage
        advantages['persistence'] = (
            dav.get('structure_persistence', 0) -
            euc.get('structure_persistence', 0)
        )
    
    if 'hyperbolic' in comparison and 'davis' in comparison:
        hyp = comparison['hyperbolic']
        dav = comparison['davis']
        
        advantages['vs_hyperbolic_detection'] = (
            len(dav.get('structure_detected_rounds', [])) -
            len(hyp.get('structure_detected_rounds', []))
        )
    
    # Overall assessment
    total_advantage = sum(advantages.values())
    advantages['total'] = total_advantage
    
    if total_advantage > 0:
        advantages['assessment'] = 'davis_superior'
    elif total_advantage < 0:
        advantages['assessment'] = 'davis_inferior'
    else:
        advantages['assessment'] = 'equivalent'
    
    return advantages


def compare_embedding_sensitivity(
    baseline_results: Dict[str, Dict],
    avalanche_results: Dict[str, Dict]
) -> Dict:
    """
    Compare sensitivity across embeddings.
    
    Returns analysis of which embedding is most sensitive to structure.
    """
    comparison = {}
    
    for emb in baseline_results.keys():
        baseline = baseline_results[emb]
        avalanche = avalanche_results.get(emb, {})
        
        comparison[emb] = {
            # From baseline
            'structure_detected_rounds': baseline.get('structure_detected_rounds', []),
            'max_significance': baseline.get('max_significance', 0),
            'structure_persistence': baseline.get('structure_persistence', -1),
            
            # From avalanche
            'anisotropy_score': avalanche.get('anisotropy_score', 0),
            'slow_bits_count': len(avalanche.get('slow_bits', [])),
            'divergence_half_life': avalanche.get('divergence_half_life', 0),
        }
    
    # Determine most sensitive embedding
    if comparison:
        by_significance = sorted(
            comparison.items(),
            key=lambda x: x[1]['max_significance'],
            reverse=True
        )
        comparison['most_sensitive'] = by_significance[0][0]
        
        by_detection = sorted(
            comparison.items(),
            key=lambda x: len(x[1]['structure_detected_rounds']),
            reverse=True
        )
        comparison['most_structure'] = by_detection[0][0]
    
    return comparison


def experiment_3_davis_comparison(config: Dict) -> Dict:
    """
    Run comparison experiment across all embeddings.
    
    Args:
        config: Dictionary with configuration
    
    Returns:
        Dictionary with comparison results
    """
    # Parse config
    cfg = ComparisonConfig(
        n_samples=config.get('n_samples', 10000),
        n_pairs=config.get('n_pairs', 5000),
        sample_rounds=config.get('sample_rounds', None),
        k_neighbors=config.get('k_neighbors', 50),
        n_eigenvalues=config.get('n_eigenvalues', 200),
        null_samples=config.get('null_samples', 100),
        seed=config.get('seed', None)
    )
    
    print(f"Running Experiment 3: Davis Manifold Comparison")
    print(f"  Samples: {cfg.n_samples}")
    print(f"  Pairs: {cfg.n_pairs}")
    print(f"  Rounds: {cfg.sample_rounds}")
    
    embeddings = ['euclidean', 'hyperbolic', 'davis']
    
    baseline_results = {}
    avalanche_results = {}
    
    for emb in embeddings:
        print(f"\n--- Embedding: {emb} ---")
        
        # Baseline experiment
        baseline_config = {
            'n_samples': cfg.n_samples,
            'sample_rounds': cfg.sample_rounds,
            'embedding': emb,
            'k_neighbors': cfg.k_neighbors,
            'n_eigenvalues': cfg.n_eigenvalues,
            'null_samples': cfg.null_samples,
            'seed': cfg.seed
        }
        
        try:
            baseline_results[emb] = experiment_1_baseline(baseline_config)
        except Exception as e:
            print(f"  Baseline failed for {emb}: {e}")
            baseline_results[emb] = {'error': str(e)}
        
        # Avalanche experiment
        avalanche_config = {
            'n_pairs': cfg.n_pairs,
            'sample_rounds': cfg.sample_rounds,
            'embedding': emb,
            'seed': cfg.seed
        }
        
        try:
            avalanche_results[emb] = experiment_2_avalanche(avalanche_config)
        except Exception as e:
            print(f"  Avalanche failed for {emb}: {e}")
            avalanche_results[emb] = {'error': str(e)}
    
    # Compare results
    comparison = compare_embedding_sensitivity(baseline_results, avalanche_results)
    davis_advantage = compute_davis_advantage(baseline_results)
    
    print(f"\n=== Experiment 3 Results ===")
    print(f"  Most sensitive embedding: {comparison.get('most_sensitive', 'unknown')}")
    print(f"  Most structure detected: {comparison.get('most_structure', 'unknown')}")
    print(f"  Davis advantage score: {davis_advantage.get('total', 0)}")
    print(f"  Assessment: {davis_advantage.get('assessment', 'unknown')}")
    
    # Summary
    for emb in embeddings:
        if emb in comparison:
            c = comparison[emb]
            print(f"\n  {emb.upper()}:")
            print(f"    Structure rounds: {c['structure_detected_rounds']}")
            print(f"    Max significance: {c['max_significance']:.4f}")
            print(f"    Anisotropy: {c['anisotropy_score']:.4f}")
    
    return {
        'config': cfg.__dict__,
        'baseline_by_embedding': baseline_results,
        'avalanche_by_embedding': avalanche_results,
        'comparison': comparison,
        'davis_advantage': davis_advantage
    }


def run_comparison_quick(
    n_samples: int = 500,
    n_pairs: int = 200
) -> Dict:
    """Quick comparison run for testing."""
    config = {
        'n_samples': n_samples,
        'n_pairs': n_pairs,
        'sample_rounds': [0, 32, 64],
        'k_neighbors': 20,
        'n_eigenvalues': 30,
        'null_samples': 10,
        'seed': 42
    }
    return experiment_3_davis_comparison(config)


if __name__ == "__main__":
    results = run_comparison_quick(n_samples=300, n_pairs=100)
    print("\nDone!")
