"""
Anomaly Detector
================

Compare observed spectral signatures against null hypothesis.

Null hypothesis (H0): SHA-256 state evolution is geometrically equivalent
to a random function on the hypercube {0,1}^256.

Calibration uses independent random-bit reference graphs processed identically.
No iid eigenmode distribution or constant heat diagonal is assumed.

If we reject H0, SHA-256 has detectable geometric structure.

Author: Bee Davis
"""

import numpy as np
from scipy import stats
from scipy import sparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .heat_kernel import HeatKernelResult, HeatKernelSolver
from .laplacian import LaplacianConstructor, LaplacianResult
from ..embeddings.base import BaseEmbedding
from ..embeddings.euclidean import EuclideanEmbedding


@dataclass
class AnomalyReport:
    """Results of anomaly detection analysis."""
    
    # Scalar metrics
    spectral_entropy: float           # Entropy of eigenvalue distribution
    spectral_gap_variance: float      # Variance in gaps (low = regular structure)
    diffusion_anisotropy: Optional[float]       # Max/min diffusion rate ratio
    
    # Statistical tests
    eigenvalue_ks_statistic: float    # Compatibility name: pooled Euclidean distance, not KS
    eigenvalue_p_value: float         # p-value for KS test
    gap_ks_statistic: float           # Compatibility name: gap distance, not KS
    gap_p_value: float
    
    # Structure detection
    structure_detected: bool          # True if p < significance_level
    significance_level: float         # α used for testing
    confidence: Optional[float]       # Deprecated: p-values are not confidence
    
    p_value: float = 1.0
    calibration: str = "exchangeable pooled-spectrum distance"

    # Detailed structure (if found)
    anomalous_modes: List[int] = field(default_factory=list)
    mode_descriptions: List[str] = field(default_factory=list)
    
    # Visualization data
    eigenvalue_histogram: np.ndarray = field(default_factory=lambda: np.array([]))
    null_histogram: np.ndarray = field(default_factory=lambda: np.array([]))
    spectral_gap_series: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class NullDistribution:
    """Null distribution statistics for comparison."""
    eigenvalue_mean: np.ndarray       # Mean eigenvalue at each index
    eigenvalue_std: np.ndarray        # Std at each index
    gap_mean: np.ndarray              # Mean spectral gap
    gap_std: np.ndarray
    heat_trace_mean: np.ndarray       # Mean heat trace at time scales
    heat_trace_std: np.ndarray
    n_samples: int                    # How many null samples
    time_scales: np.ndarray
    eigenvalue_samples: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    failed_samples: List[str] = field(default_factory=list)


class AnomalyDetector:
    """
    Detect geometric anomalies by comparing against null hypothesis.
    
    The null hypothesis is that SHA-256 behaves like a random function,
    producing states uniformly distributed on the hypercube.
    
    We generate null distributions by:
    1. Sampling random points uniformly on the embedding manifold
    2. Building k-NN Laplacian
    3. Computing eigenvalues
    4. Repeating to build distribution
    
    Then compare observed SHA-256 spectral signature to null.
    """
    
    def __init__(
        self, 
        null_samples: int = 100,
        significance_level: float = 0.01,
        n_eigenvalues: int = 100
    ):
        """
        Args:
            null_samples: Number of random graphs for null distribution
            significance_level: α for hypothesis testing
            n_eigenvalues: Number of eigenvalues to compute
        """
        self.null_samples = null_samples
        self.alpha = significance_level
        self.n_eigenvalues = n_eigenvalues
        self._null_cache: Dict[str, NullDistribution] = {}
    
    def build_null_distribution(
        self, 
        n_points: int, 
        embedding: BaseEmbedding,
        k_neighbors: int,
        time_scales: Optional[np.ndarray] = None
    ) -> NullDistribution:
        """
        Generate null distribution by sampling random points.
        
        Args:
            n_points: Number of points per sample
            embedding: Embedding to use
            k_neighbors: k for k-NN graph
            time_scales: Time scales for heat trace
        
        Returns:
            NullDistribution with statistics
        """
        if time_scales is None:
            time_scales = np.logspace(-2, 2, 20)
        
        solver = HeatKernelSolver(self.n_eigenvalues, time_scales)
        constructor = LaplacianConstructor(embedding, k_neighbors)
        
        all_eigenvalues = []
        all_gaps = []
        all_heat_traces = []
        failures = []
        
        for i in range(self.null_samples):
            # Generate random binary states (uniform on hypercube)
            random_states = np.random.randint(0, 2, size=(n_points, 256)).astype(np.uint8)
            
            # Embed
            points = embedding.embed_batch(random_states)
            
            # Build Laplacian
            try:
                lap_result = constructor.build_from_points(points)
                
                # Solve
                hk_result = solver.solve(lap_result.laplacian)
                
                all_eigenvalues.append(hk_result.eigenvalues)
                all_gaps.append(hk_result.spectral_gaps)
                all_heat_traces.append(hk_result.heat_trace_values)
            except Exception as e:
                failures.append(f"sample {i}: {type(e).__name__}: {e}")
                continue
        
        if len(all_eigenvalues) == 0:
            raise RuntimeError("All null samples failed")
        
        # Compute statistics
        eigenvalues = np.array(all_eigenvalues)
        gaps = np.array([g[:min(len(g), self.n_eigenvalues-1)] for g in all_gaps])
        heat_traces = np.array(all_heat_traces)
        
        # Pad gaps to same length
        max_gap_len = max(len(g) for g in gaps)
        padded_gaps = np.zeros((len(gaps), max_gap_len))
        for i, g in enumerate(gaps):
            padded_gaps[i, :len(g)] = g
        
        return NullDistribution(
            eigenvalue_mean=np.mean(eigenvalues, axis=0),
            eigenvalue_std=np.std(eigenvalues, axis=0),
            gap_mean=np.mean(padded_gaps, axis=0),
            gap_std=np.std(padded_gaps, axis=0),
            heat_trace_mean=np.mean(heat_traces, axis=0),
            heat_trace_std=np.std(heat_traces, axis=0),
            n_samples=len(all_eigenvalues),
            time_scales=time_scales,
            eigenvalue_samples=eigenvalues,
            failed_samples=failures
        )
    
    def analyze(
        self,
        observed: HeatKernelResult,
        null_dist: NullDistribution
    ) -> AnomalyReport:
        """
        Compare observed spectral signature to null distribution.
        
        Args:
            observed: HeatKernelResult from SHA-256 data
            null_dist: NullDistribution from random data
        
        Returns:
            AnomalyReport with statistical analysis
        """
        if null_dist.failed_samples:
            raise ValueError("Reference simulation had failures; repair before inference")
        refs = np.asarray(null_dist.eigenvalue_samples, dtype=float)
        obs = np.asarray(observed.eigenvalues, dtype=float)
        if refs.ndim != 2 or len(refs) < 2 or refs.shape[1] != len(obs):
            raise ValueError("Inference requires complete, matched independent reference spectra")
        # The pooled center is permutation symmetric in observed + references.
        # Under exchangeability, rank of this predeclared distance is valid.
        pool = np.vstack([obs, refs])
        distances = np.linalg.norm(pool-pool.mean(axis=0), axis=1)
        tolerance = 1e-12 * max(1.0, float(distances.max()))
        p_value = float(np.mean(distances >= distances[0]-tolerance))
        gap_pool = np.diff(pool,axis=1)
        gap_distances = np.linalg.norm(gap_pool-gap_pool.mean(axis=0),axis=1)
        gap_p = float(np.mean(gap_distances >= gap_distances[0]-tolerance))
        # Gap p is descriptive; do not combine dependent tests with Fisher.
        return AnomalyReport(
            spectral_entropy=self._eigenvalue_entropy(obs),
            spectral_gap_variance=float(np.var(observed.spectral_gaps)),
            diffusion_anisotropy=None,
            eigenvalue_ks_statistic=float(distances[0]),
            eigenvalue_p_value=p_value,
            gap_ks_statistic=float(gap_distances[0]),
            gap_p_value=gap_p,
            structure_detected=p_value < self.alpha,
            significance_level=self.alpha,
            confidence=None,
            p_value=p_value,
            eigenvalue_histogram=obs,
            null_histogram=refs.mean(axis=0),
            spectral_gap_series=observed.spectral_gaps,
        )

    def compare_rounds(
        self,
        results_by_round: Dict[int, HeatKernelResult]
    ) -> Dict:
        """
        Track how spectral signature evolves across rounds.
        
        Key question: does structure decay monotonically, or persist?
        
        Args:
            results_by_round: HeatKernelResult for each round
        
        Returns:
            Analysis of round-to-round evolution
        """
        rounds = sorted(results_by_round.keys())
        
        gaps = []
        entropies = []
        effective_dims_t1 = []  # at t=1
        
        for r in rounds:
            result = results_by_round[r]
            gaps.append(result.spectral_gap)
            entropies.append(self._eigenvalue_entropy(result.eigenvalues))
            effective_dims_t1.append(result.effective_dimension(1.0))
        
        gaps = np.array(gaps)
        entropies = np.array(entropies)
        effective_dims = np.array(effective_dims_t1)
        
        # Compute derivatives
        gap_velocity = np.gradient(gaps)
        entropy_velocity = np.gradient(entropies)
        
        # Find inflection points (where structure changes character)
        inflection_points = []
        for i in range(1, len(gap_velocity) - 1):
            if gap_velocity[i-1] * gap_velocity[i+1] < 0:  # Sign change
                inflection_points.append(rounds[i])
        
        return {
            'rounds': rounds,
            'spectral_gaps': gaps,
            'entropies': entropies,
            'effective_dimensions': effective_dims,
            'gap_velocity': gap_velocity,
            'entropy_velocity': entropy_velocity,
            'inflection_points': inflection_points,
            'structure_decay_rate': float(np.polyfit(rounds, gaps, 1)[0]) if len(rounds) > 1 else 0.0
        }
    
    def compare_embeddings(
        self,
        results_by_embedding: Dict[str, HeatKernelResult]
    ) -> Dict:
        """
        Compare results across different embeddings.
        
        If Davis manifold shows structure that Euclidean doesn't,
        that's the key finding.
        
        Args:
            results_by_embedding: Results keyed by embedding name
        
        Returns:
            Comparison analysis
        """
        comparison = {}
        
        for name, result in results_by_embedding.items():
            comparison[name] = {
                'spectral_gap': result.spectral_gap,
                'entropy': self._eigenvalue_entropy(result.eigenvalues),
                'effective_dim_t1': result.effective_dimension(1.0),
                'n_large_gaps': int(np.sum(result.spectral_gaps > 0.1))
            }
        
        # Compute relative advantages
        if 'euclidean' in comparison and 'davis' in comparison:
            comparison['davis_gap_advantage'] = (
                comparison['davis']['spectral_gap'] - 
                comparison['euclidean']['spectral_gap']
            )
            comparison['davis_entropy_advantage'] = (
                comparison['davis']['entropy'] -
                comparison['euclidean']['entropy']
            )
        
        return comparison
    
    def _eigenvalue_entropy(self, eigenvalues: np.ndarray) -> float:
        """Compute entropy of normalized eigenvalue distribution."""
        pos = eigenvalues[eigenvalues > 1e-10]
        if len(pos) == 0:
            return 0.0
        p = pos / pos.sum()
        return -float(np.sum(p * np.log(p + 1e-10)))
    
    def quick_test(
        self,
        observed: HeatKernelResult,
        n_points: int,
        k_neighbors: int = 50
    ) -> Tuple[bool, float]:
        """
        Quick anomaly check without full null distribution.
        
        Uses heuristic thresholds based on random graph theory.
        
        Args:
            observed: Observed spectral result
            n_points: Number of points in graph
            k_neighbors: k used for graph
        
        Returns:
            (is_anomalous, confidence)
        """
        # Expected spectral gap for random k-regular graph: ~k/n
        expected_gap = k_neighbors / n_points
        observed_gap = observed.spectral_gap
        
        # Anomaly if gap is significantly different
        gap_ratio = observed_gap / (expected_gap + 1e-10)
        
        # Expected entropy for uniform eigenvalue distribution
        n_eig = len(observed.eigenvalues)
        expected_entropy = np.log(n_eig)
        observed_entropy = self._eigenvalue_entropy(observed.eigenvalues)
        entropy_ratio = observed_entropy / (expected_entropy + 1e-10)
        
        # Heuristic: anomalous if ratios are far from 1
        anomaly_score = np.abs(gap_ratio - 1) + np.abs(entropy_ratio - 1)
        
        is_anomalous = anomaly_score > 0.5  # Threshold
        confidence = min(anomaly_score / 2, 1.0)
        
        return is_anomalous, confidence
