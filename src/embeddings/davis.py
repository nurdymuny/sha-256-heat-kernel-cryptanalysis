"""
Davis Manifold Embedding
========================

Geometry-first embedding with compositional error budgets and 
explicit distortion bounds, based on the Davis manifold framework.

Key properties from theory/manifold.tex:
- Path classes P(L) describing identity-preserving trajectories
- Bounded geodesic-Euclidean distortion ε(L) along paths
- Soft/hard margins (κ_hard, κ_soft) for abstention
- Contrastive + smoothness regularization yields explicit bounds

For SHA-256 analysis:
- "Identity" = same input message
- "Path" = trajectory through rounds
- Distortion bounds quantify how well Euclidean approximates geodesic
- Curvature-aware distance captures semantic structure

Author: Bee Davis
"""

import numpy as np
from typing import Optional, Dict, Tuple, Callable
from dataclasses import dataclass
from .base import BaseEmbedding, DistortionBounds


@dataclass
class DavisManifoldConfig:
    """Configuration for Davis manifold embedding."""
    
    # Regularization
    smoothness_lambda: float = 0.1       # Jacobian regularization strength
    curvature_penalty: float = 0.01      # Curvature regularization
    
    # Path horizons
    max_path_length: float = 10.0        # L_max for distortion bounds
    operational_horizon: float = 5.0     # L_star for typical operations
    
    # Margins
    kappa_hard: float = 0.5              # Hard decision margin
    kappa_soft: float = 0.1              # Soft abstention margin
    
    # Metric learning parameters
    temperature: float = 0.1             # InfoNCE temperature
    dimension: int = 256                 # Embedding dimension
    hidden_dim: int = 512                # Hidden layer dimension
    
    # Distortion model: ε(L) ≤ K(λ) * L^α
    distortion_K: float = 0.05           # Proportionality constant
    distortion_alpha: float = 1.0        # Path length exponent


class DavisManifoldEmbedding(BaseEmbedding):
    """
    Davis manifold embedding with explicit distortion bounds.
    
    From the Davis manifold framework:
    - Geodesic distance d_g reflects semantic change along paths
    - Euclidean computations approximate d_g within bounded-distortion regime
    - Configuration margins enable principled abstention
    
    For SHA-256 analysis, we construct a metric where:
    - Similar round-to-round transitions have small distance
    - Avalanche effect is captured by increasing distance
    - Structure (if present) appears as geometric regularity
    """
    
    def __init__(
        self, 
        config: Optional[DavisManifoldConfig] = None,
        learned_weights: Optional[Dict] = None
    ):
        """
        Args:
            config: Manifold configuration
            learned_weights: Pre-trained metric weights (if available)
        """
        self.config = config or DavisManifoldConfig()
        self.weights = learned_weights
        
        # Initialize metric components
        self._init_metric()
    
    def _init_metric(self):
        """Initialize the learned metric tensor components."""
        if self.weights is not None:
            # Use pre-trained weights
            self.W = self.weights.get('W', np.eye(256))
            self.b = self.weights.get('b', np.zeros(256))
        else:
            # Default: diagonal metric with learned scaling
            # In practice, these would be trained via contrastive learning
            self.W = np.eye(256) * 1.0
            self.b = np.zeros(256)
        
        # Precompute SVD for efficient distance computation
        self._update_metric_cache()
    
    def _update_metric_cache(self):
        """Cache metric decomposition for efficiency."""
        # For symmetric positive definite W:
        # d(x, y)² = (x-y)ᵀ W (x-y)
        # = ||L(x-y)||² where W = LᵀL
        try:
            self.L = np.linalg.cholesky(self.W)
        except np.linalg.LinAlgError:
            # If W is not positive definite, regularize
            eigvals = np.linalg.eigvalsh(self.W)
            min_eig = max(-eigvals.min() + 0.01, 0.01)
            self.W = self.W + min_eig * np.eye(256)
            self.L = np.linalg.cholesky(self.W)
    
    @property
    def dimension(self) -> int:
        return self.config.dimension
    
    def embed(self, state: np.ndarray) -> np.ndarray:
        """
        Embed state into Davis manifold coordinates.
        
        Applies learned transformation to map discrete state
        to continuous manifold with bounded distortion.
        """
        normalized = self.normalize_state(state)
        x = normalized.astype(np.float64)
        
        # Apply learned embedding transformation
        # In full implementation, this would be a neural network
        embedded = x + self.b
        
        # Smooth projection (maintains differentiability)
        # This ensures the embedding stays in a well-behaved region
        embedded = self._smooth_project(embedded)
        
        return embedded
    
    def _smooth_project(self, x: np.ndarray) -> np.ndarray:
        """
        Smooth projection to maintain bounded distortion.
        
        Uses soft clipping to keep points in regime of validity.
        """
        # Soft clipping using tanh-like function
        max_norm = 20.0  # Maximum allowed norm
        norm = np.linalg.norm(x)
        
        if norm > max_norm:
            # Smooth projection
            scale = max_norm * np.tanh(norm / max_norm) / (norm + 1e-10)
            x = x * scale
        
        return x
    
    def distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Geodesic distance on Davis manifold.
        
        Uses Mahalanobis distance with learned metric tensor.
        Approximates true geodesic within distortion bounds.
        """
        diff = p1 - p2
        
        # Mahalanobis distance: d² = diffᵀ W diff
        d_sq = diff @ self.W @ diff
        
        return np.sqrt(max(d_sq, 0))
    
    def local_metric(self, p: np.ndarray) -> np.ndarray:
        """
        Metric tensor at point p.
        
        In the current implementation, metric is constant (flat manifold
        with Mahalanobis structure). Full implementation would have
        position-dependent curvature.
        """
        # For position-dependent metric, would compute g(p)
        # Current: constant metric
        return self.W
    
    def distortion_bounds(self, path_length: float) -> DistortionBounds:
        """
        Get distortion bounds for paths of given length.
        
        From Davis manifold theory:
        ε(L) ≤ K(λ) * L^α
        
        where K(λ) depends on regularization strength.
        """
        K = self.config.distortion_K
        alpha = self.config.distortion_alpha
        lam = self.config.smoothness_lambda
        
        # Distortion bound: ε(L) = K(λ) * L^α
        # K(λ) = K_0 / (1 + λ) approximately
        effective_K = K / (1 + lam)
        epsilon = effective_K * (path_length ** alpha)
        
        return DistortionBounds(
            path_length=path_length,
            distortion_bound=epsilon,
            regularization_strength=lam
        )
    
    def is_in_regime(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[bool, float]:
        """
        Check if pair is within bounded-distortion regime.
        
        Returns:
            (is_valid, margin) where margin indicates distance to boundary
        """
        d = self.distance(p1, p2)
        L_star = self.config.operational_horizon
        
        # Estimate path length from distance
        # (assuming nearly geodesic paths)
        estimated_path_length = d
        
        is_valid = estimated_path_length <= L_star
        margin = L_star - estimated_path_length
        
        return is_valid, margin
    
    def should_abstain(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[bool, str]:
        """
        Determine if analysis should abstain for this pair.
        
        Abstention occurs when:
        - Points are in ambiguity band (between soft and hard margins)
        - Distortion regime is violated
        
        Returns:
            (should_abstain, reason)
        """
        d = self.distance(p1, p2)
        
        # Check margin conditions
        if d < self.config.kappa_soft:
            return True, "below_soft_margin"
        
        if d > self.config.kappa_hard:
            # Past hard margin - confident different
            return False, ""
        
        # In ambiguity band
        return True, "ambiguity_band"
    
    def curvature_estimate(self, points: np.ndarray) -> float:
        """
        Estimate local curvature from point cloud.
        
        Uses heat kernel asymptotics: K(x,x,t) ~ (4πt)^{-d/2}(1 + R(x)t/6 + ...)
        """
        n = points.shape[0]
        if n < 3:
            return 0.0
        
        # Estimate via comparison with flat space
        # (simplified version - full implementation would use Laplacian)
        D = self.distance_matrix(points)
        
        # Average nearest-neighbor distance
        D_sorted = np.sort(D, axis=1)
        mean_nn_dist = np.mean(D_sorted[:, 1])  # First non-self neighbor
        
        # In flat space, nearest neighbors scale as n^{-1/d}
        # Positive curvature: closer than expected
        # Negative curvature: farther than expected
        expected_nn_flat = (n ** (-1/self.dimension)) * 10  # Scaling factor
        
        curvature = (expected_nn_flat - mean_nn_dist) / (mean_nn_dist + 1e-10)
        
        return curvature
    
    def compute_distortion(
        self, 
        points: np.ndarray, 
        true_geodesics: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute actual distortion between Euclidean and geodesic distances.
        
        Args:
            points: Embedded points
            true_geodesics: Ground truth geodesic distances (if available)
        
        Returns:
            Dictionary with distortion statistics
        """
        D_manifold = self.distance_matrix(points)
        
        # Euclidean distances for comparison
        D_euclidean = np.zeros_like(D_manifold)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d = np.linalg.norm(points[i] - points[j])
                D_euclidean[i, j] = d
                D_euclidean[j, i] = d
        
        # Compute distortion metrics
        mask = D_manifold > 0
        ratios = D_euclidean[mask] / (D_manifold[mask] + 1e-10)
        
        return {
            'mean_distortion': np.mean(np.abs(ratios - 1)),
            'max_distortion': np.max(np.abs(ratios - 1)),
            'distortion_std': np.std(ratios),
            'expansion_ratio': np.mean(ratios),
        }
    
    # =========================================================================
    # Metric Learning Interface (for training)
    # =========================================================================
    
    def update_metric(self, W: np.ndarray, b: Optional[np.ndarray] = None):
        """
        Update learned metric tensor.
        
        Called during training to update the embedding.
        """
        self.W = W
        if b is not None:
            self.b = b
        self._update_metric_cache()
    
    def contrastive_loss(
        self, 
        anchor: np.ndarray, 
        positive: np.ndarray, 
        negatives: np.ndarray
    ) -> float:
        """
        InfoNCE contrastive loss for metric learning.
        
        Args:
            anchor: Anchor point
            positive: Positive example (same identity/trajectory)
            negatives: Negative examples (different identities)
        
        Returns:
            Loss value
        """
        tau = self.config.temperature
        
        # Compute distances
        d_pos = self.distance(anchor, positive)
        d_negs = np.array([self.distance(anchor, neg) for neg in negatives])
        
        # InfoNCE: -log(exp(-d_pos/τ) / (exp(-d_pos/τ) + Σ exp(-d_neg/τ)))
        numerator = np.exp(-d_pos / tau)
        denominator = numerator + np.sum(np.exp(-d_negs / tau))
        
        loss = -np.log(numerator / (denominator + 1e-10))
        
        return loss
    
    def smoothness_penalty(self, jacobian: np.ndarray) -> float:
        """
        Jacobian regularization for smoothness.
        
        Penalizes large Jacobian norms to ensure bounded distortion.
        """
        return self.config.smoothness_lambda * np.sum(jacobian ** 2)
