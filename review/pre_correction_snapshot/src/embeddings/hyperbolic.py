"""
Hyperbolic Embedding (Poincaré Ball Model)
==========================================

Embed SHA-256 states in hyperbolic space H²⁵⁶.

Rationale: Hash functions have tree-like expansion structure due to
the avalanche effect. Hyperbolic space naturally embeds trees with
low distortion (Gromov hyperbolicity).

Model: Poincaré ball - unit ball in ℝⁿ with hyperbolic metric.

Author: Bee Davis
"""

import numpy as np
from .base import BaseEmbedding, DistortionBounds


class HyperbolicEmbedding(BaseEmbedding):
    """
    Poincaré ball model of hyperbolic space H²⁵⁶.
    
    The Poincaré ball is the unit ball {x ∈ ℝⁿ : ||x|| < 1} with metric
    ds² = (2 / (1 - ||x||²))² ||dx||²
    
    Key properties:
    - Negative curvature K = -c (default c = 1)
    - Trees embed with low distortion
    - Geodesics are arcs of circles orthogonal to boundary
    - Exponential volume growth (like binary tree branching)
    """
    
    def __init__(self, curvature: float = -1.0, ball_radius: float = 0.95):
        """
        Args:
            curvature: Negative curvature (default -1.0)
            ball_radius: Maximum radius to use (< 1 for numerical stability)
        """
        if curvature >= 0:
            raise ValueError("Curvature must be negative for hyperbolic space")
        self.c = -curvature  # c > 0
        self.K = curvature   # K < 0
        self.ball_radius = ball_radius
    
    @property
    def dimension(self) -> int:
        return 256
    
    def embed(self, state: np.ndarray) -> np.ndarray:
        """
        Map state to interior of Poincaré ball.
        
        Strategy: Scale bits from {0,1} to [-1,1], then project to ball.
        """
        normalized = self.normalize_state(state)
        
        # Map bits to [-1, 1]
        x = (normalized.astype(np.float64) - 0.5) * 2
        
        # Project to interior of ball
        norm = np.linalg.norm(x)
        if norm >= self.ball_radius:
            x = x / (norm + 1e-10) * self.ball_radius
        
        return x
    
    def _conformal_factor(self, p: np.ndarray) -> float:
        """
        Conformal factor λ(p) = 2 / (1 - ||p||²).
        """
        norm_sq = np.sum(p ** 2)
        return 2.0 / (1.0 - norm_sq + 1e-10)
    
    def distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        """
        Hyperbolic distance in Poincaré ball model.
        
        d(p1, p2) = (1/√c) * arccosh(1 + 2c ||p1-p2||² / ((1-||p1||²)(1-||p2||²)))
        
        For c = 1:
        d(p1, p2) = arccosh(1 + 2 ||p1-p2||² / ((1-||p1||²)(1-||p2||²)))
        """
        diff = p1 - p2
        norm_sq_diff = np.sum(diff ** 2)
        norm_sq_p1 = np.sum(p1 ** 2)
        norm_sq_p2 = np.sum(p2 ** 2)
        
        denom = (1 - norm_sq_p1) * (1 - norm_sq_p2)
        if denom <= 0:
            # Points on or outside boundary - return large distance
            return 100.0
        
        arg = 1 + 2 * self.c * norm_sq_diff / (denom + 1e-10)
        arg = max(arg, 1.0)  # arccosh domain
        
        return np.arccosh(arg) / np.sqrt(self.c)
    
    def local_metric(self, p: np.ndarray) -> np.ndarray:
        """
        Metric tensor in Poincaré ball.
        
        g_ij = λ(p)² δ_ij where λ = 2/(1-||p||²)
        """
        lambda_p = self._conformal_factor(p)
        return (lambda_p ** 2) * np.eye(256)
    
    def mobius_addition(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """
        Möbius addition: p1 ⊕ p2 in Poincaré ball.
        
        This is the "translation" operation in hyperbolic space.
        """
        norm_sq_p1 = np.sum(p1 ** 2)
        norm_sq_p2 = np.sum(p2 ** 2)
        dot = np.dot(p1, p2)
        
        num = (1 + 2 * self.c * dot + self.c * norm_sq_p2) * p1 + (1 - self.c * norm_sq_p1) * p2
        denom = 1 + 2 * self.c * dot + (self.c ** 2) * norm_sq_p1 * norm_sq_p2
        
        return num / (denom + 1e-10)
    
    def exp_map(self, p: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Exponential map at p in direction v.
        
        Maps tangent vector to point on manifold.
        """
        lambda_p = self._conformal_factor(p)
        norm_v = np.linalg.norm(v)
        
        if norm_v < 1e-10:
            return p
        
        # Scaled exponential map
        scaled_norm = np.sqrt(self.c) * lambda_p * norm_v / 2
        coeff = np.tanh(scaled_norm) / (np.sqrt(self.c) * norm_v + 1e-10)
        
        return self.mobius_addition(p, coeff * v)
    
    def log_map(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Logarithmic map at p towards q.
        
        Maps point to tangent vector.
        """
        minus_p_plus_q = self.mobius_addition(-p, q)
        norm = np.linalg.norm(minus_p_plus_q)
        lambda_p = self._conformal_factor(p)
        
        if norm < 1e-10:
            return np.zeros_like(p)
        
        coeff = (2 / (np.sqrt(self.c) * lambda_p)) * np.arctanh(np.sqrt(self.c) * norm)
        return coeff * minus_p_plus_q / (norm + 1e-10)
    
    def geodesic(self, p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
        """
        Point on geodesic from p1 to p2 at parameter t ∈ [0, 1].
        """
        v = self.log_map(p1, p2)
        return self.exp_map(p1, t * v)
    
    def parallel_transport(
        self, 
        v: np.ndarray, 
        p: np.ndarray, 
        q: np.ndarray
    ) -> np.ndarray:
        """
        Parallel transport vector v from tangent space at p to q.
        """
        lambda_p = self._conformal_factor(p)
        lambda_q = self._conformal_factor(q)
        return (lambda_p / lambda_q) * v
    
    def curvature_at(self, p: np.ndarray) -> float:
        """
        Sectional curvature (constant in hyperbolic space).
        """
        return self.K
    
    def distance_matrix(self, points: np.ndarray) -> np.ndarray:
        """
        Compute pairwise hyperbolic distances.
        """
        n = points.shape[0]
        D = np.zeros((n, n))
        
        # Precompute norms
        norms_sq = np.sum(points ** 2, axis=1)
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                norm_sq_diff = np.sum(diff ** 2)
                denom = (1 - norms_sq[i]) * (1 - norms_sq[j])
                
                if denom <= 0:
                    d = 100.0
                else:
                    arg = 1 + 2 * self.c * norm_sq_diff / (denom + 1e-10)
                    arg = max(arg, 1.0)
                    d = np.arccosh(arg) / np.sqrt(self.c)
                
                D[i, j] = d
                D[j, i] = d
        
        return D
