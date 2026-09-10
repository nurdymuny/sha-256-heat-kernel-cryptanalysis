"""
Heat Kernel Solver
==================

Compute heat kernel and spectral invariants from graph Laplacian.

The heat equation on a manifold/graph:
    ∂u/∂t = -Lu

Solution:
    u(t) = exp(-tL) u(0)

Heat kernel:
    K(x, y, t) = Σᵢ exp(-λᵢt) φᵢ(x) φᵢ(y)

Key spectral quantities:
- Heat trace: Z(t) = Tr(exp(-tL)) = Σᵢ exp(-λᵢt)
- Spectral gap: λ₁ - λ₀ = λ₁ (controls mixing time)
- Effective dimension: participation ratio at scale t
- Heat Kernel Signature (HKS): K(x, x, t) for all x

From heat kernel.tex: curvature can be estimated from small-t asymptotics:
    K(x, x, t) ~ (4πt)^{-d/2} (1 + R(x)t/6 + O(t²))

Author: Bee Davis
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy import sparse
from typing import Callable, List, Optional, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class HeatKernelResult:
    """Complete heat kernel analysis result."""
    
    # Spectral data
    eigenvalues: np.ndarray           # Sorted ascending
    eigenvectors: np.ndarray          # Columns are eigenvectors
    
    # Derived quantities
    spectral_gap: float               # λ₁ - λ₀ = λ₁
    spectral_gaps: np.ndarray         # All consecutive gaps λᵢ₊₁ - λᵢ
    
    # Functions of t
    heat_trace: Callable[[float], float]           # Z(t) = Tr(exp(-tL))
    effective_dimension: Callable[[float], float]  # Participation ratio
    
    # Multi-scale analysis
    heat_trace_values: np.ndarray     # Z(t) at sampled time scales
    time_scales: np.ndarray           # t values used
    
    # Statistics
    n_eigenvalues: int
    n_points: int

    @property
    def truncated(self) -> bool:
        return self.n_eigenvalues < self.n_points

    def heat_trace_error_bound(self, t: float) -> float:
        """Upper bound for omitted modes of a sorted PSD spectrum."""
        if t < 0:
            raise ValueError("Diffusion time must be nonnegative")
        return float((self.n_points-self.n_eigenvalues)*np.exp(-self.eigenvalues[-1]*t))
    
    def __post_init__(self):
        """Compute derived quantities."""
        if len(self.eigenvalues) > 1:
            self.spectral_gap = float(self.eigenvalues[1] - self.eigenvalues[0])
            self.spectral_gaps = np.diff(self.eigenvalues)
        else:
            self.spectral_gap = 0.0
            self.spectral_gaps = np.array([])


class HeatKernelSolver:
    """
    Solve heat equation and compute spectral invariants.
    
    The solver computes eigendecomposition of the Laplacian and uses it
    to efficiently evaluate heat kernel quantities at any time scale.
    """
    
    def __init__(
        self, 
        n_eigenvalues: Optional[int] = 100,
        time_scales: Optional[np.ndarray] = None
    ):
        """
        Args:
            n_eigenvalues: Number of smallest eigenvalues to compute
            time_scales: Time values for multi-scale analysis
        """
        self.n_eig = n_eigenvalues
        
        if time_scales is None:
            # Default: 4 orders of magnitude
            self.time_scales = np.logspace(-2, 2, 20)
        else:
            self.time_scales = time_scales
    
    def solve(self, L: sparse.csr_matrix) -> HeatKernelResult:
        """
        Compute eigendecomposition and heat kernel quantities.
        
        Args:
            L: Normalized Laplacian (sparse or dense)
        
        Returns:
            HeatKernelResult with all spectral data
        """
        n = L.shape[0]
        if n == 0:
            raise ValueError("Laplacian must not be empty")
        k = n if self.n_eig is None else min(self.n_eig, n)
        if k < 1:
            raise ValueError("At least one eigenmode is required")
        delta = L-L.T
        asymmetry = np.max(np.abs(delta.data)) if sparse.issparse(delta) and delta.nnz else (0 if sparse.issparse(delta) else np.max(np.abs(delta)))
        if asymmetry > 1e-10:
            raise ValueError("Laplacian must be symmetric")
        
        # Compute smallest eigenvalues using Lanczos algorithm
        if sparse.issparse(L) and k < n:
            eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')
        else:
            # Dense matrix - use full decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(L.toarray() if sparse.issparse(L) else L)
            eigenvalues = eigenvalues[:k]
            eigenvectors = eigenvectors[:, :k]
        
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Clip small negative eigenvalues (numerical noise)
        if eigenvalues.min() < -1e-8:
            raise ValueError("Laplacian is not positive semidefinite")
        eigenvalues = np.maximum(eigenvalues, 0)
        
        # Create heat trace function
        def heat_trace(t: float) -> float:
            """Z(t) = Σᵢ exp(-λᵢt)"""
            return np.sum(np.exp(-eigenvalues * t))
        
        # Create effective dimension function
        def effective_dimension(t: float) -> float:
            """Participation ratio at scale t."""
            weights = np.exp(-eigenvalues * t)
            weights = weights / (weights.sum() + 1e-10)
            return 1.0 / (np.sum(weights ** 2) + 1e-10)
        
        # Compute heat trace at all time scales
        heat_trace_values = np.array([heat_trace(t) for t in self.time_scales])
        
        return HeatKernelResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            spectral_gap=0.0,  # Computed in __post_init__
            spectral_gaps=np.array([]),  # Computed in __post_init__
            heat_trace=heat_trace,
            effective_dimension=effective_dimension,
            heat_trace_values=heat_trace_values,
            time_scales=self.time_scales,
            n_eigenvalues=k,
            n_points=n
        )
    
    def heat_kernel_signature(
        self, 
        result: HeatKernelResult, 
        t_values: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute Heat Kernel Signature (HKS) at multiple time scales.
        
        HKS(x, t) = K(x, x, t) = Σᵢ exp(-λᵢt) φᵢ(x)²
        
        HKS is isometry-invariant: if SHA-256 has hidden symmetries,
        they will appear as repeated HKS profiles.
        
        Args:
            result: HeatKernelResult from solve()
            t_values: Time scales to evaluate (default: self.time_scales)
        
        Returns:
            shape (n_points, n_times) array of HKS values
        """
        if t_values is None:
            t_values = self.time_scales
        
        eigenvalues = result.eigenvalues
        eigenvectors = result.eigenvectors
        n_points = eigenvectors.shape[0]
        n_times = len(t_values)
        
        # HKS(x, t) = Σᵢ exp(-λᵢt) φᵢ(x)²
        HKS = np.zeros((n_points, n_times))
        
        for t_idx, t in enumerate(t_values):
            weights = np.exp(-eigenvalues * t)
            # φᵢ(x)² for all x and i
            phi_sq = eigenvectors ** 2  # (n_points, n_eigenvalues)
            # Weighted sum
            HKS[:, t_idx] = phi_sq @ weights
        
        return HKS
    
    def wave_kernel_signature(
        self,
        result: HeatKernelResult,
        energies: Optional[np.ndarray] = None,
        sigma: float = 0.5
    ) -> np.ndarray:
        """
        Compute Wave Kernel Signature (WKS).
        
        WKS uses log-scale energy bands, better for high-frequency features:
        WKS(x, e) = Σᵢ exp(-(e - log(λᵢ))² / (2σ²)) φᵢ(x)²
        
        Args:
            result: HeatKernelResult
            energies: Log-energy values (default: based on eigenvalue range)
            sigma: Gaussian width in log-space
        
        Returns:
            shape (n_points, n_energies) WKS array
        """
        eigenvalues = result.eigenvalues
        eigenvectors = result.eigenvectors
        
        # Use only positive eigenvalues
        pos_mask = eigenvalues > 1e-10
        pos_eigenvalues = eigenvalues[pos_mask]
        pos_eigenvectors = eigenvectors[:, pos_mask]
        
        if len(pos_eigenvalues) == 0:
            return np.zeros((eigenvectors.shape[0], 1))
        
        log_eigenvalues = np.log(pos_eigenvalues)
        
        if energies is None:
            # Default energy range
            e_min = log_eigenvalues.min()
            e_max = log_eigenvalues.max()
            energies = np.linspace(e_min - sigma, e_max + sigma, 20)
        
        n_points = pos_eigenvectors.shape[0]
        n_energies = len(energies)
        
        WKS = np.zeros((n_points, n_energies))
        
        for e_idx, e in enumerate(energies):
            # Gaussian weights in log-space
            weights = np.exp(-(e - log_eigenvalues) ** 2 / (2 * sigma ** 2))
            weights = weights / (weights.sum() + 1e-10)  # Normalize
            
            phi_sq = pos_eigenvectors ** 2
            WKS[:, e_idx] = phi_sq @ weights
        
        return WKS
    
    def spectral_curvature_estimate(
        self,
        result: HeatKernelResult,
        t_small: float = 0.1
    ) -> np.ndarray:
        """
        Estimate local curvature from small-t heat kernel asymptotics.
        
        On a Riemannian manifold:
        K(x, x, t) ~ (4πt)^{-d/2} (1 + R(x)t/6 + O(t²))
        
        For graphs, we use the ratio of HKS at two time scales to estimate
        the "curvature-like" quantity.
        
        Args:
            result: HeatKernelResult
            t_small: Small time scale for local geometry
        
        Returns:
            Per-point curvature estimates
        """
        raise NotImplementedError(
            "An unscaled graph heat-diagonal ratio is not a curvature estimator. "
            "Use heat_kernel_signature and report it as a graph observable."
        )

    def diffusion_distance(
        self,
        result: HeatKernelResult,
        t: float
    ) -> np.ndarray:
        """
        Compute diffusion distance matrix at time t.
        
        D_t(x, y)² = K(x,x,t) + K(y,y,t) - 2K(x,y,t)
                   = Σᵢ exp(-λᵢt)(φᵢ(x) - φᵢ(y))²
        
        Diffusion distance captures the geometry at scale t.
        
        Args:
            result: HeatKernelResult
            t: Diffusion time
        
        Returns:
            shape (n, n) diffusion distance matrix
        """
        eigenvalues = result.eigenvalues
        eigenvectors = result.eigenvectors
        
        # Weighted eigenvectors: ψᵢ(x) = exp(-λᵢt/2) φᵢ(x)
        weights = np.exp(-eigenvalues * t / 2)
        weighted_phi = eigenvectors * weights[None, :]
        
        # D_t(x,y)² = ||ψ(x) - ψ(y)||²
        n = weighted_phi.shape[0]
        D = np.zeros((n, n))
        
        for i in range(n):
            diff = weighted_phi - weighted_phi[i:i+1, :]
            D[i, :] = np.sqrt(np.sum(diff ** 2, axis=1))
        
        return D
    
    def commute_time_distance(
        self,
        result: HeatKernelResult
    ) -> np.ndarray:
        """
        Compute commute time distance.
        
        The expected time for random walk to go from x to y and back.
        Related to effective resistance in electrical networks.
        
        CT(x, y) = vol(G) * Σᵢ (φᵢ(x) - φᵢ(y))² / λᵢ
        
        Args:
            result: HeatKernelResult
        
        Returns:
            Commute time distance matrix
        """
        eigenvalues = result.eigenvalues
        eigenvectors = result.eigenvectors
        
        # Skip zero eigenvalue
        pos_mask = eigenvalues > 1e-10
        pos_eigenvalues = eigenvalues[pos_mask]
        pos_eigenvectors = eigenvectors[:, pos_mask]
        
        if len(pos_eigenvalues) == 0:
            n = eigenvectors.shape[0]
            return np.zeros((n, n))
        
        # Weight by 1/λ
        weights = 1.0 / pos_eigenvalues
        weighted_phi = pos_eigenvectors * np.sqrt(weights)[None, :]
        
        # CT(x,y) ∝ ||weighted_φ(x) - weighted_φ(y)||²
        n = weighted_phi.shape[0]
        CT = np.zeros((n, n))
        
        for i in range(n):
            diff = weighted_phi - weighted_phi[i:i+1, :]
            CT[i, :] = np.sum(diff ** 2, axis=1)
        
        return CT


class MultiScaleAnalyzer:
    """
    Multi-scale spectral analysis for detecting structure at different scales.
    
    SHA-256 structure might only be visible at certain scales:
    - Small t: local structure (round-to-round)
    - Large t: global structure (full compression)
    """
    
    def __init__(
        self,
        time_scales: Optional[np.ndarray] = None,
        n_eigenvalues: int = 100
    ):
        if time_scales is None:
            self.time_scales = np.logspace(-3, 3, 30)  # Wider range
        else:
            self.time_scales = time_scales
        
        self.solver = HeatKernelSolver(n_eigenvalues, self.time_scales)
    
    def analyze(self, L: sparse.csr_matrix) -> Dict:
        """
        Comprehensive multi-scale analysis.
        
        Args:
            L: Laplacian matrix
        
        Returns:
            Dictionary with all analysis results
        """
        result = self.solver.solve(L)
        
        # Heat Kernel Signature at all scales
        HKS = self.solver.heat_kernel_signature(result)
        
        # Curvature at different scales
        curvatures = {}
        for t in [0.01, 0.1, 1.0, 10.0]:
            curv = self.solver.spectral_curvature_estimate(result, t)
            curvatures[f't={t}'] = {
                'mean': float(np.mean(curv)),
                'std': float(np.std(curv)),
                'max': float(np.max(curv)),
                'min': float(np.min(curv))
            }
        
        # Effective dimension across scales
        eff_dims = np.array([result.effective_dimension(t) for t in self.time_scales])
        
        # Find "characteristic" time scales (where effective dimension changes rapidly)
        eff_dim_grad = np.abs(np.gradient(eff_dims))
        characteristic_scales = self.time_scales[eff_dim_grad > np.mean(eff_dim_grad)]
        
        return {
            'spectral_result': result,
            'HKS': HKS,
            'curvatures_by_scale': curvatures,
            'effective_dimension': eff_dims,
            'characteristic_scales': characteristic_scales,
            'spectral_gap': result.spectral_gap,
            'eigenvalue_entropy': self._eigenvalue_entropy(result.eigenvalues)
        }
    
    def _eigenvalue_entropy(self, eigenvalues: np.ndarray) -> float:
        """Compute entropy of normalized eigenvalue distribution."""
        # Normalize to probability distribution
        pos_eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(pos_eigenvalues) == 0:
            return 0.0
        
        p = pos_eigenvalues / pos_eigenvalues.sum()
        return -np.sum(p * np.log(p + 1e-10))
