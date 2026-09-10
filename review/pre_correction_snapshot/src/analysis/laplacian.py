"""
Laplacian Constructor
=====================

Build graph Laplacians from sampled SHA-256 state transitions.

The Laplacian encodes the diffusion geometry of the state space.
Its spectrum reveals:
- Clustering structure (small eigenvalues)
- Connectivity (spectral gap)
- Geometric regularity (eigenvalue distribution)

Author: Bee Davis
"""

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

from ..embeddings.base import BaseEmbedding
from ..sha256.core import SHA256Trajectory


@dataclass
class LaplacianResult:
    """Result of Laplacian construction."""
    laplacian: sparse.csr_matrix       # Normalized Laplacian
    weight_matrix: sparse.csr_matrix   # Edge weights (adjacency)
    degree_matrix: sparse.csr_matrix   # Degree matrix
    points: np.ndarray                 # Embedded points
    n_points: int                      # Number of points
    k_neighbors: int                   # k used for k-NN
    graph_connected: bool              # Is the graph connected?
    

class LaplacianConstructor:
    """
    Construct graph Laplacians from embedded SHA-256 states.
    
    The pipeline:
    1. Extract states at specified round from trajectories
    2. Embed states using chosen embedding (Euclidean, Hyperbolic, Davis)
    3. Build k-NN graph on embedded points
    4. Compute edge weights using heat kernel: w_ij = exp(-d²/(4t))
    5. Construct normalized symmetric Laplacian: L = I - D^{-1/2} W D^{-1/2}
    
    The resulting Laplacian has eigenvalues in [0, 2], with:
    - λ₀ = 0 always (trivial mode)
    - λ₁ > 0 controls mixing time (spectral gap)
    - Eigenvalue distribution reveals geometry
    """
    
    def __init__(
        self, 
        embedding: BaseEmbedding, 
        k_neighbors: int = 50,
        diffusion_time: float = 1.0,
        symmetrize: bool = True
    ):
        """
        Args:
            embedding: Manifold embedding to use
            k_neighbors: Number of neighbors for k-NN graph
            diffusion_time: Time parameter t for heat kernel weights
            symmetrize: Whether to symmetrize the k-NN graph
        """
        self.embedding = embedding
        self.k = k_neighbors
        self.t = diffusion_time
        self.symmetrize = symmetrize
    
    def build_from_trajectories(
        self, 
        trajectories: List[SHA256Trajectory],
        round_num: int
    ) -> LaplacianResult:
        """
        Build Laplacian from states at specified round.
        
        Args:
            trajectories: List of SHA-256 trajectories
            round_num: Which round to extract (0-64)
        
        Returns:
            LaplacianResult with Laplacian and embedded points
        """
        # Extract states at round_num
        states = []
        for traj in trajectories:
            for state in traj.states:
                if state.round_num == round_num:
                    states.append(state.state_vector)
                    break
        
        if len(states) == 0:
            raise ValueError(f"No states found at round {round_num}")
        
        states = np.array(states)
        
        # Embed states
        points = self.embedding.embed_batch(states)
        
        return self.build_from_points(points)
    
    def build_from_points(self, points: np.ndarray) -> LaplacianResult:
        """
        Build Laplacian from pre-embedded points.
        
        Args:
            points: shape (n, d) array of embedded points
        
        Returns:
            LaplacianResult
        """
        n = points.shape[0]
        k = min(self.k, n - 1)
        
        # Build k-NN graph
        W = self._build_knn_graph(points, k)
        
        # Compute degree matrix
        degrees = np.array(W.sum(axis=1)).flatten()
        D = sparse.diags(degrees)
        
        # Compute normalized Laplacian
        L = self._compute_normalized_laplacian(W, degrees)
        
        # Check connectivity (λ₁ > 0)
        graph_connected = self._check_connectivity(W)
        
        return LaplacianResult(
            laplacian=L.tocsr(),
            weight_matrix=W.tocsr(),
            degree_matrix=D.tocsr(),
            points=points,
            n_points=n,
            k_neighbors=k,
            graph_connected=graph_connected
        )
    
    def build_transition_laplacian(
        self,
        trajectories: List[SHA256Trajectory],
        from_round: int,
        to_round: int
    ) -> sparse.csr_matrix:
        """
        Build Laplacian capturing round-to-round transition geometry.
        
        Creates bipartite-like structure connecting states at from_round
        to their corresponding states at to_round.
        
        Args:
            trajectories: List of trajectories
            from_round: Starting round
            to_round: Ending round
        
        Returns:
            Transition Laplacian
        """
        # Extract paired states
        from_states = []
        to_states = []
        
        for traj in trajectories:
            from_state = None
            to_state = None
            
            for state in traj.states:
                if state.round_num == from_round:
                    from_state = state.state_vector
                elif state.round_num == to_round:
                    to_state = state.state_vector
            
            if from_state is not None and to_state is not None:
                from_states.append(from_state)
                to_states.append(to_state)
        
        if len(from_states) == 0:
            raise ValueError(f"No valid state pairs for rounds {from_round}->{to_round}")
        
        # Embed both sets
        from_points = self.embedding.embed_batch(np.array(from_states))
        to_points = self.embedding.embed_batch(np.array(to_states))
        
        # Build combined graph
        n = len(from_states)
        all_points = np.vstack([from_points, to_points])
        
        # Build k-NN on combined space
        result = self.build_from_points(all_points)
        
        return result.laplacian
    
    def _build_knn_graph(
        self, 
        points: np.ndarray, 
        k: int
    ) -> sparse.csr_matrix:
        """
        Build k-NN graph with heat kernel edge weights.
        
        Args:
            points: shape (n, d) points
            k: number of neighbors
        
        Returns:
            Weight matrix W as sparse matrix
        """
        n = points.shape[0]
        
        # Use KD-tree for efficient neighbor search
        # Note: This uses Euclidean distance for neighbor finding,
        # then computes true manifold distance for weights
        tree = cKDTree(points)
        
        # Query k+1 neighbors (includes self)
        distances, indices = tree.query(points, k=k+1)
        
        # Build sparse weight matrix
        rows = []
        cols = []
        weights = []
        
        for i in range(n):
            for j_idx, j in enumerate(indices[i]):
                if i == j:
                    continue  # Skip self-loops
                
                # Compute manifold distance for weight
                d = self.embedding.distance(points[i], points[j])
                
                # Heat kernel weight: w = exp(-d²/(4t))
                w = np.exp(-d**2 / (4 * self.t))
                
                rows.append(i)
                cols.append(j)
                weights.append(w)
        
        W = sparse.csr_matrix(
            (weights, (rows, cols)), 
            shape=(n, n)
        )
        
        # Symmetrize if requested
        if self.symmetrize:
            W = (W + W.T) / 2
        
        return W
    
    def _compute_normalized_laplacian(
        self, 
        W: sparse.csr_matrix,
        degrees: np.ndarray
    ) -> sparse.csr_matrix:
        """
        Compute normalized symmetric Laplacian.
        
        L_sym = I - D^{-1/2} W D^{-1/2}
        
        Eigenvalues in [0, 2].
        """
        n = W.shape[0]
        
        # Handle zero degrees (isolated vertices)
        d_inv_sqrt = np.zeros_like(degrees)
        nonzero = degrees > 1e-10
        d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
        
        D_inv_sqrt = sparse.diags(d_inv_sqrt)
        
        # L = I - D^{-1/2} W D^{-1/2}
        L = sparse.eye(n) - D_inv_sqrt @ W @ D_inv_sqrt
        
        return L
    
    def _check_connectivity(self, W: sparse.csr_matrix) -> bool:
        """
        Check if the graph is connected.
        
        Uses simple BFS/DFS approach.
        """
        from scipy.sparse.csgraph import connected_components
        n_components, _ = connected_components(W, directed=False)
        return n_components == 1
    
    def adaptive_k(
        self, 
        points: np.ndarray, 
        min_k: int = 10,
        max_k: int = 100
    ) -> int:
        """
        Adaptively choose k based on point density.
        
        Heuristic: k ≈ log(n) * sqrt(d) for n points in d dimensions.
        """
        n, d = points.shape
        k = int(np.log(n) * np.sqrt(d))
        return max(min_k, min(k, max_k, n - 1))


class TransitionGraphBuilder:
    """
    Build graphs that capture round-to-round transitions.
    
    Instead of static snapshots, this captures the dynamics
    of how states evolve through SHA-256 rounds.
    """
    
    def __init__(self, embedding: BaseEmbedding):
        self.embedding = embedding
    
    def build_transition_graph(
        self,
        trajectories: List[SHA256Trajectory],
        rounds: List[int]
    ) -> sparse.csr_matrix:
        """
        Build a graph where edges connect consecutive round states.
        
        Args:
            trajectories: List of SHA-256 trajectories
            rounds: Ordered list of rounds to include
        
        Returns:
            Sparse adjacency matrix
        """
        # Collect all states and their round/trajectory indices
        all_states = []
        state_info = []  # (trajectory_idx, round_num)
        
        for traj_idx, traj in enumerate(trajectories):
            for state in traj.states:
                if state.round_num in rounds:
                    all_states.append(state.state_vector)
                    state_info.append((traj_idx, state.round_num))
        
        if len(all_states) == 0:
            raise ValueError("No states found")
        
        # Embed all states
        points = self.embedding.embed_batch(np.array(all_states))
        n = len(points)
        
        # Build edges: connect same-trajectory consecutive rounds
        rows = []
        cols = []
        weights = []
        
        for i in range(n):
            traj_i, round_i = state_info[i]
            round_idx_i = rounds.index(round_i)
            
            if round_idx_i < len(rounds) - 1:
                # Find the next round state for same trajectory
                next_round = rounds[round_idx_i + 1]
                
                for j in range(n):
                    traj_j, round_j = state_info[j]
                    if traj_j == traj_i and round_j == next_round:
                        # Connect i -> j
                        d = self.embedding.distance(points[i], points[j])
                        w = np.exp(-d**2 / 4)  # Heat kernel weight
                        
                        rows.extend([i, j])
                        cols.extend([j, i])
                        weights.extend([w, w])
                        break
        
        W = sparse.csr_matrix(
            (weights, (rows, cols)),
            shape=(n, n)
        )
        
        return W
