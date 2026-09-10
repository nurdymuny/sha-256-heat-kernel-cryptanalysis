import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from scipy.optimize import minimize
from scipy.linalg import eigh

# Reuse your existing imports
import sys
sys.path.insert(0, '.')
from src.sha256.core import InstrumentedSHA256
from src.embeddings.hyperbolic import HyperbolicEmbedding
from src.embeddings.euclidean import EuclideanEmbedding


def compute_spectral_geometry(states: np.ndarray, k_neighbors: int = 15) -> Dict:
    """
    Compute spectral geometry metrics from a collection of states.
    Returns eigenvalues, spectral gap, effective dimension, and curvature estimate.
    """
    n_samples = len(states)
    if n_samples < 10:
        return {'error': 'insufficient samples'}
    
    # Build graph Laplacian from k-NN
    from scipy.spatial.distance import cdist
    
    # Normalize states
    states_norm = states / (np.linalg.norm(states, axis=1, keepdims=True) + 1e-10)
    
    # Pairwise distances
    dists = cdist(states_norm, states_norm)
    
    # k-NN adjacency
    W = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        neighbors = np.argsort(dists[i])[1:k_neighbors+1]
        for j in neighbors:
            W[i, j] = np.exp(-dists[i, j]**2)
            W[j, i] = W[i, j]
    
    # Degree matrix and Laplacian
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    
    # Normalized Laplacian
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-10))
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt
    
    # Eigendecomposition (smallest eigenvalues)
    try:
        eigenvalues, eigenvectors = eigh(L_norm)
        eigenvalues = np.real(eigenvalues)
        eigenvalues = np.sort(eigenvalues)[:20]  # First 20
    except:
        return {'error': 'eigendecomposition failed'}
    
    # Spectral gap (λ₂ - λ₁)
    spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0
    
    # Effective dimension (where eigenvalues plateau)
    cumvar = np.cumsum(eigenvalues) / (np.sum(eigenvalues) + 1e-10)
    eff_dim_90 = np.searchsorted(cumvar, 0.9) + 1
    
    # Curvature estimate (from eigenvalue distribution)
    # Higher variance in eigenvalues → more curvature variation
    eig_variance = np.var(eigenvalues[:10]) if len(eigenvalues) >= 10 else 0
    
    return {
        'eigenvalues': eigenvalues.tolist(),
        'spectral_gap': float(spectral_gap),
        'effective_dim_90': int(eff_dim_90),
        'eigenvalue_variance': float(eig_variance),
        'fiedler_value': float(eigenvalues[1]) if len(eigenvalues) > 1 else 0
    }

# =============================================================================
# IMPROVEMENT 1: TRUE ALGEBRAIC PROPAGATION (The "Real" Sudoku)
# =============================================================================

class AlgebraicPropagator:
    """
    Implements the actual bitwise mechanics of SHA-256 for constraint propagation.
    Replaces the placeholder logic with probabilistic XOR inversion.
    """
    
    @staticmethod
    def rotr(x: int, n: int) -> int:
        return (x >> n) | (x << (32 - n)) & 0xFFFFFFFF

    @staticmethod
    def get_sigma0_dependencies(bit_pos: int) -> List[int]:
        """
        Returns the 3 bits that sum to create Sigma0 at bit_pos.
        Sigma0(x) = ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22)
        """
        # Inverse rotation mapping
        return [(bit_pos - 2) % 32, (bit_pos - 13) % 32, (bit_pos - 22) % 32]

    @staticmethod
    def infer_bit(target_pos: int, known_bits: Dict[int, int]) -> Optional[int]:
        """
        Attempt to infer a bit value based on its Sigma0 neighbors.
        Rule: If A ^ B ^ C = Out, and we know Out, A, B -> we know C.
        """
        deps = AlgebraicPropagator.get_sigma0_dependencies(target_pos)
        
        # Check if we know the other 2 bits in the triad
        # This is a simplified probabilistic model: 
        # In a sparse state, dominating bits often dictate the output.
        
        known_vals = [known_bits.get(d) for d in deps]
        if known_vals.count(None) == 0:
            # We know all inputs, we can force the output (Forward propagation)
            return known_vals[0] ^ known_vals[1] ^ known_vals[2]
            
        return None

# =============================================================================
# IMPROVEMENT 2: HYBRID SOLVER (Logic + Optimization)
# =============================================================================

@dataclass
class ImprovedConstraintCell:
    position: int
    value: Optional[int] = None
    is_fixed: bool = False
    source_confidence: float = 1.0  # 1.0 = Certain, <1.0 = Probabilistic

class DavisHybridSolver:
    """
    Combines 'Sudoku' logic with Gradient Descent.
    1. Logic Phase: Fix bits that MUST be fixed (Algebraic).
    2. Relaxation Phase: Optimize the rest using Geodesic Loss.
    """
    def __init__(self):
        self.board: Dict[int, ImprovedConstraintCell] = {}
        for i in range(512): # 512-bit message block
            self.board[i] = ImprovedConstraintCell(position=i)
            
    def propagate(self):
        """
        Iteratively propagate constraints using the Sigma0/1 relations.
        """
        changed = True
        while changed:
            changed = False
            # Extract current known state
            known_bits = {k: v.value for k, v in self.board.items() if v.is_fixed}
            
            for i in range(512):
                if self.board[i].is_fixed:
                    continue
                
                # Try to infer from algebraic structure
                inferred = AlgebraicPropagator.infer_bit(i % 32, known_bits)
                
                if inferred is not None:
                    # In Message Schedule, bits interact across words (i, i-15, i-2, i-16)
                    # For this snippet, we stick to intra-word Sigma constraints
                    self.board[i].value = inferred
                    self.board[i].is_fixed = True
                    changed = True

    def get_initial_guess(self) -> np.ndarray:
        """
        Construct a starting vector x0 that respects all algebraic constraints.
        Fill unknowns with 0.5 (neutral) for the optimizer.
        """
        x0 = np.zeros(512)
        for i in range(512):
            if self.board[i].is_fixed:
                x0[i] = self.board[i].value
            else:
                x0[i] = 0.5 # Neutral for relaxation
        return x0

# =============================================================================
# IMPROVEMENT 3: LAG-AWARE GEODESIC LOSS
# =============================================================================

class LagAwareManifoldAttack:
    def __init__(self, target_rounds=24):
        self.target_rounds = target_rounds
        self.sha = InstrumentedSHA256(sample_rounds=[target_rounds])
        self.embedding = HyperbolicEmbedding(curvature=-1.0)
        self.solver = DavisHybridSolver()

    def _weighted_geodesic_loss(self, input_vec: np.ndarray, target_state: np.ndarray) -> float:
        """
        Loss function that knows about the 'Lag'.
        Penalizes 'Leader' registers (a,b,e,f) heavily.
        Relaxes 'Laggard' registers (d,h) to allow manifold sliding.
        """
        # 1. Discretize input for hashing (Soft-Max approximation could be used here for gradients)
        input_bits = np.clip(np.round(input_vec), 0, 1).astype(int)
        message = self._bits_to_bytes(input_bits)
        
        try:
            # Get output state
            traj = self.sha.hash_with_trajectory(message)
            current_state = traj.states[-1].state_vector
        except:
            return 1e5

        # 2. Compute component-wise distances
        # State vector is [a, b, c, d, e, f, g, h] (8 words * 32 bits = 256 bits)
        
        # Define Lag Weights based on your previous 'Red Team' finding
        # Leaders (a,b,e,f) need to be precise. Laggards (d,h) can drift.
        weights = np.ones(256)
        
        # Word indices: a=0..31, b=32..63, ..., h=224..255
        # Decrease weight for d (96-127) and h (224-255)
        weights[96:128] = 0.1  # d is loose
        weights[224:256] = 0.1 # h is loose
        
        # Increase weight for injection points a (0-31) and e (128-159)
        weights[0:32] = 2.0    # a is critical
        weights[128:160] = 2.0 # e is critical

        # 3. Weighted Euclidean/Manifold Distance
        # We perform the weighting in the raw state space, THEN embed
        
        diff = np.abs(current_state - target_state)
        weighted_diff = np.mean(diff * weights)
        
        # 4. Add 'Soft Constraint' penalty for non-binary inputs
        # Forces the optimizer to converge to valid bits (0 or 1)
        binary_penalty = np.sum(input_vec * (1 - input_vec))
        
        return weighted_diff + 0.1 * binary_penalty

    def _bits_to_bytes(self, bits):
        # Helper to convert bit array to bytes
        chars = []
        for b in range(len(bits) // 8):
            byte = bits[b*8:(b+1)*8]
            chars.append(int(''.join(map(str, byte)), 2))
        return bytes(chars)

    def find_preimage(self, target_output):
        # 1. Logic Step: Propagate constraints
        # (Assuming we have some known bits, e.g., padding or partial preimage)
        self.solver.propagate()
        x0 = self.solver.get_initial_guess()
        
        # 2. Relaxation Step: Optimize Lag-Aware Loss
        print("[*] Starting Lag-Aware Manifold Relaxation...")
        res = minimize(
            self._weighted_geodesic_loss,
            x0,
            args=(target_output,),
            method='L-BFGS-B', # Gradient-based (approximate)
            bounds=[(0, 1)] * 512,
            options={'maxiter': 100, 'eps': 1e-3} # Larger step size for discrete jumping
        )
        
        return res


def run_attack():
    """Run the Davis Hybrid Manifold Attack with spectral geometry visualization."""
    print("=" * 70)
    print("  DAVIS HYBRID MANIFOLD ATTACK")
    print("  Combining Algebraic Propagation + Lag-Aware Geodesic Relaxation")
    print("  WITH LIVE SPECTRAL GEOMETRY ANALYSIS")
    print("=" * 70)
    
    # Use full trajectory capture for spectral analysis
    sha_full = InstrumentedSHA256(sample_rounds=list(range(0, 25, 4)))
    attack = LagAwareManifoldAttack(target_rounds=24)
    
    # Generate a target output to find preimage for
    print("\n[1] Generating target state and analyzing spectral geometry...")
    np.random.seed(42)
    test_msg = np.random.bytes(55)
    traj = sha_full.hash_with_trajectory(test_msg)
    target_state = traj.states[-1].state_vector
    
    print(f"    Target message: {test_msg[:16].hex()}...")
    
    # Collect states at each round for spectral analysis
    print("\n[2] SPECTRAL GEOMETRY OF SHA-256 ROUNDS")
    print("-" * 70)
    
    # Generate many trajectories for spectral analysis
    n_samples = 100
    all_round_states = {s.round_num: [] for s in traj.states}
    
    print(f"    Generating {n_samples} trajectories for spectral analysis...")
    for i in range(n_samples):
        np.random.seed(1000 + i)
        msg = np.random.bytes(55)
        t = sha_full.hash_with_trajectory(msg)
        for state in t.states:
            if state.round_num in all_round_states:
                all_round_states[state.round_num].append(state.state_vector)
    
    print(f"\n    {'Round':<8} {'Spectral Gap':<14} {'Fiedler λ₂':<12} {'Eff Dim':<10} {'Curvature':<12}")
    print("    " + "-" * 56)
    
    spectral_by_round = {}
    for round_num in sorted(all_round_states.keys()):
        states = np.array(all_round_states[round_num])
        if len(states) < 20:
            continue
        
        spec = compute_spectral_geometry(states, k_neighbors=10)
        spectral_by_round[round_num] = spec
        
        if 'error' not in spec:
            gap = spec['spectral_gap']
            fiedler = spec['fiedler_value']
            eff_dim = spec['effective_dim_90']
            curv = spec['eigenvalue_variance']
            
            # Visual indicator
            gap_bar = "█" * min(int(gap * 20), 20)
            
            print(f"    {round_num:<8} {gap:<14.4f} {fiedler:<12.4f} {eff_dim:<10} {curv:<12.6f} {gap_bar}")
    
    # Eigenvalue spectrum visualization
    print(f"\n    EIGENVALUE SPECTRUM (first 10 values per round):")
    print("    " + "-" * 70)
    for round_num in sorted(spectral_by_round.keys()):
        spec = spectral_by_round[round_num]
        if 'eigenvalues' in spec:
            eigs = spec['eigenvalues'][:10]
            eig_str = " ".join([f"{e:.3f}" for e in eigs])
            print(f"    R{round_num:02d}: {eig_str}")
    
    # Run the attack
    print("\n" + "-" * 70)
    print("[3] Running hybrid attack with spectral monitoring...")
    
    # Custom callback to show spectral geometry during optimization
    iteration_states = []
    
    def callback_with_spectral(xk):
        """Callback that computes spectral geometry of intermediate states."""
        input_bits = np.clip(np.round(xk), 0, 1).astype(int)
        message = attack._bits_to_bytes(input_bits)
        try:
            t = sha_full.hash_with_trajectory(message)
            final_state = t.states[-1].state_vector
            iteration_states.append(final_state)
            
            if len(iteration_states) >= 10 and len(iteration_states) % 5 == 0:
                states = np.array(iteration_states[-20:])
                spec = compute_spectral_geometry(states, k_neighbors=5)
                if 'error' not in spec:
                    print(f"      Iter {len(iteration_states):3d}: gap={spec['spectral_gap']:.4f}, "
                          f"λ₂={spec['fiedler_value']:.4f}, dim={spec['effective_dim_90']}")
        except:
            pass
    
    result = minimize(
        attack._weighted_geodesic_loss,
        attack.solver.get_initial_guess(),
        args=(target_state,),
        method='L-BFGS-B',
        bounds=[(0, 1)] * 512,
        callback=callback_with_spectral,
        options={'maxiter': 50, 'eps': 1e-3}
    )
    
    print(f"\n[4] Results:")
    print(f"    Optimization success: {result.success}")
    print(f"    Final loss: {result.fun:.6f}")
    print(f"    Iterations: {result.nit}")
    
    # Check how close we got
    found_bits = np.clip(np.round(result.x), 0, 1).astype(int)
    found_msg = attack._bits_to_bytes(found_bits)
    
    try:
        found_traj = sha_full.hash_with_trajectory(found_msg)
        found_state = found_traj.states[-1].state_vector
        
        # Bit-wise match
        matches = np.sum(found_state == target_state)
        match_rate = matches / len(target_state)
        
        print(f"    Bit match rate: {match_rate:.2%} ({matches}/{len(target_state)})")
        
        # Per-word analysis
        word_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        print(f"\n    Per-word match rates:")
        for i, label in enumerate(word_labels):
            start = i * 32
            end = (i + 1) * 32
            word_matches = np.sum(found_state[start:end] == target_state[start:end])
            word_rate = word_matches / 32
            status = "⚠️" if word_rate > 0.6 else "✓"
            print(f"      {label}: {word_rate:.1%} ({word_matches}/32) {status}")
        
        # Final spectral analysis of attack trajectory
        if len(iteration_states) >= 10:
            print(f"\n[5] SPECTRAL GEOMETRY OF ATTACK TRAJECTORY")
            print("-" * 70)
            final_spec = compute_spectral_geometry(np.array(iteration_states), k_neighbors=10)
            if 'error' not in final_spec:
                print(f"    Attack manifold spectral gap: {final_spec['spectral_gap']:.4f}")
                print(f"    Attack manifold Fiedler value: {final_spec['fiedler_value']:.4f}")
                print(f"    Attack manifold eff. dimension: {final_spec['effective_dim_90']}")
                print(f"    Attack manifold curvature var: {final_spec['eigenvalue_variance']:.6f}")
                
                # Compare to SHA-256 spectral structure
                if 24 in spectral_by_round:
                    sha_spec = spectral_by_round[24]
                    gap_ratio = final_spec['spectral_gap'] / (sha_spec['spectral_gap'] + 1e-10)
                    print(f"\n    Gap ratio (attack/SHA-256): {gap_ratio:.2f}")
                    if gap_ratio > 2:
                        print("    ⚠️  Attack trajectory has HIGHER spectral gap - possible structure!")
                    else:
                        print("    ✓ Attack trajectory has similar/lower spectral gap")
        
        # Check if we found a close match
        if match_rate > 0.7:
            print(f"\n  ⚠️  HIGH MATCH RATE - Potential weakness!")
        elif match_rate > 0.55:
            print(f"\n  🟡 Elevated match rate - worth investigating")
        else:
            print(f"\n  ✓ Match rate consistent with random (~50%)")
            print("    Manifold relaxation does not find exploitable structure")
            
    except Exception as e:
        print(f"    Error evaluating result: {e}")
        import traceback
        traceback.print_exc()
    
    return result


if __name__ == "__main__":
    run_attack()