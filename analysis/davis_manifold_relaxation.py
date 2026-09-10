#!/usr/bin/env python3
"""
DAVIS MANIFOLD RELAXATION ATTACK ON SHA-256
============================================

Applying the Davis Sudoku Principle:
- Sufficient constraints + bounded curvature → unique (or constrained) solution
- Instead of brute-force search, use geometric constraint propagation
- Holonomy budget limits drift; violations indicate algebraic weakness

The Algorithm:
1. Map input/output to manifold anchor points
2. Identify high-curvature regions (slow bits) as constraint channels  
3. Propagate constraints like Sudoku - forced values cascade
4. Use manifold relaxation to find paths satisfying all constraints

Author: Bee Davis
Classification: legacy exploratory research code, superseded by the audited publication path; see review/SHA256_SUBMISSION_REVIEW.md
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '.')
from src.sha256.core import InstrumentedSHA256, SHA256Trajectory
from src.sha256.input_generator import InputGenerator
from src.embeddings.euclidean import EuclideanEmbedding
from src.embeddings.hyperbolic import HyperbolicEmbedding


# =============================================================================
# DAVIS SUDOKU FRAMEWORK FOR SHA-256
# =============================================================================

@dataclass
class CurvatureMap:
    """
    Curvature map derived from heat kernel analysis.
    Identifies constraint channels in SHA-256 state space.
    """
    # Slow bits (from our analysis) - high curvature constraint channels
    slow_bits: List[int] = field(default_factory=lambda: [
        1, 5, 7, 9, 10, 14, 16, 19, 21, 26, 28, 31,  # Word a
        39, 45, 53, 58, 59,  # Word b  
        67, 74,  # Word c
        87, 90, 91  # Beyond
    ])
    
    # Σ₀ rotation constants - define constraint orbits
    sigma0_rotations: List[int] = field(default_factory=lambda: [2, 13, 22])
    
    # Σ₁ rotation constants
    sigma1_rotations: List[int] = field(default_factory=lambda: [6, 11, 25])
    
    # Maj region (words a, b, c) - high constraint density
    maj_region: range = field(default_factory=lambda: range(0, 96))
    
    # Ch region (words e, f, g) - low constraint density  
    ch_region: range = field(default_factory=lambda: range(128, 224))
    
    def get_constraint_strength(self, bit_pos: int) -> float:
        """
        Return constraint strength at bit position.
        Higher = slower divergence = stronger geometric constraint.
        """
        if bit_pos in self.slow_bits:
            return 1.0  # Maximum constraint
        elif bit_pos in self.maj_region:
            return 0.7  # Elevated constraint (Maj region)
        elif bit_pos in self.ch_region:
            return 0.3  # Low constraint (Ch region)
        else:
            return 0.5  # Neutral
    
    def get_rotation_orbit(self, bit_pos: int, word_size: int = 32) -> Set[int]:
        """
        Get the Σ₀ rotation orbit for a bit position within a word.
        """
        local_pos = bit_pos % word_size
        orbit = {bit_pos}
        for rot in self.sigma0_rotations:
            rotated = (local_pos + rot) % word_size
            # Map back to absolute position
            word_base = (bit_pos // word_size) * word_size
            orbit.add(word_base + rotated)
        return orbit
    
    def get_constraint_graph(self) -> Dict[int, Set[int]]:
        """
        Build constraint propagation graph.
        Edges connect bits that constrain each other via Σ₀.
        """
        graph = {}
        for bit in self.slow_bits:
            if bit < 96:  # Only Maj region has Σ₀ structure
                orbit = self.get_rotation_orbit(bit)
                graph[bit] = orbit - {bit}
        return graph


@dataclass  
class HolonomyBudget:
    """
    Tracks holonomy accumulation along reasoning paths.
    When budget exceeded, constraints become inconsistent.
    """
    max_budget: float = 1.0  # τ_budget from Davis framework
    current_holonomy: float = 0.0
    path_length: int = 0
    violations: List[Tuple[int, float]] = field(default_factory=list)
    
    def accumulate(self, local_curvature: float, step_size: float = 1.0):
        """Add holonomy from traversing curved region."""
        delta = local_curvature * step_size
        self.current_holonomy += delta
        self.path_length += 1
        
        if self.current_holonomy > self.max_budget:
            self.violations.append((self.path_length, self.current_holonomy))
    
    def is_within_budget(self) -> bool:
        return self.current_holonomy <= self.max_budget
    
    def reset(self):
        self.current_holonomy = 0.0
        self.path_length = 0
        self.violations = []


@dataclass
class ConstraintCell:
    """
    A cell in the SHA-256 "Sudoku board" - represents a bit position.
    """
    position: int
    possible_values: Set[int] = field(default_factory=lambda: {0, 1})
    is_fixed: bool = False
    constraint_strength: float = 0.5
    
    def fix_value(self, value: int):
        self.possible_values = {value}
        self.is_fixed = True
    
    def eliminate(self, value: int) -> bool:
        """Eliminate a value. Returns True if cell becomes fixed."""
        if value in self.possible_values and len(self.possible_values) > 1:
            self.possible_values.discard(value)
            if len(self.possible_values) == 1:
                self.is_fixed = True
                return True
        return False
    
    def is_contradiction(self) -> bool:
        return len(self.possible_values) == 0


class DavisSudokuSolver:
    """
    Constraint propagation solver for SHA-256 using Davis manifold geometry.
    Treats the hash function as a constraint satisfaction problem where
    curvature defines constraint strength.
    """
    
    def __init__(self, target_rounds: int = 24):
        self.target_rounds = target_rounds
        self.curvature_map = CurvatureMap()
        self.sha = InstrumentedSHA256(sample_rounds=list(range(0, target_rounds + 1, 4)))
        self.embedding = HyperbolicEmbedding(curvature=-1.0)
        
        # Initialize constraint board (256 bits for state)
        self.board: List[ConstraintCell] = []
        self._init_board()
        
        self.propagation_count = 0
        self.holonomy = HolonomyBudget()
        
    def _init_board(self):
        """Initialize the constraint board."""
        self.board = []
        for i in range(256):
            cell = ConstraintCell(
                position=i,
                constraint_strength=self.curvature_map.get_constraint_strength(i)
            )
            self.board.append(cell)
    
    def reset_board(self):
        """Reset board to initial state."""
        self._init_board()
        self.propagation_count = 0
        self.holonomy.reset()
    
    def fix_bit(self, position: int, value: int):
        """Fix a bit value and trigger constraint propagation."""
        if position < len(self.board):
            self.board[position].fix_value(value)
            self._propagate_constraints(position)
    
    def _propagate_constraints(self, source: int):
        """
        Propagate constraints from a fixed cell.
        Uses the Sudoku principle: constraints cascade through high-curvature channels.
        """
        self.propagation_count += 1
        
        # Get constraint graph neighbors
        constraint_graph = self.curvature_map.get_constraint_graph()
        
        if source not in constraint_graph:
            return
        
        # Queue for BFS propagation
        queue = [source]
        visited = {source}
        
        while queue:
            current = queue.pop(0)
            current_cell = self.board[current]
            
            if not current_cell.is_fixed:
                continue
            
            # Accumulate holonomy
            self.holonomy.accumulate(current_cell.constraint_strength)
            
            # Propagate to neighbors
            neighbors = constraint_graph.get(current, set())
            for neighbor in neighbors:
                if neighbor >= len(self.board):
                    continue
                    
                neighbor_cell = self.board[neighbor]
                
                # The constraint: if current is fixed and neighbor is in rotation orbit,
                # the XOR relationship from Σ₀ constrains neighbor
                # This is simplified - real implementation would use actual Σ₀ algebra
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    
                    # Check holonomy budget
                    if not self.holonomy.is_within_budget():
                        # Constraint exceeded - this branch is forced
                        # In Sudoku terms: the value is determined
                        if not neighbor_cell.is_fixed:
                            # Force a value based on geometric consistency
                            forced_value = self._compute_forced_value(current, neighbor)
                            if forced_value is not None:
                                neighbor_cell.fix_value(forced_value)
                                queue.append(neighbor)
    
    def _compute_forced_value(self, source: int, target: int) -> Optional[int]:
        """
        Compute forced value at target based on source and Σ₀ structure.
        This is where the geometry determines the algebra.
        """
        source_cell = self.board[source]
        if not source_cell.is_fixed:
            return None
        
        source_value = list(source_cell.possible_values)[0]
        
        # Simplified: for demonstration, use XOR relationship
        # Real implementation would model full Σ₀ transformation
        return source_value  # Placeholder
    
    def get_fixed_bits(self) -> Dict[int, int]:
        """Return all fixed bit positions and values."""
        return {
            cell.position: list(cell.possible_values)[0]
            for cell in self.board
            if cell.is_fixed
        }
    
    def get_unfixed_count(self) -> int:
        """Count unfixed cells."""
        return sum(1 for cell in self.board if not cell.is_fixed)
    
    def has_contradiction(self) -> bool:
        """Check if any cell has no possible values."""
        return any(cell.is_contradiction() for cell in self.board)


class ManifoldRelaxationAttack:
    """
    Main attack: use manifold relaxation to find constrained paths through SHA-256.
    
    Instead of brute-force preimage search, we:
    1. Start with partial knowledge (some output bits)
    2. Use curvature map to identify constraint channels
    3. Propagate constraints to reduce search space
    4. Relax along geodesics to find valid inputs
    """
    
    def __init__(self, target_rounds: int = 24):
        self.target_rounds = target_rounds
        self.sha = InstrumentedSHA256(sample_rounds=[0, target_rounds])
        self.embedding = HyperbolicEmbedding(curvature=-1.0)
        self.curvature_map = CurvatureMap()
        self.solver = DavisSudokuSolver(target_rounds)
        
    def _message_to_bits(self, message: bytes) -> np.ndarray:
        """Convert message to bit array."""
        bits = []
        for byte in message:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        return np.array(bits)
    
    def _bits_to_message(self, bits: np.ndarray) -> bytes:
        """Convert bit array to message."""
        bits = np.clip(np.round(bits), 0, 1).astype(int)
        message = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte = (byte << 1) | bits[i + j]
            message.append(byte)
        return bytes(message)
    
    def _get_state_at_round(self, message: bytes) -> np.ndarray:
        """Get SHA-256 state at target round."""
        traj = self.sha.hash_with_trajectory(message)
        state = next((s for s in traj.states if s.round_num == self.target_rounds), traj.states[-1])
        return state.state_vector
    
    def _geodesic_loss(self, input_bits: np.ndarray, target_state: np.ndarray,
                       constraint_mask: np.ndarray) -> float:
        """
        Loss function for manifold relaxation.
        Combines geodesic distance with constraint satisfaction.
        """
        # Convert to message
        message = self._bits_to_message(input_bits)
        
        # Get state at target round
        try:
            state = self._get_state_at_round(message)
        except:
            return 1e10
        
        # Embed both states
        p_current = self.embedding.embed(state)
        p_target = self.embedding.embed(target_state)
        
        # Geodesic distance
        geo_dist = self.embedding.distance(p_current, p_target)
        
        # Constraint penalty: penalize deviation on constrained bits
        constraint_penalty = 0.0
        for i, (s, t, m) in enumerate(zip(state, target_state, constraint_mask)):
            if m > 0:  # Constrained position
                constraint_penalty += m * abs(s - t)
        
        # Combined loss with curvature weighting
        curvature_weight = sum(
            self.curvature_map.get_constraint_strength(i) 
            for i in range(len(state))
        ) / len(state)
        
        return geo_dist + curvature_weight * constraint_penalty
    
    def find_constrained_preimage(self, target_output: np.ndarray,
                                   known_input_bits: Dict[int, int] = None,
                                   max_iterations: int = 1000) -> Dict:
        """
        Find input that produces target output using manifold relaxation.
        
        Args:
            target_output: Desired state at target_rounds (256 bits)
            known_input_bits: Dict of position -> value for known input bits
            max_iterations: Maximum optimization iterations
            
        Returns:
            Dict with found input, distance, and statistics
        """
        print(f"\n[Manifold Relaxation] Searching for constrained preimage...")
        print(f"  Target rounds: {self.target_rounds}")
        print(f"  Known input bits: {len(known_input_bits) if known_input_bits else 0}")
        
        # Initialize solver with constraints
        self.solver.reset_board()
        
        # Fix known input bits and propagate
        if known_input_bits:
            for pos, val in known_input_bits.items():
                self.solver.fix_bit(pos, val)
        
        print(f"  After propagation: {self.solver.get_unfixed_count()} unfixed bits")
        print(f"  Holonomy violations: {len(self.solver.holonomy.violations)}")
        
        # Build constraint mask from slow bits
        constraint_mask = np.array([
            self.curvature_map.get_constraint_strength(i)
            for i in range(256)
        ])
        
        # Initial guess: random with fixed bits
        np.random.seed(42)
        x0 = np.random.rand(512)  # 512 bits for input message (64 bytes)
        
        # Fix known bits in initial guess
        if known_input_bits:
            for pos, val in known_input_bits.items():
                if pos < 512:
                    x0[pos] = val
        
        # Bounds
        bounds = [(0, 1) for _ in range(512)]
        
        # Fix known bits in bounds
        if known_input_bits:
            for pos, val in known_input_bits.items():
                if pos < 512:
                    bounds[pos] = (val, val)
        
        # Optimize using differential evolution (global) then local refinement
        print("  Running differential evolution...")
        
        best_loss = float('inf')
        best_input = None
        
        def loss_fn(x):
            return self._geodesic_loss(x, target_output, constraint_mask)
        
        # Run differential evolution
        result = differential_evolution(
            loss_fn,
            bounds,
            maxiter=max_iterations // 10,
            seed=42,
            workers=1,
            updating='deferred',
            disp=False
        )
        
        best_loss = result.fun
        best_input = result.x
        
        # Local refinement
        print("  Running local refinement...")
        try:
            local_result = minimize(
                loss_fn,
                best_input,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iterations // 2}
            )
            if local_result.fun < best_loss:
                best_loss = local_result.fun
                best_input = local_result.x
        except:
            pass
        
        # Convert to message
        found_message = self._bits_to_message(best_input)
        found_state = self._get_state_at_round(found_message)
        
        # Compute final metrics
        p_found = self.embedding.embed(found_state)
        p_target = self.embedding.embed(target_output)
        final_distance = self.embedding.distance(p_found, p_target)
        
        # Bit match rate
        bit_matches = np.sum(found_state == target_output) / 256
        
        results = {
            'success': final_distance < 1.0,
            'final_distance': float(final_distance),
            'bit_match_rate': float(bit_matches),
            'best_loss': float(best_loss),
            'propagation_count': self.solver.propagation_count,
            'holonomy_violations': len(self.solver.holonomy.violations),
            'unfixed_bits_after_propagation': self.solver.get_unfixed_count(),
            'found_message_hex': found_message[:32].hex()
        }
        
        print(f"\n  Results:")
        print(f"    Final geodesic distance: {final_distance:.4f}")
        print(f"    Bit match rate: {bit_matches:.2%}")
        print(f"    Constraint propagations: {self.solver.propagation_count}")
        print(f"    Holonomy violations: {len(self.solver.holonomy.violations)}")
        
        if results['success']:
            print(f"  ⚠️  FOUND CLOSE PREIMAGE!")
        else:
            print(f"  ✓ No close preimage found (expected for secure hash)")
        
        return results
    
    def find_geometric_collision(self, n_attempts: int = 100) -> Dict:
        """
        Search for collision using geometric guidance.
        Use slow-manifold paths to keep pairs close through rounds.
        """
        print(f"\n[Manifold Relaxation] Searching for geometric collision...")
        print(f"  Target rounds: {self.target_rounds}")
        print(f"  Attempts: {n_attempts}")
        
        best_distance = float('inf')
        best_pair = None
        
        # Strategy: start from slow bit positions, search for pairs
        # that stay close through the hash
        
        distances = []
        
        for attempt in range(n_attempts):
            if attempt % 20 == 0:
                print(f"    Attempt {attempt}/{n_attempts}...")
            
            # Generate base message
            np.random.seed(attempt)
            base_msg = np.random.bytes(55)
            base_bits = self._message_to_bits(base_msg)
            
            # Flip slow bits to create candidate pair
            modified_bits = base_bits.copy()
            
            # Flip a combination of slow bits
            slow_bits = self.curvature_map.slow_bits[:10]
            flip_pattern = np.random.choice(slow_bits, size=min(3, len(slow_bits)), replace=False)
            
            for bit in flip_pattern:
                if bit < len(modified_bits):
                    modified_bits[bit] = 1 - modified_bits[bit]
            
            modified_msg = self._bits_to_message(modified_bits)
            
            # Get states
            state1 = self._get_state_at_round(base_msg)
            state2 = self._get_state_at_round(modified_msg)
            
            # Compute geodesic distance
            p1 = self.embedding.embed(state1)
            p2 = self.embedding.embed(state2)
            dist = self.embedding.distance(p1, p2)
            
            distances.append(dist)
            
            if dist < best_distance:
                best_distance = dist
                best_pair = (base_msg, modified_msg, flip_pattern)
        
        # Statistics
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # Compute baseline for comparison
        baseline_distances = []
        for i in range(50):
            np.random.seed(1000 + i)
            m1 = np.random.bytes(55)
            np.random.seed(2000 + i)
            m2 = np.random.bytes(55)
            s1 = self._get_state_at_round(m1)
            s2 = self._get_state_at_round(m2)
            p1 = self.embedding.embed(s1)
            p2 = self.embedding.embed(s2)
            baseline_distances.append(self.embedding.distance(p1, p2))
        
        baseline_mean = np.mean(baseline_distances)
        
        # Z-score of best finding
        z_score = (best_distance - mean_dist) / std_dist if std_dist > 0 else 0
        
        results = {
            'best_distance': float(best_distance),
            'mean_distance': float(mean_dist),
            'std_distance': float(std_dist),
            'baseline_mean': float(baseline_mean),
            'improvement': float((baseline_mean - best_distance) / baseline_mean),
            'z_score': float(z_score),
            'best_flip_pattern': best_pair[2].tolist() if best_pair else [],
            'collision_found': best_distance < 0.1  # Very close
        }
        
        print(f"\n  Results:")
        print(f"    Best geodesic distance: {best_distance:.4f}")
        print(f"    Mean distance (slow bits): {mean_dist:.4f}")
        print(f"    Baseline mean (random): {baseline_mean:.4f}")
        print(f"    Improvement over baseline: {results['improvement']:.2%}")
        print(f"    Best z-score: {z_score:.2f}")
        print(f"    Best flip pattern: {results['best_flip_pattern']}")
        
        if results['collision_found']:
            print(f"  ⚠️  NEAR-COLLISION FOUND!")
        elif results['improvement'] > 0.2:
            print(f"  ⚠️  SIGNIFICANT IMPROVEMENT using geometric guidance")
        else:
            print(f"  ✓ No exploitable collision path found")
        
        return results
    
    def analyze_constraint_propagation(self, n_samples: int = 100) -> Dict:
        """
        Analyze how constraints propagate through the Sudoku board.
        """
        print(f"\n[Constraint Analysis] Analyzing propagation patterns...")
        
        propagation_depths = []
        holonomy_totals = []
        forced_bits_counts = []
        
        for i in range(n_samples):
            self.solver.reset_board()
            
            # Fix a random slow bit
            slow_bit = np.random.choice(self.curvature_map.slow_bits[:10])
            value = np.random.randint(0, 2)
            
            self.solver.fix_bit(slow_bit, value)
            
            propagation_depths.append(self.solver.propagation_count)
            holonomy_totals.append(self.solver.holonomy.current_holonomy)
            forced_bits_counts.append(256 - self.solver.get_unfixed_count())
        
        results = {
            'mean_propagation_depth': float(np.mean(propagation_depths)),
            'mean_holonomy': float(np.mean(holonomy_totals)),
            'mean_forced_bits': float(np.mean(forced_bits_counts)),
            'max_forced_bits': int(np.max(forced_bits_counts)),
            'propagation_efficiency': float(np.mean(forced_bits_counts) / 256),
            'holonomy_budget_usage': float(np.mean(holonomy_totals) / self.solver.holonomy.max_budget)
        }
        
        print(f"\n  Results:")
        print(f"    Mean propagation depth: {results['mean_propagation_depth']:.2f}")
        print(f"    Mean holonomy accumulated: {results['mean_holonomy']:.4f}")
        print(f"    Mean bits forced: {results['mean_forced_bits']:.1f} / 256")
        print(f"    Propagation efficiency: {results['propagation_efficiency']:.2%}")
        print(f"    Holonomy budget usage: {results['holonomy_budget_usage']:.2%}")
        
        return results


def run_manifold_relaxation_attacks(target_rounds: int = 24) -> Dict:
    """
    Run complete manifold relaxation attack suite.
    """
    print("=" * 70)
    print("  DAVIS MANIFOLD RELAXATION ATTACK")
    print("=" * 70)
    print(f"\n  Applying the Sudoku Principle to SHA-256")
    print(f"  Target rounds: {target_rounds}")
    
    attack = ManifoldRelaxationAttack(target_rounds=target_rounds)
    results = {}
    
    # 1. Analyze constraint propagation
    print("\n" + "-" * 70)
    results['constraint_analysis'] = attack.analyze_constraint_propagation(n_samples=50)
    
    # 2. Search for geometric collisions
    print("\n" + "-" * 70)
    results['collision_search'] = attack.find_geometric_collision(n_attempts=100)
    
    # 3. Try constrained preimage (with some known bits)
    print("\n" + "-" * 70)
    
    # Generate a target state
    np.random.seed(999)
    target_msg = np.random.bytes(55)
    sha = InstrumentedSHA256(sample_rounds=[0, target_rounds])
    traj = sha.hash_with_trajectory(target_msg)
    target_state = traj.states[-1].state_vector
    
    # Give some known input bits (simulate partial knowledge)
    known_bits = {i: int(target_msg[i // 8] >> (7 - i % 8)) & 1 for i in range(32)}
    
    results['preimage_search'] = attack.find_constrained_preimage(
        target_state, 
        known_input_bits=known_bits,
        max_iterations=500
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("  MANIFOLD RELAXATION RESULTS")
    print("=" * 70)
    
    findings = []
    
    if results['collision_search']['improvement'] > 0.2:
        findings.append(f"Geometric collision guidance: {results['collision_search']['improvement']:.1%} improvement")
    
    if results['constraint_analysis']['propagation_efficiency'] > 0.1:
        findings.append(f"Constraint propagation: {results['constraint_analysis']['propagation_efficiency']:.1%} efficiency")
    
    if results['preimage_search']['bit_match_rate'] > 0.6:
        findings.append(f"Preimage search: {results['preimage_search']['bit_match_rate']:.1%} bit match")
    
    if findings:
        print("\n  FINDINGS:")
        for f in findings:
            print(f"    ⚠️  {f}")
    else:
        print("\n  ✓ Manifold relaxation does not break SHA-256")
        print("    Geometric structure is internal, not exploitable")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Davis Manifold Relaxation Attack")
    parser.add_argument('--rounds', type=int, default=24, help='Target rounds')
    parser.add_argument('--output', type=str, default='manifold_relaxation_results.json')
    args = parser.parse_args()
    
    results = run_manifold_relaxation_attacks(args.rounds)
    
    import json
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {args.output}")
