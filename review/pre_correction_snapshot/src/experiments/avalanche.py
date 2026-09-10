"""
Experiment 2: Avalanche Geometry
================================

Objective: Determine if the avalanche effect is geometrically isotropic.

Protocol:
1. Generate Hamming-1 pairs (messages differing by exactly 1 bit)
2. Collect trajectories for both elements of each pair
3. Compute divergence curves (distance vs round)
4. Analyze isotropy (does divergence depend on which bit was flipped?)
5. Look for "slow directions" (bits whose flipping causes slower divergence)

Success Criteria:
- If anisotropy_score > threshold, avalanche has geometric bias
- If slow_bits is non-empty, certain input bits have exploitable properties
- Expected (null): divergence should be uniform across all bit positions

Author: Bee Davis
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

from ..sha256.core import InstrumentedSHA256, SHA256Trajectory
from ..sha256.input_generator import InputGenerator, HammingPair
from ..embeddings import get_embedding, BaseEmbedding


@dataclass
class AvalancheConfig:
    """Configuration for avalanche geometry experiment."""
    n_pairs: int = 5000                 # Number of 1-bit-flip pairs
    bit_positions: Optional[List[int]] = None  # Which bits to flip (None = all)
    sample_rounds: List[int] = None     # Which rounds to track
    embedding: str = 'euclidean'        # Embedding to use
    anisotropy_threshold: float = 0.1   # Threshold for detecting anisotropy
    slow_bit_threshold: float = 0.05    # Threshold for slow bits (percentile)
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.sample_rounds is None:
            self.sample_rounds = list(range(0, 65, 4))  # Every 4 rounds


@dataclass 
class AvalancheResult:
    """Results from avalanche geometry experiment."""
    config: AvalancheConfig
    
    # Divergence data
    divergence_curves: np.ndarray       # Shape (n_pairs, n_rounds)
    mean_divergence: np.ndarray         # Mean across pairs for each round
    divergence_variance: np.ndarray     # Variance at each round
    
    # Anisotropy analysis
    anisotropy_score: Optional[float]   # Measure of non-uniformity
    divergence_by_bit_position: Dict[int, np.ndarray]  # Grouped by flipped bit
    
    # Slow bits
    slow_bits: List[int]                # Bits with slow divergence
    bit_divergence_rates: np.ndarray    # Rate per bit position
    
    # Statistical summary
    mean_final_divergence: float
    divergence_half_life: float         # Round at which half max divergence


def get_state_at_round(
    trajectory: SHA256Trajectory, 
    round_num: int
) -> Optional[np.ndarray]:
    """Extract state vector at specific round from trajectory."""
    for state in trajectory.states:
        if state.round_num == round_num:
            return state.state_vector
    return None


def compute_divergence_rate(curve: np.ndarray, rounds: List[int]) -> float:
    """Compute linear divergence rate from curve."""
    if len(curve) < 2:
        return 0.0
    # Linear regression slope
    x = np.array(rounds)
    y = curve
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def find_half_life(curve: np.ndarray, rounds: List[int], max_divergence: float) -> float:
    """Find round at which divergence reaches half of maximum."""
    target = max_divergence / 2
    for i, (r, d) in enumerate(zip(rounds, curve)):
        if d >= target:
            # Interpolate
            if i > 0:
                prev_d = curve[i-1]
                prev_r = rounds[i-1]
                frac = (target - prev_d) / (d - prev_d + 1e-10)
                return prev_r + frac * (r - prev_r)
            return float(r)
    return float(rounds[-1])


def experiment_2_avalanche(config: Dict) -> Dict:
    """
    Run avalanche geometry experiment.
    
    Args:
        config: Dictionary with configuration
    
    Returns:
        Dictionary with all results
    """
    # Parse config
    cfg = AvalancheConfig(
        n_pairs=config.get('n_pairs', 5000),
        bit_positions=config.get('bit_positions', None),
        sample_rounds=config.get('sample_rounds', None),
        embedding=config.get('embedding', 'euclidean'),
        anisotropy_threshold=config.get('anisotropy_threshold', 0.1),
        slow_bit_threshold=config.get('slow_bit_threshold', 0.05),
        seed=config.get('seed', None)
    )
    
    print(f"Running Experiment 2: Avalanche Geometry")
    print(f"  Pairs: {cfg.n_pairs}")
    print(f"  Embedding: {cfg.embedding}")
    print(f"  Rounds: {cfg.sample_rounds}")
    
    # 1. Generate Hamming-1 pairs
    gen = InputGenerator(seed=cfg.seed)
    pairs = gen.hamming_pairs(cfg.n_pairs, cfg.bit_positions)
    print(f"  Generated {len(pairs)} Hamming-1 pairs")
    
    # 2. Collect trajectories
    sha = InstrumentedSHA256(sample_rounds=cfg.sample_rounds)
    trajectories_a = sha.hash_batch([p.message_a for p in pairs])
    trajectories_b = sha.hash_batch([p.message_b for p in pairs])
    print(f"  Collected {len(trajectories_a) * 2} trajectories")
    
    # 3. Setup embedding
    embedding = get_embedding(cfg.embedding)
    
    # 4. Compute divergence curves
    divergences = []
    pair_bit_positions = []
    
    for i, (ta, tb, pair) in enumerate(zip(trajectories_a, trajectories_b, pairs)):
        curve = []
        for round_num in cfg.sample_rounds:
            state_a = get_state_at_round(ta, round_num)
            state_b = get_state_at_round(tb, round_num)
            
            if state_a is None or state_b is None:
                curve.append(0.0)
                continue
            
            p_a = embedding.embed(state_a)
            p_b = embedding.embed(state_b)
            d = embedding.distance(p_a, p_b)
            curve.append(d)
        
        divergences.append(curve)
        pair_bit_positions.append(pair.flipped_bit)
    
    divergences = np.array(divergences)  # Shape (n_pairs, n_rounds)
    pair_bit_positions = np.array(pair_bit_positions)
    
    print(f"  Computed divergence curves")
    
    # 5. Analyze by bit position
    divergence_by_bit = defaultdict(list)
    for i, bit_pos in enumerate(pair_bit_positions):
        divergence_by_bit[bit_pos].append(divergences[i])
    
    # Average divergence rate per bit position
    bit_rates = {}
    for bit_pos, curves in divergence_by_bit.items():
        curves = np.array(curves)
        mean_curve = curves.mean(axis=0)
        rate = compute_divergence_rate(mean_curve, cfg.sample_rounds)
        bit_rates[bit_pos] = rate
    
    # Convert to array (fill missing with mean)
    max_bit = max(bit_rates.keys()) + 1 if bit_rates else 440
    bit_divergence_rates = np.zeros(max_bit)
    mean_rate = np.mean(list(bit_rates.values())) if bit_rates else 0
    
    for bit_pos in range(max_bit):
        bit_divergence_rates[bit_pos] = bit_rates.get(bit_pos, mean_rate)
    
    # 6. Compute anisotropy score
    if len(bit_rates) > 1:
        rates = np.array(list(bit_rates.values()))
        anisotropy_score = float(np.std(rates) / (np.mean(rates) + 1e-10))
    else:
        anisotropy_score = 0.0
    
    # 7. Find slow bits
    threshold_rate = np.percentile(bit_divergence_rates, cfg.slow_bit_threshold * 100)
    slow_bits = list(np.where(bit_divergence_rates < threshold_rate)[0])
    
    # 8. Summary statistics
    mean_divergence = divergences.mean(axis=0)
    divergence_variance = divergences.var(axis=0)
    mean_final_divergence = float(mean_divergence[-1])
    
    # Half-life
    max_div = mean_divergence.max()
    divergence_half_life = find_half_life(mean_divergence, cfg.sample_rounds, max_div)
    
    print(f"\n=== Experiment 2 Results ===")
    print(f"  Anisotropy score: {anisotropy_score:.4f}")
    print(f"  Slow bits found: {len(slow_bits)}")
    print(f"  Mean final divergence: {mean_final_divergence:.4f}")
    print(f"  Divergence half-life: {divergence_half_life:.1f} rounds")
    
    if anisotropy_score > cfg.anisotropy_threshold:
        print(f"  ⚠️  ANISOTROPY DETECTED (score > threshold)")
    
    if slow_bits:
        print(f"  ⚠️  SLOW BITS: {slow_bits[:10]}...")
    
    return {
        'config': cfg.__dict__,
        'divergence_curves': divergences.tolist(),
        'mean_divergence': mean_divergence.tolist(),
        'divergence_variance': divergence_variance.tolist(),
        'anisotropy_score': anisotropy_score,
        'slow_bits': slow_bits,
        'bit_divergence_rates': bit_divergence_rates.tolist(),
        'mean_final_divergence': mean_final_divergence,
        'divergence_half_life': divergence_half_life,
        'sample_rounds': cfg.sample_rounds
    }


def run_avalanche_quick(
    n_pairs: int = 500,
    embedding: str = 'euclidean'
) -> Dict:
    """Quick avalanche run for testing."""
    config = {
        'n_pairs': n_pairs,
        'sample_rounds': list(range(0, 65, 8)),
        'embedding': embedding,
        'seed': 42
    }
    return experiment_2_avalanche(config)


if __name__ == "__main__":
    results = run_avalanche_quick(n_pairs=200)
    print("\nDone!")
