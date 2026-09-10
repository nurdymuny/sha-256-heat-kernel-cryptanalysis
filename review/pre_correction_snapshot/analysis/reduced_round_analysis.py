#!/usr/bin/env python3
"""
Reduced-Round SHA-256 Analysis
==============================
Find exactly when slow bits dissolve into noise.

Test rounds: 16, 20, 24, 28, 32, 40, 48, 56, 64

Key question: At which round does anisotropy disappear?

Author: Bee Davis
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

import sys
sys.path.insert(0, '.')
from src.sha256.core import InstrumentedSHA256
from src.sha256.input_generator import InputGenerator
from src.embeddings.euclidean import EuclideanEmbedding
from src.analysis.laplacian import LaplacianConstructor
from src.analysis.heat_kernel import HeatKernelSolver


@dataclass 
class RoundAnalysisResult:
    """Results for a single round analysis."""
    round_num: int
    anisotropy_score: float
    n_slow_bits: int
    slow_bit_positions: List[int]
    mean_divergence: float
    std_divergence: float
    per_bit_rates: np.ndarray
    spectral_gap: float


def analyze_at_round(
    trajectories_a: List,
    trajectories_b: List, 
    round_num: int,
    embedding,
    n_pairs: int
) -> RoundAnalysisResult:
    """Analyze divergence at a specific round."""
    
    # Compute per-bit divergence rates
    # Group pairs by which bit was flipped
    bit_divergences = {}
    
    for i in range(n_pairs):
        traj_a = trajectories_a[i]
        traj_b = trajectories_b[i]
        
        # Get state at round_num
        state_a = None
        state_b = None
        for s in traj_a.states:
            if s.round_num == round_num:
                state_a = s.state_vector
                break
        for s in traj_b.states:
            if s.round_num == round_num:
                state_b = s.state_vector
                break
                
        if state_a is None or state_b is None:
            continue
            
        p_a = embedding.embed(state_a)
        p_b = embedding.embed(state_b)
        dist = embedding.distance(p_a, p_b)
        
        # Get flipped bit from pair metadata (stored in trajectory)
        # Since we're using sequential bit flips, pair i flipped bit i % 512
        flipped_bit = i % 512
        
        if flipped_bit not in bit_divergences:
            bit_divergences[flipped_bit] = []
        bit_divergences[flipped_bit].append(dist)
    
    # Compute per-bit mean divergence rates
    per_bit_rates = np.zeros(512)
    for bit, dists in bit_divergences.items():
        per_bit_rates[bit] = np.mean(dists)
    
    # Normalize by round number to get rate
    if round_num > 0:
        per_bit_rates = per_bit_rates / round_num
    
    # Compute anisotropy (coefficient of variation of rates)
    mean_rate = np.mean(per_bit_rates[per_bit_rates > 0])
    std_rate = np.std(per_bit_rates[per_bit_rates > 0])
    anisotropy = std_rate / mean_rate if mean_rate > 0 else 0
    
    # Find slow bits (bottom 10% of divergence rates)
    nonzero_rates = per_bit_rates[per_bit_rates > 0]
    if len(nonzero_rates) > 0:
        threshold = np.percentile(nonzero_rates, 10)
        slow_bits = [i for i in range(512) if 0 < per_bit_rates[i] <= threshold]
    else:
        slow_bits = []
    
    # Compute spectral gap on state cloud
    constructor = LaplacianConstructor(embedding, k_neighbors=30)
    solver = HeatKernelSolver(n_eigenvalues=50)
    
    # Get all states at this round
    states = []
    for traj in trajectories_a[:500]:
        for s in traj.states:
            if s.round_num == round_num:
                states.append(s.state_vector)
                break
    
    if len(states) > 50:
        points = np.array([embedding.embed(s) for s in states])
        lap_result = constructor.build_from_points(points)
        hk_result = solver.solve(lap_result.laplacian)
        spectral_gap = hk_result.spectral_gap
    else:
        spectral_gap = 0.0
    
    return RoundAnalysisResult(
        round_num=round_num,
        anisotropy_score=anisotropy,
        n_slow_bits=len(slow_bits),
        slow_bit_positions=slow_bits,
        mean_divergence=mean_rate,
        std_divergence=std_rate,
        per_bit_rates=per_bit_rates,
        spectral_gap=spectral_gap
    )


def run_reduced_round_analysis(
    n_pairs: int = 2000,
    test_rounds: List[int] = None,
    output_file: str = "results/reduced_round_analysis.json"
):
    """
    Run analysis at multiple rounds to find when slow bits dissolve.
    """
    if test_rounds is None:
        test_rounds = [8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64]
    
    print("=" * 70)
    print("  REDUCED-ROUND SHA-256 ANALYSIS")
    print("  Finding when slow bits dissolve")
    print("=" * 70)
    
    # Generate Hamming-1 pairs with known bit positions
    print(f"\n[1/3] Generating {n_pairs} Hamming-1 pairs...")
    gen = InputGenerator(seed=42)
    pairs = gen.hamming_pairs(n_pairs)
    
    messages_a = [p.message_a for p in pairs]
    messages_b = [p.message_b for p in pairs]
    
    # Hash with all rounds captured
    print("[2/3] Computing SHA-256 trajectories...")
    sha = InstrumentedSHA256(sample_rounds=test_rounds)
    trajectories_a = sha.hash_batch(messages_a)
    trajectories_b = sha.hash_batch(messages_b)
    
    # Analyze at each round
    print("[3/3] Analyzing divergence at each round...")
    embedding = EuclideanEmbedding()
    
    results = []
    for r in test_rounds:
        print(f"  Round {r}...")
        result = analyze_at_round(
            trajectories_a, trajectories_b, r, embedding, n_pairs
        )
        results.append(result)
    
    # Print summary
    print("\n" + "=" * 70)
    print("  RESULTS: ANISOTROPY BY ROUND")
    print("=" * 70)
    print(f"\n  {'Round':<8} {'Anisotropy':<12} {'Slow Bits':<12} {'Spectral Gap':<14} {'Status'}")
    print("  " + "-" * 60)
    
    for r in results:
        # Status based on anisotropy threshold
        if r.anisotropy_score > 0.20:
            status = "🔴 HIGH ANISOTROPY"
        elif r.anisotropy_score > 0.10:
            status = "🟡 MODERATE"
        elif r.anisotropy_score > 0.05:
            status = "🟢 LOW"
        else:
            status = "✅ ISOTROPIC"
            
        print(f"  {r.round_num:<8} {r.anisotropy_score:<12.4f} {r.n_slow_bits:<12} {r.spectral_gap:<14.4f} {status}")
    
    # Find transition point
    print("\n  " + "-" * 60)
    transition_round = None
    for i, r in enumerate(results):
        if r.anisotropy_score < 0.10:
            transition_round = r.round_num
            if i > 0:
                prev = results[i-1]
                print(f"\n  📍 TRANSITION: Anisotropy drops from {prev.anisotropy_score:.3f} to {r.anisotropy_score:.3f}")
                print(f"     between rounds {prev.round_num} and {r.round_num}")
            break
    
    if transition_round is None:
        print("\n  ⚠️  Anisotropy persists through all tested rounds!")
    
    # Track slow bit evolution
    print("\n" + "=" * 70)
    print("  SLOW BIT EVOLUTION")
    print("=" * 70)
    
    # Find bits that are consistently slow
    bit_slow_count = np.zeros(512)
    for r in results:
        for bit in r.slow_bit_positions:
            bit_slow_count[bit] += 1
    
    persistent_slow = [i for i in range(512) if bit_slow_count[i] >= len(results) // 2]
    print(f"\n  Bits slow in ≥50% of rounds: {len(persistent_slow)}")
    
    if persistent_slow:
        # Map to SHA-256 words
        word_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        for bit in persistent_slow[:20]:  # Show first 20
            if bit < 256:
                # State space bit
                word = bit // 32
                bit_pos = bit % 32
                print(f"    Bit {bit}: state word {word_names[word]}[{bit_pos}]")
            else:
                # Message bit
                print(f"    Bit {bit}: message bit {bit - 256}")
    
    # Save results
    output = {
        'test_rounds': test_rounds,
        'n_pairs': n_pairs,
        'results': [
            {
                'round': r.round_num,
                'anisotropy_score': r.anisotropy_score,
                'n_slow_bits': r.n_slow_bits,
                'slow_bits': r.slow_bit_positions[:50],  # Top 50
                'mean_divergence': r.mean_divergence,
                'std_divergence': r.std_divergence,
                'spectral_gap': r.spectral_gap
            }
            for r in results
        ],
        'transition_round': transition_round,
        'persistent_slow_bits': persistent_slow[:50]
    }
    
    Path(output_file).parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Results saved to: {output_file}")
    
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reduced-Round SHA-256 Analysis")
    parser.add_argument('--n-pairs', type=int, default=2000, help='Number of pairs')
    parser.add_argument('--output', type=str, default='results/reduced_round_analysis.json')
    args = parser.parse_args()
    
    run_reduced_round_analysis(args.n_pairs, output_file=args.output)
