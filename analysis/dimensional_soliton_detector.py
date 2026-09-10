#!/usr/bin/env python3
"""
DIMENSIONAL SOLITON DETECTOR
============================

Hypothesis: SHA-256 maintains a low-dimensional "tube" (19-D) through 24 rounds
when starting from slow-bit perturbations. A true random oracle should expand
to ~72 effective dimensions by round 4.

This script:
1. Uses correct Participation Ratio metric for effective dimension
2. Scans ALL rounds (0-64) to find the "snap" point
3. Compares slow-bit trajectories vs random trajectories
4. Identifies the "event horizon" where the soliton collapses

Author: Bee Davis
"""

import numpy as np
from scipy.linalg import eigh
from typing import Dict, List, Tuple
import sys
sys.path.insert(0, '.')

from src.sha256.core import InstrumentedSHA256


def effective_dimension_participation_ratio(eigenvalues: np.ndarray) -> float:
    """
    Participation Ratio = (Σλ)² / Σλ²
    
    This is the correct metric for effective dimension.
    For n equal eigenvalues: PR = n
    For 1 dominant eigenvalue: PR = 1
    """
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if len(eigenvalues) == 0:
        return 0
    s1 = np.sum(eigenvalues)
    s2 = np.sum(eigenvalues**2)
    return (s1**2) / s2 if s2 > 0 else 0


def effective_dimension_stable_rank(eigenvalues: np.ndarray) -> float:
    """
    Stable Rank = Σλ / max(λ)
    
    Alternative metric, less sensitive to noise.
    """
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    if len(eigenvalues) == 0:
        return 0
    return np.sum(eigenvalues) / np.max(eigenvalues) if np.max(eigenvalues) > 0 else 0


def compute_covariance_spectrum(states: np.ndarray) -> np.ndarray:
    """
    Compute eigenvalue spectrum of covariance matrix.
    """
    # Center the data
    centered = states - np.mean(states, axis=0)
    
    # Covariance matrix
    cov = np.cov(centered, rowvar=False)
    
    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
    
    return eigenvalues


def generate_slow_bit_cloud(n_samples: int, slow_bits: List[int], seed: int = 42) -> List[bytes]:
    """
    Generate messages that differ only in slow bit positions.
    This creates a cloud constrained to the slow-bit subspace.
    """
    np.random.seed(seed)
    base_msg = np.random.bytes(55)
    
    messages = []
    for i in range(n_samples):
        msg = bytearray(base_msg)
        # Flip random subset of slow bits
        for bit in slow_bits:
            if np.random.rand() > 0.5:
                byte_idx = bit // 8
                bit_idx = bit % 8
                if byte_idx < len(msg):
                    msg[byte_idx] ^= (1 << bit_idx)
        messages.append(bytes(msg))
    
    return messages


def generate_random_cloud(n_samples: int, seed: int = 42) -> List[bytes]:
    """
    Generate completely random messages.
    """
    np.random.seed(seed)
    return [np.random.bytes(55) for _ in range(n_samples)]


def scan_dimensional_evolution(messages: List[bytes], 
                               sample_rounds: List[int]) -> Dict[int, Dict]:
    """
    Track effective dimension through all specified rounds.
    """
    sha = InstrumentedSHA256(sample_rounds=sample_rounds)
    
    # Collect states at each round
    round_states = {r: [] for r in sample_rounds}
    
    for msg in messages:
        traj = sha.hash_with_trajectory(msg)
        for state in traj.states:
            if state.round_num in round_states:
                round_states[state.round_num].append(state.state_vector)
    
    # Compute metrics for each round
    results = {}
    for r in sample_rounds:
        states = np.array(round_states[r])
        if len(states) < 10:
            continue
            
        eigenvalues = compute_covariance_spectrum(states)
        
        pr = effective_dimension_participation_ratio(eigenvalues)
        sr = effective_dimension_stable_rank(eigenvalues)
        
        # Top eigenvalue dominance
        top_ratio = eigenvalues[0] / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else 0
        
        results[r] = {
            'participation_ratio': pr,
            'stable_rank': sr,
            'top_eigenvalue_ratio': top_ratio,
            'eigenvalues': eigenvalues[:20],  # Top 20
            'n_samples': len(states)
        }
    
    return results


def main():
    print("=" * 80)
    print("  DIMENSIONAL SOLITON DETECTOR")
    print("  Finding the 'Event Horizon' of SHA-256")
    print("=" * 80)
    
    # Slow bits from heat kernel analysis
    SLOW_BITS = [1, 5, 7, 9, 10, 14, 16, 19, 21, 26, 28, 31, 39, 45, 53, 58, 59, 67, 74]
    
    n_samples = 100
    
    # Scan every 2 rounds from 0 to 64
    sample_rounds = list(range(0, 65, 2))
    
    # Baseline: Random cloud
    print("\n[1] BASELINE: Random 256-bit vectors (no SHA-256)")
    print("-" * 80)
    np.random.seed(42)
    random_bits = np.random.randint(0, 2, size=(n_samples, 256))
    baseline_eigs = compute_covariance_spectrum(random_bits.astype(float))
    baseline_pr = effective_dimension_participation_ratio(baseline_eigs)
    baseline_sr = effective_dimension_stable_rank(baseline_eigs)
    print(f"    Participation Ratio: {baseline_pr:.2f}")
    print(f"    Stable Rank: {baseline_sr:.2f}")
    print(f"    Expected for random: ~{n_samples * 0.72:.0f} (72% of n_samples)")
    
    # Test 1: Slow-bit constrained cloud
    print("\n[2] SLOW-BIT CLOUD through SHA-256")
    print("-" * 80)
    print(f"    Slow bits: {SLOW_BITS[:10]}... ({len(SLOW_BITS)} total)")
    print(f"    Input cloud dimension: ~{len(SLOW_BITS)} (constrained to slow-bit subspace)")
    
    slow_messages = generate_slow_bit_cloud(n_samples, SLOW_BITS)
    slow_results = scan_dimensional_evolution(slow_messages, sample_rounds)
    
    print(f"\n    {'Round':<8} {'Part. Ratio':<14} {'Stable Rank':<14} {'Top λ %':<12} {'Status'}")
    print("    " + "-" * 56)
    
    snap_round = None
    for r in sorted(slow_results.keys()):
        res = slow_results[r]
        pr = res['participation_ratio']
        sr = res['stable_rank']
        top = res['top_eigenvalue_ratio'] * 100
        
        # Detect the "snap"
        if pr > 40 and snap_round is None and r > 0:
            snap_round = r
            status = "🔴 SNAP!"
        elif pr < 25:
            status = "⚠️ TUBE"
        else:
            status = "✓"
        
        bar = "█" * min(int(pr / 2), 40)
        print(f"    {r:<8} {pr:<14.2f} {sr:<14.2f} {top:<12.1f} {status} {bar}")
    
    # Test 2: Random cloud (control)
    print("\n[3] RANDOM CLOUD through SHA-256 (Control)")
    print("-" * 80)
    
    random_messages = generate_random_cloud(n_samples)
    random_results = scan_dimensional_evolution(random_messages, sample_rounds)
    
    print(f"\n    {'Round':<8} {'Part. Ratio':<14} {'Stable Rank':<14} {'Top λ %':<12}")
    print("    " + "-" * 48)
    
    for r in sorted(random_results.keys()):
        res = random_results[r]
        pr = res['participation_ratio']
        sr = res['stable_rank']
        top = res['top_eigenvalue_ratio'] * 100
        
        bar = "█" * min(int(pr / 2), 40)
        print(f"    {r:<8} {pr:<14.2f} {sr:<14.2f} {top:<12.1f} {bar}")
    
    # Comparison
    print("\n[4] DIMENSIONAL SOLITON ANALYSIS")
    print("=" * 80)
    
    if snap_round:
        print(f"\n    🔴 DIMENSIONAL SNAP DETECTED AT ROUND {snap_round}")
        print(f"       Before: PR ≈ {slow_results[snap_round-2]['participation_ratio']:.1f}")
        print(f"       After:  PR ≈ {slow_results[snap_round]['participation_ratio']:.1f}")
        print(f"\n       The 'Soliton Tube' collapses at round {snap_round}.")
    else:
        # Check if tube persists
        final_pr = slow_results[max(slow_results.keys())]['participation_ratio']
        if final_pr < 30:
            print(f"\n    ⚠️  DIMENSIONAL SOLITON PERSISTS THROUGH ROUND 64!")
            print(f"       Final PR: {final_pr:.1f} (expected ~72 for random)")
            print(f"       This indicates a structural invariant in SHA-256.")
        else:
            print(f"\n    ✓ Dimensional expansion occurred normally.")
    
    # Compare slow vs random at key rounds
    print("\n    Round-by-Round Comparison (Slow vs Random):")
    print(f"    {'Round':<8} {'Slow PR':<14} {'Random PR':<14} {'Ratio':<12} {'Meaning'}")
    print("    " + "-" * 56)
    
    for r in [0, 8, 16, 24, 32, 48, 64]:
        if r in slow_results and r in random_results:
            slow_pr = slow_results[r]['participation_ratio']
            rand_pr = random_results[r]['participation_ratio']
            ratio = slow_pr / rand_pr if rand_pr > 0 else 0
            
            if ratio < 0.5:
                meaning = "🔴 TUBE (slow << random)"
            elif ratio < 0.8:
                meaning = "⚠️ Constrained"
            else:
                meaning = "✓ Normal"
            
            print(f"    {r:<8} {slow_pr:<14.2f} {rand_pr:<14.2f} {ratio:<12.2f} {meaning}")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("  VERDICT")
    print("=" * 80)
    
    # Check round 24 specifically
    if 24 in slow_results and 24 in random_results:
        slow_24 = slow_results[24]['participation_ratio']
        rand_24 = random_results[24]['participation_ratio']
        
        if slow_24 < rand_24 * 0.5:
            print(f"""
    🔴 DIMENSIONAL SOLITON CONFIRMED
    
    At Round 24:
      - Slow-bit cloud: PR = {slow_24:.1f}
      - Random cloud:   PR = {rand_24:.1f}
      - Ratio: {slow_24/rand_24:.2f}x (should be ~1.0 for good mixing)
    
    The slow-bit perturbation cloud propagates through {max(slow_results.keys())} rounds
    while maintaining dimensional confinement. This is a structural invariant
    that should not exist in a random oracle.
    
    INTERPRETATION:
    SHA-256 has a "low-dimensional attractor" along the slow-bit manifold.
    Input variations in this subspace do not fully avalanche.
            """)
        else:
            print(f"""
    ✓ NO DIMENSIONAL SOLITON
    
    At Round 24:
      - Slow-bit cloud: PR = {slow_24:.1f}
      - Random cloud:   PR = {rand_24:.1f}
      - Ratio: {slow_24/rand_24:.2f}x
    
    Both clouds expand to similar dimensions. The avalanche effect
    is working as expected.
            """)


if __name__ == "__main__":
    main()
