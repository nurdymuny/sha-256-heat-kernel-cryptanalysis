#!/usr/bin/env python3
"""
Per-Bit Divergence Analysis for SHA-256 Avalanche Experiment
=============================================================

Analyzes whether the anisotropy is:
1. EXPLOITABLE: Specific bits consistently diverge slower, enabling prediction
2. FINGERPRINT: Statistical artifact with no practical exploitation path

Key metrics:
- Per-bit mean divergence rate
- Per-bit divergence variance (consistency)
- Bit correlation structure (do slow bits cluster?)
- Round-by-round divergence profiles

Author: Bee Davis
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

# SHA-256 working variable names
WORKING_VARS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

def load_experiment_results(filepath: str) -> dict:
    """Load pickled experiment results."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def bit_to_word(bit_position: int) -> Tuple[str, int]:
    """Map 256-bit position to SHA-256 word and bit within word."""
    word_idx = bit_position // 32
    bit_in_word = bit_position % 32
    return WORKING_VARS[word_idx], bit_in_word


def analyze_per_bit_divergence(results: dict) -> Dict:
    """
    Extract and analyze per-bit divergence statistics.
    
    Returns detailed per-bit metrics to assess exploitability.
    """
    # The avalanche experiment tracks divergence curves
    # Shape should be (n_pairs, n_rounds) or similar
    
    analysis = {
        'config': {},
        'per_bit_stats': {},
        'exploitability_assessment': {},
        'round_profiles': {},
    }
    
    # Print what we have in the results
    print("=" * 70)
    print("  Experiment Result Structure")
    print("=" * 70)
    
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: ndarray shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, (list, tuple)):
            print(f"  {key}: {type(value).__name__} len={len(value)}")
            if len(value) > 0 and isinstance(value[0], np.ndarray):
                print(f"       first element: ndarray shape={value[0].shape}")
        elif isinstance(value, dict):
            print(f"  {key}: dict with keys: {list(value.keys())[:5]}...")
        else:
            print(f"  {key}: {type(value).__name__} = {str(value)[:50]}")
    
    print()
    
    # Look for divergence data
    divergence_curves = None
    bit_positions_flipped = None
    
    if 'divergence_curves' in results:
        divergence_curves = results['divergence_curves']
    if 'per_bit_divergence' in results:
        divergence_curves = results['per_bit_divergence']
    if 'bit_divergence' in results:
        divergence_curves = results['bit_divergence']
        
    if 'bit_positions' in results:
        bit_positions_flipped = results['bit_positions']
    if 'flipped_bits' in results:
        bit_positions_flipped = results['flipped_bits']
    
    # Check if we have per-bit data or need to reconstruct
    if divergence_curves is not None and isinstance(divergence_curves, np.ndarray):
        print("=" * 70)
        print("  Divergence Curve Analysis")
        print("=" * 70)
        print(f"  Shape: {divergence_curves.shape}")
        
        if len(divergence_curves.shape) == 2:
            n_pairs, n_rounds = divergence_curves.shape
            print(f"  Pairs: {n_pairs}, Rounds: {n_rounds}")
            
            # Compute per-pair final divergence
            final_divergence = divergence_curves[:, -1]
            
            # If we have bit position data, compute per-bit stats
            if bit_positions_flipped is not None:
                bit_positions_flipped = np.array(bit_positions_flipped)
                compute_per_bit_stats(divergence_curves, bit_positions_flipped, final_divergence)
            else:
                print("\n  No per-bit mapping available.")
                print("  Analyzing overall divergence distribution...\n")
                analyze_divergence_distribution(divergence_curves)
    
    # Check for slow_bits directly
    if 'slow_bits' in results:
        slow_bits = results['slow_bits']
        print("\n" + "=" * 70)
        print("  Slow Bits Analysis (from experiment)")
        print("=" * 70)
        print(f"  Total slow bits: {len(slow_bits)}")
        print(f"  Positions: {slow_bits[:20]}{'...' if len(slow_bits) > 20 else ''}")
        
        # Map to words
        word_counts = {v: 0 for v in WORKING_VARS}
        for bit in slow_bits:
            word, _ = bit_to_word(bit)
            word_counts[word] += 1
        
        print("\n  Distribution by SHA-256 word:")
        for word, count in word_counts.items():
            bar = '█' * count + '░' * (10 - count)
            print(f"    {word}: {bar} {count}")
    
    # Check for anisotropy score
    if 'anisotropy_score' in results:
        print(f"\n  Anisotropy Score: {results['anisotropy_score']:.4f}")
        
    return analysis


def compute_per_bit_stats(divergence_curves: np.ndarray, 
                          bit_positions: np.ndarray,
                          final_divergence: np.ndarray):
    """Compute statistics grouped by which bit was flipped."""
    
    print("\n" + "-" * 60)
    print("  Per-Bit Statistics")
    print("-" * 60)
    
    # Group by bit position
    unique_bits = np.unique(bit_positions)
    print(f"  Unique bits flipped: {len(unique_bits)}")
    
    bit_stats = {}
    for bit in unique_bits:
        mask = bit_positions == bit
        bit_divergences = final_divergence[mask]
        
        if len(bit_divergences) > 0:
            bit_stats[int(bit)] = {
                'mean': float(np.mean(bit_divergences)),
                'std': float(np.std(bit_divergences)),
                'count': int(np.sum(mask)),
                'min': float(np.min(bit_divergences)),
                'max': float(np.max(bit_divergences)),
            }
    
    # Sort by mean divergence
    sorted_bits = sorted(bit_stats.items(), key=lambda x: x[1]['mean'])
    
    print("\n  Slowest-diverging bits (potential vulnerabilities):")
    print("  " + "-" * 56)
    print("  {:>6} {:>6} {:>10} {:>10} {:>10} {:>10}".format(
        "Bit", "Word", "Mean", "Std", "Count", "Concern"))
    print("  " + "-" * 56)
    
    for bit, stats in sorted_bits[:15]:
        word, bit_in_word = bit_to_word(bit)
        concern = "⚠️ LOW" if stats['mean'] < np.median([s['mean'] for s in bit_stats.values()]) * 0.9 else ""
        print("  {:>6} {:>6} {:>10.4f} {:>10.4f} {:>10} {:>10}".format(
            bit, f"{word}[{bit_in_word}]", stats['mean'], stats['std'], stats['count'], concern))
    
    print("\n  Fastest-diverging bits (expected behavior):")
    print("  " + "-" * 56)
    for bit, stats in sorted_bits[-10:]:
        word, bit_in_word = bit_to_word(bit)
        print("  {:>6} {:>6} {:>10.4f} {:>10.4f} {:>10}".format(
            bit, f"{word}[{bit_in_word}]", stats['mean'], stats['std'], stats['count']))
    
    # Exploitability assessment
    print("\n" + "=" * 70)
    print("  EXPLOITABILITY ASSESSMENT")
    print("=" * 70)
    
    means = [s['mean'] for s in bit_stats.values()]
    overall_mean = np.mean(means)
    overall_std = np.std(means)
    cv = overall_std / overall_mean  # Coefficient of variation
    
    # Bits significantly below mean
    threshold = overall_mean - 2 * overall_std
    vulnerable_bits = [b for b, s in bit_stats.items() if s['mean'] < threshold]
    
    print(f"\n  Overall divergence mean: {overall_mean:.4f}")
    print(f"  Overall divergence std:  {overall_std:.4f}")
    print(f"  Coefficient of variation: {cv:.4f}")
    print(f"  Bits below 2σ threshold: {len(vulnerable_bits)}")
    
    if cv < 0.05:
        print("\n  ✅ VERDICT: Low anisotropy - likely just statistical noise")
        print("     The divergence rates are nearly uniform across all bits.")
    elif cv < 0.15:
        print("\n  ⚠️  VERDICT: Moderate anisotropy - fingerprint detected")
        print("     Some bits diverge slower, but effect size may be too small to exploit.")
        print("     Would need many samples to distinguish bit positions.")
    else:
        print("\n  🚨 VERDICT: High anisotropy - potentially exploitable")
        print("     Significant variation in divergence rates.")
        print(f"     {len(vulnerable_bits)} bits show notably slower divergence.")
        print("     Further investigation recommended.")
    
    if vulnerable_bits:
        print(f"\n  Vulnerable bit positions: {sorted(vulnerable_bits)}")
        
        # Check if they cluster in specific words
        word_vuln = {v: 0 for v in WORKING_VARS}
        for bit in vulnerable_bits:
            word, _ = bit_to_word(bit)
            word_vuln[word] += 1
        
        print("  Clustering by SHA-256 word:")
        for word, count in word_vuln.items():
            if count > 0:
                print(f"    {word}: {count} vulnerable bits")


def analyze_divergence_distribution(divergence_curves: np.ndarray):
    """Analyze the overall divergence distribution when per-bit data unavailable."""
    
    n_pairs, n_rounds = divergence_curves.shape
    
    print("  Round-by-round divergence:")
    print("  " + "-" * 40)
    
    for r in range(n_rounds):
        round_div = divergence_curves[:, r]
        mean = np.mean(round_div)
        std = np.std(round_div)
        print(f"    Round {r:2d}: mean={mean:.4f}, std={std:.4f}")
    
    # Final round analysis
    final = divergence_curves[:, -1]
    
    print("\n  Final round distribution:")
    print(f"    Mean:   {np.mean(final):.4f}")
    print(f"    Std:    {np.std(final):.4f}")
    print(f"    Min:    {np.min(final):.4f}")
    print(f"    Max:    {np.max(final):.4f}")
    print(f"    Median: {np.median(final):.4f}")
    
    # Check for bimodality
    from scipy import stats as scipy_stats
    
    # Kurtosis - high positive = heavy tails, negative = light tails
    kurt = scipy_stats.kurtosis(final)
    skew = scipy_stats.skew(final)
    
    print(f"\n  Distribution shape:")
    print(f"    Skewness: {skew:.4f} (0 = symmetric)")
    print(f"    Kurtosis: {kurt:.4f} (0 = normal, >0 = heavy tails)")
    
    if abs(skew) > 0.5:
        print("    ⚠️  Distribution is skewed - may indicate structure")
    if kurt > 1:
        print("    ⚠️  Heavy tails - outliers present")


def main():
    """Main analysis entry point."""
    
    results_dir = Path(__file__).parent.parent / "results"
    pkl_files = list(results_dir.glob("exp2_*.pkl"))
    
    if not pkl_files:
        print("No experiment 2 pickle files found in results/")
        print("Looking for:", results_dir / "exp2_*.pkl")
        return
    
    # Use the most recent
    pkl_file = sorted(pkl_files)[-1]
    print(f"Loading: {pkl_file.name}\n")
    
    try:
        results = load_experiment_results(pkl_file)
        analyze_per_bit_divergence(results)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        print("\nTrying to inspect raw structure...")
        
        import pickle
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Type: {type(data)}")
        if isinstance(data, dict):
            for k, v in data.items():
                print(f"  {k}: {type(v)}")


if __name__ == "__main__":
    main()
