#!/usr/bin/env python3
"""
Per-Bit Rate Correlation Analysis
=================================

Instead of comparing binary slow/fast classification,
compare the actual per-bit divergence RATES between sample sizes.

If the same bits are consistently slowest/fastest, the RANKING should
be stable even if the binary threshold changes.

Author: Bee Davis
"""

import pickle
import numpy as np
from pathlib import Path
from scipy import stats


def main():
    results_dir = Path(__file__).parent.parent / "results"
    
    pkl_10k = results_dir / "exp2_euclidean_10000_20251206_231713.pkl"
    pkl_50k = results_dir / "exp2_euclidean_50000_20251206_232417.pkl"
    
    with open(pkl_10k, 'rb') as f:
        results_10k = pickle.load(f)
    
    with open(pkl_50k, 'rb') as f:
        results_50k = pickle.load(f)
    
    print("=" * 70)
    print("  PER-BIT DIVERGENCE RATE ANALYSIS")
    print("=" * 70)
    
    # Get the per-bit rates
    rates_10k = np.array(results_10k.get('bit_divergence_rates', []))
    rates_50k = np.array(results_50k.get('bit_divergence_rates', []))
    
    print(f"\n  10k rates: {len(rates_10k)} values")
    print(f"  50k rates: {len(rates_50k)} values")
    
    if len(rates_10k) != len(rates_50k):
        print("  ⚠️  Different number of rate values - can't directly compare")
        # Still analyze each independently
    
    # Check if rates are per-unique-bit or per-pair
    # If 440 values, that's likely ~440 unique bits tested
    # If rates are per-bit, we can correlate them
    
    if len(rates_10k) == len(rates_50k):
        print("\n" + "-" * 70)
        print("  Rate Correlation Analysis")
        print("-" * 70)
        
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(rates_10k, rates_50k)
        
        # Spearman rank correlation (more robust to outliers)
        spearman_r, spearman_p = stats.spearmanr(rates_10k, rates_50k)
        
        print(f"\n  Pearson correlation:  r = {pearson_r:.4f}, p = {pearson_p:.2e}")
        print(f"  Spearman correlation: ρ = {spearman_r:.4f}, p = {spearman_p:.2e}")
        
        if spearman_r > 0.7:
            print("\n  ✅ STRONG CORRELATION: Per-bit rates are stable across sample sizes")
            print("     The same bits are consistently slow/fast.")
        elif spearman_r > 0.3:
            print("\n  ⚠️  MODERATE CORRELATION: Some stability in per-bit rates")
        else:
            print("\n  ❌ WEAK CORRELATION: Per-bit rates vary significantly between runs")
            print("     The slow/fast classification may be noise.")
        
        # Find the most stable slow bits (low rate in both)
        print("\n" + "-" * 70)
        print("  Most Consistently Slow Bits")
        print("-" * 70)
        
        # Combine rankings
        rank_10k = stats.rankdata(rates_10k)
        rank_50k = stats.rankdata(rates_50k)
        avg_rank = (rank_10k + rank_50k) / 2
        
        # Get top 20 slowest by average rank
        top_slow_idx = np.argsort(avg_rank)[:20]
        
        print("\n  {:>4} {:>10} {:>10} {:>10} {:>10}".format(
            "Bit", "Rate 10k", "Rate 50k", "Rank 10k", "Rank 50k"))
        print("  " + "-" * 48)
        
        for idx in top_slow_idx:
            print("  {:>4} {:>10.4f} {:>10.4f} {:>10.0f} {:>10.0f}".format(
                idx, rates_10k[idx], rates_50k[idx], rank_10k[idx], rank_50k[idx]))
    
    # Analyze slow bit classification stability
    print("\n" + "=" * 70)
    print("  SLOW BIT CLASSIFICATION STABILITY")
    print("=" * 70)
    
    slow_10k = set(int(b) for b in results_10k['slow_bits'])
    slow_50k = set(int(b) for b in results_50k['slow_bits'])
    
    overlap = slow_10k & slow_50k
    jaccard = len(overlap) / len(slow_10k | slow_50k) if slow_10k | slow_50k else 0
    
    print(f"\n  Slow bits at 10k: {sorted(slow_10k)}")
    print(f"  Slow bits at 50k: {sorted(slow_50k)}")
    print(f"\n  Overlap:          {sorted(overlap)}")
    print(f"  Jaccard index:    {jaccard:.3f}")
    
    if jaccard < 0.3:
        print("\n  ⚠️  LOW OVERLAP: The specific slow bits change with sample size")
        print("     This suggests the binary classification threshold is sensitive.")
        print("     The continuous rates may be more informative.")
    
    # Check if overlap bits are in word 'a' and related to Σ₀
    print("\n" + "-" * 70)
    print("  Overlap Bit Analysis (stable across sample sizes)")
    print("-" * 70)
    
    for bit in sorted(overlap):
        word_idx = bit // 32
        bit_in_word = bit % 32
        word = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'][word_idx]
        
        # Σ₀ input bits for this output position (if in word 'a')
        if word == 'a':
            in1 = (bit_in_word + 2) % 32
            in2 = (bit_in_word + 13) % 32
            in3 = (bit_in_word + 22) % 32
            print(f"  Bit {bit:3d} = {word}[{bit_in_word:2d}] ← Σ₀ inputs: [{in1}, {in2}, {in3}]")
        else:
            print(f"  Bit {bit:3d} = {word}[{bit_in_word:2d}]")
    
    # Final assessment
    print("\n" + "=" * 70)
    print("  FINAL ASSESSMENT")
    print("=" * 70)
    
    print("""
  The key question: Is this exploitable or just a fingerprint?
  
  FINGERPRINT indicators:
  - Effect size is consistent (~24% CV at both 10k and 50k)
  - Slow bits concentrated in Maj(a,b,c) region (expected from structure)
  - All bits still diverge fully by round 64
  
  EXPLOITABILITY indicators (would need):
  - Consistent specific bit positions (Jaccard > 0.5)
  - Effect persisting to output hash
  - Ability to predict output bits from input
  
  Current evidence suggests: GEOMETRIC FINGERPRINT
  The heat kernel can detect Σ₀'s algebraic structure as geometric anisotropy,
  but this structure is inherent to the compression function design,
  not a weakness that enables attacks.
  """)


if __name__ == "__main__":
    main()
