#!/usr/bin/env python3
"""
Deep Per-Bit Divergence Analysis
================================

Investigates whether the anisotropy is exploitable or just a fingerprint.

Author: Bee Davis
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

WORKING_VARS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

def bit_to_word(bit_position: int) -> Tuple[str, int]:
    """Map 256-bit position to SHA-256 word and bit within word."""
    word_idx = bit_position // 32
    bit_in_word = bit_position % 32
    return WORKING_VARS[word_idx], bit_in_word


def analyze_bit_divergence_rates(results: dict):
    """Deep analysis of bit_divergence_rates."""
    
    rates = results.get('bit_divergence_rates', [])
    slow_bits = [int(b) for b in results.get('slow_bits', [])]
    config = results.get('config', {})
    
    print("=" * 70)
    print("  BIT DIVERGENCE RATE ANALYSIS")
    print("=" * 70)
    print(f"\n  Total bit positions tested: {len(rates)}")
    print(f"  Anisotropy score: {results.get('anisotropy_score', 'N/A'):.4f}")
    
    if not rates:
        print("  No rate data available.")
        return
    
    rates = np.array(rates)
    
    # Basic statistics
    print(f"\n  Rate statistics:")
    print(f"    Mean:   {np.mean(rates):.6f}")
    print(f"    Std:    {np.std(rates):.6f}")
    print(f"    Min:    {np.min(rates):.6f}")
    print(f"    Max:    {np.max(rates):.6f}")
    print(f"    Range:  {np.max(rates) - np.min(rates):.6f}")
    print(f"    CV:     {np.std(rates) / np.mean(rates):.4f} (coefficient of variation)")
    
    # The 440 rates - let's see if we can figure out the mapping
    # 10k pairs / 440 ≈ 22.7 samples per bit position
    # OR: 440 could be samples_per_round × unique_bits
    
    # Check if we have bit position info in config
    bit_positions = config.get('bit_positions', None)
    
    if bit_positions is None:
        # Reconstruct: assume rates are per unique bit tested
        # With 10k pairs and uniform sampling across 512 input bits...
        print(f"\n  Mapping interpretation:")
        print(f"    440 rates / 10000 pairs = maybe grouped by bit?")
        print(f"    512 bits × ~20 samples each would give ~10k")
    
    # Analyze the slow bits
    print("\n" + "=" * 70)
    print("  SLOW BITS DETAILED ANALYSIS")
    print("=" * 70)
    
    print(f"\n  Total slow bits: {len(slow_bits)}")
    print(f"  Slow bit positions: {slow_bits}")
    
    # Group by word
    by_word = defaultdict(list)
    for bit in slow_bits:
        word, bit_in_word = bit_to_word(bit)
        by_word[word].append(bit_in_word)
    
    print("\n  Distribution by SHA-256 working variable:")
    print("  " + "-" * 50)
    
    total = len(slow_bits)
    expected_per_word = total / 8
    
    chi_sq = 0
    for word in WORKING_VARS:
        count = len(by_word[word])
        pct = count / total * 100 if total > 0 else 0
        bar_len = int(count / max(1, max(len(v) for v in by_word.values())) * 20)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        
        bits_str = ', '.join(map(str, sorted(by_word[word]))) if by_word[word] else '-'
        print(f"    {word}: {bar} {count:2d} ({pct:5.1f}%)  bits: [{bits_str}]")
        
        chi_sq += (count - expected_per_word) ** 2 / expected_per_word if expected_per_word > 0 else 0
    
    print(f"\n  Chi-squared: {chi_sq:.2f} (critical value ~14.07 for p=0.05, df=7)")
    if chi_sq > 14.07:
        print("  ⚠️  SIGNIFICANT clustering - slow bits are NOT uniformly distributed")
    else:
        print("  ✓  No significant clustering detected")
    
    # Analyze bit positions within words
    print("\n" + "-" * 50)
    print("  Bit position patterns within words:")
    
    all_positions = [bit % 32 for bit in slow_bits]
    
    # Check for rotation-related patterns
    # SHA-256 Σ₀ uses rotations by 2, 13, 22
    # SHA-256 Σ₁ uses rotations by 6, 11, 25
    sigma0_bits = {2, 13, 22, 32-2, 32-13, 32-22}  # Rotation positions and complements
    sigma1_bits = {6, 11, 25, 32-6, 32-11, 32-25}
    
    sigma0_overlap = len([p for p in all_positions if p in sigma0_bits])
    sigma1_overlap = len([p for p in all_positions if p in sigma1_bits])
    
    print(f"    Positions in Σ₀ rotation set (2,13,22): {sigma0_overlap}")
    print(f"    Positions in Σ₁ rotation set (6,11,25): {sigma1_overlap}")
    
    # High vs low bit positions
    high_bits = len([p for p in all_positions if p >= 24])
    low_bits = len([p for p in all_positions if p < 8])
    mid_bits = len([p for p in all_positions if 8 <= p < 24])
    
    print(f"\n    High bits (24-31): {high_bits}")
    print(f"    Mid bits (8-23):   {mid_bits}")
    print(f"    Low bits (0-7):    {low_bits}")
    
    # Exploitability analysis
    print("\n" + "=" * 70)
    print("  EXPLOITABILITY ASSESSMENT")
    print("=" * 70)
    
    # Key questions:
    # 1. Are slow bits concentrated in specific functional regions?
    # 2. Is the effect size large enough to be useful?
    # 3. Does the pattern suggest a structural weakness?
    
    anisotropy = results.get('anisotropy_score', 0)
    
    print(f"""
  1. EFFECT SIZE:
     Anisotropy score: {anisotropy:.4f}
     {"⚠️  HIGH - notable deviation from isotropy" if anisotropy > 0.15 else "Moderate anisotropy"}
     
  2. CONCENTRATION:
     Words a,b,c contain {len(by_word['a']) + len(by_word['b']) + len(by_word['c'])}/{len(slow_bits)} slow bits
     This is the Maj(a,b,c) computation region of SHA-256.
     Words e,f,g contain {len(by_word['e']) + len(by_word['f']) + len(by_word['g'])}/{len(slow_bits)} slow bits
     This is the Ch(e,f,g) computation region.
     
  3. FUNCTIONAL ANALYSIS:
     The Maj function: Maj(a,b,c) = (a AND b) XOR (a AND c) XOR (b AND c)
     The Ch function:  Ch(e,f,g) = (e AND f) XOR (NOT e AND g)
     
     Slow bits concentrated in Maj inputs suggest the majority function
     may have slight bias in how it propagates changes.
     """)
    
    # Final verdict
    print("=" * 70)
    print("  VERDICT")
    print("=" * 70)
    
    if anisotropy < 0.10:
        verdict = "FINGERPRINT ONLY"
        explanation = """
     The anisotropy is too small to be practically exploitable.
     This is likely a geometric fingerprint of SHA-256's structure
     but not a cryptographic weakness."""
    elif len(by_word['a']) > len(slow_bits) * 0.4:
        verdict = "STRUCTURAL SIGNATURE"
        explanation = f"""
     Strong concentration in word 'a' ({len(by_word['a'])}/{len(slow_bits)} slow bits).
     Word 'a' receives the most complex update each round (Σ₀ + Maj + temp).
     This is expected from the compression function structure.
     
     Likely NOT exploitable because:
     - Effect is on internal state, not output
     - By round 64, all words have been in position 'a' many times
     - The "slow" bits still diverge, just slightly slower"""
    else:
        verdict = "NEEDS FURTHER INVESTIGATION"
        explanation = """
     Unexpected pattern detected.
     The slow bits don't follow the expected structural distribution.
     Recommend additional experiments with different input classes."""
    
    print(f"\n  {verdict}")
    print(explanation)
    
    # Recommendation
    print("\n  RECOMMENDED NEXT STEPS:")
    if anisotropy > 0.15:
        print("  1. Run with 50k+ pairs to verify stability")
        print("  2. Test if slow bits are consistent across different input classes")
        print("  3. Trace the exact round where divergence slows for these bits")
        print("  4. Check if the pattern persists in full SHA-256 (not just single block)")


def main():
    results_dir = Path(__file__).parent.parent / "results"
    pkl_file = results_dir / "exp2_euclidean_10000_20251206_231713.pkl"
    
    if not pkl_file.exists():
        print(f"File not found: {pkl_file}")
        return
    
    print(f"Loading: {pkl_file.name}\n")
    
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)
    
    analyze_bit_divergence_rates(results)


if __name__ == "__main__":
    main()
