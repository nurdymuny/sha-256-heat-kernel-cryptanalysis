#!/usr/bin/env python3
"""
Σ₀ Rotation Chain Analysis
==========================

Analyzes whether slow bits form chains under the Σ₀ rotation structure.
Σ₀(a) = ROTR²(a) ⊕ ROTR¹³(a) ⊕ ROTR²²(a)

If slow bits are algebraically related via rotations, this indicates
structure in how XOR cancellation occurs across the three rotation terms.

Author: Bee Davis
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Set, List, Dict, Tuple
from collections import defaultdict

# SHA-256 Σ₀ rotation constants
SIGMA0_ROTATIONS = [2, 13, 22]

# SHA-256 Σ₁ rotation constants (for comparison)
SIGMA1_ROTATIONS = [6, 11, 25]

WORKING_VARS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


def rotate_position(bit: int, rotation: int, word_size: int = 32) -> int:
    """Compute where a bit position rotates to."""
    return (bit + rotation) % word_size


def inverse_rotate_position(bit: int, rotation: int, word_size: int = 32) -> int:
    """Compute where a bit position rotates from."""
    return (bit - rotation) % word_size


def find_rotation_chains(slow_bits: Set[int], rotations: List[int]) -> Dict[int, List[int]]:
    """
    Find chains of slow bits connected by rotation relationships.
    
    A chain is a set of bits where each bit rotates to another slow bit.
    """
    chains = defaultdict(list)
    visited = set()
    
    for bit in sorted(slow_bits):
        if bit in visited:
            continue
            
        # Find all bits connected to this one via rotations
        chain = [bit]
        visited.add(bit)
        
        # Forward rotation
        current = bit
        for _ in range(32):  # Max iterations
            found_next = False
            for rot in rotations:
                next_bit = rotate_position(current, rot)
                if next_bit in slow_bits and next_bit not in visited:
                    chain.append(next_bit)
                    visited.add(next_bit)
                    current = next_bit
                    found_next = True
                    break
            if not found_next:
                break
        
        # Backward rotation
        current = bit
        for _ in range(32):
            found_prev = False
            for rot in rotations:
                prev_bit = inverse_rotate_position(current, rot)
                if prev_bit in slow_bits and prev_bit not in visited:
                    chain.insert(0, prev_bit)
                    visited.add(prev_bit)
                    current = prev_bit
                    found_prev = True
                    break
            if not found_prev:
                break
        
        if len(chain) > 1:
            chains[min(chain)] = chain
        else:
            chains[bit] = chain
    
    return dict(chains)


def analyze_rotation_structure(slow_bits_10k: List[int], slow_bits_50k: List[int]):
    """Compare slow bits between sample sizes and analyze rotation relationships."""
    
    print("=" * 70)
    print("  Σ₀ ROTATION CHAIN ANALYSIS")
    print("=" * 70)
    
    # Convert to sets for easier analysis
    set_10k = set(slow_bits_10k)
    set_50k = set(slow_bits_50k)
    
    print("\n  Sample size comparison:")
    print(f"    10k pairs: {len(set_10k)} slow bits")
    print(f"    50k pairs: {len(set_50k)} slow bits")
    
    overlap = set_10k & set_50k
    print(f"    Overlap:   {len(overlap)} bits")
    print(f"    Overlap positions: {sorted(overlap)}")
    
    only_10k = set_10k - set_50k
    only_50k = set_50k - set_10k
    print(f"\n    Only in 10k: {sorted(only_10k)}")
    print(f"    Only in 50k: {sorted(only_50k)}")
    
    # Analyze rotation relationships in each set
    print("\n" + "-" * 70)
    print("  Rotation-linked pairs in 50k slow bits:")
    print("-" * 70)
    
    # For each slow bit, check if any of its rotated positions are also slow
    rotation_links = []
    for bit in sorted(set_50k):
        for rot in SIGMA0_ROTATIONS:
            target = rotate_position(bit, rot)
            if target in set_50k and target != bit:
                rotation_links.append((bit, target, rot))
    
    if rotation_links:
        print(f"\n  Found {len(rotation_links)} rotation links:")
        for src, dst, rot in rotation_links:
            print(f"    Bit {src:2d} --[+{rot:2d}]--> Bit {dst:2d}")
    else:
        print("\n  No direct rotation links found.")
    
    # Check for XOR cancellation patterns
    print("\n" + "-" * 70)
    print("  XOR Cancellation Analysis:")
    print("-" * 70)
    
    # In Σ₀, bits can partially cancel if:
    # bit_i ⊕ (bit_i rotated by r1) ⊕ (bit_i rotated by r2) = 0 for some input patterns
    # This happens when the same bit appears in multiple rotation terms
    
    print("""
  Σ₀(a) = ROTR²(a) ⊕ ROTR¹³(a) ⊕ ROTR²²(a)
  
  For output bit position p, inputs come from:
    - Bit (p + 2) mod 32  via ROTR²
    - Bit (p + 13) mod 32 via ROTR¹³
    - Bit (p + 22) mod 32 via ROTR²²
  
  Slow bits are OUTPUT positions with slower divergence.
  These are positions where the THREE input bits have correlated behavior.
  """)
    
    print("  Input bit mapping for each slow output bit:")
    print("  " + "-" * 56)
    print("  {:>6} | {:>8} {:>8} {:>8} | Input bits".format(
        "Output", "+2", "+13", "+22"))
    print("  " + "-" * 56)
    
    for bit in sorted(set_50k):
        in1 = rotate_position(bit, 2)
        in2 = rotate_position(bit, 13)
        in3 = rotate_position(bit, 22)
        # Check if any inputs are the same or also slow
        markers = []
        if in1 in set_50k: markers.append(f"in1({in1})∈S")
        if in2 in set_50k: markers.append(f"in2({in2})∈S")
        if in3 in set_50k: markers.append(f"in3({in3})∈S")
        
        print("  {:>6} | {:>8} {:>8} {:>8} | {}".format(
            bit, in1, in2, in3, 
            " ".join(markers) if markers else "-"))
    
    # Analyze chains
    print("\n" + "-" * 70)
    print("  Rotation Chains (bits connected via +2 mod 32):")
    print("-" * 70)
    
    # Build chains for the +2 rotation (most direct)
    chains_2 = []
    used = set()
    for bit in sorted(set_50k):
        if bit in used:
            continue
        chain = [bit]
        used.add(bit)
        # Forward
        current = bit
        while rotate_position(current, 2) in set_50k:
            next_bit = rotate_position(current, 2)
            if next_bit in used:
                break
            chain.append(next_bit)
            used.add(next_bit)
            current = next_bit
        # Backward
        current = bit
        while inverse_rotate_position(current, 2) in set_50k:
            prev_bit = inverse_rotate_position(current, 2)
            if prev_bit in used:
                break
            chain.insert(0, prev_bit)
            used.add(prev_bit)
            current = prev_bit
        
        chains_2.append(chain)
    
    print("\n  Chains found:")
    for i, chain in enumerate(sorted(chains_2, key=len, reverse=True)):
        if len(chain) > 1:
            chain_str = " → ".join(map(str, chain))
            print(f"    Chain {i+1} (len={len(chain)}): {chain_str}")
        else:
            print(f"    Singleton: {chain[0]}")
    
    # Statistical significance of chains
    print("\n" + "=" * 70)
    print("  STATISTICAL SIGNIFICANCE")
    print("=" * 70)
    
    n_slow = len(set_50k)
    n_total = 32  # Only looking at word 'a'
    
    # Probability of random overlap
    p_random = n_slow / n_total
    
    # Expected number of rotation links by chance
    expected_links = n_slow * 3 * p_random  # 3 rotations, each slow bit can link
    observed_links = len(rotation_links)
    
    print(f"""
  Under null hypothesis (slow bits are random positions):
    P(any position is slow) = {p_random:.3f}
    Expected rotation links = {expected_links:.1f}
    Observed rotation links = {observed_links}
    
  Ratio: {observed_links / expected_links:.2f}x expected
  """)
    
    if observed_links > expected_links * 1.5:
        print("  ⚠️  SIGNIFICANT: More rotation links than expected by chance")
        print("      This suggests algebraic structure in Σ₀ affects divergence rates.")
    else:
        print("  ✓ Rotation links consistent with random distribution")
    
    # Final summary
    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)
    
    print("""
  The slow bits are OUTPUT positions of word 'a' where the avalanche
  effect propagates more slowly. For Σ₀:
  
    output_bit_p = input_(p+2) ⊕ input_(p+13) ⊕ input_(p+22)
  
  If the three input bits have correlated values (e.g., two are equal),
  the XOR output has lower entropy → slower divergence.
  
  This is NOT a cryptographic weakness because:
  1. Correlation is in geometric embedding, not probability
  2. Effect size is small (~24% variation in divergence rate)
  3. All bits still fully diverge by round 64
  4. The heat kernel detects structure, but structure ≠ vulnerability
  
  What we've found: A geometric fingerprint of the Σ₀ mixing function
  that traditional differential cryptanalysis may not characterize.
  """)


def main():
    results_dir = Path(__file__).parent.parent / "results"
    
    # Load 10k results
    pkl_10k = results_dir / "exp2_euclidean_10000_20251206_231713.pkl"
    pkl_50k = results_dir / "exp2_euclidean_50000_20251206_232417.pkl"
    
    if not pkl_10k.exists() or not pkl_50k.exists():
        print("Missing result files. Need both 10k and 50k experiments.")
        return
    
    with open(pkl_10k, 'rb') as f:
        results_10k = pickle.load(f)
    
    with open(pkl_50k, 'rb') as f:
        results_50k = pickle.load(f)
    
    # Extract slow bits (just from word 'a' = positions 0-31)
    slow_bits_10k = [int(b) for b in results_10k['slow_bits'] if int(b) < 32]
    slow_bits_50k = [int(b) for b in results_50k['slow_bits'] if int(b) < 32]
    
    print(f"Loaded results:")
    print(f"  10k pairs: {len(results_10k['slow_bits'])} total slow bits, {len(slow_bits_10k)} in word 'a'")
    print(f"  50k pairs: {len(results_50k['slow_bits'])} total slow bits, {len(slow_bits_50k)} in word 'a'")
    print()
    
    analyze_rotation_structure(slow_bits_10k, slow_bits_50k)


if __name__ == "__main__":
    main()
