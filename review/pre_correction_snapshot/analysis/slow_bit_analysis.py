#!/usr/bin/env python3
"""
Slow Bit Analysis for SHA-256 Heat Kernel Study
================================================

Maps detected slow bits back to SHA-256 internal structure:
- 256-bit state = 8 × 32-bit words (a, b, c, d, e, f, g, h)
- Bit positions 0-31 = word 'a', 32-63 = word 'b', etc.

Author: Bee Davis
"""

import numpy as np
from typing import List, Dict, Tuple

# SHA-256 working variable names
WORKING_VARS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

def bit_to_word(bit_position: int) -> Tuple[str, int]:
    """
    Map a 256-bit position to its SHA-256 working variable and bit within that word.
    
    The 256-bit state is organized as:
    - Bits 0-31:   word 'a'
    - Bits 32-63:  word 'b'  
    - Bits 64-95:  word 'c'
    - Bits 96-127: word 'd'
    - Bits 128-159: word 'e'
    - Bits 160-191: word 'f'
    - Bits 192-223: word 'g'
    - Bits 224-255: word 'h'
    
    Returns:
        (word_name, bit_within_word)
    """
    word_idx = bit_position // 32
    bit_in_word = bit_position % 32
    return WORKING_VARS[word_idx], bit_in_word


def analyze_slow_bits(slow_bits: List[int]) -> Dict:
    """
    Analyze which SHA-256 working variables the slow bits fall into.
    """
    # Map each bit
    by_word = {v: [] for v in WORKING_VARS}
    mappings = []
    
    for bit in slow_bits:
        word, pos = bit_to_word(bit)
        by_word[word].append(pos)
        mappings.append({
            'bit': bit,
            'word': word,
            'position_in_word': pos,
            'is_high_bit': pos >= 24,  # MSB region
            'is_low_bit': pos < 8,      # LSB region
        })
    
    # Count per word
    word_counts = {v: len(bits) for v, bits in by_word.items()}
    
    # Chi-squared test for uniform distribution
    expected = len(slow_bits) / 8
    chi_sq = sum((count - expected) ** 2 / expected for count in word_counts.values())
    
    return {
        'mappings': mappings,
        'by_word': by_word,
        'word_counts': word_counts,
        'chi_squared': chi_sq,
        'expected_per_word': expected,
    }


def print_analysis(slow_bits: List[int], label: str = "Slow Bits"):
    """Print detailed analysis of slow bits."""
    print(f"\n{'='*60}")
    print(f"  {label} Analysis")
    print(f"{'='*60}")
    
    result = analyze_slow_bits(slow_bits)
    
    print(f"\nTotal slow bits: {len(slow_bits)}")
    print(f"\nDistribution by SHA-256 working variable:")
    print("-" * 40)
    
    for word in WORKING_VARS:
        bits = result['by_word'][word]
        count = len(bits)
        bar = '█' * count + '░' * (8 - min(count, 8))
        pct = count / len(slow_bits) * 100 if slow_bits else 0
        
        print(f"  {word}: {bar} {count:3d} ({pct:5.1f}%)  bits: {bits[:5]}{'...' if len(bits) > 5 else ''}")
    
    print(f"\nChi-squared statistic: {result['chi_squared']:.2f}")
    print(f"(Expected under uniform: ~14.07 for p=0.05 with df=7)")
    
    # Check for patterns
    print(f"\nBit position patterns:")
    high_bits = [m for m in result['mappings'] if m['is_high_bit']]
    low_bits = [m for m in result['mappings'] if m['is_low_bit']]
    
    print(f"  High bits (pos >= 24): {len(high_bits)} ({len(high_bits)/len(slow_bits)*100:.1f}%)")
    print(f"  Low bits (pos < 8):    {len(low_bits)} ({len(low_bits)/len(slow_bits)*100:.1f}%)")
    
    # Which words are overrepresented?
    expected = result['expected_per_word']
    print(f"\nOverrepresented words (> {expected:.1f} expected):")
    for word, count in sorted(result['word_counts'].items(), key=lambda x: -x[1]):
        if count > expected:
            excess = count - expected
            print(f"  {word}: {count} (excess: +{excess:.1f})")
    
    # Significance for SHA-256 structure
    print(f"\n{'='*60}")
    print("  SHA-256 Structural Significance")
    print("="*60)
    
    # Check e, f, g (used in Ch function)
    ch_vars = ['e', 'f', 'g']
    ch_count = sum(result['word_counts'][v] for v in ch_vars)
    
    # Check a, b, c (used in Maj function)
    maj_vars = ['a', 'b', 'c']
    maj_count = sum(result['word_counts'][v] for v in maj_vars)
    
    print(f"\n  Ch function variables (e, f, g): {ch_count} slow bits")
    print(f"  Maj function variables (a, b, c): {maj_count} slow bits")
    print(f"  Other (d, h): {result['word_counts']['d'] + result['word_counts']['h']} slow bits")
    
    # Variables with special roles
    print(f"\n  Key variable roles:")
    print(f"    'a' (most updated): {result['word_counts']['a']} slow bits")
    print(f"    'e' (gates Ch):     {result['word_counts']['e']} slow bits")
    print(f"    'h' (least updated):{result['word_counts']['h']} slow bits")
    
    return result


if __name__ == "__main__":
    # Slow bits from 10k run (first 10 shown in output)
    # Let's analyze a representative sample
    
    # From first run (2k pairs):
    slow_bits_2k = [35, 41, 63, 78, 80, 87, 106, 119, 121, 140, 146]
    
    # From 10k run (these are the first 10):
    slow_bits_10k = [1, 5, 7, 9, 10, 14, 16, 19, 21, 26]
    
    print("\n" + "="*70)
    print("  SHA-256 Slow Bit Mapping Analysis")
    print("="*70)
    
    print("\n2k pairs run slow bits:")
    result_2k = print_analysis(slow_bits_2k, "2k Pairs Run")
    
    print("\n\n10k pairs run slow bits:")
    result_10k = print_analysis(slow_bits_10k, "10k Pairs Run")
    
    # Compare the two runs
    print("\n" + "="*70)
    print("  Comparison Between Runs")
    print("="*70)
    
    overlap = set(slow_bits_2k) & set(slow_bits_10k)
    print(f"\nOverlapping slow bits: {overlap}")
    print(f"Overlap size: {len(overlap)} / {len(slow_bits_2k)} (2k) vs {len(slow_bits_10k)} (10k)")
    
    # Word distribution comparison
    print("\nWord distribution comparison:")
    for word in WORKING_VARS:
        count_2k = result_2k['word_counts'][word]
        count_10k = result_10k['word_counts'][word]
        print(f"  {word}: 2k={count_2k}, 10k={count_10k}")
