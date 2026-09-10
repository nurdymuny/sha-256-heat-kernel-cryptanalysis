#!/usr/bin/env python3
"""
Combined Geometric-Algebraic Analysis
======================================

This analysis bridges the gap between:
1. Geometric isotropy (achieved at round 16)
2. Algebraic security (achieved at round 19)

Key findings to investigate:
- Slow-manifold gradient attack found bit patterns [1,5,7,9,10,14] cause slow divergence
- 16-bit cube attack breaks through round 18, round 19 secure
- 3-round gap between geometric and algebraic security

Hypothesis: The slow-diverging bit patterns may correspond to exploitable
algebraic structure that persists even after geometric isotropy is achieved.
"""

import struct
import numpy as np
from pathlib import Path
import json

# The slow-diverging bit patterns from Attack 1
SLOW_PATTERNS = [
    (5, 10),   # z = -34.89
    (1, 9),    # z = -34.45  
    (5, 7),    # z = -34.34
    (1, 10),   # z = -33.72
    (10, 14),  # z = -33.92
    (1, 7),    # z = -33.82
    (1, 5),    # z = -33.52
]

# All unique slow bits
SLOW_BITS_SET = {1, 5, 7, 9, 10, 14}

# Σ₀ rotation constants
SIGMA0_ROTATIONS = [2, 13, 22]


# --- Compact SHA-256 Implementation ---
def rotr(x, n):
    return (x >> n) | (x << (32 - n)) & 0xFFFFFFFF


def sha256_reduced(message_bytes, rounds=64):
    """SHA-256 with configurable rounds (actual reduced computation)."""
    K = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    ]
    h_init = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

    # Padding
    msg = bytearray(message_bytes)
    length = len(msg) * 8
    msg.append(0x80)
    while (len(msg) * 8 + 64) % 512 != 0:
        msg.append(0x00)
    msg += struct.pack('>Q', length)
    
    chunk = bytes(msg[:64])
    w = [0] * 64
    for i in range(16):
        w[i] = struct.unpack('>I', chunk[i*4:(i+1)*4])[0]

    # Message Schedule
    for i in range(16, min(rounds, 64)): 
        s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3)
        s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10)
        w[i] = (w[i-16] + s0 + w[i-7] + s1) & 0xFFFFFFFF

    a, b, c, d, e, f, g, h = h_init

    for i in range(rounds):
        s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
        ch = (e & f) ^ ((~e) & g)
        temp1 = (h + s1 + ch + K[i] + w[i]) & 0xFFFFFFFF
        s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        temp2 = (s0 + maj) & 0xFFFFFFFF

        h = g
        g = f
        f = e
        e = (d + temp1) & 0xFFFFFFFF
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & 0xFFFFFFFF

    return [a, b, c, d, e, f, g, h]


def get_output_bit(state, bit_idx):
    """Extract a specific bit from the 256-bit state."""
    word_idx = bit_idx // 32
    bit_in_word = bit_idx % 32
    return (state[word_idx] >> bit_in_word) & 1


def analyze_pattern_algebra():
    """
    Check if slow-diverging patterns align with SHA-256 algebraic structure.
    """
    print("=" * 70)
    print("  PATTERN-ALGEBRA CORRELATION ANALYSIS")
    print("=" * 70)
    
    slow_bits = sorted(SLOW_BITS_SET)
    print(f"\n  Slow-diverging bits: {slow_bits}")
    
    # Check relationship to Σ₀ rotation constants
    print("\n  Relationship to Σ₀ (rotations 2, 13, 22):")
    for rot in SIGMA0_ROTATIONS:
        aligned_bits = [b for b in slow_bits if b == rot or (b - rot) in slow_bits or (b + rot) in slow_bits]
        print(f"    Rotation {rot}: aligned bits = {aligned_bits}")
    
    # Message schedule analysis
    print("\n  Relationship to message schedule σ functions:")
    sigma0_rots = [7, 18]  # rotations in σ₀
    sigma1_rots = [17, 19]  # rotations in σ₁
    
    for bit in slow_bits:
        s0_aligned = [(bit + r) % 32 for r in sigma0_rots]
        s1_aligned = [(bit + r) % 32 for r in sigma1_rots]
        print(f"    Bit {bit:2d}: σ₀ → {s0_aligned}, σ₁ → {s1_aligned}")
    
    # Word-level analysis
    print("\n  Slow bits by message word (32-bit chunks):")
    for word in range(2):
        word_bits = [b for b in slow_bits if b // 32 == word or b < 32]
        print(f"    Word {word}: bits {word_bits}")


def cube_attack_with_slow_patterns(target_rounds=[16, 17, 18, 19], n_trials=50):
    """
    Use slow-diverging bit patterns as cube variables.
    """
    print("\n" + "=" * 70)
    print("  PATTERN-GUIDED CUBE ATTACK")
    print("=" * 70)
    
    np.random.seed(42)
    results = {}
    
    for rounds in target_rounds:
        print(f"\n  --- Round {rounds} ---")
        pattern_results = []
        
        for pattern in SLOW_PATTERNS[:5]:  # Test top 5 patterns
            zeros = 0
            total = 0
            
            for trial in range(n_trials):
                # Random base message
                base = np.random.bytes(55)
                cube_vars = list(pattern)
                
                # Test multiple output bits
                for out_bit in [0, 32, 64, 128, 160, 192, 224]:
                    cube_sum = 0
                    for corner in range(1 << len(cube_vars)):
                        msg = bytearray(base)
                        for i, var in enumerate(cube_vars):
                            if corner & (1 << i):
                                byte_idx = var // 8
                                bit_idx = var % 8
                                if byte_idx < len(msg):
                                    msg[byte_idx] ^= (1 << bit_idx)
                        
                        state = sha256_reduced(bytes(msg), rounds=rounds)
                        bit_val = get_output_bit(state, out_bit)
                        cube_sum ^= bit_val
                    
                    total += 1
                    if cube_sum == 0:
                        zeros += 1
            
            rate = zeros / total if total > 0 else 0
            bias = abs(rate - 0.5)
            pattern_results.append({
                "pattern": list(pattern),
                "zero_rate": rate,
                "bias": bias,
                "broken": bias > 0.1
            })
            
            status = "🔴 BROKEN" if bias > 0.1 else ("🟡 PARTIAL" if bias > 0.05 else "✓ Secure")
            print(f"    Pattern {pattern}: {rate:.1%} zeros, bias={bias:.3f} {status}")
        
        results[rounds] = pattern_results
    
    return results


def exhaustive_round_scan():
    """
    Fine-grained scan of the geometric-algebraic transition zone.
    Using pattern-derived cube variables.
    """
    print("\n" + "=" * 70)
    print("  FINE-GRAINED TRANSITION SCAN (Rounds 14-22)")
    print("=" * 70)
    
    np.random.seed(42)
    results = {}
    
    # Pattern-derived cube variables (from slow bits)
    cube_vars = [1, 5, 9, 10]  # 4-bit cube using slow pattern bits
    
    for rounds in range(14, 23):
        # Geometric: anisotropy (using full SHA-256 trajectories)
        states = []
        for _ in range(100):
            msg = np.random.bytes(55)
            state = sha256_reduced(msg, rounds=rounds)
            states.append(state)
        
        states = np.array(states, dtype=np.float64)
        centered = states - np.mean(states, axis=0)
        cov = np.cov(centered.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 0]
        anisotropy = np.std(eigvals) / np.mean(eigvals) if len(eigvals) > 0 else 0
        
        # Algebraic: 4-bit cube attack using slow bits
        cube_zeros = 0
        cube_total = 0
        
        for trial in range(30):
            base = np.random.bytes(55)
            
            for out_bit in [0, 64, 128, 192]:
                cube_sum = 0
                for corner in range(16):  # 2^4
                    msg = bytearray(base)
                    for i, var in enumerate(cube_vars):
                        if corner & (1 << i):
                            byte_idx = var // 8
                            bit_idx = var % 8
                            if byte_idx < len(msg):
                                msg[byte_idx] ^= (1 << bit_idx)
                    
                    state = sha256_reduced(bytes(msg), rounds=rounds)
                    bit_val = get_output_bit(state, out_bit)
                    cube_sum ^= bit_val
                
                cube_total += 1
                if cube_sum == 0:
                    cube_zeros += 1
        
        cube_rate = cube_zeros / cube_total
        cube_bias = abs(cube_rate - 0.5)
        
        geo_status = "Isotropic" if anisotropy < 0.1 else "Anisotropic"
        alg_status = "🔴 BROKEN" if cube_bias > 0.15 else ("🟡 Partial" if cube_bias > 0.05 else "✓ Secure")
        
        results[rounds] = {
            "anisotropy": float(anisotropy),
            "cube_bias": float(cube_bias),
            "geometric": geo_status,
            "algebraic": alg_status
        }
        
        print(f"  Round {rounds:2d}: aniso={anisotropy:.3f} ({geo_status:10s}) | "
              f"bias={cube_bias:.3f} ({alg_status})")
    
    return results


def heavy_cube_with_all_slow_bits(target_rounds=[17, 18, 19, 20]):
    """
    Use ALL slow bits (6 bits = 64 iterations) as cube variables.
    """
    print("\n" + "=" * 70)
    print("  HEAVY CUBE: ALL SLOW BITS (6-dimensional)")
    print("=" * 70)
    
    np.random.seed(42)
    cube_vars = sorted(SLOW_BITS_SET)  # [1, 5, 7, 9, 10, 14]
    print(f"\n  Cube variables: {cube_vars}")
    print(f"  Iterations per test: {2**len(cube_vars)}")
    
    results = {}
    
    for rounds in target_rounds:
        print(f"\n  --- Round {rounds} ---")
        
        zeros = 0
        total = 0
        word_zeros = [0] * 8
        word_total = [0] * 8
        
        for trial in range(50):
            base = np.random.bytes(55)
            
            for word_idx in range(8):
                out_bit = word_idx * 32 + 16  # Middle bit of each word
                cube_sum = 0
                
                for corner in range(1 << len(cube_vars)):
                    msg = bytearray(base)
                    for i, var in enumerate(cube_vars):
                        if corner & (1 << i):
                            byte_idx = var // 8
                            bit_idx = var % 8
                            if byte_idx < len(msg):
                                msg[byte_idx] ^= (1 << bit_idx)
                    
                    state = sha256_reduced(bytes(msg), rounds=rounds)
                    bit_val = get_output_bit(state, out_bit)
                    cube_sum ^= bit_val
                
                total += 1
                word_total[word_idx] += 1
                if cube_sum == 0:
                    zeros += 1
                    word_zeros[word_idx] += 1
        
        overall_rate = zeros / total
        overall_bias = abs(overall_rate - 0.5)
        
        word_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        broken_words = []
        
        print(f"    Overall: {overall_rate:.1%} zeros, bias={overall_bias:.3f}")
        print(f"    Per-word breakdown:")
        
        for i, label in enumerate(word_labels):
            if word_total[i] > 0:
                rate = word_zeros[i] / word_total[i]
                bias = abs(rate - 0.5)
                status = "🔴 BROKEN" if bias > 0.15 else ("🟡" if bias > 0.05 else "✓")
                if bias > 0.15:
                    broken_words.append(label)
                print(f"      {label}: {rate:.1%} zeros, bias={bias:.3f} {status}")
        
        results[rounds] = {
            "overall_rate": overall_rate,
            "overall_bias": overall_bias,
            "broken_words": broken_words,
            "secure": len(broken_words) == 0
        }
    
    return results


def main():
    print("\n" + "=" * 70)
    print("  SHA-256 COMBINED GEOMETRIC-ALGEBRAIC ANALYSIS")
    print("  Investigating the 3-round gap between geometry and algebra")
    print("=" * 70)
    
    # Analysis 1: Pattern-algebra correlation
    analyze_pattern_algebra()
    
    # Analysis 2: Fine-grained transition scan
    scan_results = exhaustive_round_scan()
    
    # Analysis 3: Pattern-guided cube attack
    cube_results = cube_attack_with_slow_patterns(
        target_rounds=[16, 17, 18, 19],
        n_trials=30
    )
    
    # Analysis 4: Heavy cube with all slow bits
    heavy_results = heavy_cube_with_all_slow_bits(
        target_rounds=[17, 18, 19, 20]
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY: GEOMETRIC-ALGEBRAIC GAP ANALYSIS")
    print("=" * 70)
    
    # Find transition points
    geo_transition = None
    alg_transition = None
    
    for r, data in sorted(scan_results.items()):
        if geo_transition is None and data["anisotropy"] < 0.1:
            geo_transition = r
        if alg_transition is None and data["cube_bias"] < 0.05:
            alg_transition = r
    
    print(f"\n  Geometric isotropy achieved: Round {geo_transition}")
    print(f"  Algebraic security achieved: Round {alg_transition}")
    
    if geo_transition and alg_transition:
        gap = alg_transition - geo_transition
        print(f"  Gap: {gap} rounds")
        print(f"\n  Interpretation:")
        print(f"    - Geometric structure dissolves at round {geo_transition}")
        print(f"    - Algebraic structure (exploitable by cube attacks) persists until round {alg_transition}")
        print(f"    - The {gap}-round gap represents 'hidden' algebraic structure")
    
    # Heavy cube results
    print(f"\n  Heavy Cube Attack (6 slow bits):")
    for r, data in sorted(heavy_results.items()):
        status = "✅ SECURE" if data["secure"] else f"🔴 BROKEN ({','.join(data['broken_words'])})"
        print(f"    Round {r}: {status}")
    
    # Find security margin
    for r in sorted(heavy_results.keys()):
        if heavy_results[r]["secure"]:
            print(f"\n  ⚡ Security margin: Round {r} is the first fully secure round")
            print(f"     with slow-bit-derived cube attack (6 bits, 64 iterations)")
            break
    
    # Save results
    all_results = {
        "transition_scan": {str(k): v for k, v in scan_results.items()},
        "pattern_cube": {str(k): v for k, v in cube_results.items()},
        "heavy_cube": {str(k): v for k, v in heavy_results.items()}
    }
    
    with open("combined_analysis_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n  Results saved to: combined_analysis_results.json")


if __name__ == "__main__":
    main()
