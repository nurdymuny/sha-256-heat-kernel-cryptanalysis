#!/usr/bin/env python3
"""
High-Precision Slow-Bit Cube Attack
====================================

Use the 6 slow bits derived from heat kernel analysis as cube variables.
Run many more trials to reduce statistical noise.

Key question: What is the true security margin when using 
geometrically-derived slow bits?
"""

import struct
import numpy as np
from scipy import stats

# Slow bits from heat kernel slow-manifold gradient attack
SLOW_BITS = [1, 5, 7, 9, 10, 14]


def rotr(x, n):
    return (x >> n) | (x << (32 - n)) & 0xFFFFFFFF


def sha256_reduced(message_bytes, rounds=64):
    """SHA-256 with configurable rounds."""
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

        h, g, f, e = g, f, e, (d + temp1) & 0xFFFFFFFF
        d, c, b, a = c, b, a, (temp1 + temp2) & 0xFFFFFFFF

    return [a, b, c, d, e, f, g, h]


def run_cube_attack(rounds, cube_vars, n_trials=200):
    """
    Run cube attack and return statistical analysis.
    """
    np.random.seed(42 + rounds)  # Reproducible per-round
    
    word_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    cube_size = len(cube_vars)
    iterations = 2 ** cube_size
    
    # Per-word and per-bit statistics
    word_zeros = {w: 0 for w in word_labels}
    word_total = {w: 0 for w in word_labels}
    
    for trial in range(n_trials):
        base = np.random.bytes(55)
        
        for word_idx, word_label in enumerate(word_labels):
            # Test bit 16 of each word (middle bit)
            out_bit = word_idx * 32 + 16
            cube_sum = 0
            
            for corner in range(iterations):
                msg = bytearray(base)
                for i, var in enumerate(cube_vars):
                    if corner & (1 << i):
                        byte_idx = var // 8
                        bit_idx = var % 8
                        if byte_idx < len(msg):
                            msg[byte_idx] ^= (1 << bit_idx)
                
                state = sha256_reduced(bytes(msg), rounds=rounds)
                bit_val = (state[word_idx] >> 16) & 1
                cube_sum ^= bit_val
            
            word_total[word_label] += 1
            if cube_sum == 0:
                word_zeros[word_label] += 1
    
    # Statistical analysis
    results = {}
    for word in word_labels:
        zeros = word_zeros[word]
        total = word_total[word]
        rate = zeros / total
        
        # Binomial test
        binom_result = stats.binomtest(zeros, total, 0.5, alternative='two-sided')
        p_value = binom_result.pvalue
        
        # Chi-squared
        expected = total / 2
        chi2 = ((zeros - expected) ** 2 + (total - zeros - expected) ** 2) / expected
        
        results[word] = {
            'zeros': zeros,
            'total': total,
            'rate': rate,
            'bias': abs(rate - 0.5),
            'p_value': p_value,
            'chi2': chi2,
            'significant': p_value < 0.01  # 99% confidence
        }
    
    return results


def main():
    print("=" * 70)
    print("  HIGH-PRECISION SLOW-BIT CUBE ATTACK")
    print("=" * 70)
    print(f"\n  Cube variables (slow bits): {SLOW_BITS}")
    print(f"  Cube dimension: {len(SLOW_BITS)} → {2**len(SLOW_BITS)} iterations")
    print(f"  Trials per word: 200")
    print(f"  Significance threshold: p < 0.01")
    
    test_rounds = list(range(14, 25))
    
    print("\n" + "-" * 70)
    print(f"  {'Round':<6} | {'Word':<4} | {'Rate':<6} | {'Bias':<6} | {'χ²':<7} | {'p-value':<10} | Status")
    print("-" * 70)
    
    summary = {}
    
    for rounds in test_rounds:
        results = run_cube_attack(rounds, SLOW_BITS, n_trials=200)
        
        broken_words = []
        for word, data in results.items():
            status = "🔴 SIG" if data['significant'] else "✓ OK"
            if data['significant']:
                broken_words.append(word)
            
            print(f"  {rounds:<6} | {word:<4} | {data['rate']:.3f}  | {data['bias']:.3f}  | "
                  f"{data['chi2']:<7.2f} | {data['p_value']:<10.4f} | {status}")
        
        summary[rounds] = {
            'broken_words': broken_words,
            'n_broken': len(broken_words),
            'secure': len(broken_words) == 0
        }
        
        print("-" * 70)
    
    # Final summary
    print("\n" + "=" * 70)
    print("  SUMMARY: SECURITY BY ROUND")
    print("=" * 70)
    
    print(f"\n  {'Round':<6} | {'Status':<12} | Broken Words")
    print("-" * 50)
    
    for rounds, data in summary.items():
        status = "✅ SECURE" if data['secure'] else "🔴 BROKEN"
        broken = ', '.join(data['broken_words']) if data['broken_words'] else '-'
        print(f"  {rounds:<6} | {status:<12} | {broken}")
    
    # Find security margin
    first_secure = None
    for r in sorted(summary.keys()):
        if summary[r]['secure']:
            first_secure = r
            break
    
    if first_secure:
        print(f"\n  ⚡ SECURITY MARGIN: Round {first_secure}")
        print(f"     Using 6-bit cube with geometrically-derived slow bits,")
        print(f"     SHA-256 becomes indistinguishable from random at round {first_secure}.")
        print(f"\n     This is {64 - first_secure} rounds of security margin ({(64-first_secure)/64*100:.0f}%)")


if __name__ == "__main__":
    main()
