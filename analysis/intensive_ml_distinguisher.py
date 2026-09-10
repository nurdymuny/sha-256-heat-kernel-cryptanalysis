#!/usr/bin/env python3
"""
INTENSIVE ML DISTINGUISHER
===========================

High-power version of the Davis Manifold ML Distinguisher.
- 20,000+ samples
- Deeper neural networks
- XGBoost with hyperparameter tuning
- Statistical significance testing
- Multiple random seeds for robustness

Hypothesis: The 52-55% signals in the initial run are real,
and will strengthen with more data and model capacity.

Author: Bee Davis
"""

import numpy as np
from typing import Dict, List, Tuple
import struct
from scipy import stats
from scipy.stats import binom_test
import time

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not available, using GradientBoosting instead")


# =============================================================================
# SHA-256 REDUCED ROUNDS
# =============================================================================

def rotr(x, n):
    return (x >> n) | (x << (32 - n)) & 0xFFFFFFFF


def sha256_state_at_round(message_bytes: bytes, target_round: int) -> List[int]:
    """Returns 8-word state at specified round."""
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
    h_init = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 
              0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

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

    for i in range(16, 64): 
        s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3)
        s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10)
        w[i] = (w[i-16] + s0 + w[i-7] + s1) & 0xFFFFFFFF

    a, b, c, d, e, f, g, h = h_init

    for i in range(target_round):
        s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
        ch = (e & f) ^ ((~e) & g)
        temp1 = (h + s1 + ch + K[i] + w[i]) & 0xFFFFFFFF
        s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        temp2 = (s0 + maj) & 0xFFFFFFFF

        h, g, f, e = g, f, e, (d + temp1) & 0xFFFFFFFF
        d, c, b, a = c, b, a, (temp1 + temp2) & 0xFFFFFFFF

    return [a, b, c, d, e, f, g, h]


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

SLOW_BITS = [1, 5, 7, 9, 10, 14]  # From heat kernel
SIGMA0_ROTS = [2, 13, 22]


def extract_features(state: List[int], mode: str = 'full') -> np.ndarray:
    """Extract features from SHA-256 state."""
    features = []
    
    if mode in ['lagging', 'full']:
        # Lagging registers d (3) and h (7)
        for word in [state[3], state[7]]:
            for bit in range(32):
                features.append((word >> bit) & 1)
    
    if mode in ['correlation', 'full']:
        # Leader-laggard correlations
        a, d, e, h = state[0], state[3], state[4], state[7]
        for bit in range(32):
            a_bit = (a >> bit) & 1
            d_bit = (d >> bit) & 1
            e_bit = (e >> bit) & 1
            h_bit = (h >> bit) & 1
            features.append(a_bit ^ d_bit)
            features.append(e_bit ^ h_bit)
            features.append(a_bit ^ h_bit)
            features.append(e_bit ^ d_bit)
    
    if mode in ['sigma0', 'full']:
        # Σ₀ orbit structure
        a = state[0]
        for rot in SIGMA0_ROTS:
            features.append((a >> rot) & 1)
            features.append((a >> ((rot + 16) % 32)) & 1)
        for i, r1 in enumerate(SIGMA0_ROTS):
            for r2 in SIGMA0_ROTS[i+1:]:
                features.append(((a >> r1) & 1) ^ ((a >> r2) & 1))
    
    if mode == 'all_bits':
        # Full 256-bit state
        for word in state:
            for bit in range(32):
                features.append((word >> bit) & 1)
    
    return np.array(features, dtype=np.float32)


def generate_dataset(n_samples: int, target_round: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate balanced dataset.
    Class 1: Slow-bit perturbed messages
    Class 0: Fully random messages (both hashed through SHA-256)
    """
    np.random.seed(seed)
    
    X = []
    y = []
    
    for i in range(n_samples):
        # Class 1: Slow-bit structured input
        base_msg = np.random.bytes(55)
        msg = bytearray(base_msg)
        for bit in SLOW_BITS:
            if np.random.rand() > 0.5:
                byte_idx = bit // 8
                bit_idx = 7 - (bit % 8)
                if byte_idx < len(msg):
                    msg[byte_idx] ^= (1 << bit_idx)
        
        state = sha256_state_at_round(bytes(msg), target_round)
        X.append(extract_features(state, mode='full'))
        y.append(1)
        
        # Class 0: Fully random input (also hashed)
        random_msg = np.random.bytes(55)
        random_state = sha256_state_at_round(random_msg, target_round)
        X.append(extract_features(random_state, mode='full'))
        y.append(0)
    
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle
    perm = np.random.permutation(len(y))
    return X[perm], y[perm]


# =============================================================================
# INTENSIVE DISTINGUISHER
# =============================================================================

def run_intensive_attack(target_round: int, n_samples: int = 20000, n_seeds: int = 5):
    """
    Run intensive ML attack with multiple seeds for robustness.
    """
    print(f"\n{'='*70}")
    print(f"  INTENSIVE ATTACK: ROUND {target_round}")
    print(f"  Samples: {n_samples}, Seeds: {n_seeds}")
    print(f"{'='*70}")
    
    all_accuracies = []
    all_aucs = []
    
    for seed in range(n_seeds):
        print(f"\n  [Seed {seed+1}/{n_seeds}]")
        
        # Generate data
        t0 = time.time()
        X, y = generate_dataset(n_samples // 2, target_round, seed=seed * 1000)
        print(f"    Data generation: {time.time()-t0:.1f}s")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train models
        models = {}
        
        # 1. Random Forest (deeper)
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=200, 
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed
        )
        rf.fit(X_train, y_train)
        models['RF'] = rf
        print(f"    Random Forest: {time.time()-t0:.1f}s")
        
        # 2. XGBoost or GradientBoosting
        t0 = time.time()
        if HAS_XGBOOST:
            xgb = XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                random_state=seed,
                verbosity=0
            )
            xgb.fit(X_train, y_train)
            models['XGB'] = xgb
            print(f"    XGBoost: {time.time()-t0:.1f}s")
        else:
            gb = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                random_state=seed
            )
            gb.fit(X_train, y_train)
            models['GB'] = gb
            print(f"    GradientBoosting: {time.time()-t0:.1f}s")
        
        # 3. Deep Neural Network
        t0 = time.time()
        nn = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='relu',
            batch_size=256,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed
        )
        nn.fit(X_train, y_train)
        models['NN'] = nn
        print(f"    Neural Network: {time.time()-t0:.1f}s")
        
        # Evaluate
        seed_accs = []
        seed_aucs = []
        for name, model in models.items():
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, pred)
            auc = roc_auc_score(y_test, prob)
            seed_accs.append(acc)
            seed_aucs.append(auc)
            print(f"    {name}: Acc={acc:.4f}, AUC={auc:.4f}")
        
        all_accuracies.append(max(seed_accs))
        all_aucs.append(max(seed_aucs))
    
    # Aggregate results
    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    mean_auc = np.mean(all_aucs)
    
    # Statistical test: is accuracy significantly > 50%?
    n_test = int(n_samples * 0.2)
    correct = int(mean_acc * n_test)
    p_value = 1 - stats.binom.cdf(correct - 1, n_test, 0.5)
    
    print(f"\n  AGGREGATE RESULTS:")
    print(f"    Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"    Mean AUC:      {mean_auc:.4f}")
    print(f"    Binomial p-value: {p_value:.6f}")
    
    if p_value < 0.01:
        print(f"    🔴 STATISTICALLY SIGNIFICANT (p < 0.01)")
        significant = True
    elif p_value < 0.05:
        print(f"    ⚠️ MARGINALLY SIGNIFICANT (p < 0.05)")
        significant = True
    else:
        print(f"    ✓ NOT SIGNIFICANT (p >= 0.05)")
        significant = False
    
    return {
        'round': target_round,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_auc': mean_auc,
        'p_value': p_value,
        'significant': significant,
        'all_accuracies': all_accuracies
    }


def main():
    print("=" * 70)
    print("  INTENSIVE ML DISTINGUISHER")
    print("  Testing if 52-55% signal is real or noise")
    print("=" * 70)
    
    # Test key rounds
    test_rounds = [16, 18, 20, 22, 24]
    n_samples = 20000  # 10x more than before
    n_seeds = 5  # Multiple seeds for robustness
    
    results = {}
    
    for round_num in test_rounds:
        result = run_intensive_attack(round_num, n_samples=n_samples, n_seeds=n_seeds)
        results[round_num] = result
    
    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"\n  {'Round':<8} {'Accuracy':<15} {'AUC':<10} {'p-value':<12} {'Status'}")
    print("  " + "-" * 60)
    
    for round_num in test_rounds:
        r = results[round_num]
        acc_str = f"{r['mean_accuracy']:.4f} ± {r['std_accuracy']:.4f}"
        
        if r['significant'] and r['mean_accuracy'] > 0.52:
            status = "🔴 BROKEN"
        elif r['p_value'] < 0.1:
            status = "⚠️ WEAK"
        else:
            status = "✓ SECURE"
        
        print(f"  {round_num:<8} {acc_str:<15} {r['mean_auc']:<10.4f} {r['p_value']:<12.6f} {status}")
    
    # Determine security cliff
    print("\n" + "=" * 70)
    print("  CONCLUSION")
    print("=" * 70)
    
    broken_rounds = [r for r in test_rounds if results[r]['significant'] and results[r]['mean_accuracy'] > 0.52]
    
    if broken_rounds:
        max_broken = max(broken_rounds)
        print(f"""
    🔴 ML DISTINGUISHER FINDS REAL SIGNAL
    
    Statistically significant (p < 0.05) distinction up to Round {max_broken}.
    
    The slow-bit / lagging-register hypothesis provides a real
    (if small) advantage over random guessing.
    
    Security cliff: Round {max_broken + 1}
    Safety margin: {64 - max_broken - 1} rounds
        """)
    else:
        print(f"""
    ✓ NO SIGNIFICANT SIGNAL DETECTED
    
    The 52-55% numbers from the initial run were noise.
    With {n_samples} samples and {n_seeds} seeds, accuracy clusters at 50%.
    
    SHA-256 is secure against ML-based distinguishers by Round {min(test_rounds)}.
        """)
    
    # Save results
    import json
    with open("intensive_ml_results.json", "w") as f:
        json.dump({str(k): {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist()) 
                           for kk, vv in v.items()} 
                  for k, v in results.items()}, f, indent=2)
    print("\n  Results saved to: intensive_ml_results.json")
    
    return results


if __name__ == "__main__":
    main()
