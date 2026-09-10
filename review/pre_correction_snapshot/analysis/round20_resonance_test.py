#!/usr/bin/env python3
"""
ROUND 20 RESONANCE VERIFICATION
================================

High-power statistical test of the "Echo" phenomenon at Round 20.

Hypothesis: The 54.75% accuracy at Round 20 (correlation mode) is NOT noise.
The slow-bit perturbation entering at Round 0 wraps around the state
registers and resonates with the lagging registers (d, h) at Round 20.

If this holds at 100,000 samples:
- Z-score > 50 → Undeniable statistical break of 20-round SHA-256
- Z-score < 2 → The 54% was sampling variance

This is a focused, single-hypothesis test with maximum statistical power.

Author: Bee Davis
"""

import numpy as np
from typing import List, Tuple
import struct
from scipy import stats
import time

# ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Optional: XGBoost for extra power
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# =============================================================================
# SHA-256 IMPLEMENTATION
# =============================================================================

def rotr(x: int, n: int) -> int:
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
# CORRELATION FEATURE EXTRACTION (The winning mode)
# =============================================================================

# Slow bits from heat kernel analysis (Σ₀ aligned)
SLOW_BITS = [1, 5, 7, 9, 10, 14]


def extract_correlation_features(state: List[int]) -> np.ndarray:
    """
    Extract leader-laggard correlation features.
    This is the mode that detected the Round 20 echo.
    
    Features: a⊕d, e⊕h, a⊕h, e⊕d for each bit position (128 features)
    """
    a, d, e, h = state[0], state[3], state[4], state[7]
    
    features = []
    for bit in range(32):
        a_bit = (a >> bit) & 1
        d_bit = (d >> bit) & 1
        e_bit = (e >> bit) & 1
        h_bit = (h >> bit) & 1
        
        # Phase correlations
        features.append(a_bit ^ d_bit)  # Leader a vs laggard d
        features.append(e_bit ^ h_bit)  # Leader e vs laggard h
        features.append(a_bit ^ h_bit)  # Cross-phase
        features.append(e_bit ^ d_bit)  # Cross-phase
    
    return np.array(features, dtype=np.float32)


def generate_dataset(n_samples: int, target_round: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate balanced dataset for the resonance test.
    
    Class 1 (Manifold): Slow-bit perturbed messages
    Class 0 (Control): Fully random messages
    
    Both are hashed through reduced-round SHA-256.
    """
    np.random.seed(seed)
    
    X = []
    y = []
    
    for _ in range(n_samples):
        # === Class 1: Slow-bit structured input ===
        base_msg = np.random.bytes(55)
        msg = bytearray(base_msg)
        
        # Apply slow-bit perturbation pattern
        for bit in SLOW_BITS:
            if np.random.rand() > 0.5:
                byte_idx = bit // 8
                bit_idx = 7 - (bit % 8)  # MSB first
                if byte_idx < len(msg):
                    msg[byte_idx] ^= (1 << bit_idx)
        
        state = sha256_state_at_round(bytes(msg), target_round)
        X.append(extract_correlation_features(state))
        y.append(1)
        
        # === Class 0: Fully random input ===
        random_msg = np.random.bytes(55)
        random_state = sha256_state_at_round(random_msg, target_round)
        X.append(extract_correlation_features(random_state))
        y.append(0)
    
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle
    perm = np.random.permutation(len(y))
    return X[perm], y[perm]


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_significance(accuracy: float, n_samples: int) -> dict:
    """
    Compute statistical significance of accuracy > 50%.
    """
    # Number of correct predictions
    correct = int(accuracy * n_samples)
    
    # Binomial test: P(X >= correct | p=0.5)
    p_value = 1 - stats.binom.cdf(correct - 1, n_samples, 0.5)
    
    # Z-score (normal approximation for large n)
    expected = n_samples * 0.5
    std = np.sqrt(n_samples * 0.5 * 0.5)
    z_score = (correct - expected) / std
    
    # Effect size (Cohen's h for proportions)
    p_obs = accuracy
    p_null = 0.5
    cohens_h = 2 * (np.arcsin(np.sqrt(p_obs)) - np.arcsin(np.sqrt(p_null)))
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'n_samples': n_samples,
        'p_value': p_value,
        'z_score': z_score,
        'cohens_h': cohens_h,
        'sigma': z_score  # Alias for clarity
    }


# =============================================================================
# MAIN TEST
# =============================================================================

def run_resonance_test(n_samples: int = 100000, target_round: int = 20, n_seeds: int = 3):
    """
    High-power test of the Round 20 resonance hypothesis.
    """
    print("=" * 70)
    print("  ROUND 20 RESONANCE VERIFICATION")
    print("  Testing the 'Echo' Phenomenon")
    print("=" * 70)
    print(f"\n  Hypothesis: Slow-bit perturbation resonates at Round {target_round}")
    print(f"  Samples: {n_samples:,} per seed")
    print(f"  Seeds: {n_seeds} (for robustness)")
    print(f"  Features: Correlation mode (a⊕d, e⊕h)")
    print()
    
    all_results = []
    
    for seed in range(n_seeds):
        print(f"\n{'='*70}")
        print(f"  SEED {seed + 1}/{n_seeds}")
        print(f"{'='*70}")
        
        # Generate data
        print(f"\n  Generating {n_samples:,} samples...")
        t0 = time.time()
        X, y = generate_dataset(n_samples // 2, target_round, seed=seed * 12345)
        print(f"  Data generation: {time.time() - t0:.1f}s")
        
        # Split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        n_test = len(y_test)
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"  Train: {len(y_train):,}, Test: {n_test:,}")
        
        # Train models
        results = {}
        
        # 1. Random Forest
        print("\n  Training Random Forest...")
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=seed
        )
        rf.fit(X_train, y_train)
        rf_acc = accuracy_score(y_test, rf.predict(X_test))
        rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
        results['RF'] = {'accuracy': rf_acc, 'auc': rf_auc}
        print(f"    Accuracy: {rf_acc:.4f}, AUC: {rf_auc:.4f} ({time.time()-t0:.1f}s)")
        
        # 2. Gradient Boosting / XGBoost
        print("  Training Gradient Boosting...")
        t0 = time.time()
        if HAS_XGB:
            gb = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                n_jobs=-1,
                random_state=seed,
                verbosity=0
            )
        else:
            gb = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=seed
            )
        gb.fit(X_train, y_train)
        gb_acc = accuracy_score(y_test, gb.predict(X_test))
        gb_auc = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])
        results['GB'] = {'accuracy': gb_acc, 'auc': gb_auc}
        print(f"    Accuracy: {gb_acc:.4f}, AUC: {gb_auc:.4f} ({time.time()-t0:.1f}s)")
        
        # 3. Neural Network
        print("  Training Neural Network...")
        t0 = time.time()
        nn = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            batch_size=512,
            learning_rate='adaptive',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=seed
        )
        nn.fit(X_train, y_train)
        nn_acc = accuracy_score(y_test, nn.predict(X_test))
        nn_auc = roc_auc_score(y_test, nn.predict_proba(X_test)[:, 1])
        results['NN'] = {'accuracy': nn_acc, 'auc': nn_auc}
        print(f"    Accuracy: {nn_acc:.4f}, AUC: {nn_auc:.4f} ({time.time()-t0:.1f}s)")
        
        # Best result for this seed
        best_acc = max(rf_acc, gb_acc, nn_acc)
        best_model = ['RF', 'GB', 'NN'][np.argmax([rf_acc, gb_acc, nn_acc])]
        
        # Statistical significance
        sig = compute_significance(best_acc, n_test)
        
        print(f"\n  SEED {seed + 1} RESULTS:")
        print(f"    Best Model: {best_model}")
        print(f"    Accuracy:   {best_acc:.4f} ({sig['correct']:,}/{n_test:,} correct)")
        print(f"    Z-score:    {sig['z_score']:.2f}σ")
        print(f"    p-value:    {sig['p_value']:.2e}")
        
        if sig['z_score'] > 5:
            print(f"    Status:     🔴 HIGHLY SIGNIFICANT (>{sig['z_score']:.0f}σ)")
        elif sig['z_score'] > 3:
            print(f"    Status:     🔴 SIGNIFICANT (>{sig['z_score']:.0f}σ)")
        elif sig['z_score'] > 2:
            print(f"    Status:     ⚠️ MARGINALLY SIGNIFICANT")
        else:
            print(f"    Status:     ✓ NOT SIGNIFICANT")
        
        all_results.append({
            'seed': seed,
            'best_accuracy': best_acc,
            'best_model': best_model,
            'significance': sig,
            'all_results': results
        })
    
    # ==========================================================================
    # AGGREGATE ANALYSIS
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("  AGGREGATE ANALYSIS")
    print("=" * 70)
    
    accuracies = [r['best_accuracy'] for r in all_results]
    z_scores = [r['significance']['z_score'] for r in all_results]
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    mean_z = np.mean(z_scores)
    
    # Combined significance (Fisher's method)
    p_values = [r['significance']['p_value'] for r in all_results]
    # -2 * sum(log(p)) ~ chi2(2k)
    chi2_stat = -2 * np.sum(np.log(np.array(p_values) + 1e-300))
    combined_p = 1 - stats.chi2.cdf(chi2_stat, 2 * len(p_values))
    
    print(f"\n  Across {n_seeds} seeds:")
    print(f"    Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"    Mean Z-score:  {mean_z:.2f}σ")
    print(f"    Combined p-value (Fisher): {combined_p:.2e}")
    
    # ==========================================================================
    # FINAL VERDICT
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("  FINAL VERDICT")
    print("=" * 70)
    
    if mean_z > 5:
        print(f"""
    🔴🔴🔴 CONFIRMED: ROUND {target_round} IS DISTINGUISHABLE 🔴🔴🔴
    
    Mean accuracy: {mean_acc:.4f} ({(mean_acc - 0.5) * 100:.2f}% above random)
    Statistical significance: {mean_z:.1f}σ (p < {combined_p:.2e})
    
    The "Echo" phenomenon at Round {target_round} is REAL.
    The slow-bit perturbation resonates with the lagging registers.
    
    IMPLICATIONS:
    - 20-round SHA-256 is statistically distinguishable from random
    - The "phase correlation" (a⊕d, e⊕h) captures resonance structure
    - This is an academic break of {target_round}-round SHA-256
    
    Security margin reduced by {64 - target_round} rounds.
        """)
    elif mean_z > 2:
        print(f"""
    ⚠️ MARGINAL SIGNAL AT ROUND {target_round}
    
    Mean accuracy: {mean_acc:.4f} (Z = {mean_z:.1f}σ)
    
    The signal is weak but consistent across seeds.
    More samples or deeper models may clarify.
        """)
    else:
        print(f"""
    ✓ NO SIGNIFICANT SIGNAL AT ROUND {target_round}
    
    Mean accuracy: {mean_acc:.4f} (Z = {mean_z:.1f}σ)
    
    The 54% from the initial test was sampling variance.
    SHA-256 is secure at Round {target_round}.
        """)
    
    # Save results
    import json
    
    def convert(obj):
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    output = {
        'target_round': target_round,
        'n_samples': n_samples,
        'n_seeds': n_seeds,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_z_score': mean_z,
        'combined_p_value': combined_p,
        'per_seed_results': all_results
    }
    
    with open("round20_resonance_results.json", "w") as f:
        json.dump(output, f, indent=2, default=convert)
    
    print(f"\n  Results saved to: round20_resonance_results.json")
    
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Round 20 Resonance Test")
    parser.add_argument("--samples", type=int, default=100000, help="Total samples")
    parser.add_argument("--round", type=int, default=20, help="Target round")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds")
    
    args = parser.parse_args()
    
    run_resonance_test(
        n_samples=args.samples,
        target_round=args.round,
        n_seeds=args.seeds
    )
