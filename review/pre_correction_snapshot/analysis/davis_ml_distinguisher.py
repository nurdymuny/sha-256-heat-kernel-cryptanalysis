#!/usr/bin/env python3
"""
DAVIS MANIFOLD ML DISTINGUISHER
================================

Combines three discoveries:
1. Slow Bits (from Heat Kernel): {1, 5, 7, 9, 10, 14} aligned with Σ₀ rotations
2. Lagging Registers (from Red Team): d and h trail behind a, e
3. Spectral Geometry: Low-dimensional structure persists to round 4

Strategy:
- Feed slow-bit-derived inputs through reduced-round SHA-256
- Train ML models ONLY on lagging registers (d, h)
- Test if AI can distinguish Round 18-22 outputs from true random

If accuracy > 50% at Round 19+, we've pushed beyond the algebraic cliff.

Author: Bee Davis
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import struct
from scipy.linalg import eigh

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '.')
from src.embeddings.hyperbolic import HyperbolicEmbedding


# =============================================================================
# SHA-256 REDUCED ROUNDS (Direct Implementation)
# =============================================================================

def rotr(x, n):
    return (x >> n) | (x << (32 - n)) & 0xFFFFFFFF


def sha256_state_at_round(message_bytes: bytes, target_round: int) -> Tuple[List[int], np.ndarray]:
    """
    Returns (8-word state, 256-bit vector) at specified round.
    """
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

    state = [a, b, c, d, e, f, g, h]
    
    # Convert to 256-bit vector
    bits = []
    for word in state:
        for bit_pos in range(32):
            bits.append((word >> bit_pos) & 1)
    
    return state, np.array(bits)


# =============================================================================
# DAVIS MANIFOLD FEATURE EXTRACTOR
# =============================================================================

class DavisManifoldFeatures:
    """
    Extract features based on Davis manifold insights:
    - Lagging register values (d, h)
    - Slow bit correlations
    - Σ₀ rotation orbit structure
    """
    
    # Slow bits from heat kernel analysis
    SLOW_BITS = [1, 5, 7, 9, 10, 14]
    
    # Σ₀ rotation constants
    SIGMA0_ROTS = [2, 13, 22]
    
    # Σ₁ rotation constants  
    SIGMA1_ROTS = [6, 11, 25]
    
    def __init__(self):
        self.embedding = HyperbolicEmbedding(curvature=-1.0)
    
    def extract_lagging_registers(self, state: List[int]) -> np.ndarray:
        """
        Extract only the lagging registers d (index 3) and h (index 7).
        These trail behind and may retain structure longer.
        """
        d = state[3]
        h = state[7]
        
        # Extract all 64 bits from d and h
        features = []
        for word in [d, h]:
            for bit_pos in range(32):
                features.append((word >> bit_pos) & 1)
        
        return np.array(features)
    
    def extract_sigma0_orbits(self, state: List[int]) -> np.ndarray:
        """
        Extract bits at Σ₀ rotation positions from word 'a'.
        These are the "constraint channels" from heat kernel.
        """
        a = state[0]
        
        features = []
        for rot in self.SIGMA0_ROTS:
            features.append((a >> rot) & 1)
            features.append((a >> ((rot + 16) % 32)) & 1)  # Opposite bit
        
        # Also XOR combinations (Σ₀ structure)
        for i, r1 in enumerate(self.SIGMA0_ROTS):
            for r2 in self.SIGMA0_ROTS[i+1:]:
                b1 = (a >> r1) & 1
                b2 = (a >> r2) & 1
                features.append(b1 ^ b2)
        
        return np.array(features)
    
    def extract_injection_correlation(self, state: List[int]) -> np.ndarray:
        """
        Extract correlation between injection points (a, e) and laggards (d, h).
        The "standing wave" pattern suggests these are out of phase.
        """
        a, d, e, h = state[0], state[3], state[4], state[7]
        
        features = []
        
        # XOR correlations between leader and laggard pairs
        for bit_pos in range(32):
            a_bit = (a >> bit_pos) & 1
            d_bit = (d >> bit_pos) & 1
            e_bit = (e >> bit_pos) & 1
            h_bit = (h >> bit_pos) & 1
            
            # a-d correlation (should be out of phase)
            features.append(a_bit ^ d_bit)
            # e-h correlation (should be out of phase)
            features.append(e_bit ^ h_bit)
            # Cross correlations
            features.append(a_bit ^ h_bit)
            features.append(e_bit ^ d_bit)
        
        return np.array(features)
    
    def extract_all_features(self, state: List[int], mode: str = 'full') -> np.ndarray:
        """
        Extract features based on mode:
        - 'lagging': Only d and h (64 bits)
        - 'sigma0': Only Σ₀ orbit structure (9 features)
        - 'correlation': Leader-laggard correlations (128 features)
        - 'full': All combined
        """
        if mode == 'lagging':
            return self.extract_lagging_registers(state)
        elif mode == 'sigma0':
            return self.extract_sigma0_orbits(state)
        elif mode == 'correlation':
            return self.extract_injection_correlation(state)
        elif mode == 'full':
            return np.concatenate([
                self.extract_lagging_registers(state),
                self.extract_sigma0_orbits(state),
                self.extract_injection_correlation(state)
            ])
        else:
            raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# ML DISTINGUISHER
# =============================================================================

class DavisMLDistinguisher:
    """
    Train ML models to distinguish SHA-256 outputs from random.
    Uses Davis manifold features targeting lagging registers.
    """
    
    def __init__(self, target_round: int = 19):
        self.target_round = target_round
        self.feature_extractor = DavisManifoldFeatures()
        self.slow_bits = DavisManifoldFeatures.SLOW_BITS
        
    def generate_slow_bit_samples(self, n_samples: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate samples using slow-bit-constrained inputs.
        Returns (SHA-256 features, random features).
        
        CRITICAL FIX: Generate FRESH random base message for EVERY sample.
        The old code used a static base, causing the model to learn
        "this one cluster vs everything else" instead of universal structure.
        """
        np.random.seed(seed)
        
        sha_features = []
        random_features = []
        
        for i in range(n_samples):
            # FIX: Fresh random base for EVERY sample
            base_msg = np.random.bytes(55)
            msg = bytearray(base_msg)
            
            # Apply slow-bit perturbation pattern
            for bit in self.slow_bits:
                if np.random.rand() > 0.5:
                    byte_idx = bit // 8
                    bit_idx = 7 - (bit % 8)  # Fix bit ordering (MSB first)
                    if byte_idx < len(msg):
                        msg[byte_idx] ^= (1 << bit_idx)
            
            # Get SHA-256 state at target round
            state, _ = sha256_state_at_round(bytes(msg), self.target_round)
            sha_features.append(state)
            
            # For random class: also hash a DIFFERENT random message
            # This tests manifold vs manifold, not manifold vs uniform noise
            random_msg = np.random.bytes(55)
            random_state, _ = sha256_state_at_round(random_msg, self.target_round)
            random_features.append(random_state)
        
        return sha_features, random_features
    
    def prepare_dataset(self, n_samples: int = 2000, 
                       feature_mode: str = 'lagging') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare labeled dataset for training.
        Label 1 = SHA-256, Label 0 = Random
        """
        sha_states, random_states = self.generate_slow_bit_samples(n_samples)
        
        X = []
        y = []
        
        for state in sha_states:
            features = self.feature_extractor.extract_all_features(state, mode=feature_mode)
            X.append(features)
            y.append(1)
        
        for state in random_states:
            features = self.feature_extractor.extract_all_features(state, mode=feature_mode)
            X.append(features)
            y.append(0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Shuffle
        perm = np.random.permutation(len(y))
        return X[perm], y[perm]
    
    def train_and_evaluate(self, n_samples: int = 2000, 
                          feature_mode: str = 'lagging') -> Dict:
        """
        Train multiple ML models and evaluate distinguishing power.
        """
        print(f"\n    Preparing dataset (n={n_samples}, features={feature_mode})...")
        X, y = self.prepare_dataset(n_samples, feature_mode)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split
        split_idx = int(0.8 * len(y))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        results = {}
        
        # Model 1: Random Forest
        print("    Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_prob = rf.predict_proba(X_test)[:, 1]
        results['random_forest'] = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'auc': roc_auc_score(y_test, rf_prob)
        }
        
        # Model 2: Gradient Boosting
        print("    Training Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        gb_prob = gb.predict_proba(X_test)[:, 1]
        results['gradient_boosting'] = {
            'accuracy': accuracy_score(y_test, gb_pred),
            'auc': roc_auc_score(y_test, gb_prob)
        }
        
        # Model 3: Neural Network (deeper for subtle patterns)
        print("    Training Neural Network...")
        nn = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        nn.fit(X_train, y_train)
        nn_pred = nn.predict(X_test)
        nn_prob = nn.predict_proba(X_test)[:, 1]
        results['neural_network'] = {
            'accuracy': accuracy_score(y_test, nn_pred),
            'auc': roc_auc_score(y_test, nn_prob)
        }
        
        # Cross-validation for robustness
        print("    Running cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='accuracy')
        results['cv_mean'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()
        
        return results


def run_ml_distinguisher_attack():
    """
    Run the Davis Manifold ML Distinguisher across multiple rounds.
    """
    print("=" * 80)
    print("  DAVIS MANIFOLD ML DISTINGUISHER")
    print("  Targeting Lagging Registers with Slow-Bit Inputs")
    print("=" * 80)
    
    # Test rounds around the algebraic cliff
    test_rounds = [16, 17, 18, 19, 20, 21, 22, 24]
    feature_modes = ['lagging', 'correlation', 'full']
    
    all_results = {}
    
    for mode in feature_modes:
        print(f"\n{'='*80}")
        print(f"  FEATURE MODE: {mode.upper()}")
        print(f"{'='*80}")
        
        mode_results = {}
        
        for round_num in test_rounds:
            print(f"\n  --- Round {round_num} ---")
            
            distinguisher = DavisMLDistinguisher(target_round=round_num)
            results = distinguisher.train_and_evaluate(n_samples=2000, feature_mode=mode)
            
            best_acc = max(
                results['random_forest']['accuracy'],
                results['gradient_boosting']['accuracy'],
                results['neural_network']['accuracy']
            )
            best_auc = max(
                results['random_forest']['auc'],
                results['gradient_boosting']['auc'],
                results['neural_network']['auc']
            )
            
            mode_results[round_num] = {
                'best_accuracy': best_acc,
                'best_auc': best_auc,
                'cv_mean': results['cv_mean'],
                'cv_std': results['cv_std'],
                'details': results
            }
            
            # Status
            if best_acc > 0.55:
                status = "🔴 DISTINGUISHABLE"
            elif best_acc > 0.52:
                status = "⚠️ WEAK SIGNAL"
            else:
                status = "✓ Random"
            
            print(f"\n    Best Accuracy: {best_acc:.4f}")
            print(f"    Best AUC:      {best_auc:.4f}")
            print(f"    CV Mean:       {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")
            print(f"    Status:        {status}")
        
        all_results[mode] = mode_results
    
    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY: ML DISTINGUISHER RESULTS")
    print("=" * 80)
    
    print(f"\n  {'Round':<8}", end="")
    for mode in feature_modes:
        print(f" {mode[:8]:<12}", end="")
    print(" Status")
    print("  " + "-" * 60)
    
    for round_num in test_rounds:
        print(f"  {round_num:<8}", end="")
        max_acc = 0
        for mode in feature_modes:
            acc = all_results[mode][round_num]['best_accuracy']
            max_acc = max(max_acc, acc)
            print(f" {acc:<12.4f}", end="")
        
        if max_acc > 0.55:
            print(" 🔴 BROKEN")
        elif max_acc > 0.52:
            print(" ⚠️ WEAK")
        else:
            print(" ✓ OK")
    
    # Find the security cliff
    print("\n" + "=" * 80)
    print("  SECURITY CLIFF ANALYSIS")
    print("=" * 80)
    
    cliff_round = None
    for round_num in sorted(test_rounds, reverse=True):
        max_acc = max(all_results[mode][round_num]['best_accuracy'] for mode in feature_modes)
        if max_acc > 0.52:
            cliff_round = round_num
            break
    
    if cliff_round:
        print(f"""
    🔴 ML DISTINGUISHER EXTENDS BEYOND ALGEBRAIC CLIFF
    
    The algebraic cube attack fails at Round 19.
    However, ML can still detect structure at Round {cliff_round}.
    
    This means:
    - The lagging registers (d, h) retain exploitable correlations
    - Neural networks can detect patterns invisible to algebraic tests
    - The TRUE security margin is Round {cliff_round + 1}
    
    Effective security reduction: {64 - cliff_round - 1} rounds lost
        """)
    else:
        print(f"""
    ✓ ML DISTINGUISHER CONFIRMS ALGEBRAIC CLIFF
    
    No model achieved >52% accuracy beyond Round {min(test_rounds)}.
    
    The lagging register hypothesis does not provide additional attack surface.
    SHA-256's avalanche effect defeats both algebraic AND statistical attacks
    by Round {min(r for r in test_rounds if all_results['full'][r]['best_accuracy'] < 0.52)}.
        """)
    
    return all_results


if __name__ == "__main__":
    results = run_ml_distinguisher_attack()
    
    # Save results
    import json
    
    # Convert to JSON-serializable format
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open("ml_distinguisher_results.json", "w") as f:
        json.dump(results, f, indent=2, default=convert)
    
    print("\n  Results saved to: ml_distinguisher_results.json")
