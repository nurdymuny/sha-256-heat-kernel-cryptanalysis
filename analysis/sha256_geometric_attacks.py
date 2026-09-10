#!/usr/bin/env python3
"""
SHA-256 ADVANCED GEOMETRIC ATTACKS
==================================
Four targeted attacks based on heat kernel analysis findings:

1. Slow-Manifold Gradient Attack - geodesic-guided differential search
2. Σ-Rotational Cube Attack - targeted cube attack on slow bits
3. Round 24 ML Distinguisher - exploit the spectral gap bump
4. Subspace "Pancake" Probe - find dimensional collapse

Author: Bee Davis
Classification: CONFIDENTIAL - CRYPTANALYTIC RESEARCH
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '.')
from src.sha256.core import InstrumentedSHA256, SHA256Trajectory
from src.sha256.input_generator import InputGenerator
from src.embeddings.euclidean import EuclideanEmbedding
from src.embeddings.hyperbolic import HyperbolicEmbedding
from src.embeddings.davis import DavisManifoldEmbedding


# =============================================================================
# ATTACK 1: SLOW-MANIFOLD GRADIENT ATTACK
# =============================================================================

class SlowManifoldGradientAttack:
    """
    Use geodesic distance on the discovered manifold as a loss function.
    Goal: Find input pairs that diverge slower than random - "surf" the manifold.
    """
    
    def __init__(self, target_round: int = 24, embedding_type: str = 'hyperbolic'):
        self.target_round = target_round
        self.sha = InstrumentedSHA256(sample_rounds=[0, 8, 16, 24, 32, 48, 64])
        
        if embedding_type == 'hyperbolic':
            self.embedding = HyperbolicEmbedding(curvature=-1.0)
        elif embedding_type == 'davis':
            self.embedding = DavisManifoldEmbedding()
        else:
            self.embedding = EuclideanEmbedding()
            
        self.embedding_type = embedding_type
        self.results = {}
        
    def _message_to_bits(self, message: bytes) -> np.ndarray:
        """Convert message bytes to bit array."""
        bits = []
        for byte in message:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        return np.array(bits, dtype=np.float64)
    
    def _bits_to_message(self, bits: np.ndarray) -> bytes:
        """Convert bit array back to message bytes."""
        bits = np.clip(np.round(bits), 0, 1).astype(int)
        message = []
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                if i + j < len(bits):
                    byte = (byte << 1) | bits[i + j]
            message.append(byte)
        return bytes(message)
    
    def _compute_divergence(self, m1: bytes, m2: bytes, round_num: int) -> float:
        """Compute geodesic divergence between two messages at specified round."""
        traj1 = self.sha.hash_with_trajectory(m1)
        traj2 = self.sha.hash_with_trajectory(m2)
        
        state1 = next((s for s in traj1.states if s.round_num == round_num), None)
        state2 = next((s for s in traj2.states if s.round_num == round_num), None)
        
        if state1 is None or state2 is None:
            # Find closest round
            available = [s.round_num for s in traj1.states]
            closest = min(available, key=lambda x: abs(x - round_num))
            state1 = next(s for s in traj1.states if s.round_num == closest)
            state2 = next(s for s in traj2.states if s.round_num == closest)
        
        p1 = self.embedding.embed(state1.state_vector)
        p2 = self.embedding.embed(state2.state_vector)
        
        return self.embedding.distance(p1, p2)
    
    def _loss_function(self, delta_bits: np.ndarray, base_message: bytes) -> float:
        """
        Loss = geodesic distance at target round.
        We want to MINIMIZE this - find pairs that stay close.
        """
        # Create modified message
        base_bits = self._message_to_bits(base_message)
        
        # Apply delta (treating delta as which bits to flip)
        # Use sigmoid to make it differentiable-ish
        flip_probs = 1 / (1 + np.exp(-delta_bits))
        modified_bits = base_bits.copy()
        
        # Flip bits where delta > 0
        flip_mask = delta_bits > 0
        modified_bits[flip_mask] = 1 - modified_bits[flip_mask]
        
        modified_message = self._bits_to_message(modified_bits)
        
        # Compute divergence
        divergence = self._compute_divergence(base_message, modified_message, self.target_round)
        
        # We want small divergence, but not zero (trivial case)
        # Add penalty for no change
        n_flips = np.sum(flip_mask)
        if n_flips == 0:
            return 1000.0  # Penalty for no change
        
        # Normalize by number of flips to find efficient flip patterns
        return divergence / np.sqrt(n_flips)
    
    def find_slow_diverging_pair(self, n_attempts: int = 100, 
                                   message_length: int = 55) -> Dict:
        """
        Search for input pairs with anomalously slow divergence.
        """
        print(f"\n[Attack 1] Slow-Manifold Gradient Attack")
        print(f"  Target round: {self.target_round}")
        print(f"  Embedding: {self.embedding_type}")
        print(f"  Searching for slow-diverging pairs...")
        
        gen = InputGenerator(seed=42)
        best_results = []
        baseline_divergences = []
        
        # First, establish baseline divergence for random 1-bit flips
        print("  Computing baseline divergence...")
        for _ in range(100):
            pair = gen.hamming_pairs(1)[0]
            m1, m2 = pair.message_a, pair.message_b
            div = self._compute_divergence(m1, m2, self.target_round)
            baseline_divergences.append(div)
        
        baseline_mean = np.mean(baseline_divergences)
        baseline_std = np.std(baseline_divergences)
        print(f"  Baseline: {baseline_mean:.4f} ± {baseline_std:.4f}")
        
        # Now search for slow pairs
        print(f"  Running {n_attempts} optimization attempts...")
        
        for attempt in range(n_attempts):
            if attempt % 20 == 0:
                print(f"    Attempt {attempt}/{n_attempts}...")
            
            # Random base message
            base_message = gen.random_batch(1)[0][:message_length]
            
            # Try different flip patterns
            best_div = float('inf')
            best_pattern = None
            best_modified = None
            
            # Strategy 1: Flip slow bits only
            slow_bits = [1, 5, 7, 9, 10, 14, 16, 19, 21, 26, 28, 31]  # From our analysis
            for bit in slow_bits:
                if bit < len(base_message) * 8:
                    # Flip this bit
                    base_bits = self._message_to_bits(base_message)
                    base_bits[bit] = 1 - base_bits[bit]
                    modified = self._bits_to_message(base_bits)
                    
                    div = self._compute_divergence(base_message, modified, self.target_round)
                    if div < best_div:
                        best_div = div
                        best_pattern = [bit]
                        best_modified = modified
            
            # Strategy 2: Flip combinations of slow bits
            for i, bit1 in enumerate(slow_bits[:5]):
                for bit2 in slow_bits[i+1:6]:
                    if bit1 < len(base_message) * 8 and bit2 < len(base_message) * 8:
                        base_bits = self._message_to_bits(base_message)
                        base_bits[bit1] = 1 - base_bits[bit1]
                        base_bits[bit2] = 1 - base_bits[bit2]
                        modified = self._bits_to_message(base_bits)
                        
                        div = self._compute_divergence(base_message, modified, self.target_round)
                        # Normalize by sqrt(2) for fair comparison
                        normalized_div = div  # Raw distance on both sides; no unmatched sqrt(2) scaling
                        if normalized_div < best_div:
                            best_div = normalized_div
                            best_pattern = [bit1, bit2]
                            best_modified = modified
            
            if best_modified is not None:
                best_results.append({
                    'divergence': best_div,
                    'pattern': best_pattern,
                    'z_score': (best_div - baseline_mean) / baseline_std
                })
        
        # Analyze results
        divergences = [r['divergence'] for r in best_results]
        z_scores = [r['z_score'] for r in best_results]
        
        # Find anomalously slow pairs
        anomalous = [r for r in best_results if r['z_score'] < -2.0]
        
        results = {
            'baseline_mean': baseline_mean,
            'baseline_std': baseline_std,
            'search_mean': np.mean(divergences),
            'search_min': np.min(divergences),
            'search_max': np.max(divergences),
            'n_anomalous': len(anomalous),
            'best_z_score': min(z_scores),
            'anomalous_pairs': anomalous[:10],  # Top 10
            'success': len(anomalous) > 0
        }
        
        print(f"\n  Results:")
        print(f"    Baseline mean: {baseline_mean:.4f}")
        print(f"    Search min:    {results['search_min']:.4f}")
        print(f"    Best z-score:  {results['best_z_score']:.4f}")
        print(f"    Anomalous pairs (z < -2): {len(anomalous)}")
        
        if results['success']:
            print(f"  ⚠️  FOUND SLOW-DIVERGING PAIRS!")
        else:
            print(f"  ✓ No anomalous pairs found")
        
        self.results = results
        return results


# =============================================================================
# ATTACK 2: Σ-ROTATIONAL CUBE ATTACK
# =============================================================================

class SigmaRotationalCubeAttack:
    """
    Cube attack targeting the slow bits in Maj region.
    Check if algebraic degree is lower than expected.
    """
    
    def __init__(self, target_rounds: int = 24):
        self.target_rounds = target_rounds
        self.sha = InstrumentedSHA256(sample_rounds=[target_rounds])
        
        # Slow bits from our analysis (Σ₀ rotation related)
        self.slow_bits = [1, 5, 7, 9, 10, 14, 16, 19, 21, 26, 28, 31]
        
        # Σ₀ rotation constants
        self.sigma0_constants = [2, 13, 22]
        
        self.results = {}
        
    def _create_cube_inputs(self, cube_vars: List[int], base_message: bytes) -> List[bytes]:
        """
        Generate all 2^k inputs for cube variables.
        """
        k = len(cube_vars)
        inputs = []
        
        base_bits = []
        for byte in base_message:
            for i in range(8):
                base_bits.append((byte >> (7 - i)) & 1)
        base_bits = np.array(base_bits)
        
        for i in range(2 ** k):
            bits = base_bits.copy()
            for j, var in enumerate(cube_vars):
                if var < len(bits):
                    bits[var] = (i >> j) & 1
            
            # Convert back to bytes
            message = []
            for idx in range(0, len(bits), 8):
                byte = 0
                for b in range(8):
                    if idx + b < len(bits):
                        byte = (byte << 1) | bits[idx + b]
                message.append(byte)
            inputs.append(bytes(message))
        
        return inputs
    
    def _compute_cube_sum(self, cube_vars: List[int], base_message: bytes,
                          output_bit: int) -> int:
        """
        Compute the cube sum for a specific output bit.
        If sum = 0, algebraic degree is lower than expected.
        """
        inputs = self._create_cube_inputs(cube_vars, base_message)
        
        bit_sum = 0
        for inp in inputs:
            traj = self.sha.hash_with_trajectory(inp)
            state = traj.states[-1]  # Get state at target round
            
            # Get output bit
            if output_bit < len(state.state_vector):
                bit_sum ^= state.state_vector[output_bit]
        
        return bit_sum
    
    def run_cube_attack(self, cube_size: int = 4, n_trials: int = 50,
                        use_slow_bits: bool = True) -> Dict:
        """
        Run cube attack with either slow bits or random bits.
        """
        print(f"\n[Attack 2] Σ-Rotational Cube Attack")
        print(f"  Target rounds: {self.target_rounds}")
        print(f"  Cube size: {cube_size}")
        print(f"  Using slow bits: {use_slow_bits}")
        
        gen = InputGenerator(seed=42)
        
        zero_sums_slow = 0
        zero_sums_random = 0
        total_tests = 0
        
        results_detail = []
        
        for trial in range(n_trials):
            if trial % 10 == 0:
                print(f"    Trial {trial}/{n_trials}...")
            
            base_message = gen.random_batch(1)[0][:55]
            
            # Test with slow bits
            if use_slow_bits:
                cube_vars = self.slow_bits[:cube_size]
            else:
                np.random.seed(trial)
                cube_vars = list(np.random.choice(range(64), cube_size, replace=False))
            
            # Test multiple output bits
            for output_bit in [0, 32, 64, 128, 160, 192, 224]:  # Sample from each word
                cube_sum = self._compute_cube_sum(cube_vars, base_message, output_bit)
                total_tests += 1
                
                if cube_sum == 0:
                    if use_slow_bits:
                        zero_sums_slow += 1
                    else:
                        zero_sums_random += 1
                    
                    results_detail.append({
                        'trial': trial,
                        'cube_vars': cube_vars,
                        'output_bit': output_bit,
                        'sum': cube_sum,
                        'type': 'slow' if use_slow_bits else 'random'
                    })
        
        # Expected: for random function, P(sum=0) = 0.5
        expected_zeros = total_tests * 0.5
        
        if use_slow_bits:
            observed_zeros = zero_sums_slow
        else:
            observed_zeros = zero_sums_random
        
        # Chi-squared test
        observed = [observed_zeros, total_tests - observed_zeros]
        expected = [expected_zeros, expected_zeros]
        
        if expected_zeros > 0:
            chi2 = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
        else:
            chi2 = 0
        
        results = {
            'cube_size': cube_size,
            'n_trials': n_trials,
            'total_tests': total_tests,
            'zero_sums': observed_zeros,
            'expected_zeros': expected_zeros,
            'zero_rate': observed_zeros / total_tests if total_tests > 0 else 0,
            'expected_rate': 0.5,
            'chi_squared': chi2,
            'bias': (observed_zeros / total_tests - 0.5) if total_tests > 0 else 0,
            'significant': chi2 > 3.84,  # p < 0.05
            'details': results_detail[:20]
        }
        
        print(f"\n  Results:")
        print(f"    Total tests: {total_tests}")
        print(f"    Zero sums: {observed_zeros} ({results['zero_rate']:.2%})")
        print(f"    Expected: {expected_zeros:.0f} (50%)")
        print(f"    Bias: {results['bias']:+.4f}")
        print(f"    χ²: {chi2:.4f}")
        
        if results['significant']:
            print(f"  ⚠️  SIGNIFICANT BIAS DETECTED!")
        else:
            print(f"  ✓ No significant bias")
        
        self.results = results
        return results
    
    def compare_slow_vs_random(self, cube_size: int = 4, n_trials: int = 30) -> Dict:
        """
        Compare cube attack success with slow bits vs random bits.
        """
        print(f"\n[Attack 2b] Comparing Slow Bits vs Random Bits")
        
        # Run with slow bits
        print("\n  --- Using SLOW bits ---")
        slow_results = self.run_cube_attack(cube_size, n_trials, use_slow_bits=True)
        
        # Run with random bits
        print("\n  --- Using RANDOM bits ---")
        random_results = self.run_cube_attack(cube_size, n_trials, use_slow_bits=False)
        
        comparison = {
            'slow_zero_rate': slow_results['zero_rate'],
            'random_zero_rate': random_results['zero_rate'],
            'slow_bias': slow_results['bias'],
            'random_bias': random_results['bias'],
            'slow_advantage': slow_results['zero_rate'] - random_results['zero_rate']
        }
        
        print(f"\n  Comparison:")
        print(f"    Slow bit zero rate:   {comparison['slow_zero_rate']:.2%}")
        print(f"    Random bit zero rate: {comparison['random_zero_rate']:.2%}")
        print(f"    Slow bit advantage:   {comparison['slow_advantage']:+.4f}")
        
        return comparison


# =============================================================================
# ATTACK 3: ROUND 24 ML DISTINGUISHER
# =============================================================================

class Round24MLDistinguisher:
    """
    Train ML classifier to distinguish Round 24 state from random.
    Exploits the spectral gap bump at round 24.
    """
    
    def __init__(self, target_round: int = 24):
        self.target_round = target_round
        self.sha = InstrumentedSHA256(sample_rounds=[0, 8, 16, 24, 32, 48, 64])
        self.gen = InputGenerator()
        self.results = {}
        
    def generate_data(self, n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate labeled dataset:
        - Class 1: SHA-256 state at target round
        - Class 0: Truly random 256-bit vectors
        """
        print(f"  Generating {n_samples} samples each...")
        
        # SHA-256 states at target round
        gen_seeded = InputGenerator(seed=42)
        inputs = gen_seeded.random_batch(n_samples)
        trajectories = self.sha.hash_batch(inputs)
        
        sha_states = []
        for traj in trajectories:
            state = next((s for s in traj.states if s.round_num == self.target_round), None)
            if state is None:
                # Find closest
                available = [s.round_num for s in traj.states]
                closest = min(available, key=lambda x: abs(x - self.target_round))
                state = next(s for s in traj.states if s.round_num == closest)
            sha_states.append(state.state_vector)
        
        sha_states = np.array(sha_states)
        
        # Random states
        np.random.seed(123)
        random_states = np.random.randint(0, 2, size=(n_samples, 256)).astype(np.uint8)
        
        # Combine
        X = np.vstack([sha_states, random_states])
        y = np.array([1] * n_samples + [0] * n_samples)
        
        return X, y
    
    def train_and_evaluate(self, n_samples: int = 5000) -> Dict:
        """
        Train multiple classifiers and evaluate distinguishing power.
        """
        print(f"\n[Attack 3] Round {self.target_round} ML Distinguisher")
        
        X, y = self.generate_data(n_samples)
        
        # Shuffle
        idx = np.random.permutation(len(y))
        X, y = X[idx], y[idx]
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        results = {}
        
        # Classifier 1: Random Forest
        print("  Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_prob = rf.predict_proba(X_test)[:, 1]
        
        results['random_forest'] = {
            'accuracy': float(accuracy_score(y_test, rf_pred)),
            'auc': float(roc_auc_score(y_test, rf_prob))
        }
        
        # Classifier 2: Gradient Boosting
        print("  Training Gradient Boosting...")
        gb = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
        gb.fit(X_train, y_train)
        gb_pred = gb.predict(X_test)
        gb_prob = gb.predict_proba(X_test)[:, 1]
        
        results['gradient_boosting'] = {
            'accuracy': float(accuracy_score(y_test, gb_pred)),
            'auc': float(roc_auc_score(y_test, gb_prob))
        }
        
        # Classifier 3: Neural Network
        print("  Training Neural Network...")
        nn = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42)
        nn.fit(X_train, y_train)
        nn_pred = nn.predict(X_test)
        nn_prob = nn.predict_proba(X_test)[:, 1]
        
        results['neural_network'] = {
            'accuracy': float(accuracy_score(y_test, nn_pred)),
            'auc': float(roc_auc_score(y_test, nn_prob))
        }
        
        # Cross-validation for robustness
        print("  Running cross-validation...")
        cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
        results['cross_validation'] = {
            'mean': float(cv_scores.mean()),
            'std': float(cv_scores.std()),
            'scores': cv_scores.tolist()
        }
        
        # Feature importance analysis
        importance = rf.feature_importances_
        top_bits = np.argsort(importance)[-20:][::-1]
        
        # Map to SHA-256 words
        word_importance = np.zeros(8)
        for i in range(256):
            word_importance[i // 32] += importance[i]
        
        results['feature_analysis'] = {
            'top_predictive_bits': top_bits.tolist(),
            'top_importances': importance[top_bits].tolist(),
            'word_importance': {
                'words': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
                'importance': word_importance.tolist()
            }
        }
        
        # Summary
        best_accuracy = max(results['random_forest']['accuracy'],
                           results['gradient_boosting']['accuracy'],
                           results['neural_network']['accuracy'])
        
        results['summary'] = {
            'best_accuracy': best_accuracy,
            'baseline': 0.5,
            'advantage': best_accuracy - 0.5,
            'distinguishable': best_accuracy > 0.55
        }
        
        print(f"\n  Results for Round {self.target_round}:")
        print(f"    Random Forest:     {results['random_forest']['accuracy']:.4f}")
        print(f"    Gradient Boosting: {results['gradient_boosting']['accuracy']:.4f}")
        print(f"    Neural Network:    {results['neural_network']['accuracy']:.4f}")
        print(f"    Cross-val mean:    {results['cross_validation']['mean']:.4f} ± {results['cross_validation']['std']:.4f}")
        print(f"    Baseline:          0.5000")
        
        if results['summary']['distinguishable']:
            print(f"\n  ⚠️  ROUND {self.target_round} IS DISTINGUISHABLE FROM RANDOM!")
            print(f"      Advantage: {results['summary']['advantage']:.4f}")
        else:
            print(f"\n  ✓ Round {self.target_round} appears random")
        
        self.results = results
        return results
    
    def compare_rounds(self, rounds: List[int] = [8, 16, 24, 32, 48, 64],
                       n_samples: int = 3000) -> Dict:
        """
        Compare distinguishability across different rounds.
        """
        print(f"\n[Attack 3b] Comparing Distinguishability Across Rounds")
        
        comparison = {}
        
        for r in rounds:
            print(f"\n  --- Round {r} ---")
            self.target_round = r
            self.sha = InstrumentedSHA256(sample_rounds=[r])
            result = self.train_and_evaluate(n_samples)
            comparison[r] = result['summary']
        
        # Summary table
        print(f"\n  Round Comparison Summary:")
        print(f"  {'Round':<8} {'Accuracy':<12} {'Advantage':<12} {'Status'}")
        print(f"  {'-'*50}")
        for r in rounds:
            acc = comparison[r]['best_accuracy']
            adv = comparison[r]['advantage']
            status = "⚠️  WEAK" if comparison[r]['distinguishable'] else "✓ OK"
            print(f"  {r:<8} {acc:<12.4f} {adv:<+12.4f} {status}")
        
        return comparison


# =============================================================================
# ATTACK 4: SUBSPACE "PANCAKE" PROBE
# =============================================================================

class SubspacePancakeProbe:
    """
    Analyze if intermediate states live in a lower-dimensional subspace.
    If so, collision search complexity drops.
    """
    
    def __init__(self):
        self.sha = InstrumentedSHA256(sample_rounds=[0, 8, 16, 24, 28, 32, 48, 64])
        self.gen = InputGenerator()
        self.results = {}
        
    def analyze_subspace(self, n_samples: int = 5000,
                         target_rounds: List[int] = [24, 28, 32]) -> Dict:
        """
        Perform PCA on intermediate states to find effective dimensionality.
        """
        print(f"\n[Attack 4] Subspace 'Pancake' Probe")
        print(f"  Analyzing {n_samples} samples...")
        
        gen_seeded = InputGenerator(seed=42)
        inputs = gen_seeded.random_batch(n_samples)
        trajectories = self.sha.hash_batch(inputs)
        
        results = {}
        
        for target_round in target_rounds:
            print(f"\n  --- Round {target_round} ---")
            
            # Collect states
            states = []
            for traj in trajectories:
                state = next((s for s in traj.states if s.round_num == target_round), None)
                if state is None:
                    available = [s.round_num for s in traj.states]
                    closest = min(available, key=lambda x: abs(x - target_round))
                    state = next(s for s in traj.states if s.round_num == closest)
                states.append(state.state_vector)
            
            states = np.array(states, dtype=np.float64)
            
            # PCA analysis
            pca = PCA()
            pca.fit(states)
            
            explained_var = pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            # Find effective dimension (95% variance)
            eff_dim_95 = np.argmax(cumulative_var >= 0.95) + 1
            eff_dim_99 = np.argmax(cumulative_var >= 0.99) + 1
            
            # Rank analysis
            # Check if data lies near a subspace
            singular_values = np.sqrt(pca.explained_variance_ * (n_samples - 1))
            rank_ratio = singular_values[0] / singular_values[-1] if singular_values[-1] > 0 else float('inf')
            
            # Compression test: can we reconstruct with fewer dimensions?
            reconstruction_errors = []
            for n_components in [50, 100, 150, 200]:
                pca_reduced = PCA(n_components=n_components)
                transformed = pca_reduced.fit_transform(states)
                reconstructed = pca_reduced.inverse_transform(transformed)
                error = np.mean((states - reconstructed) ** 2)
                reconstruction_errors.append({
                    'n_components': n_components,
                    'mse': float(error),
                    'compression_ratio': n_components / 256
                })
            
            results[target_round] = {
                'effective_dim_95': int(eff_dim_95),
                'effective_dim_99': int(eff_dim_99),
                'top_10_variance': float(cumulative_var[9]) if len(cumulative_var) > 9 else 1.0,
                'top_50_variance': float(cumulative_var[49]) if len(cumulative_var) > 49 else 1.0,
                'rank_ratio': float(rank_ratio),
                'reconstruction_errors': reconstruction_errors,
                'variance_curve': explained_var[:50].tolist()
            }
            
            print(f"    Effective dim (95%): {eff_dim_95}")
            print(f"    Effective dim (99%): {eff_dim_99}")
            print(f"    Top 50 components:   {results[target_round]['top_50_variance']:.2%} variance")
            print(f"    Rank ratio:          {rank_ratio:.2f}")
        
        # Random baseline
        print(f"\n  --- Random Baseline ---")
        random_states = np.random.randint(0, 2, size=(n_samples, 256)).astype(np.float64)
        pca_random = PCA()
        pca_random.fit(random_states)
        cumulative_random = np.cumsum(pca_random.explained_variance_ratio_)
        
        results['random_baseline'] = {
            'effective_dim_95': int(np.argmax(cumulative_random >= 0.95) + 1),
            'effective_dim_99': int(np.argmax(cumulative_random >= 0.99) + 1),
            'top_50_variance': float(cumulative_random[49]) if len(cumulative_random) > 49 else 1.0
        }
        
        print(f"    Effective dim (95%): {results['random_baseline']['effective_dim_95']}")
        print(f"    Effective dim (99%): {results['random_baseline']['effective_dim_99']}")
        
        # Summary
        print(f"\n  Summary:")
        dimensional_collapse = False
        for r in target_rounds:
            if results[r]['effective_dim_95'] < results['random_baseline']['effective_dim_95'] * 0.8:
                dimensional_collapse = True
                print(f"    ⚠️  Round {r}: Dimensional collapse detected!")
                print(f"       SHA-256 dim: {results[r]['effective_dim_95']} vs Random: {results['random_baseline']['effective_dim_95']}")
        
        if not dimensional_collapse:
            print(f"    ✓ No significant dimensional collapse detected")
        
        results['dimensional_collapse'] = dimensional_collapse
        
        self.results = results
        return results


# =============================================================================
# MASTER ATTACK RUNNER
# =============================================================================

def run_all_attacks(n_samples: int = 3000, output_file: str = "attack_results.json"):
    """
    Run all four attacks and compile results.
    """
    print("=" * 70)
    print("  SHA-256 GEOMETRIC ATTACK SUITE")
    print("=" * 70)
    
    all_results = {}
    
    # Attack 1: Slow-Manifold Gradient
    attack1 = SlowManifoldGradientAttack(target_round=24, embedding_type='hyperbolic')
    all_results['slow_manifold'] = attack1.find_slow_diverging_pair(n_attempts=50)
    
    # Attack 2: Σ-Rotational Cube Attack
    attack2 = SigmaRotationalCubeAttack(target_rounds=24)
    all_results['cube_attack'] = attack2.compare_slow_vs_random(cube_size=4, n_trials=20)
    
    # Attack 3: Round 24 ML Distinguisher
    attack3 = Round24MLDistinguisher(target_round=24)
    all_results['ml_distinguisher'] = attack3.compare_rounds(
        rounds=[16, 24, 32, 64], 
        n_samples=min(n_samples, 3000)
    )
    
    # Attack 4: Subspace Pancake Probe
    attack4 = SubspacePancakeProbe()
    all_results['subspace_probe'] = attack4.analyze_subspace(
        n_samples=min(n_samples, 5000),
        target_rounds=[24, 28, 32]
    )
    
    # Overall summary
    print("\n" + "=" * 70)
    print("  ATTACK RESULTS SUMMARY")
    print("=" * 70)
    
    findings = []
    
    if all_results['slow_manifold'].get('success'):
        findings.append("Slow-manifold: Found anomalous pairs")
    
    if all_results['cube_attack'].get('slow_advantage', 0) > 0.05:
        findings.append(f"Cube attack: Slow bit advantage = {all_results['cube_attack']['slow_advantage']:.4f}")
    
    for r, data in all_results['ml_distinguisher'].items():
        if isinstance(data, dict) and data.get('distinguishable'):
            findings.append(f"ML distinguisher: Round {r} distinguishable (acc={data['best_accuracy']:.4f})")
    
    if all_results['subspace_probe'].get('dimensional_collapse'):
        findings.append("Subspace probe: Dimensional collapse detected")
    
    if findings:
        print("\n  ⚠️  POTENTIAL WEAKNESSES FOUND:")
        for f in findings:
            print(f"    - {f}")
    else:
        print("\n  ✓ No exploitable weaknesses found")
        print("    The geometric structure is internal only")
    
    print("\n" + "=" * 70)
    
    # Save results
    import json
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SHA-256 Geometric Attack Suite")
    parser.add_argument('--n-samples', type=int, default=3000, help='Number of samples')
    parser.add_argument('--attack', type=str, default='all', 
                       choices=['all', '1', '2', '3', '4'],
                       help='Which attack to run')
    parser.add_argument('--output', type=str, default='attack_results.json')
    args = parser.parse_args()
    
    if args.attack == 'all':
        run_all_attacks(args.n_samples, args.output)
    elif args.attack == '1':
        attack = SlowManifoldGradientAttack(target_round=24)
        attack.find_slow_diverging_pair(n_attempts=100)
    elif args.attack == '2':
        attack = SigmaRotationalCubeAttack(target_rounds=24)
        attack.compare_slow_vs_random(cube_size=4, n_trials=30)
    elif args.attack == '3':
        attack = Round24MLDistinguisher(target_round=24)
        attack.compare_rounds(n_samples=args.n_samples)
    elif args.attack == '4':
        attack = SubspacePancakeProbe()
        attack.analyze_subspace(n_samples=args.n_samples)
