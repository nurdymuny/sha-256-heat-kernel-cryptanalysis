#!/usr/bin/env python3
"""
SHA-256 DISTINGUISHER TEST
==========================
The critical question: Can we tell SHA-256 output from true random
using the geometric signature we found?

If YES → Random oracle assumption violated → Real cryptographic weakness
If NO  → Geometric fingerprint is internal only → Not exploitable

Author: Bee Davis
Classification: CONFIDENTIAL
"""

import numpy as np
from typing import List, Tuple
from scipy.stats import ks_2samp, mannwhitneyu, ttest_ind
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '.')
from src.sha256.core import InstrumentedSHA256
from src.sha256.input_generator import InputGenerator
from src.embeddings.euclidean import EuclideanEmbedding
from src.analysis.laplacian import LaplacianConstructor
from src.analysis.heat_kernel import HeatKernelSolver


class DistinguisherTest:
    """
    Test whether SHA-256 outputs can be distinguished from random.
    """
    
    def __init__(self, n_samples: int = 5000):
        self.n_samples = n_samples
        self.sha = InstrumentedSHA256(sample_rounds=[64])  # Only need final state
        self.gen = InputGenerator(seed=42)
        self.embedding = EuclideanEmbedding()
        
    def generate_sha256_outputs(self) -> np.ndarray:
        """Generate SHA-256 hash outputs as bit vectors."""
        print(f"  Generating {self.n_samples} SHA-256 outputs...")
        inputs = self.gen.random_batch(self.n_samples)
        trajectories = self.sha.hash_batch(inputs)
        
        outputs = []
        for traj in trajectories:
            final_state = traj.states[-1]
            outputs.append(final_state.state_vector)
        
        return np.array(outputs)
    
    def generate_random_outputs(self) -> np.ndarray:
        """Generate uniformly random 256-bit strings."""
        print(f"  Generating {self.n_samples} random 256-bit strings...")
        np.random.seed(123)  # Different seed than SHA-256 inputs
        return np.random.randint(0, 2, size=(self.n_samples, 256)).astype(np.uint8)
    
    def test_statistical_distinguisher(self, sha_outputs: np.ndarray, 
                                        random_outputs: np.ndarray) -> dict:
        """
        Test 1: Basic statistical tests on bit distributions.
        """
        print("\n  [Test 1] Statistical Distribution Tests...")
        
        results = {}
        
        # Test 1a: Bit-wise mean comparison
        sha_bit_means = sha_outputs.mean(axis=0)
        random_bit_means = random_outputs.mean(axis=0)
        
        # Should both be ~0.5 for random
        results['sha_mean_of_means'] = float(sha_bit_means.mean())
        results['random_mean_of_means'] = float(random_bit_means.mean())
        results['sha_std_of_means'] = float(sha_bit_means.std())
        results['random_std_of_means'] = float(random_bit_means.std())
        
        # KS test on bit means
        ks_stat, ks_p = ks_2samp(sha_bit_means, random_bit_means)
        results['ks_bit_means'] = {'statistic': float(ks_stat), 'p_value': float(ks_p)}
        
        # Test 1b: Hamming weight distribution
        sha_hamming = sha_outputs.sum(axis=1)
        random_hamming = random_outputs.sum(axis=1)
        
        results['sha_hamming_mean'] = float(sha_hamming.mean())
        results['random_hamming_mean'] = float(random_hamming.mean())
        results['expected_hamming'] = 128.0  # n/2 for random
        
        ks_stat, ks_p = ks_2samp(sha_hamming, random_hamming)
        results['ks_hamming'] = {'statistic': float(ks_stat), 'p_value': float(ks_p)}
        
        # Test 1c: Pairwise correlation
        sha_corr = np.corrcoef(sha_outputs.T)
        random_corr = np.corrcoef(random_outputs.T)
        
        # Compare off-diagonal correlations
        sha_off_diag = sha_corr[np.triu_indices(256, k=1)]
        random_off_diag = random_corr[np.triu_indices(256, k=1)]
        
        results['sha_mean_correlation'] = float(np.mean(sha_off_diag))
        results['random_mean_correlation'] = float(np.mean(random_off_diag))
        
        ks_stat, ks_p = ks_2samp(sha_off_diag, random_off_diag)
        results['ks_correlations'] = {'statistic': float(ks_stat), 'p_value': float(ks_p)}
        
        return results
    
    def test_geometric_distinguisher(self, sha_outputs: np.ndarray,
                                      random_outputs: np.ndarray) -> dict:
        """
        Test 2: Can the geometric signature distinguish them?
        """
        print("\n  [Test 2] Geometric Signature Tests...")
        
        results = {}
        
        # Build heat kernel analysis for both
        constructor = LaplacianConstructor(self.embedding, k_neighbors=50)
        solver = HeatKernelSolver(n_eigenvalues=100)
        
        # Use subset for tractability
        n_subset = min(2000, self.n_samples)
        
        print("    Analyzing SHA-256 outputs...")
        sha_points = np.array([self.embedding.embed(s) for s in sha_outputs[:n_subset]])
        sha_lap = constructor.build_from_points(sha_points)
        sha_hk = solver.solve(sha_lap.laplacian)
        
        print("    Analyzing random outputs...")
        random_points = np.array([self.embedding.embed(s) for s in random_outputs[:n_subset]])
        random_lap = constructor.build_from_points(random_points)
        random_hk = solver.solve(random_lap.laplacian)
        
        # Compare spectral properties
        results['sha_spectral_gap'] = float(sha_hk.spectral_gap)
        results['random_spectral_gap'] = float(random_hk.spectral_gap)
        results['spectral_gap_ratio'] = float(sha_hk.spectral_gap / random_hk.spectral_gap)
        
        # Compare eigenvalue distributions
        ks_stat, ks_p = ks_2samp(sha_hk.eigenvalues[:50], random_hk.eigenvalues[:50])
        results['ks_eigenvalues'] = {'statistic': float(ks_stat), 'p_value': float(ks_p)}
        
        # Compare heat traces at multiple time scales
        t_values = [0.1, 1.0, 10.0]
        for t in t_values:
            sha_trace = sha_hk.heat_trace(t)
            random_trace = random_hk.heat_trace(t)
            results[f'heat_trace_t{t}'] = {
                'sha': float(sha_trace),
                'random': float(random_trace),
                'ratio': float(sha_trace / random_trace)
            }
        
        # Effective dimension comparison
        results['sha_eff_dim'] = float(sha_hk.effective_dimension(1.0))
        results['random_eff_dim'] = float(random_hk.effective_dimension(1.0))
        
        return results
    
    def test_ml_distinguisher(self, sha_outputs: np.ndarray,
                               random_outputs: np.ndarray) -> dict:
        """
        Test 3: Can a classifier learn to distinguish them?
        """
        print("\n  [Test 3] Machine Learning Distinguisher...")
        
        results = {}
        
        # Prepare labeled dataset
        n_each = min(2000, len(sha_outputs), len(random_outputs))
        X = np.vstack([sha_outputs[:n_each], random_outputs[:n_each]])
        y = np.array([1] * n_each + [0] * n_each)  # 1 = SHA-256, 0 = random
        
        # Shuffle
        idx = np.random.permutation(len(y))
        X, y = X[idx], y[idx]
        
        # Split
        split = int(0.8 * len(y))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Test 3a: Random Forest
        print("    Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_prob = rf.predict_proba(X_test)[:, 1]
        
        results['random_forest'] = {
            'accuracy': float(accuracy_score(y_test, rf_pred)),
            'auc': float(roc_auc_score(y_test, rf_prob)),
            'baseline': 0.5  # Random guessing
        }
        
        # Test 3b: SVM on first few principal components
        print("    Training SVM...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=50)
        X_pca = pca.fit_transform(X)
        X_train_pca, X_test_pca = X_pca[:split], X_pca[split:]
        
        svm = SVC(kernel='rbf', probability=True, random_state=42)
        svm.fit(X_train_pca, y_train)
        svm_pred = svm.predict(X_test_pca)
        svm_prob = svm.predict_proba(X_test_pca)[:, 1]
        
        results['svm'] = {
            'accuracy': float(accuracy_score(y_test, svm_pred)),
            'auc': float(roc_auc_score(y_test, svm_prob)),
            'baseline': 0.5
        }
        
        # Test 3c: Cross-validation scores
        print("    Running cross-validation...")
        cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
        results['cv_accuracy'] = {
            'mean': float(cv_scores.mean()),
            'std': float(cv_scores.std()),
            'scores': cv_scores.tolist()
        }
        
        # Feature importance (which bits are most predictive?)
        importance = rf.feature_importances_
        top_bits = np.argsort(importance)[-20:][::-1]
        results['most_predictive_bits'] = {
            'positions': top_bits.tolist(),
            'importances': importance[top_bits].tolist()
        }
        
        # Map to SHA-256 words
        word_importance = np.zeros(8)
        for i in range(256):
            word_importance[i // 32] += importance[i]
        results['word_importance'] = {
            'words': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
            'importance': word_importance.tolist()
        }
        
        return results
    
    def test_heat_kernel_signature_distinguisher(self, sha_outputs: np.ndarray,
                                                   random_outputs: np.ndarray) -> dict:
        """
        Test 4: Use Heat Kernel Signature (HKS) as feature for classification.
        """
        print("\n  [Test 4] Heat Kernel Signature Distinguisher...")
        
        results = {}
        
        n_subset = min(1000, len(sha_outputs))
        
        # Compute HKS for each sample
        constructor = LaplacianConstructor(self.embedding, k_neighbors=30)
        solver = HeatKernelSolver(n_eigenvalues=50)
        
        print("    Computing HKS for SHA-256 outputs...")
        sha_points = np.array([self.embedding.embed(s) for s in sha_outputs[:n_subset]])
        sha_lap = constructor.build_from_points(sha_points)
        sha_hk = solver.solve(sha_lap.laplacian)
        
        t_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        sha_hks = solver.heat_kernel_signature(sha_hk, t_values)
        
        print("    Computing HKS for random outputs...")
        random_points = np.array([self.embedding.embed(s) for s in random_outputs[:n_subset]])
        random_lap = constructor.build_from_points(random_points)
        random_hk = solver.solve(random_lap.laplacian)
        random_hks = solver.heat_kernel_signature(random_hk, t_values)
        
        # Compare HKS distributions
        sha_hks_flat = sha_hks.flatten()
        random_hks_flat = random_hks.flatten()
        
        ks_stat, ks_p = ks_2samp(sha_hks_flat, random_hks_flat)
        results['ks_hks_overall'] = {'statistic': float(ks_stat), 'p_value': float(ks_p)}
        
        # Per time scale comparison
        for i, t in enumerate(t_values):
            sha_hks_t = sha_hks[:, i]
            random_hks_t = random_hks[:, i]
            
            ks_stat, ks_p = ks_2samp(sha_hks_t, random_hks_t)
            mw_stat, mw_p = mannwhitneyu(sha_hks_t, random_hks_t)
            
            results[f'hks_t{t}'] = {
                'sha_mean': float(sha_hks_t.mean()),
                'random_mean': float(random_hks_t.mean()),
                'ks': {'statistic': float(ks_stat), 'p_value': float(ks_p)},
                'mann_whitney': {'statistic': float(mw_stat), 'p_value': float(mw_p)}
            }
        
        # Use HKS as features for classification
        print("    Training classifier on HKS features...")
        X = np.vstack([sha_hks, random_hks])
        y = np.array([1] * len(sha_hks) + [0] * len(random_hks))
        
        idx = np.random.permutation(len(y))
        X, y = X[idx], y[idx]
        
        split = int(0.8 * len(y))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        
        results['hks_classifier'] = {
            'accuracy': float(accuracy_score(y_test, rf.predict(X_test))),
            'auc': float(roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])),
            'baseline': 0.5
        }
        
        return results
    
    def run_all_tests(self) -> dict:
        """Run complete distinguisher test suite."""
        print("=" * 70)
        print("  SHA-256 OUTPUT DISTINGUISHER TEST")
        print("=" * 70)
        print(f"\n  Testing with {self.n_samples} samples each")
        
        # Generate data
        sha_outputs = self.generate_sha256_outputs()
        random_outputs = self.generate_random_outputs()
        
        # Run all tests
        results = {
            'n_samples': self.n_samples,
            'statistical': self.test_statistical_distinguisher(sha_outputs, random_outputs),
            'geometric': self.test_geometric_distinguisher(sha_outputs, random_outputs),
            'ml': self.test_ml_distinguisher(sha_outputs, random_outputs),
            'hks': self.test_heat_kernel_signature_distinguisher(sha_outputs, random_outputs)
        }
        
        # Summary
        print("\n" + "=" * 70)
        print("  DISTINGUISHER TEST RESULTS")
        print("=" * 70)
        
        # Statistical tests
        print("\n  [Statistical Tests]")
        stat = results['statistical']
        print(f"    Bit means KS p-value:      {stat['ks_bit_means']['p_value']:.4f}")
        print(f"    Hamming weight KS p-value: {stat['ks_hamming']['p_value']:.4f}")
        print(f"    Correlations KS p-value:   {stat['ks_correlations']['p_value']:.4f}")
        
        # Geometric tests
        print("\n  [Geometric Tests]")
        geo = results['geometric']
        print(f"    SHA-256 spectral gap:  {geo['sha_spectral_gap']:.4f}")
        print(f"    Random spectral gap:   {geo['random_spectral_gap']:.4f}")
        print(f"    Eigenvalue KS p-value: {geo['ks_eigenvalues']['p_value']:.4f}")
        
        # ML tests
        print("\n  [Machine Learning Tests]")
        ml = results['ml']
        print(f"    Random Forest accuracy: {ml['random_forest']['accuracy']:.4f}")
        print(f"    Random Forest AUC:      {ml['random_forest']['auc']:.4f}")
        print(f"    SVM accuracy:           {ml['svm']['accuracy']:.4f}")
        print(f"    Cross-val accuracy:     {ml['cv_accuracy']['mean']:.4f} ± {ml['cv_accuracy']['std']:.4f}")
        
        # HKS tests
        print("\n  [Heat Kernel Signature Tests]")
        hks = results['hks']
        print(f"    HKS overall KS p-value: {hks['ks_hks_overall']['p_value']:.4f}")
        print(f"    HKS classifier accuracy: {hks['hks_classifier']['accuracy']:.4f}")
        print(f"    HKS classifier AUC:      {hks['hks_classifier']['auc']:.4f}")
        
        # Overall verdict
        print("\n" + "-" * 70)
        print("  VERDICT:")
        
        distinguishable = False
        reasons = []
        
        # Check if any test shows significant distinguishability
        if ml['random_forest']['accuracy'] > 0.55:
            distinguishable = True
            reasons.append(f"RF accuracy {ml['random_forest']['accuracy']:.2%} > 55%")
        
        if ml['cv_accuracy']['mean'] > 0.55:
            distinguishable = True
            reasons.append(f"CV accuracy {ml['cv_accuracy']['mean']:.2%} > 55%")
        
        if hks['hks_classifier']['accuracy'] > 0.55:
            distinguishable = True
            reasons.append(f"HKS classifier {hks['hks_classifier']['accuracy']:.2%} > 55%")
        
        if geo['ks_eigenvalues']['p_value'] < 0.01:
            distinguishable = True
            reasons.append(f"Geometric KS p={geo['ks_eigenvalues']['p_value']:.4f} < 0.01")
        
        results['verdict'] = {
            'distinguishable': distinguishable,
            'reasons': reasons
        }
        
        if distinguishable:
            print("  ⚠️  SHA-256 OUTPUTS MAY BE DISTINGUISHABLE FROM RANDOM")
            print("  Reasons:")
            for r in reasons:
                print(f"    - {r}")
            print("\n  ⚠️  This warrants further investigation!")
        else:
            print("  ✅ SHA-256 outputs appear indistinguishable from random")
            print("  The geometric structure is internal only - not exploitable")
        
        print("-" * 70)
        
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SHA-256 Distinguisher Test")
    parser.add_argument('--n-samples', type=int, default=5000, help='Number of samples')
    parser.add_argument('--output', type=str, default='distinguisher_results.json', help='Output file')
    args = parser.parse_args()
    
    tester = DistinguisherTest(args.n_samples)
    results = tester.run_all_tests()
    
    import json
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
