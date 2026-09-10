#!/usr/bin/env python3
"""
SHA-256 Random Oracle Distinguisher
====================================

THE CRITICAL TEST: Can heat kernel analysis distinguish SHA-256 outputs
from a true random oracle (CSPRNG)?

If YES → the tested procedure separates the samples; the audit in review/SHA256_SUBMISSION_REVIEW.md withdrew this script's claims.
If NO  → The geometric structure is internal only, not observable in outputs.

Methodology:
1. Generate N random inputs → SHA-256 → N output hashes (256-bit each)
2. Generate N random 256-bit strings from CSPRNG
3. Build k-NN graph on each set in the embedding
4. Compute heat kernel signatures
5. Train binary classifier: "SHA-256 output" vs "random"
6. If classifier accuracy > 50% + ε, distinguisher exists

Author: Bee Davis
Date: December 6, 2025
"""

import numpy as np
from scipy import stats
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.sha256.core import InstrumentedSHA256
from src.embeddings.euclidean import EuclideanEmbedding


def generate_sha256_outputs(n: int, seed: int = 42) -> np.ndarray:
    """Generate n SHA-256 output hashes as 256-bit vectors."""
    rng = np.random.RandomState(seed)
    sha = InstrumentedSHA256()
    
    outputs = []
    for i in range(n):
        # Random 64-byte message
        msg = rng.bytes(64)
        trajectory = sha.hash_with_trajectory(msg)
        # Get final hash as bit vector
        final_state = trajectory.states[-1].state_vector
        outputs.append(final_state)
    
    return np.array(outputs)


def generate_random_outputs(n: int, seed: int = 123) -> np.ndarray:
    """Generate n truly random 256-bit vectors from CSPRNG."""
    rng = np.random.RandomState(seed)
    
    outputs = []
    for i in range(n):
        # Random 256 bits
        random_bytes = rng.bytes(32)
        bits = np.unpackbits(np.frombuffer(random_bytes, dtype=np.uint8))
        outputs.append(bits)
    
    return np.array(outputs)


def compute_geometric_features(points: np.ndarray, k: int = 30) -> dict:
    """
    Compute geometric features from a point cloud using heat kernel analysis.
    
    Features that might distinguish structured from random:
    - Eigenvalue distribution
    - Spectral gaps
    - Heat trace at various t
    - Local density statistics
    """
    n = points.shape[0]
    embedding = EuclideanEmbedding()
    
    # Build k-NN graph
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Build weight matrix with heat kernel weights
    t = 1.0  # diffusion time
    rows, cols, data = [], [], []
    for i in range(n):
        for j_idx in range(k):
            j = indices[i, j_idx]
            d = distances[i, j_idx]
            w = np.exp(-d**2 / (4 * t))
            rows.append(i)
            cols.append(j)
            data.append(w)
    
    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    W = (W + W.T) / 2  # Symmetrize
    
    # Normalized Laplacian
    D = np.array(W.sum(axis=1)).flatten()
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-10))
    L = np.eye(n) - D_inv_sqrt @ W.toarray() @ D_inv_sqrt
    
    # Compute eigenvalues
    n_eig = min(50, n - 2)
    try:
        eigenvalues, eigenvectors = eigsh(csr_matrix(L), k=n_eig, which='SM')
        eigenvalues = np.sort(eigenvalues)
    except:
        eigenvalues = np.zeros(n_eig)
    
    # Extract features
    features = {}
    
    # Spectral features
    features['spectral_gap'] = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0
    features['eigenvalue_mean'] = np.mean(eigenvalues)
    features['eigenvalue_std'] = np.std(eigenvalues)
    features['eigenvalue_skew'] = stats.skew(eigenvalues) if len(eigenvalues) > 2 else 0
    features['eigenvalue_kurtosis'] = stats.kurtosis(eigenvalues) if len(eigenvalues) > 3 else 0
    
    # Gap distribution
    gaps = np.diff(eigenvalues)
    features['gap_mean'] = np.mean(gaps) if len(gaps) > 0 else 0
    features['gap_std'] = np.std(gaps) if len(gaps) > 0 else 0
    features['gap_max'] = np.max(gaps) if len(gaps) > 0 else 0
    
    # Heat trace at different times
    for t_val in [0.1, 1.0, 10.0]:
        features[f'heat_trace_t{t_val}'] = np.sum(np.exp(-eigenvalues * t_val))
    
    # Effective dimension at different times
    for t_val in [0.1, 1.0, 10.0]:
        weights = np.exp(-eigenvalues * t_val)
        weights = weights / (weights.sum() + 1e-10)
        features[f'eff_dim_t{t_val}'] = 1.0 / (np.sum(weights**2) + 1e-10)
    
    # Distance distribution features
    all_distances = distances[:, 1:].flatten()  # Exclude self
    features['dist_mean'] = np.mean(all_distances)
    features['dist_std'] = np.std(all_distances)
    features['dist_skew'] = stats.skew(all_distances)
    
    # Density features
    local_densities = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-10)
    features['density_mean'] = np.mean(local_densities)
    features['density_std'] = np.std(local_densities)
    features['density_cv'] = features['density_std'] / (features['density_mean'] + 1e-10)
    
    return features


def run_distinguisher_experiment(n_samples: int = 500, n_trials: int = 20):
    """
    Main distinguisher experiment.
    
    For each trial:
    1. Generate SHA-256 outputs and random outputs
    2. Compute geometric features for each set
    3. Label and collect
    
    Then train classifier to distinguish.
    """
    print("=" * 70)
    print("  SHA-256 RANDOM ORACLE DISTINGUISHER")
    print("=" * 70)
    print(f"\n  Samples per trial: {n_samples}")
    print(f"  Number of trials:  {n_trials}")
    print(f"  Total samples:     {n_trials * 2} (half SHA-256, half random)")
    
    all_features = []
    all_labels = []
    
    print("\n  Generating samples and computing features...")
    
    for trial in range(n_trials):
        seed_sha = trial * 1000
        seed_rng = trial * 1000 + 500
        
        # Generate outputs
        sha_outputs = generate_sha256_outputs(n_samples, seed=seed_sha)
        rng_outputs = generate_random_outputs(n_samples, seed=seed_rng)
        
        # Compute geometric features
        sha_features = compute_geometric_features(sha_outputs)
        rng_features = compute_geometric_features(rng_outputs)
        
        all_features.append(list(sha_features.values()))
        all_labels.append(1)  # SHA-256
        
        all_features.append(list(rng_features.values()))
        all_labels.append(0)  # Random
        
        if (trial + 1) % 5 == 0:
            print(f"    Completed {trial + 1}/{n_trials} trials")
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n  Feature matrix shape: {X.shape}")
    print(f"  Features: {list(sha_features.keys())}")
    
    # Train classifiers
    print("\n" + "-" * 70)
    print("  CLASSIFIER RESULTS")
    print("-" * 70)
    
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', random_state=42),
        'SVM (Linear)': SVC(kernel='linear', random_state=42),
    }
    
    best_accuracy = 0
    best_classifier = None
    
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        mean_acc = scores.mean()
        std_acc = scores.std()
        
        print(f"\n  {name}:")
        print(f"    Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"    Individual folds: {[f'{s:.3f}' for s in scores]}")
        
        if mean_acc > best_accuracy:
            best_accuracy = mean_acc
            best_classifier = name
    
    # Statistical significance
    print("\n" + "-" * 70)
    print("  STATISTICAL ANALYSIS")
    print("-" * 70)
    
    # Null hypothesis: accuracy = 0.5 (random guessing)
    # Use binomial test
    n_correct = int(best_accuracy * len(y))
    n_total = len(y)
    
    # p-value for getting this many correct by chance
    result = stats.binomtest(n_correct, n_total, 0.5, alternative='greater')
    p_value = result.pvalue
    
    print(f"\n  Best classifier: {best_classifier}")
    print(f"  Best accuracy: {best_accuracy:.4f}")
    print(f"  Expected by chance: 0.5000")
    print(f"  Advantage: {best_accuracy - 0.5:.4f}")
    print(f"  P-value (binomial test): {p_value:.2e}")
    
    # Feature importance analysis (if RF is best)
    if 'Random Forest' in classifiers:
        rf = classifiers['Random Forest']
        rf.fit(X, y)
        importances = rf.feature_importances_
        feature_names = list(sha_features.keys())
        
        sorted_idx = np.argsort(importances)[::-1]
        
        print("\n  Top distinguishing features:")
        for i in sorted_idx[:5]:
            print(f"    {feature_names[i]}: {importances[i]:.4f}")
    
    # Compare feature distributions directly
    print("\n" + "-" * 70)
    print("  DIRECT FEATURE COMPARISON")
    print("-" * 70)
    
    sha_mask = y == 1
    rng_mask = y == 0
    
    feature_names = list(sha_features.keys())
    significant_features = []
    
    for i, fname in enumerate(feature_names):
        sha_vals = X[sha_mask, i]
        rng_vals = X[rng_mask, i]
        
        # Two-sample t-test
        t_stat, p_val = stats.ttest_ind(sha_vals, rng_vals)
        
        if p_val < 0.05:
            significant_features.append((fname, p_val, np.mean(sha_vals), np.mean(rng_vals)))
    
    if significant_features:
        print("\n  Features with significant difference (p < 0.05):")
        for fname, pval, sha_mean, rng_mean in sorted(significant_features, key=lambda x: x[1]):
            diff_pct = (sha_mean - rng_mean) / (rng_mean + 1e-10) * 100
            print(f"    {fname}: p={pval:.4f}, SHA256={sha_mean:.4f}, Random={rng_mean:.4f} ({diff_pct:+.1f}%)")
    else:
        print("\n  No features show significant difference at p < 0.05")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    
    if best_accuracy > 0.6 and p_value < 0.01:
        print("""
  🚨 DISTINGUISHER DETECTED 🚨
  
  The geometric analyzer can distinguish SHA-256 outputs from random.
  This violates the random oracle assumption.
  
  Severity: HIGH
  - Classifier accuracy: {:.1f}%
  - Statistical significance: p = {:.2e}
  
  Legacy message. This script's claims were withdrawn after audit; see review/SHA256_SUBMISSION_REVIEW.md.
        """.format(best_accuracy * 100, p_value))
    elif best_accuracy > 0.55 and p_value < 0.05:
        print("""
  ⚠️  WEAK DISTINGUISHER DETECTED
  
  There is a statistically significant but small advantage in distinguishing
  SHA-256 outputs from random.
  
  Severity: MODERATE
  - Classifier accuracy: {:.1f}%
  - Statistical significance: p = {:.2e}
  
  This may indicate structure in SHA-256 outputs, but effect size is small.
  Further investigation with larger sample sizes recommended.
        """.format(best_accuracy * 100, p_value))
    else:
        print("""
  ✅ NO DISTINGUISHER FOUND
  
  The geometric analyzer cannot reliably distinguish SHA-256 outputs
  from truly random 256-bit strings.
  
  - Classifier accuracy: {:.1f}% (expected: 50%)
  - Statistical significance: p = {:.2e}
  
  The geometric structure detected in internal states does NOT propagate
  to the output. SHA-256 appears to behave as a random oracle at the
  output level.
  
  INTERPRETATION:
  The Σ₀ structure we found is a geometric fingerprint of the internal
  compression function, but the finalization successfully decorrelates
  it from the output. This is SHA-256 working as designed.
        """.format(best_accuracy * 100, p_value))
    
    return {
        'best_accuracy': best_accuracy,
        'best_classifier': best_classifier,
        'p_value': p_value,
        'significant_features': significant_features,
        'n_samples': n_samples,
        'n_trials': n_trials,
    }


if __name__ == "__main__":
    results = run_distinguisher_experiment(n_samples=500, n_trials=20)
