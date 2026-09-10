#!/usr/bin/env python3
"""
Large-scale SHA-256 Random Oracle Distinguisher
================================================

Runs the distinguisher test with higher statistical power on Modal.

Author: Bee Davis
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy import stats
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


def generate_sha256_outputs(n: int, seed: int = 42) -> np.ndarray:
    """Generate n SHA-256 output hashes as 256-bit vectors."""
    from src.sha256.core import InstrumentedSHA256
    
    rng = np.random.RandomState(seed)
    sha = InstrumentedSHA256()
    
    outputs = []
    for i in range(n):
        msg = rng.bytes(64)
        trajectory = sha.hash_with_trajectory(msg)
        final_state = trajectory.states[-1].state_vector
        outputs.append(final_state)
    
    return np.array(outputs)


def generate_random_outputs(n: int, seed: int = 123) -> np.ndarray:
    """Generate n truly random 256-bit vectors from CSPRNG."""
    rng = np.random.RandomState(seed)
    
    outputs = []
    for i in range(n):
        random_bytes = rng.bytes(32)
        bits = np.unpackbits(np.frombuffer(random_bytes, dtype=np.uint8))
        outputs.append(bits)
    
    return np.array(outputs)


def compute_geometric_features(points: np.ndarray, k: int = 30) -> dict:
    """Compute geometric features from point cloud."""
    n = points.shape[0]
    
    # Build k-NN graph
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Build weight matrix
    t = 1.0
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
    W = (W + W.T) / 2
    
    # Laplacian
    D = np.array(W.sum(axis=1)).flatten()
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-10))
    L = np.eye(n) - D_inv_sqrt @ W.toarray() @ D_inv_sqrt
    
    # Eigenvalues
    n_eig = min(100, n - 2)
    try:
        eigenvalues, _ = eigsh(csr_matrix(L), k=n_eig, which='SM')
        eigenvalues = np.sort(eigenvalues)
    except:
        eigenvalues = np.zeros(n_eig)
    
    features = {}
    
    # Spectral features
    features['spectral_gap'] = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0
    features['eigenvalue_mean'] = np.mean(eigenvalues)
    features['eigenvalue_std'] = np.std(eigenvalues)
    features['eigenvalue_skew'] = stats.skew(eigenvalues) if len(eigenvalues) > 2 else 0
    features['eigenvalue_kurtosis'] = stats.kurtosis(eigenvalues) if len(eigenvalues) > 3 else 0
    
    # Eigenvalue percentiles
    for p in [10, 25, 50, 75, 90]:
        features[f'eigenvalue_p{p}'] = np.percentile(eigenvalues, p)
    
    # Gap distribution
    gaps = np.diff(eigenvalues)
    features['gap_mean'] = np.mean(gaps) if len(gaps) > 0 else 0
    features['gap_std'] = np.std(gaps) if len(gaps) > 0 else 0
    features['gap_max'] = np.max(gaps) if len(gaps) > 0 else 0
    features['gap_entropy'] = stats.entropy(gaps / (gaps.sum() + 1e-10)) if len(gaps) > 0 else 0
    
    # Heat traces
    for t_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
        features[f'heat_trace_t{t_val}'] = np.sum(np.exp(-eigenvalues * t_val))
    
    # Effective dimension
    for t_val in [0.1, 1.0, 10.0]:
        weights = np.exp(-eigenvalues * t_val)
        weights = weights / (weights.sum() + 1e-10)
        features[f'eff_dim_t{t_val}'] = 1.0 / (np.sum(weights**2) + 1e-10)
    
    # Distance statistics
    all_distances = distances[:, 1:].flatten()
    features['dist_mean'] = np.mean(all_distances)
    features['dist_std'] = np.std(all_distances)
    features['dist_skew'] = stats.skew(all_distances)
    features['dist_kurtosis'] = stats.kurtosis(all_distances)
    
    for p in [10, 25, 50, 75, 90]:
        features[f'dist_p{p}'] = np.percentile(all_distances, p)
    
    # Density features
    local_densities = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-10)
    features['density_mean'] = np.mean(local_densities)
    features['density_std'] = np.std(local_densities)
    features['density_cv'] = features['density_std'] / (features['density_mean'] + 1e-10)
    features['density_entropy'] = stats.entropy(local_densities / local_densities.sum())
    
    return features


def run_distinguisher(n_samples: int = 1000, n_trials: int = 50):
    """Run the distinguisher experiment."""
    
    print("=" * 70)
    print("  SHA-256 RANDOM ORACLE DISTINGUISHER (LARGE SCALE)")
    print("=" * 70)
    print(f"\n  Samples per trial: {n_samples}")
    print(f"  Number of trials:  {n_trials}")
    print(f"  Total samples:     {n_trials * 2}")
    
    all_features = []
    all_labels = []
    
    print("\n  Generating samples...")
    
    for trial in range(n_trials):
        seed_sha = trial * 1000
        seed_rng = trial * 1000 + 500
        
        sha_outputs = generate_sha256_outputs(n_samples, seed=seed_sha)
        rng_outputs = generate_random_outputs(n_samples, seed=seed_rng)
        
        sha_features = compute_geometric_features(sha_outputs)
        rng_features = compute_geometric_features(rng_outputs)
        
        all_features.append(list(sha_features.values()))
        all_labels.append(1)
        
        all_features.append(list(rng_features.values()))
        all_labels.append(0)
        
        if (trial + 1) % 10 == 0:
            print(f"    Completed {trial + 1}/{n_trials} trials")
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n  Feature matrix: {X.shape}")
    
    # Classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', C=1.0, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    }
    
    print("\n" + "-" * 70)
    print("  CLASSIFIER RESULTS (5-fold CV)")
    print("-" * 70)
    
    best_accuracy = 0
    best_classifier = None
    
    for name, clf in classifiers.items():
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
        mean_acc = scores.mean()
        std_acc = scores.std()
        
        print(f"\n  {name}:")
        print(f"    Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        
        if mean_acc > best_accuracy:
            best_accuracy = mean_acc
            best_classifier = name
    
    # Statistical test
    n_correct = int(best_accuracy * len(y))
    n_total = len(y)
    result = stats.binomtest(n_correct, n_total, 0.5, alternative='greater')
    p_value = result.pvalue
    
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"\n  Best classifier: {best_classifier}")
    print(f"  Best accuracy: {best_accuracy:.4f}")
    print(f"  P-value: {p_value:.2e}")
    
    if best_accuracy > 0.6 and p_value < 0.01:
        print("\n  🚨 DISTINGUISHER DETECTED - Random oracle assumption violated")
    elif best_accuracy > 0.55 and p_value < 0.05:
        print("\n  ⚠️  WEAK SIGNAL - Possible distinguisher, needs more samples")
    else:
        print("\n  ✅ NO DISTINGUISHER - SHA-256 outputs indistinguishable from random")
    
    return {
        'best_accuracy': best_accuracy,
        'p_value': p_value,
        'n_samples': n_samples,
        'n_trials': n_trials,
    }


if __name__ == "__main__":
    run_distinguisher(n_samples=1000, n_trials=50)
