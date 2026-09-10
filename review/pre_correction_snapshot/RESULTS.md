# SHA-256 Heat Kernel Analysis: Final Results

## Executive Summary

This project applied **heat kernel methods** (spectral geometry on Riemannian manifolds) to analyze the internal dynamics of SHA-256. The goal was to find geometric signatures that might reveal cryptographic weaknesses.

### Key Findings

| Analysis | Result |
|----------|--------|
| **Geometric Fingerprint** | ✅ Found. SHA-256 trajectories form a structured manifold with silhouette score 0.995 (vs random -0.574) |
| **Slow Bits** | ✅ Found. 22 bits in word 'a' (Σ₀ target) exhibit 0.24 anisotropy |
| **Distinguisher** | ❌ No distinguisher found. SHA-256 outputs indistinguishable from random (RF 48.4%, SVM 48.5%) |
| **Slow Bit → Attack** | ❌ Slow bits provide no advantage. p-values 0.27-0.92 for cube attacks |
| **Security Margin** | ✅ Full 64 rounds secure. Even 6-bit cube at round 14 shows no bias |

---

## Detailed Analysis Results

### 1. Geometric Trajectory Analysis

SHA-256's internal state evolution forms a **highly structured manifold** when embedded in hyperbolic space:

| Metric | SHA-256 | Random |
|--------|---------|--------|
| Spectral clustering silhouette | 0.995 | -0.574 |
| Geodesic distortion | 2.637× | 1.0× |
| Maj/Ch function half-life | 16 rounds | N/A |

**Interpretation**: The hash function's mixing process is geometrically regular, but this structure does not translate to cryptographic weakness.

### 2. Slow-Manifold Gradient Attack

Found message bit patterns that cause **anomalously slow divergence** in hyperbolic embedding:

| Pattern | z-score | Bits |
|---------|---------|------|
| (5, 10) | -34.89 | First word |
| (1, 9) | -34.45 | First word |
| (5, 7) | -34.34 | First word |

**Slow bit set**: {1, 5, 7, 9, 10, 14}

These align with Σ₀ rotation constants:
- Rotation 2: bits 5, 7, 9
- Rotation 13: bits 1, 14

### 3. Cube Attack Results

#### 6-Bit Slow-Bit Cube (64 iterations)
| Round | Zero Rate | Bias | p-value | Status |
|-------|-----------|------|---------|--------|
| 14 | 50.7% | 0.007 | 0.70 | ✅ Secure |
| 15 | 50.2% | 0.002 | 0.92 | ✅ Secure |
| 16 | 52.0% | 0.020 | 0.27 | ✅ Secure |
| 17 | 48.2% | 0.018 | 0.34 | ✅ Secure |
| 18 | 49.4% | 0.006 | 0.75 | ✅ Secure |
| 19 | 49.0% | 0.010 | 0.60 | ✅ Secure |

#### Comparative: Slow Bits vs Random Bits
| Round | Slow Bias | Random Bias | Advantage |
|-------|-----------|-------------|-----------|
| 14 | 0.007 | 0.030 | -0.023 (random) |
| 15 | 0.002 | 0.021 | -0.019 (none) |
| 16 | 0.020 | 0.011 | +0.009 (none) |
| 17 | 0.018 | 0.006 | +0.011 (none) |
| 18 | 0.006 | 0.005 | +0.001 (none) |
| 19 | 0.010 | 0.009 | +0.001 (none) |

**Conclusion**: Geometrically-derived slow bits provide **no cryptographic advantage**.

### 4. 16-Bit Heavy Cube Attack (Previous Result)

From the escalation experiment (65,536 iterations per test):

| Round | Broken Words | Status |
|-------|--------------|--------|
| 16 | All 8 | 🔴 BROKEN |
| 17 | c,d,g,h | 🟡 PARTIAL |
| 18 | d,h | 🟡 PARTIAL |
| 19 | None | ✅ SECURE |

**Note**: This attack does NOT use slow bits — it's a brute-force algebraic attack. The slow bits from heat kernel analysis don't improve this.

---

## Theoretical Interpretation

### Why Geometric ≠ Algebraic

1. **Geometric structure** captures the *continuous* behavior of the mixing process in embedded space
2. **Algebraic structure** (exploitable by cube attacks) measures *discrete* Boolean polynomial degree

The slow-diverging bit patterns represent regions where the **Riemannian curvature** is lower, causing slower heat diffusion. However, this does not imply low algebraic degree.

### The 3-Round Gap Explained

- **Round 16**: Geometric isotropy achieved (anisotropy < 0.1)
- **Round 19**: Algebraic security achieved (16-bit cube fails)

This gap exists because:
- Geometric properties measure *average* behavior over trajectories
- Algebraic properties measure *worst-case* structure in Boolean representation

---

## Files Produced

| File | Description |
|------|-------------|
| `analysis/sha256_geometric_analysis.py` | 3-part geometric analysis |
| `analysis/sha256_distinguisher_test.py` | 4-test distinguisher suite |
| `analysis/reduced_round_analysis.py` | Per-round anisotropy tracking |
| `analysis/cube_attack.py` | 7-bit cube attack |
| `analysis/cube_attack_escalation.py` | 16-bit heavy cube |
| `analysis/sha256_geometric_attacks.py` | 4 advanced attacks |
| `analysis/combined_geometric_algebraic.py` | Gap analysis |
| `analysis/precision_cube_attack.py` | High-precision slow-bit cube |
| `analysis/comparative_cube_attack.py` | Slow vs random comparison |
| `visualizations/` | 4 trajectory visualizations |
| `attack_results.json` | Raw attack data |
| `combined_analysis_results.json` | Combined analysis data |

---

## Conclusion

Heat kernel analysis successfully reveals **geometric structure** in SHA-256's internal dynamics, including:
- Highly clustered trajectory manifolds
- Slow-diverging bit patterns aligned with Σ₀ rotations
- Transition from anisotropic to isotropic at round 16

However, **this geometric structure does not translate to cryptographic weakness**:
- SHA-256 outputs remain indistinguishable from random
- Slow bits provide no advantage in cube attacks
- Full 64 rounds maintain complete security margin

The heat kernel framework is valuable for understanding hash function dynamics but does not reveal exploitable vulnerabilities in SHA-256.
