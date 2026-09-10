# SHA-256 Heat Kernel Geometric Analysis

Spectral geometry analysis of SHA-256's internal state space using heat kernel methods and the Davis manifold framework.

## Overview

This project applies geometric analysis techniques—heat kernels, spectral decomposition, and the Davis manifold framework—to study the internal structure of SHA-256's state space. The goal is to determine whether SHA-256's round function exhibits any geometric regularities that differ from a random function.

**Core Question**: Does SHA-256 have hidden geometric structure that could be exploited?

### Theoretical Foundation

From `theory/heat kernel.tex`: The heat kernel $K(x, y, t)$ reveals geometric structure through its spectral decomposition:

$$K(x, y, t) = \sum_i e^{-\lambda_i t} \phi_i(x) \phi_i(y)$$

where $(\lambda_i, \phi_i)$ are eigenvalue/eigenvector pairs of the graph Laplacian. Structure appears as:
- **Clustered eigenvalues** → geometric regularity
- **Spectral gaps** → disconnected structure
- **Persistent modes** → robust geometric features

From `theory/manifold.tex`: The Davis manifold provides bounded distortion $\varepsilon(L) \leq K(\lambda) \cdot L^\alpha$ for paths of length $L$, enabling principled detection with:
- Explicit error bounds
- Soft/hard margins for abstention
- Compositional error tracking through rounds

## Project Structure

```
sha-256-heat_kernel/
├── src/
│   ├── sha256/           # Instrumented SHA-256 implementation
│   │   ├── core.py       # Full SHA-256 from FIPS 180-4 with trajectory capture
│   │   └── input_generator.py  # Random, Hamming pairs, structured inputs
│   ├── embeddings/       # Manifold embeddings
│   │   ├── base.py       # Abstract base class with DistortionBounds
│   │   ├── euclidean.py  # Baseline ℝ²⁵⁶ embedding
│   │   ├── hyperbolic.py # Poincaré ball H²⁵⁶ (tree-like structure)
│   │   └── davis.py      # Davis manifold with distortion bounds
│   ├── analysis/         # Geometric analysis tools
│   │   ├── laplacian.py  # k-NN graph Laplacian construction
│   │   ├── heat_kernel.py # Spectral solver, HKS, WKS, curvature
│   │   └── anomaly.py    # Null hypothesis testing, KS tests
│   ├── experiments/      # Experiment protocols
│   │   ├── baseline.py   # Exp 1: Diffusion profile analysis
│   │   ├── avalanche.py  # Exp 2: Avalanche geometry & isotropy
│   │   └── comparison.py # Exp 3: Davis vs Euclidean vs Hyperbolic
│   └── utils/            # Visualization utilities
├── theory/               # Mathematical foundations
│   ├── heat kernel.tex   # Heat kernel theory & spectral geometry
│   └── manifold.tex      # Davis manifold framework
├── heat-kernel-samples/  # Reference implementations
├── results/              # Experiment outputs (JSON)
├── main.py               # CLI entry point
├── modal_app.py          # Modal GPU deployment
├── tests/                # Unit tests
├── spec.MD               # Engineering specification
└── pyproject.toml        # Package configuration
```

## Quick Start

### Installation

```bash
cd sha-256-heat_kernel

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy scipy scikit-learn matplotlib

# Or install as package
pip install -e .
```

### Validate SHA-256 Implementation

Verify the implementation against FIPS 180-4 test vectors:

```bash
python main.py --validate
# Output: ✓ All SHA-256 test vectors pass
```

### Run Demo

See the system in action:

```bash
python main.py --demo
```

Output shows:
1. Trajectory capture through SHA-256 rounds
2. Embeddings in all three manifolds
3. Spectral analysis (gap, effective dimension)

### Run Experiments

```bash
# Experiment 1: Baseline diffusion profile
python main.py --experiment 1 --n-samples 1000 --embedding euclidean

# Experiment 2: Avalanche geometry (isotropy analysis)
python main.py --experiment 2 --n-pairs 500

# Experiment 3: Full Davis comparison (all embeddings)
python main.py --experiment 3 --quick

# Full analysis (longer runtime)
python main.py --experiment 3 --n-samples 10000 --n-pairs 5000
```

Results are saved to `results/exp{N}_{timestamp}.json`.

## Experiments

### Experiment 1: Baseline Diffusion Profile

Characterizes heat kernel behavior on SHA-256 state space.

**What it does**:
1. Generates `n_samples` random inputs
2. Collects state trajectories through SHA-256 rounds
3. Builds k-NN graph Laplacians at each sampled round
4. Computes spectral invariants (eigenvalues, heat trace, etc.)
5. Compares to null distribution (random function on hypercube)

**Interpretation**:
- `structure_detected_rounds`: Rounds where p < 0.01 (statistically significant)
- `max_significance`: Highest confidence of structure detection
- `structure_persistence`: Last round with detectable structure

**Success criteria**: 
- p < 0.01 for any round → evidence of structure
- Structure past round 32 → cryptographically interesting
- Structure at round 0 but gone by round 16 → SHA-256 working as designed

### Experiment 2: Avalanche Geometry

Tests whether the avalanche effect is geometrically isotropic.

**What it does**:
1. Generates Hamming-1 pairs (messages differing by 1 bit)
2. Tracks divergence of state vectors across rounds
3. Analyzes isotropy (does divergence depend on *which* bit was flipped?)
4. Identifies "slow bits" (potential vulnerabilities)

**Interpretation**:
- `anisotropy_score`: 0 = perfectly isotropic; higher = geometric bias
- `slow_bits`: Bit positions with slower-than-average divergence
- `divergence_curves`: Per-pair distance evolution

**Success criteria**:
- High anisotropy → avalanche has geometric bias
- Non-empty slow_bits → certain input bits have exploitable properties

### Experiment 3: Davis Manifold Comparison

Compares structure detection across embeddings.

**What it does**:
1. Runs experiments 1 and 2 with all three embeddings
2. **Euclidean**: Baseline (flat geometry)
3. **Hyperbolic**: Tree-like structure detection
4. **Davis**: Semantic coherence with distortion bounds (from `theory/manifold.tex`)

**Key insight**: If Davis sees structure that Euclidean doesn't, hidden geometry exists.

**Interpretation**:
- `davis_advantage`: Metrics where Davis outperforms other embeddings
- `comparison`: Cross-embedding sensitivity analysis

**Success criteria**:
- Davis shows structure, Euclidean doesn't → hidden geometry confirmed
- All three agree (no structure) → SHA-256 is robust to geometric analysis
- Hyperbolic shows structure → tree-like structure in state transitions

## Modal Deployment

For large-scale GPU-accelerated runs:

```bash
# Install Modal
pip install modal

# Configure Modal authentication
modal token new

# Run single experiment on GPU
modal run modal_app.py --experiment 1 --n-samples 100000

# Full analysis on A100
modal run modal_app.py --experiment all
```

The Modal deployment (`modal_app.py`) provides:
- A100/H100 GPU acceleration for eigendecomposition
- Parallel execution of multiple configurations
- Persistent storage of results
- ~10-50x speedup over CPU for n_samples > 5000

## Mathematical Background

See `theory/` for full derivations.

### Heat Kernel on Graphs (`theory/heat kernel.tex`)

Given a k-NN graph from embedded SHA-256 states, we construct:

1. **Symmetrized adjacency**: $W = (A + A^T)/2$
2. **Normalized Laplacian**: $L = I - D^{-1/2} W D^{-1/2}$
3. **Heat kernel**: $K(x,y,t) = \sum_i e^{-\lambda_i t} \phi_i(x) \phi_i(y)$

**Key spectral quantities**:
- **Spectral gap** ($\lambda_1$): Controls mixing time; larger = faster mixing
- **Heat trace**: $Z(t) = \text{Tr}(e^{-tL}) = \sum_i e^{-\lambda_i t}$ (spectral fingerprint)
- **HKS**: $K(x,x,t)$ (isometry-invariant signature per point)
- **Curvature**: From small-$t$ asymptotics (Theorem 2.3, heat kernel.tex):
  $$K(x,x,t) \sim (4\pi t)^{-d/2}\left(1 + \frac{R(x)}{6}t + O(t^2)\right)$$

### Davis Manifold Framework (`theory/manifold.tex`)

The Davis manifold provides a geometry-first detection framework with:

- **Bounded distortion**: $\varepsilon(L) \leq K(\lambda) \cdot L^\alpha$ for paths of length $L$ (Lemma 3.1)
- **Margins**: Soft margin $\kappa_{soft}$ and hard margin $\kappa_{hard}$ for principled abstention (Theorem 4.2)
- **Training**: InfoNCE loss + Jacobian regularization (Eq. 5.1):
  $$\mathcal{L} = \mathcal{L}_{InfoNCE} + \lambda \|\nabla f\|_F^2$$

**Application to SHA-256**:
- "Identity" = same input message
- "Path" = trajectory through rounds
- Distortion bounds quantify when Euclidean approximations are valid
- Curvature-aware distance captures semantic structure in state transitions

## Security Protocol

**If structure is found:**

1. Do not publish publicly
2. Assess severity (see spec.MD Section 6):
   - Minor: Document and consider responsible disclosure timeline
   - Moderate: Contact cryptography experts privately
   - Severe (practical attack): Contact NIST immediately
3. For critical findings (effective search space reduced by >2^10): Contact NIST

## References

1. FIPS 180-4: Secure Hash Standard (SHA-256 specification)
2. Grigor'yan, A. "Heat Kernel and Analysis on Manifolds" - AMS/IP Studies
3. Bronstein, M. et al. "Geometric Deep Learning" - arXiv:2104.13478
4. Davis, B. "Geometry-First Detection with Compositional Error Bounds" - `theory/manifold.tex`
5. Davis, B. "Heat Kernel Methods for Discrete Structures" - `theory/heat kernel.tex`

## Author

Bee Davis <bee_davis@alumni.brown.edu>
