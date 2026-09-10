"""
Visualization Utilities
=======================

Generate plots for SHA-256 heat kernel analysis.

Plots:
1. Eigenvalue spectrum by round (line plot, log scale)
2. Spectral gap distribution vs null (histogram overlay)
3. Heat trace decay curves by round
4. Avalanche divergence curves (mean ± std)
5. Anisotropy heatmap (bit position vs round)
6. Embedding comparison (3-panel: Euclidean, Hyperbolic, Davis)

Author: Bee Davis
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import io


def _check_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib
        return True
    except ImportError:
        return False


def plot_eigenvalue_spectrum(
    results_by_round: Dict[int, Any],
    title: str = "Eigenvalue Spectrum by Round",
    log_scale: bool = True
) -> Optional[bytes]:
    """
    Plot eigenvalue spectrum for each round.
    
    Args:
        results_by_round: Dict mapping round -> HeatKernelResult
        title: Plot title
        log_scale: Use log scale for y-axis
    
    Returns:
        PNG bytes if matplotlib available, else None
    """
    if not _check_matplotlib():
        return None
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rounds = sorted(results_by_round.keys())
    cmap = plt.cm.viridis(np.linspace(0, 1, len(rounds)))
    
    for i, r in enumerate(rounds):
        result = results_by_round[r]
        eigenvalues = result.eigenvalues if hasattr(result, 'eigenvalues') else result.get('eigenvalues', [])
        if len(eigenvalues) > 0:
            ax.plot(eigenvalues, color=cmap[i], label=f'Round {r}', alpha=0.7)
    
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(title)
    ax.legend(loc='upper right')
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.grid(True, alpha=0.3)
    
    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf.read()


def plot_spectral_gap_distribution(
    observed_gaps: np.ndarray,
    null_gaps: np.ndarray,
    title: str = "Spectral Gap Distribution"
) -> Optional[bytes]:
    """
    Plot spectral gap distribution vs null.
    
    Args:
        observed_gaps: Observed spectral gaps
        null_gaps: Null distribution gaps
        title: Plot title
    
    Returns:
        PNG bytes
    """
    if not _check_matplotlib():
        return None
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram for null
    ax.hist(null_gaps, bins=30, alpha=0.5, label='Null Distribution', 
            color='gray', density=True)
    
    # Histogram for observed
    ax.hist(observed_gaps, bins=30, alpha=0.7, label='Observed (SHA-256)',
            color='blue', density=True)
    
    ax.set_xlabel('Spectral Gap')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf.read()


def plot_divergence_curves(
    divergences: np.ndarray,
    rounds: List[int],
    title: str = "Avalanche Divergence Curves"
) -> Optional[bytes]:
    """
    Plot avalanche divergence curves with mean and std.
    
    Args:
        divergences: Shape (n_pairs, n_rounds) divergence data
        rounds: List of round numbers
        title: Plot title
    
    Returns:
        PNG bytes
    """
    if not _check_matplotlib():
        return None
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mean = np.mean(divergences, axis=0)
    std = np.std(divergences, axis=0)
    
    # Plot all curves with low alpha
    for curve in divergences[:100]:  # Limit for visibility
        ax.plot(rounds, curve, alpha=0.05, color='blue')
    
    # Plot mean with confidence band
    ax.plot(rounds, mean, color='red', linewidth=2, label='Mean')
    ax.fill_between(rounds, mean - std, mean + std, 
                    color='red', alpha=0.2, label='±1 std')
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Divergence (distance)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf.read()


def plot_structure_decay(
    decay_analysis: Dict,
    title: str = "Structure Decay Across Rounds"
) -> Optional[bytes]:
    """
    Plot how spectral structure decays across rounds.
    
    Args:
        decay_analysis: Output from AnomalyDetector.compare_rounds()
        title: Plot title
    
    Returns:
        PNG bytes
    """
    if not _check_matplotlib():
        return None
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    rounds = decay_analysis['rounds']
    
    # Spectral gaps
    ax = axes[0, 0]
    ax.plot(rounds, decay_analysis['spectral_gaps'], 'b-o', linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Spectral Gap')
    ax.set_title('Spectral Gap vs Round')
    ax.grid(True, alpha=0.3)
    
    # Entropy
    ax = axes[0, 1]
    ax.plot(rounds, decay_analysis['entropies'], 'g-o', linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Eigenvalue Entropy')
    ax.set_title('Spectral Entropy vs Round')
    ax.grid(True, alpha=0.3)
    
    # Effective dimension
    ax = axes[1, 0]
    ax.plot(rounds, decay_analysis['effective_dimensions'], 'r-o', linewidth=2)
    ax.set_xlabel('Round')
    ax.set_ylabel('Effective Dimension')
    ax.set_title('Effective Dimension vs Round')
    ax.grid(True, alpha=0.3)
    
    # Gap velocity (rate of change)
    ax = axes[1, 1]
    ax.plot(rounds, decay_analysis['gap_velocity'], 'm-o', linewidth=2)
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Round')
    ax.set_ylabel('Gap Velocity')
    ax.set_title('Rate of Gap Change')
    ax.grid(True, alpha=0.3)
    
    # Mark inflection points
    for ip in decay_analysis.get('inflection_points', []):
        for ax in axes.flat:
            ax.axvline(ip, color='orange', linestyle=':', alpha=0.7)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf.read()


def plot_anisotropy_heatmap(
    divergences_by_bit: Dict[int, np.ndarray],
    rounds: List[int],
    title: str = "Avalanche Anisotropy Heatmap"
) -> Optional[bytes]:
    """
    Plot heatmap of divergence by bit position and round.
    
    Args:
        divergences_by_bit: Dict mapping bit position -> divergence curves
        rounds: List of round numbers
        title: Plot title
    
    Returns:
        PNG bytes
    """
    if not _check_matplotlib():
        return None
    
    import matplotlib.pyplot as plt
    
    # Build matrix
    bit_positions = sorted(divergences_by_bit.keys())
    n_bits = len(bit_positions)
    n_rounds = len(rounds)
    
    heatmap = np.zeros((n_bits, n_rounds))
    for i, bp in enumerate(bit_positions):
        curves = divergences_by_bit[bp]
        if len(curves) > 0:
            heatmap[i, :] = np.mean(curves, axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(heatmap, aspect='auto', cmap='viridis')
    
    ax.set_xlabel('Round')
    ax.set_ylabel('Bit Position')
    ax.set_title(title)
    
    # Set tick labels
    ax.set_xticks(np.arange(n_rounds)[::max(1, n_rounds//10)])
    ax.set_xticklabels([rounds[i] for i in range(0, n_rounds, max(1, n_rounds//10))])
    
    if n_bits <= 50:
        ax.set_yticks(np.arange(n_bits))
        ax.set_yticklabels(bit_positions)
    
    plt.colorbar(im, ax=ax, label='Mean Divergence')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf.read()


def plot_embedding_comparison(
    results_by_embedding: Dict[str, Dict],
    metric: str = 'spectral_gap',
    title: str = "Embedding Comparison"
) -> Optional[bytes]:
    """
    Compare results across embeddings.
    
    Args:
        results_by_embedding: Dict mapping embedding name -> results
        metric: Which metric to compare
        title: Plot title
    
    Returns:
        PNG bytes
    """
    if not _check_matplotlib():
        return None
    
    import matplotlib.pyplot as plt
    
    n_embeddings = len(results_by_embedding)
    fig, axes = plt.subplots(1, n_embeddings, figsize=(4 * n_embeddings, 5))
    
    if n_embeddings == 1:
        axes = [axes]
    
    for i, (name, results) in enumerate(results_by_embedding.items()):
        ax = axes[i]
        
        # Get decay analysis
        decay = results.get('decay_analysis', {})
        if 'rounds' in decay and metric == 'spectral_gap':
            ax.plot(decay['rounds'], decay.get('spectral_gaps', []), 'b-o')
            ax.set_xlabel('Round')
            ax.set_ylabel('Spectral Gap')
        
        ax.set_title(name.upper())
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf.read()


def save_figure(fig_bytes: bytes, filepath: str):
    """Save figure bytes to file."""
    with open(filepath, 'wb') as f:
        f.write(fig_bytes)
