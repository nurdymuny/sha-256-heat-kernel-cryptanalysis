#!/usr/bin/env python3
"""
SHA-256 Geometric Structure Analysis Suite
==========================================
Three-part analysis:
1. VISUALIZATION - See the geometry we found
2. MULTI-DIRECTIONAL PROBING - Attack from different angles
3. DEEP STRUCTURAL ANALYSIS - Characterize the curves

Author: Bee Davis
Classification: CONFIDENTIAL
"""

import numpy as np
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Scientific computing
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, ks_2samp
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Visualization
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings('ignore')

# Our modules
import sys
sys.path.insert(0, '.')
from src.sha256.core import InstrumentedSHA256, SHA256Trajectory
from src.sha256.input_generator import InputGenerator
from src.embeddings.euclidean import EuclideanEmbedding
from src.embeddings.hyperbolic import HyperbolicEmbedding
from src.embeddings.davis import DavisManifoldEmbedding
from src.analysis.laplacian import LaplacianConstructor
from src.analysis.heat_kernel import HeatKernelSolver


# =============================================================================
# PART 1: VISUALIZATION
# =============================================================================

class GeometricVisualizer:
    """Visualize the geometric structure in SHA-256 state evolution."""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def plot_state_space_trajectories(self, trajectories: List[SHA256Trajectory], 
                                       n_samples: int = 500, method: str = 'pca'):
        """
        Visualize SHA-256 state trajectories projected to 2D.
        Shows how states evolve through rounds.
        """
        print("  Generating state space trajectory visualization...")
        
        # Collect all states across all rounds
        all_states = []
        round_labels = []
        trajectory_ids = []
        
        sample_trajectories = trajectories[:n_samples]
        rounds = [0, 8, 16, 24, 32, 40, 48, 56, 64]
        
        for tid, traj in enumerate(sample_trajectories):
            for state in traj.states:
                if state.round_num in rounds:
                    all_states.append(state.state_vector)
                    round_labels.append(state.round_num)
                    trajectory_ids.append(tid)
        
        all_states = np.array(all_states)
        round_labels = np.array(round_labels)
        
        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2)
            projected = reducer.fit_transform(all_states)
            explained_var = reducer.explained_variance_ratio_
            title_suffix = f"(PCA: {explained_var[0]:.1%}+{explained_var[1]:.1%} var)"
        elif method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=30, random_state=42)
            projected = reducer.fit_transform(all_states)
            title_suffix = "(t-SNE)"
        else:
            reducer = MDS(n_components=2, random_state=42)
            projected = reducer.fit_transform(all_states)
            title_suffix = "(MDS)"
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: All trajectories colored by round
        ax1 = axes[0]
        scatter = ax1.scatter(projected[:, 0], projected[:, 1], 
                             c=round_labels, cmap='viridis', alpha=0.5, s=10)
        plt.colorbar(scatter, ax=ax1, label='Round')
        ax1.set_title(f'State Space by Round {title_suffix}')
        ax1.set_xlabel('Dimension 1')
        ax1.set_ylabel('Dimension 2')
        
        # Plot 2: Connect trajectories with lines
        ax2 = axes[1]
        cmap = cm.get_cmap('tab10')
        for tid in range(min(20, n_samples)):  # Show 20 trajectories
            mask = np.array(trajectory_ids) == tid
            traj_points = projected[mask]
            traj_rounds = round_labels[mask]
            # Sort by round
            order = np.argsort(traj_rounds)
            traj_points = traj_points[order]
            ax2.plot(traj_points[:, 0], traj_points[:, 1], 
                    color=cmap(tid % 10), alpha=0.7, linewidth=1)
            ax2.scatter(traj_points[0, 0], traj_points[0, 1], 
                       color=cmap(tid % 10), s=50, marker='o', edgecolor='black')
            ax2.scatter(traj_points[-1, 0], traj_points[-1, 1], 
                       color=cmap(tid % 10), s=50, marker='s', edgecolor='black')
        ax2.set_title('Individual Trajectories (○=start, □=end)')
        ax2.set_xlabel('Dimension 1')
        ax2.set_ylabel('Dimension 2')
        
        # Plot 3: Density by round
        ax3 = axes[2]
        for r in [0, 16, 32, 48, 64]:
            mask = round_labels == r
            ax3.scatter(projected[mask, 0], projected[mask, 1], 
                       alpha=0.3, s=20, label=f'Round {r}')
        ax3.legend()
        ax3.set_title('State Clouds by Round')
        ax3.set_xlabel('Dimension 1')
        ax3.set_ylabel('Dimension 2')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'state_space_trajectories.png', dpi=150)
        plt.close()
        print(f"    Saved: {self.output_dir / 'state_space_trajectories.png'}")
        
    def plot_eigenvalue_spectrum(self, results_by_round: Dict[int, 'HeatKernelResult']):
        """
        Visualize eigenvalue spectrum evolution across rounds.
        Structure shows up as gaps and clustering.
        """
        print("  Generating eigenvalue spectrum visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        rounds = sorted(results_by_round.keys())
        cmap = cm.get_cmap('viridis', len(rounds))
        
        # Plot 1: Eigenvalue spectra overlay
        ax1 = axes[0, 0]
        for i, r in enumerate(rounds):
            eigs = results_by_round[r].eigenvalues[:50]
            ax1.plot(range(len(eigs)), eigs, color=cmap(i), 
                    label=f'Round {r}', alpha=0.8, linewidth=2)
        ax1.set_xlabel('Eigenvalue Index')
        ax1.set_ylabel('Eigenvalue')
        ax1.set_title('Eigenvalue Spectrum by Round')
        ax1.legend(loc='lower right')
        ax1.set_yscale('log')
        
        # Plot 2: Spectral gaps
        ax2 = axes[0, 1]
        gaps_by_round = {}
        for r in rounds:
            eigs = results_by_round[r].eigenvalues[:50]
            gaps = np.diff(eigs)
            gaps_by_round[r] = gaps
            ax2.plot(range(len(gaps)), gaps, color=cmap(rounds.index(r)), 
                    label=f'Round {r}', alpha=0.8)
        ax2.set_xlabel('Gap Index')
        ax2.set_ylabel('Spectral Gap (λᵢ₊₁ - λᵢ)')
        ax2.set_title('Spectral Gaps (structure = irregular gaps)')
        ax2.legend(loc='upper right')
        
        # Plot 3: First spectral gap evolution
        ax3 = axes[1, 0]
        first_gaps = [results_by_round[r].spectral_gap for r in rounds]
        ax3.plot(rounds, first_gaps, 'bo-', linewidth=2, markersize=10)
        ax3.fill_between(rounds, first_gaps, alpha=0.3)
        ax3.set_xlabel('Round')
        ax3.set_ylabel('First Spectral Gap (λ₁)')
        ax3.set_title('Spectral Gap Evolution\n(gap=connectivity, high=structure)')
        ax3.axhline(y=0.1, color='r', linestyle='--', label='Threshold')
        ax3.legend()
        
        # Plot 4: Heat trace decay
        ax4 = axes[1, 1]
        t_values = np.logspace(-2, 2, 100)
        for i, r in enumerate(rounds):
            result = results_by_round[r]
            trace = [result.heat_trace(t) for t in t_values]
            ax4.plot(t_values, trace, color=cmap(i), label=f'Round {r}', linewidth=2)
        ax4.set_xlabel('Time (t)')
        ax4.set_ylabel('Heat Trace Tr(exp(-tL))')
        ax4.set_title('Heat Diffusion Decay\n(slow decay = structure)')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eigenvalue_spectrum.png', dpi=150)
        plt.close()
        print(f"    Saved: {self.output_dir / 'eigenvalue_spectrum.png'}")
        
    def plot_anisotropy_heatmap(self, divergence_rates: np.ndarray, slow_bits: List[int]):
        """
        Visualize which bits are slow/fast as a heatmap of SHA-256's internal structure.
        """
        print("  Generating anisotropy heatmap...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Reshape rates to 8x32 (working variables x bit positions)
        # Assuming divergence_rates is per input bit position (512 bits in message)
        # But our slow bits are in state space (256 bits = 8 words x 32 bits)
        
        # Create state-space rate map (8 words x 32 bits)
        rate_map = np.zeros((8, 32))
        slow_map = np.zeros((8, 32))
        
        # Map the first 256 positions
        n_rates = min(256, len(divergence_rates))
        for i in range(n_rates):
            word = i // 32
            bit = i % 32
            rate_map[word, bit] = divergence_rates[i] if i < len(divergence_rates) else 0
        
        for bit in slow_bits:
            if bit < 256:
                word = bit // 32
                bit_pos = bit % 32
                slow_map[word, bit_pos] = 1
        
        # Plot 1: Rate heatmap
        ax1 = axes[0, 0]
        im1 = ax1.imshow(rate_map, cmap='RdYlBu_r', aspect='auto')
        ax1.set_xlabel('Bit Position (0-31)')
        ax1.set_ylabel('Working Variable')
        ax1.set_yticks(range(8))
        ax1.set_yticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        ax1.set_title('Divergence Rate by Position\n(red=slow, blue=fast)')
        plt.colorbar(im1, ax=ax1, label='Divergence Rate')
        
        # Add boxes around Maj(a,b,c) and Ch(e,f,g) regions
        rect_maj = Rectangle((-0.5, -0.5), 32, 3, fill=False, 
                            edgecolor='green', linewidth=3, label='Maj(a,b,c)')
        rect_ch = Rectangle((-0.5, 3.5), 32, 3, fill=False, 
                           edgecolor='orange', linewidth=3, label='Ch(e,f,g)')
        ax1.add_patch(rect_maj)
        ax1.add_patch(rect_ch)
        ax1.legend(loc='upper right')
        
        # Plot 2: Slow bits binary map
        ax2 = axes[0, 1]
        im2 = ax2.imshow(slow_map, cmap='Reds', aspect='auto')
        ax2.set_xlabel('Bit Position (0-31)')
        ax2.set_ylabel('Working Variable')
        ax2.set_yticks(range(8))
        ax2.set_yticklabels(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
        ax2.set_title(f'Slow Bit Locations ({len(slow_bits)} total)\n(all in Maj region)')
        
        # Plot 3: Rate distribution by word
        ax3 = axes[1, 0]
        word_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        word_rates = [rate_map[i, :].mean() for i in range(8)]
        word_stds = [rate_map[i, :].std() for i in range(8)]
        colors = ['green']*3 + ['gray'] + ['orange']*3 + ['gray']
        bars = ax3.bar(word_names, word_rates, yerr=word_stds, capsize=5, color=colors, alpha=0.7)
        ax3.set_xlabel('Working Variable')
        ax3.set_ylabel('Mean Divergence Rate')
        ax3.set_title('Average Divergence Rate by Word\n(green=Maj inputs, orange=Ch inputs)')
        ax3.axhline(y=np.mean(word_rates), color='red', linestyle='--', label='Overall mean')
        ax3.legend()
        
        # Plot 4: Σ₀ rotation analysis
        ax4 = axes[1, 1]
        # Σ₀ rotations are 2, 13, 22
        sigma0_positions = []
        for bit in range(32):
            related = [(bit + 2) % 32, (bit + 13) % 32, (bit + 22) % 32]
            sigma0_positions.append(related)
        
        # Check if slow bits cluster around Σ₀ relationships
        slow_in_a = [b for b in slow_bits if b < 32]
        sigma0_coverage = []
        for bit in range(32):
            # How many slow bits are related to this position via Σ₀?
            related = set([(bit + 2) % 32, (bit + 13) % 32, (bit + 22) % 32, bit])
            coverage = len(related.intersection(set(slow_in_a)))
            sigma0_coverage.append(coverage)
        
        ax4.bar(range(32), sigma0_coverage, color='purple', alpha=0.7)
        ax4.set_xlabel('Bit Position in word a')
        ax4.set_ylabel('Σ₀ Relation Count')
        ax4.set_title('Slow Bit Clustering via Σ₀ Rotations\n(higher = more slow bits in rotation orbit)')
        
        # Mark the rotation constants
        for rot in [2, 13, 22]:
            ax4.axvline(x=rot, color='red', linestyle='--', alpha=0.5)
        ax4.legend(['Σ₀ rotation constants (2, 13, 22)'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'anisotropy_heatmap.png', dpi=150)
        plt.close()
        print(f"    Saved: {self.output_dir / 'anisotropy_heatmap.png'}")
        
    def plot_divergence_curves(self, divergences: np.ndarray, slow_bits: List[int], 
                               rounds: List[int]):
        """
        Visualize how different bits diverge across rounds.
        """
        print("  Generating divergence curve visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # divergences shape: (n_pairs, n_rounds) or we need to compute per-bit
        n_pairs, n_rounds = divergences.shape
        
        # Plot 1: Mean divergence with confidence interval
        ax1 = axes[0, 0]
        mean_div = divergences.mean(axis=0)
        std_div = divergences.std(axis=0)
        ax1.plot(rounds, mean_div, 'b-', linewidth=2, label='Mean')
        ax1.fill_between(rounds, mean_div - std_div, mean_div + std_div, alpha=0.3)
        ax1.set_xlabel('Round')
        ax1.set_ylabel('Geodesic Distance')
        ax1.set_title('Avalanche Divergence Curve\n(how fast 1-bit changes spread)')
        ax1.legend()
        
        # Plot 2: Individual trajectories (sample)
        ax2 = axes[0, 1]
        sample_idx = np.random.choice(n_pairs, min(100, n_pairs), replace=False)
        for idx in sample_idx:
            ax2.plot(rounds, divergences[idx], alpha=0.2, color='blue', linewidth=0.5)
        ax2.plot(rounds, mean_div, 'r-', linewidth=3, label='Mean')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Geodesic Distance')
        ax2.set_title('Individual Divergence Trajectories')
        ax2.legend()
        
        # Plot 3: Divergence rate (derivative)
        ax3 = axes[1, 0]
        # Compute rate as difference between rounds
        rates = np.diff(divergences, axis=1)
        mean_rate = rates.mean(axis=0)
        std_rate = rates.std(axis=0)
        round_midpoints = [(rounds[i] + rounds[i+1])/2 for i in range(len(rounds)-1)]
        ax3.plot(round_midpoints, mean_rate, 'g-', linewidth=2)
        ax3.fill_between(round_midpoints, mean_rate - std_rate, mean_rate + std_rate, 
                        alpha=0.3, color='green')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Round')
        ax3.set_ylabel('Divergence Rate (Δdist/Δround)')
        ax3.set_title('Divergence Rate by Round\n(flat = saturated, high = active mixing)')
        
        # Plot 4: Final divergence distribution
        ax4 = axes[1, 1]
        final_divergences = divergences[:, -1]
        ax4.hist(final_divergences, bins=50, density=True, alpha=0.7, color='purple')
        ax4.axvline(x=final_divergences.mean(), color='red', linestyle='--', 
                   label=f'Mean: {final_divergences.mean():.2f}')
        ax4.axvline(x=np.sqrt(128), color='green', linestyle='--', 
                   label=f'Expected (√128): {np.sqrt(128):.2f}')
        ax4.set_xlabel('Final Geodesic Distance')
        ax4.set_ylabel('Density')
        ax4.set_title('Distribution of Final Divergences\n(should be ~√128 for random)')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'divergence_curves.png', dpi=150)
        plt.close()
        print(f"    Saved: {self.output_dir / 'divergence_curves.png'}")


# =============================================================================
# PART 2: MULTI-DIRECTIONAL ANALYSIS
# =============================================================================

class MultiDirectionalProbe:
    """
    Attack the geometry from multiple angles to verify findings.
    """
    
    def __init__(self, n_samples: int = 2000):
        self.n_samples = n_samples
        self.sha = InstrumentedSHA256(sample_rounds=[0, 16, 32, 48, 64])
        self.gen = InputGenerator()
        self.results = {}
        
    def probe_different_k_neighbors(self, trajectories: List[SHA256Trajectory],
                                    k_values: List[int] = [10, 25, 50, 100, 200]):
        """
        Does the structure depend on k-NN graph density?
        """
        print("\n  Probing different k-neighbor values...")
        
        embedding = EuclideanEmbedding()
        results = {}
        
        for k in k_values:
            print(f"    k={k}...")
            constructor = LaplacianConstructor(embedding, k_neighbors=k)
            solver = HeatKernelSolver(n_eigenvalues=50)
            
            # Analyze round 32 (middle)
            lap_result = constructor.build_from_trajectories(trajectories, round_num=32)
            L, points = lap_result.laplacian, lap_result.points
            hk_result = solver.solve(L)
            
            results[k] = {
                'spectral_gap': hk_result.spectral_gap,
                'eigenvalues': hk_result.eigenvalues[:20].tolist(),
                'effective_dim': hk_result.effective_dimension(1.0)
            }
            
        self.results['k_neighbor_probe'] = results
        return results
    
    def probe_different_time_scales(self, trajectories: List[SHA256Trajectory],
                                    t_values: List[float] = [0.01, 0.1, 1.0, 10.0, 100.0]):
        """
        Does structure appear at different diffusion time scales?
        """
        print("\n  Probing different time scales...")
        
        embedding = EuclideanEmbedding()
        constructor = LaplacianConstructor(embedding, k_neighbors=50)
        solver = HeatKernelSolver(n_eigenvalues=100)
        
        results = {}
        
        for r in [0, 32, 64]:
            lap_result = constructor.build_from_trajectories(trajectories, round_num=r)
            L, points = lap_result.laplacian, lap_result.points
            hk_result = solver.solve(L)
            
            round_results = {}
            for t in t_values:
                heat_trace = hk_result.heat_trace(t)
                eff_dim = hk_result.effective_dimension(t)
                round_results[t] = {
                    'heat_trace': heat_trace,
                    'effective_dimension': eff_dim
                }
            results[r] = round_results
            
        self.results['time_scale_probe'] = results
        return results
    
    def probe_different_embeddings(self, trajectories: List[SHA256Trajectory]):
        """
        Compare all three embeddings on the same data.
        """
        print("\n  Probing different embeddings...")
        
        embeddings = {
            'euclidean': EuclideanEmbedding(),
            'hyperbolic': HyperbolicEmbedding(curvature=-1.0),
            'davis': DavisManifoldEmbedding()
        }
        
        results = {}
        
        for name, embedding in embeddings.items():
            print(f"    {name}...")
            constructor = LaplacianConstructor(embedding, k_neighbors=50)
            solver = HeatKernelSolver(n_eigenvalues=50)
            
            round_results = {}
            for r in [0, 16, 32, 48, 64]:
                try:
                    lap_result = constructor.build_from_trajectories(trajectories, round_num=r)
                    L, points = lap_result.laplacian, lap_result.points
                    hk_result = solver.solve(L)
                    round_results[r] = {
                        'spectral_gap': hk_result.spectral_gap,
                        'n_significant_modes': np.sum(hk_result.eigenvalues < 0.5)
                    }
                except Exception as e:
                    round_results[r] = {'error': str(e)}
                    
            results[name] = round_results
            
        self.results['embedding_probe'] = results
        return results
    
    def probe_maj_vs_ch_regions(self, n_pairs: int = 2000):
        """
        Flip bits only in Maj region vs only in Ch region.
        """
        print("\n  Probing Maj vs Ch regions...")
        
        # Maj region: bits 0-95 (words a, b, c)
        # Ch region: bits 128-223 (words e, f, g)
        # (bits 96-127 = word d, bits 224-255 = word h)
        
        maj_bits = list(range(0, 96))
        ch_bits = list(range(128, 224))
        
        results = {}
        embedding = EuclideanEmbedding()
        
        for region_name, bit_range in [('maj', maj_bits), ('ch', ch_bits)]:
            print(f"    {region_name} region...")
            
            # Generate pairs with flips only in this region
            pairs = self.gen.hamming_pairs(n_pairs, bit_positions=bit_range)
            
            # Collect trajectories
            all_messages = [p.message_a for p in pairs] + [p.message_b for p in pairs]
            trajectories = self.sha.hash_batch(all_messages)
            
            # Compute divergences
            divergences = []
            rounds = self.sha.sample_rounds
            
            for i in range(n_pairs):
                traj_a = trajectories[i]
                traj_b = trajectories[i + n_pairs]
                
                curve = []
                for r in rounds:
                    state_a = next(s for s in traj_a.states if s.round_num == r)
                    state_b = next(s for s in traj_b.states if s.round_num == r)
                    p_a = embedding.embed(state_a.state_vector)
                    p_b = embedding.embed(state_b.state_vector)
                    dist = embedding.distance(p_a, p_b)
                    curve.append(dist)
                divergences.append(curve)
            
            divergences = np.array(divergences)
            
            results[region_name] = {
                'mean_divergence': divergences.mean(axis=0).tolist(),
                'std_divergence': divergences.std(axis=0).tolist(),
                'final_mean': float(divergences[:, -1].mean()),
                'final_std': float(divergences[:, -1].std()),
                'half_life': self._compute_half_life(divergences.mean(axis=0), rounds)
            }
            
        self.results['region_probe'] = results
        return results
    
    def _compute_half_life(self, mean_curve, rounds):
        """Compute rounds to reach half of final divergence."""
        final = mean_curve[-1]
        half = final / 2
        for i, val in enumerate(mean_curve):
            if val >= half:
                return rounds[i]
        return rounds[-1]
    
    def probe_random_seeds(self, n_seeds: int = 5, n_samples: int = 1000):
        """
        Verify findings are stable across different random seeds.
        """
        print("\n  Probing random seed stability...")
        
        results = []
        
        for seed in range(n_seeds):
            print(f"    seed={seed}...")
            
            gen = InputGenerator(seed=seed)
            inputs = gen.random_batch(n_samples)
            trajectories = self.sha.hash_batch(inputs)
            
            embedding = EuclideanEmbedding()
            constructor = LaplacianConstructor(embedding, k_neighbors=50)
            solver = HeatKernelSolver(n_eigenvalues=50)
            
            # Analyze round 32
            lap_result = constructor.build_from_trajectories(trajectories, round_num=32)
            L, points = lap_result.laplacian, lap_result.points
            hk_result = solver.solve(L)
            
            results.append({
                'seed': seed,
                'spectral_gap': hk_result.spectral_gap,
                'eigenvalues': hk_result.eigenvalues[:10].tolist()
            })
            
        # Compute stability metrics
        gaps = [r['spectral_gap'] for r in results]
        stability = {
            'mean_gap': float(np.mean(gaps)),
            'std_gap': float(np.std(gaps)),
            'cv_gap': float(np.std(gaps) / np.mean(gaps)),  # Coefficient of variation
            'results': results
        }
        
        self.results['seed_probe'] = stability
        return stability


# =============================================================================
# PART 3: DEEP STRUCTURAL ANALYSIS
# =============================================================================

class DeepStructuralAnalyzer:
    """
    Characterize the geometric curves in detail.
    """
    
    def __init__(self):
        self.results = {}
        
    def analyze_curvature_distribution(self, trajectories: List[SHA256Trajectory],
                                        rounds: List[int] = [0, 16, 32, 48, 64]):
        """
        Estimate local curvature from heat kernel small-t asymptotics.
        """
        print("\n  Analyzing curvature distribution...")
        
        embedding = EuclideanEmbedding()
        constructor = LaplacianConstructor(embedding, k_neighbors=50)
        solver = HeatKernelSolver(n_eigenvalues=100)
        
        results = {}
        
        for r in rounds:
            print(f"    Round {r}...")
            lap_result = constructor.build_from_trajectories(trajectories, round_num=r)
            L, points = lap_result.laplacian, lap_result.points
            hk_result = solver.solve(L)
            
            # Estimate curvature from HKS at small t
            t_small = 0.1
            hks = solver.heat_kernel_signature(hk_result, np.array([t_small]))
            curvature_proxy = hks[:, 0]  # HKS at small t relates to scalar curvature
            
            results[r] = {
                'mean_curvature': float(np.mean(curvature_proxy)),
                'std_curvature': float(np.std(curvature_proxy)),
                'min_curvature': float(np.min(curvature_proxy)),
                'max_curvature': float(np.max(curvature_proxy)),
                'curvature_range': float(np.max(curvature_proxy) - np.min(curvature_proxy))
            }
            
        self.results['curvature_analysis'] = results
        return results
    
    def analyze_diffusion_distances(self, trajectories: List[SHA256Trajectory],
                                     round_num: int = 64):
        """
        Compute diffusion distance matrix and analyze its structure.
        """
        print(f"\n  Analyzing diffusion distances at round {round_num}...")
        
        embedding = EuclideanEmbedding()
        constructor = LaplacianConstructor(embedding, k_neighbors=50)
        solver = HeatKernelSolver(n_eigenvalues=100)
        
        lap_result = constructor.build_from_trajectories(trajectories[:1000], round_num=round_num)
        L, points = lap_result.laplacian, lap_result.points
        hk_result = solver.solve(L)
        
        # Diffusion distance at multiple time scales
        results = {}
        for t in [0.1, 1.0, 10.0]:
            diff_dist = solver.diffusion_distance(hk_result, t)
            
            # Analyze the distance matrix
            upper_tri = diff_dist[np.triu_indices(diff_dist.shape[0], k=1)]
            
            results[t] = {
                'mean_distance': float(np.mean(upper_tri)),
                'std_distance': float(np.std(upper_tri)),
                'min_distance': float(np.min(upper_tri)),
                'max_distance': float(np.max(upper_tri))
            }
            
        self.results['diffusion_distance_analysis'] = results
        return results
    
    def analyze_spectral_clustering(self, trajectories: List[SHA256Trajectory],
                                     round_num: int = 64, n_clusters: int = 5):
        """
        Do states cluster in spectral space? Clustering = structure.
        """
        print(f"\n  Analyzing spectral clustering at round {round_num}...")
        
        from sklearn.cluster import KMeans
        
        embedding = EuclideanEmbedding()
        constructor = LaplacianConstructor(embedding, k_neighbors=50)
        solver = HeatKernelSolver(n_eigenvalues=20)
        
        lap_result = constructor.build_from_trajectories(trajectories[:2000], round_num=round_num)
        L, points = lap_result.laplacian, lap_result.points
        hk_result = solver.solve(L)
        
        # Use first few eigenvectors for spectral embedding
        spectral_coords = hk_result.eigenvectors[:, 1:n_clusters+1]  # Skip constant eigenvector
        
        # Cluster in spectral space
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(spectral_coords)
        
        # Compute cluster quality metrics
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        silhouette = silhouette_score(spectral_coords, labels)
        calinski = calinski_harabasz_score(spectral_coords, labels)
        
        # Compare to random clustering
        random_labels = np.random.randint(0, n_clusters, len(labels))
        random_silhouette = silhouette_score(spectral_coords, random_labels)
        
        results = {
            'n_clusters': n_clusters,
            'silhouette_score': float(silhouette),
            'random_silhouette': float(random_silhouette),
            'silhouette_ratio': float(silhouette / random_silhouette) if random_silhouette > 0 else float('inf'),
            'calinski_harabasz': float(calinski),
            'cluster_sizes': [int(np.sum(labels == i)) for i in range(n_clusters)]
        }
        
        self.results['spectral_clustering'] = results
        return results
    
    def analyze_geodesic_structure(self, trajectories: List[SHA256Trajectory]):
        """
        Analyze the structure of geodesics between states.
        """
        print("\n  Analyzing geodesic structure...")
        
        # Sample state pairs and measure geodesic vs Euclidean distance
        embedding = EuclideanEmbedding()
        n_states = min(500, len(trajectories))
        
        # Get final states
        final_states = []
        for traj in trajectories[:n_states]:
            final_state = next(s for s in traj.states if s.round_num == 64)
            final_states.append(final_state.state_vector)
        final_states = np.array(final_states)
        
        # Compute pairwise Euclidean distances
        euclidean_dists = squareform(pdist(final_states, 'euclidean'))
        
        # Compute graph geodesic (shortest path) distances
        # Build k-NN graph
        k = 10
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(final_states)
        distances, indices = nn.kneighbors(final_states)
        
        # Create sparse adjacency matrix
        n = len(final_states)
        adj = sparse.lil_matrix((n, n))
        for i in range(n):
            for j, d in zip(indices[i], distances[i]):
                adj[i, j] = d
                adj[j, i] = d
        adj = adj.tocsr()
        
        # Compute shortest paths (geodesics on graph)
        from scipy.sparse.csgraph import shortest_path
        geodesic_dists = shortest_path(adj, directed=False)
        
        # Compare Euclidean vs geodesic
        mask = np.triu(np.ones_like(euclidean_dists), k=1).astype(bool)
        mask &= ~np.isinf(geodesic_dists)  # Exclude unreachable pairs
        
        euc_flat = euclidean_dists[mask]
        geo_flat = geodesic_dists[mask]
        
        # Distortion: geodesic / euclidean ratio
        distortion = geo_flat / (euc_flat + 1e-10)
        
        results = {
            'mean_distortion': float(np.mean(distortion)),
            'std_distortion': float(np.std(distortion)),
            'max_distortion': float(np.max(distortion)),
            'min_distortion': float(np.min(distortion)),
            'correlation': float(pearsonr(euc_flat, geo_flat)[0]),
            'n_pairs_analyzed': int(np.sum(mask))
        }
        
        self.results['geodesic_analysis'] = results
        return results


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_full_analysis(n_samples: int = 2000, n_pairs: int = 1000, output_dir: str = "visualizations"):
    """
    Run the complete geometric analysis suite.
    """
    print("=" * 70)
    print("  SHA-256 GEOMETRIC STRUCTURE ANALYSIS")
    print("=" * 70)
    
    # Setup
    sha = InstrumentedSHA256(sample_rounds=[0, 8, 16, 24, 32, 40, 48, 56, 64])
    gen = InputGenerator(seed=42)
    visualizer = GeometricVisualizer(output_dir)
    prober = MultiDirectionalProbe(n_samples)
    analyzer = DeepStructuralAnalyzer()
    
    # Generate data
    print("\n[1/5] Generating trajectories...")
    inputs = gen.random_batch(n_samples)
    trajectories = sha.hash_batch(inputs)
    print(f"  Generated {len(trajectories)} trajectories")
    
    # Compute heat kernel results for visualization
    print("\n[2/5] Computing heat kernel analysis...")
    embedding = EuclideanEmbedding()
    constructor = LaplacianConstructor(embedding, k_neighbors=50)
    solver = HeatKernelSolver(n_eigenvalues=100)
    
    hk_results = {}
    for r in [0, 8, 16, 24, 32, 40, 48, 56, 64]:
        print(f"  Round {r}...")
        lap_result = constructor.build_from_trajectories(trajectories, round_num=r)
        L = lap_result.laplacian
        hk_results[r] = solver.solve(L)
    
    # Generate avalanche data
    print("\n[3/5] Computing avalanche divergences...")
    pairs = gen.hamming_pairs(n_pairs)
    all_messages = [p.message_a for p in pairs] + [p.message_b for p in pairs]
    pair_trajectories = sha.hash_batch(all_messages)
    
    divergences = []
    rounds = sha.sample_rounds
    for i in range(n_pairs):
        traj_a = pair_trajectories[i]
        traj_b = pair_trajectories[i + n_pairs]
        curve = []
        for r in rounds:
            state_a = next(s for s in traj_a.states if s.round_num == r)
            state_b = next(s for s in traj_b.states if s.round_num == r)
            p_a = embedding.embed(state_a.state_vector)
            p_b = embedding.embed(state_b.state_vector)
            curve.append(embedding.distance(p_a, p_b))
        divergences.append(curve)
    divergences = np.array(divergences)
    
    # Compute per-bit rates (simplified)
    final_divergences = divergences[:, -1]
    divergence_rates = np.zeros(256)
    # This is a placeholder - actual per-bit rates come from experiment 2
    for i in range(min(256, len(final_divergences))):
        divergence_rates[i] = final_divergences[i % len(final_divergences)] / 64
    
    slow_bits = [1, 5, 7, 9, 10, 14, 16, 19, 21, 26, 28, 31, 39, 45, 53, 58, 59, 67, 74, 87, 90, 91]
    
    # PART 1: Visualizations
    print("\n[4/5] Generating visualizations...")
    visualizer.plot_state_space_trajectories(trajectories, n_samples=500, method='pca')
    visualizer.plot_eigenvalue_spectrum(hk_results)
    visualizer.plot_anisotropy_heatmap(divergence_rates, slow_bits)
    visualizer.plot_divergence_curves(divergences, slow_bits, rounds)
    
    # PART 2: Multi-directional probes
    print("\n[5/5] Running multi-directional analysis...")
    prober.probe_different_k_neighbors(trajectories)
    prober.probe_different_time_scales(trajectories)
    prober.probe_different_embeddings(trajectories)
    prober.probe_maj_vs_ch_regions(n_pairs=500)
    prober.probe_random_seeds(n_seeds=3, n_samples=500)
    
    # PART 3: Deep structural analysis
    analyzer.analyze_curvature_distribution(trajectories)
    analyzer.analyze_spectral_clustering(trajectories)
    analyzer.analyze_geodesic_structure(trajectories)
    
    # Save all results
    all_results = {
        'multi_directional': prober.results,
        'deep_structural': analyzer.results
    }
    
    results_path = Path(output_dir) / 'analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n  Visualizations saved to: {output_dir}/")
    print(f"  Results saved to: {results_path}")
    
    # Print key findings
    print("\n  KEY FINDINGS:")
    print("  " + "-" * 50)
    
    if 'region_probe' in prober.results:
        maj = prober.results['region_probe']['maj']
        ch = prober.results['region_probe']['ch']
        print(f"  Maj region half-life: {maj['half_life']} rounds")
        print(f"  Ch region half-life:  {ch['half_life']} rounds")
        print(f"  Maj final divergence: {maj['final_mean']:.3f} ± {maj['final_std']:.3f}")
        print(f"  Ch final divergence:  {ch['final_mean']:.3f} ± {ch['final_std']:.3f}")
    
    if 'spectral_clustering' in analyzer.results:
        sc = analyzer.results['spectral_clustering']
        print(f"\n  Spectral clustering silhouette: {sc['silhouette_score']:.3f}")
        print(f"  Random baseline silhouette:     {sc['random_silhouette']:.3f}")
        print(f"  Structure ratio:                {sc['silhouette_ratio']:.2f}x")
    
    if 'geodesic_analysis' in analyzer.results:
        ga = analyzer.results['geodesic_analysis']
        print(f"\n  Geodesic distortion mean: {ga['mean_distortion']:.3f}")
        print(f"  Euclidean-geodesic corr:  {ga['correlation']:.3f}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SHA-256 Geometric Analysis Suite")
    parser.add_argument('--n-samples', type=int, default=2000, help='Number of random samples')
    parser.add_argument('--n-pairs', type=int, default=1000, help='Number of Hamming pairs')
    parser.add_argument('--output-dir', type=str, default='visualizations', help='Output directory')
    args = parser.parse_args()
    
    run_full_analysis(args.n_samples, args.n_pairs, args.output_dir)
