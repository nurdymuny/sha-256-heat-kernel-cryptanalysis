"""One-time, guarded source corrections following submission review."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def edit(path, old, new):
    p=ROOT/path; s=p.read_text(encoding='utf-8-sig')
    if old not in s: raise RuntimeError(f'Missing source anchor: {path}: {old[:60]}')
    p.write_text(s.replace(old,new),encoding='utf-8')

p=ROOT/'src/analysis/anomaly.py'; s=p.read_text(encoding='utf-8-sig')
s=s.replace('confidence: float                 # 1 - p_value','confidence: Optional[float]       # Deprecated: p-values are not confidence')
s=s.replace('    # Detailed structure (if found)', '    p_value: float = 1.0\n    calibration: str = "exchangeable pooled-spectrum distance"\n\n    # Detailed structure (if found)')
s=s.replace('    time_scales: np.ndarray\n\n\nclass AnomalyDetector:', '    time_scales: np.ndarray\n    eigenvalue_samples: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))\n    failed_samples: List[str] = field(default_factory=list)\n\n\nclass AnomalyDetector:')
s=s.replace('        all_heat_traces = []','        all_heat_traces = []\n        failures = []')
s=s.replace('                # Skip failed samples\n                continue','                failures.append(f"sample {i}: {type(e).__name__}: {e}")\n                continue')
s=s.replace('            time_scales=time_scales\n        )','            time_scales=time_scales,\n            eigenvalue_samples=eigenvalues,\n            failed_samples=failures\n        )',1)
a=s.index('        # Eigenvalue comparison',s.index('    def analyze(')); b=s.index('    def compare_rounds(',a)
s=s[:a]+'''        if null_dist.failed_samples:
            raise ValueError("Reference simulation had failures; repair before inference")
        refs = np.asarray(null_dist.eigenvalue_samples, dtype=float)
        obs = np.asarray(observed.eigenvalues, dtype=float)
        if refs.ndim != 2 or len(refs) < 2 or refs.shape[1] != len(obs):
            raise ValueError("Inference requires complete, matched independent reference spectra")
        # The pooled center is permutation symmetric in observed + references.
        # Under exchangeability, rank of this predeclared distance is valid.
        pool = np.vstack([obs, refs])
        distances = np.linalg.norm(pool-pool.mean(axis=0), axis=1)
        tolerance = 1e-12 * max(1.0, float(distances.max()))
        p_value = float(np.mean(distances >= distances[0]-tolerance))
        gap_pool = np.diff(pool,axis=1)
        gap_distances = np.linalg.norm(gap_pool-gap_pool.mean(axis=0),axis=1)
        gap_p = float(np.mean(gap_distances >= gap_distances[0]-tolerance))
        # Gap p is descriptive; do not combine dependent tests with Fisher.
        return AnomalyReport(
            spectral_entropy=self._eigenvalue_entropy(obs),
            spectral_gap_variance=float(np.var(observed.spectral_gaps)),
            diffusion_anisotropy=1.0,
            eigenvalue_ks_statistic=float(distances[0]),
            eigenvalue_p_value=p_value,
            gap_ks_statistic=float(gap_distances[0]),
            gap_p_value=gap_p,
            structure_detected=p_value < self.alpha,
            significance_level=self.alpha,
            confidence=None,
            p_value=p_value,
            eigenvalue_histogram=obs,
            null_histogram=refs.mean(axis=0),
            spectral_gap_series=observed.spectral_gaps,
        )

'''+s[b:]
s=s.replace('    confidence: Optional[float]', '    confidence: Optional[float]')
s=s.replace('eigenvalue_ks_statistic: float    # KS test vs null distribution','eigenvalue_ks_statistic: float    # Compatibility name: pooled Euclidean distance, not KS')
s=s.replace('gap_ks_statistic: float           # KS test on spectral gaps','gap_ks_statistic: float           # Compatibility name: gap distance, not KS')
s=s.replace('Under H0:\n- Eigenvalue distribution follows Marchenko-Pastur (for random graphs)\n- Spectral gaps are uniformly distributed  \n- Heat kernel signature is constant across states\n- Diffusion rate is isotropic', 'Calibration uses independent random-bit reference graphs processed identically.\nNo iid eigenmode distribution or constant heat diagonal is assumed.')
p.write_text(s,encoding='utf-8')
edit('src/experiments/baseline.py','report.confidence for report in anomaly_reports.values()', '1-report.p_value for report in anomaly_reports.values()')
edit('src/experiments/baseline.py','Highest confidence of structure','Largest 1-p score; not posterior confidence')

p=ROOT/'src/analysis/heat_kernel.py'; s=p.read_text(); a=s.index('        t1 = t_small',s.index('    def spectral_curvature_estimate(')); b=s.index('    def diffusion_distance(',a); s=s[:a]+ '\n'+s[b:]; p.write_text(s)

# Pass real coordinates into reduced-round grouping, never infer from row order.
edit('analysis/reduced_round_analysis.py','    n_pairs: int\n) -> RoundAnalysisResult:', '    n_pairs: int,\n    pair_bit_positions: List[int]\n) -> RoundAnalysisResult:')
edit('analysis/reduced_round_analysis.py','        flipped_bit = i % 512','        flipped_bit = pair_bit_positions[i]')
edit('analysis/reduced_round_analysis.py','    per_bit_rates = np.zeros(512)','    per_bit_rates = np.zeros(440)')
edit('analysis/reduced_round_analysis.py','trajectories_a, trajectories_b, r, embedding, n_pairs\n', 'trajectories_a, trajectories_b, r, embedding, n_pairs, [p.flipped_bit for p in pairs]\n')

# Correct the actual bit selected by all byte-oriented cube tools.
for name in ['analysis/comparative_cube_attack.py','analysis/precision_cube_attack.py']:
    p=ROOT/name; s=p.read_text(encoding='utf-8-sig')
    s=s.replace('bit_idx = var % 8','bit_idx = 7 - (var % 8)')
    p.write_text(s,encoding='utf-8')

# Integer-based historical cubes become canonical MSB-first message positions.
for name in ['analysis/cube_attack.py','analysis/cube_attack_escalation.py']:
    p=ROOT/name; s=p.read_text(encoding='utf-8-sig')
    s=s.replace('msg_int |= (1 << cube_pos)', 'msg_int |= (1 << (511 - cube_pos))')
    s=s.replace('BROKEN','ZERO SUM').replace('SECURE','NONZERO SUM').replace('broken','zero-sum').replace('secure','nonzero-sum')
    if name.endswith('/cube_attack.py'):
        a=s.index('for r in [15, 16, 17]:')
        s=s[:a]+'if __name__ == "__main__":\n'+''.join('    '+line if line.strip() else line for line in s[a:].splitlines(keepends=True))
    else:
        s=s.replace('rank = np.linalg.matrix_rank(matrix)', 'from src.sha256.batch import gf2_rank\n    rank = gf2_rank(matrix)')
    p.write_text(s,encoding='utf-8')

# Match physical distance scales; a fresh matched search study supplies inference.
edit('analysis/sha256_geometric_attacks.py','normalized_div = div / np.sqrt(2)','normalized_div = div  # Raw distance on both sides; no unmatched sqrt(2) scaling')
edit('analysis/sha256_distinguisher_test.py','outputs.append(final_state.state_vector)','outputs.append(np.unpackbits(np.frombuffer(traj.final_hash, dtype=np.uint8)))')
edit('analysis/sha256_distinguisher_test.py','X_pca = pca.fit_transform(X)\n        X_train_pca, X_test_pca = X_pca[:split], X_pca[split:]','X_train_pca = pca.fit_transform(X_train)\n        X_test_pca = pca.transform(X_test)')

# Keep every observation and coordinate used for aggregation.
edit('src/experiments/avalanche.py',"        'divergence_curves': divergences.tolist(),", "        'divergence_curves': divergences.tolist(),\n        'pair_bit_positions': pair_bit_positions.tolist(),\n        'per_bit_counts': {int(k):len(v) for k,v in divergence_by_bit.items()},\n        'coordinate_system': 'input-message MSB-first, 55 bytes',\n        'slow_bit_interpretation': 'lower empirical quantile; not a significance test',")

# Explicit JSON serializer for dataclasses; no repr strings or callables.
edit('main.py',"        if hasattr(obj, 'tolist'):","        from dataclasses import is_dataclass, fields\n        if is_dataclass(obj):\n            return {f.name: convert(getattr(obj,f.name)) for f in fields(obj) if not callable(getattr(obj,f.name))}\n        if hasattr(obj, 'tolist'):")
edit('main.py','json.dump(serializable, f, indent=2, default=str)','json.dump(serializable, f, indent=2, allow_nan=False)')
print('Applied shared-code corrections. Publication runner supplies matched controls and inference.')
