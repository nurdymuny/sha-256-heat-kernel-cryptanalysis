"""Small local diagnostics for the submission review; does not alter research receipts.

Run from any directory: python -X utf8 <path>/review/submission_audit.py
This intentionally tests the CURRENT implementation, including its failures.
Assertions confirm counterexamples, not production correctness. No cloud jobs.
"""
from pathlib import Path
import ast
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
import sklearn
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from src.sha256.core import InstrumentedSHA256
from src.sha256.input_generator import InputGenerator
from src.embeddings.euclidean import EuclideanEmbedding
from src.embeddings.hyperbolic import HyperbolicEmbedding
from src.analysis.laplacian import LaplacianConstructor
from src.analysis.heat_kernel import HeatKernelSolver
from src.analysis.anomaly import AnomalyDetector, NullDistribution
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SEED = 20260909


def normalized_without_absolute_cutoff(w):
    d = np.asarray(w.sum(axis=1)).ravel()
    assert np.all(d > 0), "Diagnostic requires positive-degree vertices"
    di = sparse.diags(1 / np.sqrt(d))
    return (sparse.eye(w.shape[0]) - di @ w @ di).toarray()


def graph_diagnostic(points):
    c = LaplacianConstructor(EuclideanEmbedding(), k_neighbors=30)
    result = c.build_from_points(points)
    w = result.weight_matrix
    d = np.asarray(w.sum(axis=1)).ravel()
    current = result.laplacian.toarray()
    fixed = normalized_without_absolute_cutoff(w)
    scaled = c._compute_normalized_laplacian(w * 1e12, d * 1e12).toarray()
    v = np.sqrt(d) / np.linalg.norm(np.sqrt(d))
    return {
        "n": len(points), "k": 30, "diffusion_time": 1.0,
        "connected_components": int(connected_components(w, directed=False)[0]),
        "positive_degrees_discarded": int(np.sum((d > 0) & (d <= 1e-10))),
        "degree_min": float(d.min()), "degree_max": float(d.max()),
        "current_min_eigenvalue": float(np.linalg.eigvalsh(current)[0]),
        "corrected_min_eigenvalue": float(np.linalg.eigvalsh(fixed)[0]),
        "current_zero_mode_residual": float(np.linalg.norm(current @ v)),
        "corrected_zero_mode_residual": float(np.linalg.norm(fixed @ v)),
        "current_common_rescaling_max_change": float(np.max(np.abs(current - scaled))),
        "corrected_common_rescaling_max_change": float(np.max(np.abs(fixed - normalized_without_absolute_cutoff(w * 1e12)))),
    }, fixed


def load_reduced_function_without_running_experiments():
    path = ROOT / 'analysis/cube_attack.py'
    module = ast.parse(path.read_text(encoding='utf-8-sig'))
    allowed = [n for n in module.body if isinstance(n, (ast.Import, ast.ImportFrom))
               or isinstance(n, ast.FunctionDef) and n.name in ('rotr', 'sha256_reduced')]
    scope = {}
    exec(compile(ast.Module(body=allowed, type_ignores=[]), str(path), 'exec'), scope)
    return scope['sha256_reduced']


def gf2_rank(a):
    a = a.copy() % 2
    row = 0
    for col in range(a.shape[1]):
        pivots = np.flatnonzero(a[row:, col])
        if not len(pivots):
            continue
        pivot = row + pivots[0]
        a[[row, pivot]] = a[[pivot, row]]
        for j in range(a.shape[0]):
            if j != row and a[j, col]:
                a[j] ^= a[row]
        row += 1
        if row == a.shape[0]:
            break
    return row


def wilson(successes, n):
    z = 1.959963984540054
    p = successes / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [float(center-half), float(center+half)]


def matched_clustering_control():
    """New controlled experiment at the headline sample size, not a frozen replay."""
    sha = InstrumentedSHA256(sample_rounds=[64])
    messages = InputGenerator(seed=42).random_batch(1000)
    states = np.array([sha.hash_with_trajectory(m).states[-1].state_vector for m in messages])
    random_states = np.random.default_rng(SEED).integers(0, 2, (1000, 256), dtype=np.uint8)
    out = {"n": 1000, "k": 50, "modes": 20, "clusters": 5,
           "sha_input_seed": 42, "random_input_seed": SEED,
           "note": "Both datasets receive fitted KMeans. Corrected branch changes only degree normalization; this is not a calibrated significance test."}
    for name, data in [('sha', states), ('random', random_states)]:
        c = LaplacianConstructor(EuclideanEmbedding(), k_neighbors=50)
        lr = c.build_from_points(data.astype(float))
        w = lr.weight_matrix
        d = np.asarray(w.sum(axis=1)).ravel()
        out[name] = {"discarded_positive_degrees": int(np.sum((d > 0) & (d <= 1e-10)))}
        for version, lap in [('current', lr.laplacian), ('corrected', sparse.csr_matrix(normalized_without_absolute_cutoff(w)))]:
            np.random.seed(SEED)
            hk = HeatKernelSolver(n_eigenvalues=20).solve(lap)
            coords = hk.eigenvectors[:,1:6]
            labels = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(coords)
            out[name][version] = {"min_eigenvalue": float(hk.eigenvalues[0]),
                "silhouette": float(silhouette_score(coords, labels)),
                "cluster_sizes": [int(np.sum(labels==i)) for i in range(5)]}
        print(f'Matched clustering: {name} complete.', flush=True)
    return out


def main():
    rng = np.random.default_rng(SEED)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
              "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
              "scope": "Targeted current-code counterexamples, not a rerun of all historical experiments."}
    sha = InstrumentedSHA256(sample_rounds=list(range(65)))
    lengths = [0, 1, 3, 55, 56, 63, 64, 65, 119, 120, 128]
    checked = []
    for length in lengths:
        for trial in range(3):
            msg = rng.bytes(length)
            actual = sha.hash(msg)
            assert actual == hashlib.sha256(msg).digest()
            checked.append([length, trial])
    abc = sha.hash_with_trajectory(b'abc')
    multi = sha.hash_with_trajectory(bytes(64))
    duplicates = {str(k): v for k, v in Counter(s.round_num for s in multi.states).items() if v > 1}
    report['sha_core'] = {"hashlib_cases_passed": len(checked), "message_lengths": lengths,
                          "abc_digest": abc.final_hash.hex(), "abc_last_working_state": abc.states[-1].to_bytes().hex(),
                          "working_state_is_digest": abc.states[-1].to_bytes() == abc.final_hash,
                          "64_byte_message_blocks": multi.num_blocks, "duplicate_round_labels": duplicates}
    print('SHA and padding-boundary checks complete.', flush=True)

    bits = rng.integers(0, 2, (256, 256), dtype=np.uint8)
    report['random_graph'], fixed = graph_diagnostic(bits.astype(float))
    messages = InputGenerator(seed=42).random_batch(256)
    sha64 = InstrumentedSHA256(sample_rounds=[64])
    states = np.array([sha64.hash_with_trajectory(m).states[-1].state_vector for m in messages])
    report['sha_graph'], _ = graph_diagnostic(states.astype(float))
    assert report['random_graph']['current_zero_mode_residual'] > 1e-3
    assert report['random_graph']['corrected_zero_mode_residual'] < 1e-12
    print('Graph normalization counterexamples complete.', flush=True)

    times = np.array([0.0, 0.01, 0.1, 1.0, 10.0])
    solver = HeatKernelSolver(n_eigenvalues=50, time_scales=times)
    hk = solver.solve(fixed)
    full_eigenvalues = np.linalg.eigvalsh(fixed)
    full = np.array([np.exp(-full_eigenvalues*t).sum() for t in times])
    report['truncated_heat'] = {"n_vertices": len(fixed), "retained_modes": hk.n_eigenvalues,
        "times": times.tolist(), "reported_trace": hk.heat_trace_values.tolist(), "full_trace": full.tolist(),
        "relative_trace_underestimate": (1-hk.heat_trace_values/full).tolist()}
    curv = solver.spectral_curvature_estimate(hk)
    report['curvature_statistic'] = {"min": float(curv.min()), "max": float(curv.max()),
                                    "all_nonpositive": bool(np.all(curv <= 0)),
                                    "reason": "Each PSD heat-kernel diagonal is nonincreasing; this ratio cannot establish curvature sign."}

    # A degenerate empirical null: every reference graph has exactly this spectrum.
    null = NullDistribution(hk.eigenvalues.copy(), np.zeros_like(hk.eigenvalues),
        hk.spectral_gaps.copy(), np.zeros_like(hk.spectral_gaps), hk.heat_trace_values.copy(),
        np.zeros_like(times), 20, times)
    anomaly = AnomalyDetector().analyze(hk, null)
    report['identical_to_degenerate_null'] = {
        "description": "Observed spectrum equals every reference spectrum; a valid comparison must not call it anomalous.",
        "eigenvalue_p": anomaly.eigenvalue_p_value, "gap_p": anomaly.gap_p_value,
        "structure_detected": bool(anomaly.structure_detected), "reported_confidence": float(anomaly.confidence),
        "anomalous_modes": [int(x) for x in anomaly.anomalous_modes]}
    assert anomaly.structure_detected and len(anomaly.anomalous_modes) == 0

    hyp = HyperbolicEmbedding()
    r = 0.95 * 16 / (16 + 1e-10)
    errors = []
    for a, b in zip(bits[:64], bits[64:128]):
        h = np.count_nonzero(a != b)
        predicted = np.arccosh(1 + 8*r*r*h / (256*((1-r*r)**2 + 1e-10)))
        errors.append(abs(hyp.distance(hyp.embed(a), hyp.embed(b)) - predicted))
    report['binary_hyperbolic_embedding'] = {"pairs": len(errors), "fixed_radius": r,
        "max_error_vs_function_of_hamming_only": float(max(errors))}
    print('Heat, null-calibration and embedding checks complete.', flush=True)

    reduced = load_reduced_function_without_running_experiments()
    base = (0xDEADBEEF << 64).to_bytes(64, 'big')
    varied = ((0xDEADBEEF << 64) | (1 << 2)).to_bytes(64, 'big')
    first_difference = next(r for r in range(1, 21) if reduced(base, r) != reduced(varied, r))
    cube_bits = [2, 13, 22, 32, 39, 34, 45]
    sums = {}
    for rounds in (15, 16, 17, 18):
        acc = [0]*8
        for corner in range(1 << len(cube_bits)):
            value = 0xDEADBEEF << 64
            for i, bit in enumerate(cube_bits):
                value |= ((corner >> i) & 1) << bit
            for j, word in enumerate(reduced(value.to_bytes(64, 'big'), rounds)):
                acc[j] ^= word
        sums[str(rounds)] = acc
    report['cube_coordinates'] = {"integer_bit_to_message_word": {str(b): 15-b//32 for b in cube_bits},
        "integer_bit_2_first_influences_state_after_round": first_difference,
        "cube_dimension": len(cube_bits), "fixed_base_hex": base.hex(), "cube_xor_by_round": sums,
        "observable": "First raw-block working state before feed-forward, not the padded two-block hash."}
    assert first_difference == 16
    m = np.array([[1,1,0], [1,0,1], [0,1,1]], dtype=int)
    report['rank_field_counterexample'] = {"matrix": m.tolist(), "real_rank": int(np.linalg.matrix_rank(m)), "gf2_rank": gf2_rank(m)}
    rates = rng.normal(1, 0.01, 440)
    report['quantile_selection'] = {"iid_rates": len(rates), "percentile": 5,
        "selected_count": int(np.sum(rates < np.percentile(rates, 5))),
        "meaning": "The lower 5% selects 22 even without a special population; this is not a null test."}

    generated_pairs = InputGenerator(seed=42).hamming_pairs(2000)
    wrong = [i for i,p in enumerate(generated_pairs) if p.flipped_bit != i % 512]
    report['reduced_round_metadata'] = {"pairs": 2000, "seed": 42,
        "actual_input_bits": 440, "code_assumes_bits": 512,
        "incorrect_pair_bit_labels": len(wrong),
        "first_ten_actual": [int(p.flipped_bit) for p in generated_pairs[:10]],
        "first_ten_assumed": list(range(10)),
        "source": "analysis/reduced_round_analysis.py:80"}
    attacks = json.loads((ROOT/'attack_results.json').read_text(encoding='utf-8-sig'))['slow_manifold']
    mu, sd = attacks['baseline_mean'], attacks['baseline_std']
    report['slow_pair_normalization'] = {"saved_baseline_mean": mu, "saved_baseline_std": sd,
        "z_created_by_dividing_baseline_mean_by_sqrt2": float((mu/np.sqrt(2)-mu)/sd),
        "interpretation": "Normalization alone creates this score without any reduction in raw distance; search multiplicity is additional.",
        "saved_pairs_with_raw_distance_restored": [
            {"pattern": p['pattern'], "saved_z": p['z_score'],
             "raw_distance": float(p['divergence']*np.sqrt(len(p['pattern']))),
             "raw_distance_standardized_vs_single_flip": float((p['divergence']*np.sqrt(len(p['pattern']))-mu)/sd)}
            for p in attacks['anomalous_pairs'][:3]],
        "qualification": "Restored scores are diagnostics, not valid significance tests; two-bit controls and matched search remain necessary."}

    saved = json.loads((ROOT/'visualizations/analysis_results.json').read_text(encoding='utf-8-sig'))
    report['saved_geometric_sections'] = {k:v for k,v in saved.items() if k in ('spectral_clustering', 'geodesic_analysis')}
    # Some receipts nest analyzer sections. Capture the exact matching objects recursively.
    def collect(obj, prefix=''):
        found = {}
        if isinstance(obj, dict):
            for key, val in obj.items():
                path = prefix + '/' + key
                if key in ('spectral_clustering', 'geodesic_analysis'):
                    found[path] = val
                elif isinstance(val, (dict, list)):
                    found.update(collect(val, path))
        elif isinstance(obj, list):
            for i, val in enumerate(obj):
                found.update(collect(val, prefix+'/'+str(i)))
        return found
    # Preserve nonstandard Infinity in the historical JSON as explicit text,
    # so this audit itself remains strict JSON.
    def strict_values(obj):
        if isinstance(obj, float) and not np.isfinite(obj):
            return str(obj)
        if isinstance(obj, dict):
            return {k: strict_values(v) for k,v in obj.items()}
        if isinstance(obj, list):
            return [strict_values(v) for v in obj]
        return obj
    report['saved_geometric_sections'] = strict_values(collect(saved))
    report['illustrative_ml_intervals'] = {name: {"correct": correct, "test_n": 800, "accuracy": correct/800,
        "wilson_95_percent": wilson(correct, 800), "qualification": "Single fixed classifier, independent test observations; not multiplicity-adjusted or a security bound."}
        for name,correct in [('random_forest',387),('svm',388)]}
    report['matched_clustering_control'] = matched_clustering_control()
    sources = [
        'src/sha256/core.py','src/sha256/input_generator.py','src/embeddings/euclidean.py','src/embeddings/hyperbolic.py',
        'src/analysis/laplacian.py','src/analysis/heat_kernel.py','src/analysis/anomaly.py','src/experiments/avalanche.py',
        'analysis/sha256_geometric_analysis.py','analysis/sha256_distinguisher_test.py','analysis/slow_bit_analysis.py',
        'analysis/cube_attack.py','analysis/cube_attack_escalation.py','analysis/comparative_cube_attack.py',
        'analysis/reduced_round_analysis.py','analysis/sha256_geometric_attacks.py','attack_results.json',
        'RESULTS.md','visualizations/analysis_results.json','results/distinguisher_comprehensive.json',
        'results/exp1_20251206_144431.json','review/submission_audit.py']
    report['source_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    output = ROOT/'review/submission_audit_results.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('sha_core','random_graph','sha_graph','truncated_heat','identical_to_degenerate_null','cube_coordinates')}, indent=2))
    print(f'Wrote {output}', flush=True)


if __name__ == '__main__':
    main()
