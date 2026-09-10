"""Remove unsupported figure inputs and interpretations from legacy entry points."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'analysis/sha256_geometric_analysis.py'; s=p.read_text(encoding='utf-8-sig')
a=s.index('        # Compare to random clustering'); b=s.index('        results = {',a)
s=s[:a]+'''        # Fit the same pipeline to independent random-bit data.
        rng = np.random.default_rng(123)
        null_bits = rng.integers(0, 2, (len(labels), 256), dtype=np.uint8)
        null_lap = constructor.build_from_points(embedding.embed_batch(null_bits))
        null_hk = solver.solve(null_lap.laplacian)
        null_coords = null_hk.eigenvectors[:, 1:n_clusters+1]
        random_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(null_coords)
        random_silhouette = silhouette_score(null_coords, random_labels)

'''+s[b:]
s=s.replace("float('inf')",'None') # undefined silhouette ratio represented as JSON null
# Replace the placeholder runner with the canonical, receipt-producing runner.
a=s.index('def run_full_analysis('); b=s.index("if __name__",a)
s=s[:a]+'''def run_full_analysis(n_samples=2000, n_pairs=1000, output_dir="visualizations"):
    """Legacy orchestration replaced by the controlled publication protocol."""
    raise RuntimeError(
        "Use python analysis/publication_study.py --output <new-directory>. "
        "It saves actual bit metadata, matched controls and measured figure inputs. "
        "The old placeholder figure runner is intentionally unavailable."
    )


'''+s[b:]
# Heat-diagonal observations must be named as such, not as curvature.
s=s.replace('curvature_proxy','heat_diagonal').replace('mean_curvature','mean_heat_diagonal').replace('std_curvature','std_heat_diagonal').replace('min_curvature','min_heat_diagonal').replace('max_curvature','max_heat_diagonal').replace('curvature_range','heat_diagonal_range').replace("self.results['curvature_analysis']","self.results['heat_diagonal_analysis']")
s=s.replace('Estimate local curvature from heat kernel small-t asymptotics.','Compatibility method: return a truncated graph heat diagonal, not curvature.')
s=s.replace('# HKS at small t relates to scalar curvature','# Truncated graph heat diagonal; no intrinsic curvature claim')
# Include a measured comparator for the detour statistic with identical settings.
a=s.index('        # Sample state pairs and measure geodesic'); b=s.index('\n\n# =============================================================================',a)
s=s[:a]+'''        from analysis.publication_study import detour
        states = np.array([next(s for s in t.states if s.round_num == 64).state_vector
                           for t in trajectories[:500]])
        null = np.random.default_rng(123).integers(0,2,states.shape,dtype=np.uint8)
        results = {'mean_distortion':detour(states),'random_mean_distortion':detour(null),
                   'n_states':len(states),'k_nonself':10,'interpretation':'graph detour ratio'}
        self.results['geodesic_analysis'] = results
        return results
'''+s[b:]
p.write_text(s,encoding='utf-8')

p=ROOT/'analysis/slow_bit_analysis.py'
p.write_text('''"""Map input-message bit positions; descriptive counts, not register inference."""
from collections import Counter

def bit_to_word(bit_position):
    if not 0 <= bit_position < 440:
        raise ValueError("Expected an MSB-first bit in a 55-byte message")
    return f"W{bit_position//32}", bit_position%32

def analyze_slow_bits(slow_bits):
    mappings=[]
    for bit in slow_bits:
        word,offset=bit_to_word(int(bit))
        mappings.append({'input_bit':int(bit),'message_word':word,
                         'msb_first_offset':offset,'lsb_number':31-offset})
    return {'mappings':mappings,'counts':dict(Counter(m['message_word'] for m in mappings)),
            'interpretation':'selected input positions; selection is not a significance test'}

def print_analysis(slow_bits,label="Selected input bits"):
    result=analyze_slow_bits(slow_bits)
    print(label,result)
    return result
''',encoding='utf-8')

p=ROOT/'analysis/comparative_cube_attack.py'; s=p.read_text(encoding='utf-8-sig')
s=s.replace('No significant difference','Descriptive bias difference')
a=s.index('    print("""\n  The heat-kernel-derived'); b=s.index('\n\n\nif __name__',a)
s=s[:a]+'''    print("These are exploratory single-bit tests. Use publication_study.py for matched controls, saved observations and multiplicity-aware inference.")
'''+s[b:]; p.write_text(s,encoding='utf-8')

p=ROOT/'analysis/reduced_round_analysis.py'; s=p.read_text(encoding='utf-8-sig')
s=s.replace("        # Get flipped bit from pair metadata (stored in trajectory)\n        # Since we're using sequential bit flips, pair i flipped bit i % 512", "        # Actual input-message bit supplied by the generator metadata.")
s=s.replace("'slow_bits': r.slow_bit_positions[:50],  # Top 50", "'slow_bits': r.slow_bit_positions,\n                'coordinate_system': 'input-message MSB-first',")
p.write_text(s,encoding='utf-8')
print('Removed placeholder orchestration; repaired legacy comparators and coordinate labels.')
