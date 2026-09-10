"""Derived manuscript quantities computed from the frozen receipts.

Every macro in numbers.tex and followup_numbers.tex, and every row of the four
table files, is produced here. publication/build_paper.py and
publication/followup_tables.py write the files from these functions, and
publication/verify_evidence.py regenerates them from the receipts and requires
byte equality with the files on disk. Design constants (sample sizes, rounds,
seeds) are typed in the manuscript; every measured quantity passes through here.
"""
from pathlib import Path
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from analysis.publication_study import interval


def read(E, name): return json.loads((Path(E) / name).read_text())
def ci4(v): return f"[{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]"
def ci3(v, scale=1): return f"[{scale*v['ci95'][0]:+.3f}, {scale*v['ci95'][1]:+.3f}]"
def ci2(v, scale=1, nd=2): return '[' + f"{scale*v['ci95'][0]:+.{nd}f}" + ', ' + f"{scale*v['ci95'][1]:+.{nd}f}" + ']'
def pformat(p):
    if p >= .001: return f'{p:.3f}'
    power = int(np.floor(np.log10(p))); return f'${p/10.**power:.2f}\\times10^{{{power}}}$'


def rank_bootstrap(candidate, controls, relation, seed=915, resamples=5000, chunk=250):
    """Percentile bootstrap over bases of the number of control sets whose mean is at or above ('ge')
    or below ('lt') the candidate mean. Returns integer 2.5 and 97.5 percentiles of that count."""
    cand = np.asarray(candidate, float); ctrl = np.asarray(controls, float); n = len(cand)
    rng = np.random.default_rng(seed); counts = []
    for start in range(0, resamples, chunk):
        idx = rng.integers(0, n, (min(chunk, resamples - start), n))
        cm = cand[idx].mean(1); sm = ctrl[:, idx].mean(2)
        counts.append(((sm >= cm) if relation == 'ge' else (sm < cm)).sum(0))
    lo, hi = np.quantile(np.concatenate(counts), [.025, .975])
    return int(round(lo)), int(round(hi))


def primary_macros(E):
    """Macros and table rows for numbers.tex, from the primary, confirmation and sensitivity receipts."""
    g = read(E, 'graphs.json'); c = read(E, 'confirmation/graphs.json'); a = read(E, 'avalanche.json')
    s = read(E, 'search.json'); q = read(E, 'cubes.json'); m = read(E, 'ml.json'); z = read(E, 'sensitivity/detour.json')
    tables = {}; rows = []
    for cohort, data in [('Primary', g), ('Confirmation', c)]:
        for h in ['1.0', '8.0']:
            for metric, label in [('silhouette', 'Silhouette'), ('spectral_gap', 'Gap')]:
                v = data['summary'][h][metric]
                rows.append(f"{cohort} & {h} & {label} & {v['sha_mean']:.4f} & {v['random_mean']:.4f} & {v['difference']['mean']:+.4f} {ci4(v['difference'])}")
        v = data['summary']['detour']
        rows.append(f"{cohort} & -- & Detour & {v['sha_mean']:.4f} & {v['random_mean']:.4f} & {v['difference']['mean']:+.4f} {ci4(v['difference'])}")
    tables['GraphRows'] = rows
    tables['SensitivityRows'] = [f"{key.replace(' minus ', ' -- ')} & {v['mean']:+.5f} & {ci4(v)}" for key, v in z['summary'].items()]
    tables['CubeRows'] = [f"{v['round']} & {100*v['target_zero_rate']:.2f} & {100*v['control_zero_rate']:.2f} & {100*v['difference']['mean']:+.2f} & [{100*v['difference']['ci95'][0]:.2f}, {100*v['difference']['ci95'][1]:.2f}] & {v['holm_p']:.3f}" for v in q['records']]
    tables['MLRows'] = [f"{name.replace('_', ' + ')} & {100*v['mean_accuracy']:.3f} & {100*v['range'][0]:.3f}--{100*v['range'][1]:.3f} & {v['mean_auc']:.4f}" for name, v in m['summary'].items()]
    macros = {'HoldoutDifference': f"{a['holdout_selected_minus_others']['mean']:+.3f}",
              'HoldoutCI': ci4(a['holdout_selected_minus_others']),
              'SearchDifference': f"{s['target_minus_mean_control']['mean']:+.3f}",
              'SearchCI': ci4(s['target_minus_mean_control']),
              'FinalHamming': f"{a['round64_mean_hamming']:.3f}",
              'RFAccuracy': f"{100*m['summary']['RF']['mean_accuracy']:.3f}",
              'SVMAccuracy': f"{100*m['summary']['PCA_SVM']['mean_accuracy']:.3f}"}
    return macros, tables


TABLE_FILES = [('GraphRows', 'graph_table.tex'), ('SensitivityRows', 'sensitivity_table.tex'), ('CubeRows', 'cube_table.tex'), ('MLRows', 'ml_table.tex')]


def followup_macros(E):
    """Macros and table rows for followup_numbers.tex, from all receipts."""
    E = Path(E); macros = {}; tables = {}
    heat = []
    graphs = {'Primary': read(E, 'graphs.json'), 'Confirmation': read(E, 'confirmation/graphs.json')}
    for group, g in graphs.items():
        for h in ['1.0', '8.0']:
            for j, t in enumerate([.01, .1, 1., 10.]):
                v = interval([r['kernels'][h]['sha']['heat_trace'][j] - r['kernels'][h]['random']['heat_trace'][j] for r in g['records']])
                power = int(np.floor(np.log10(max(abs(x) for x in v['ci95'])))); scale = 10. ** (-power)
                heat.append(f"{group} & {float(h):g} & {t:g} & $10^{{{power}}}$ & {v['mean']*scale:+.3f} & {ci3(v, scale)}")
    tables['HeatTraceRows'] = heat
    arr = np.load(E / 'avalanche.npz'); D = arr['discovery']; H = arr['holdout']; rounds = arr['rounds']
    early = []; corr = {}
    for r in range(1, 8):
        j = int(np.flatnonzero(rounds == r)[0]); d = D[:, :32, j].mean(0); h = H[:, :32, j].mean(0)
        corr[r] = float(np.corrcoef(d, h)[0, 1])
        early.append(f"{r} & {corr[r]:.3f} & {h[0]:.2f} & {h.mean():.2f} & {int(d.argmin())} & {int(h.argmin())}")
    tables['EarlyCoordinateRows'] = early
    x = H[:, :, 23].astype(float); center = x - x.mean(1, keepdims=True)
    macros.update({'PositionSD': f'{x.mean(0).std(ddof=1):.3f}',
                   'NoiseFloor': f'{np.sqrt(center.var(0, ddof=1).sum() / ((x.shape[1] - 1) * len(x))):.3f}',
                   'RoundCorrelation': f'{np.corrcoef(D[:, :, 23].mean(0), x.mean(0))[0, 1]:.3f}'})
    s = read(E, 'followup/search.json'); q = read(E, 'followup/cubes.json'); z = read(E, 'followup/detour.json')
    ms = read(E, 'matched/search.json'); mq = read(E, 'matched/cubes.json'); ud = read(E, 'matched/uniform_diagnostics.json')
    rows = []
    for u, m in zip(s['records'], ms['records']):
        assert u['round'] == m['round'] and abs(u['target_mean'] - m['candidate_mean']) < 1e-9
        rows.append(f"{u['round']} & {u['target_mean']:.2f} & {u['control_mean']:.2f} & {u['difference']['mean']:+.2f} {ci2(u['difference'])} & {m['matched_mean']:.2f} & {m['difference']['mean']:+.2f} {ci2(m['difference'])} & {m['matched_sets_below_candidate']}/40")
    tables['EarlySearchRows'] = rows
    rows = []
    for u, m in zip(q['records'], mq['records']):
        assert u['round'] == m['round'] and abs(u['target_mean'] - m['candidate_zero_rate']) < 1e-9
        rows.append(f"{u['round']} & {100*u['target_mean']:.2f} & {100*u['control_mean']:.2f} & {100*u['difference']['mean']:+.2f} {ci2(u['difference'], 100, 1)} & {100*m['matched_zero_rate']:.2f} & {100*m['difference']['mean']:+.2f} {ci2(m['difference'], 100, 1)} & {m['matched_sets_at_or_above_candidate']}/40 & {pformat(u['holm_p'])}")
    tables['EarlyCubeRows'] = rows
    tables['FreshDetourRows'] = [f"{k.replace(' minus ', ' -- ')} & {v['mean']:+.5f} & [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]" for k, v in z['summary'].items()]
    v = z['summary']['state minus PCG64']
    macros.update({'FreshDetourMean': f"{v['mean']:+.5f}", 'FreshDetourCI': f"[{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}]"})
    v = q['records'][0]
    macros.update({'EarlyCubeTarget': f"{100*v['target_mean']:.2f}", 'EarlyCubeControl': f"{100*v['control_mean']:.2f}",
                   'EarlyCubeDifference': f"{100*v['difference']['mean']:.2f}", 'EarlyCubeCI': ci3(v['difference'], 100), 'EarlyCubeP': pformat(v['holm_p'])})
    m2 = ms['records'][0]; c2 = mq['records'][0]
    macros.update({'MatchedSearchDiffTwo': f"{m2['difference']['mean']:+.2f}", 'MatchedSearchCITwo': ci2(m2['difference']),
                   'MatchedSearchBelowTwo': str(m2['matched_sets_below_candidate']),
                   'UniformSearchOffsetCorr': f"{ud['search_corr_control_mean_with_mean_offset']['2']:.2f}",
                   'UniformCubeMinTwo': f"{100*ud['cube_round2_control_zero_rate_range'][0]:.2f}",
                   'UniformCubeMaxTwo': f"{100*ud['cube_round2_control_zero_rate_range'][1]:.2f}",
                   'UniformCubeAboveCorr': f"{ud['cube_round2_corr_zero_rate_with_bits_above_16']:.2f}",
                   'MatchedCubeMeanTwo': f"{100*c2['matched_zero_rate']:.2f}",
                   'MatchedCubeRangeTwo': f"{100*c2['matched_range'][0]:.1f}--{100*c2['matched_range'][1]:.1f}",
                   'MatchedCubeAtOrAboveTwo': str(c2['matched_sets_at_or_above_candidate']),
                   'MatchedCubeDiffTwo': f"{100*c2['difference']['mean']:+.2f}", 'MatchedCubeCITwo': ci2(c2['difference'], 100, 1),
                   'WholeWordATwo': f"{100*c2['whole_word_a_zero']['candidate']:.2f}",
                   'WholeWordAControlTwo': f"{100*ud['cube_round2_whole_word_a_zero']['uniform_controls']:.2f}"})
    # Bootstrap bounds on the descriptive rank counts at round 2, over bases, on the saved matched arrays.
    mc = np.load(E / 'matched/cubes.npz'); msz = np.load(E / 'matched/search.npz')
    jc = list(mc['rounds']).index(2); js = list(msz['rounds']).index(2)
    lo, hi = rank_bootstrap(mc['zero_indicators'][0, :, jc], mc['zero_indicators'][1:, :, jc], 'ge')
    macros['MatchedCubeRankCITwo'] = f"{lo}--{hi}"
    lo, hi = rank_bootstrap(msz['scores'][0, :, js], msz['scores'][1:, :, js], 'lt')
    macros['MatchedSearchRankCITwo'] = f"{lo}--{hi}"
    c7 = mq['records'][[r['round'] for r in mq['records']].index(7)]
    macros.update({'MatchedCubeSevenDiff': f"{100*c7['difference']['mean']:+.2f}", 'MatchedCubeSevenCI': ci2(c7['difference'], 100, 1)})
    # Early-coordinate correlation range and later values, for the abstract and Section 4.1.
    macros.update({'EarlyCorrRange': f"{min(corr[r] for r in range(1, 6)):.3f}--{max(corr[r] for r in range(1, 6)):.3f}",
                   'EarlyCorrSix': f"{corr[6]:.3f}", 'EarlyCorrSeven': f"{corr[7]:.3f}"})
    # Cluster-size prose for Section 3.3.
    def sorted_sizes(g, h, label): return np.array([sorted(r['kernels'][h][label]['cluster_sizes']) for r in g['records']], float)
    for label, key in [('sha', 'SHA'), ('random', 'Random')]:
        mean_sizes = sorted_sizes(graphs['Primary'], '1.0', label).mean(0)
        macros['PrimaryOneSizes' + key] = '(' + ','.join(f'{v:.2f}' for v in mean_sizes) + ')'
        macros['ConfirmationLargest' + key] = f"{sorted_sizes(graphs['Confirmation'], '1.0', label)[:, -1].mean():.2f}"
        mean8 = sorted_sizes(graphs['Primary'], '8.0', label).mean(0)
        macros['PrimaryEightRange' + key] = f"{mean8.min():.2f} to {mean8.max():.2f}"
    return macros, tables


def render(macros, tables):
    """Exact text of a macro file: one \\newcommand per macro, then one row-block per table."""
    lines = ['\\newcommand{\\' + k + '}{' + v + '}' for k, v in macros.items()]
    for k, rows in tables.items():
        lines.append('\\newcommand{\\' + k + '}{%\n' + '\n'.join(r + ' \\\\' for r in rows) + '%\n}')
    return '\n'.join(lines) + '\n'


def table_text(rows): return '\n'.join(r + ' \\\\' for r in rows) + '\n'
