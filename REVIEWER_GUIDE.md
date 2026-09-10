# Reviewer guide

This guide takes you from a clone to an independent check of every number in the paper. The minimum path (steps 1 to 3) takes about five minutes and needs only Python. Rebuilding the PDF (step 4) needs `pdflatex`. Re-running experiments from scratch (step 5) takes a few minutes more.

Nothing in the paper is typed by hand. Every number enters the manuscript through macros that `publication/build_paper.py` and `publication/followup_tables.py` generate from the receipts under `publication/evidence/`, and `publication/verify_evidence.py` recomputes each receipt's summary from its raw arrays. So a reviewer can check the chain at three points: the raw arrays, the summaries, and the manuscript macros.

## 1. Clone and install

```
git clone https://github.com/nurdymuny/sha-256-heat-kernel-cryptanalysis.git
cd sha-256-heat-kernel-cryptanalysis
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS or Linux
python -m pip install -r publication/release/requirements.txt
```

Python 3.12 is what produced the receipts (`publication/evidence/environment.json`). The pinned versions are `numpy==2.4.0`, `scipy==1.16.3`, `scikit-learn==1.9.0`, `matplotlib==3.10.8`, `pytest==9.0.2`. Other recent versions will almost certainly work for steps 2 and 3, because those steps recompute from saved arrays; they matter only if you re-run the graph experiments (step 5), where eigenvalue solvers may differ in the last digits.

The repository stores every file byte-exactly (`.gitattributes` sets `* -text`). Do not run any tool that rewrites line endings; the receipts record SHA-256 hashes of the source files as stored.

## 2. Run the tests (about five seconds)

```
python -m pytest tests -q
```

Expected: `42 passed`. The tests establish that the primitive is SHA-256 (FIPS 180-4 vectors and `hashlib` agreement at padding boundaries up to 128 bytes), that the vectorized raw-block compression used for every measurement matches the scalar reference round by round, that input coordinates are placed where the paper says (MSB-first within each byte, word index = bit // 32), that a variable in W15 cannot influence the state before round 16, that the graph normalization keeps tiny positive degrees and is scale invariant, that full heat traces obey the truncation bound, that the legacy anomaly detector no longer rejects an exact match to its own null, that no continuum curvature is inferred, and that the follow-up and matched-control constructions are what the paper describes.

## 3. Run the verifier (about ten seconds)

```
python publication/verify_evidence.py
```

Expected: a JSON block with `"passed": true` and sixteen check names. The verifier reads only `publication/evidence/` and recomputes, from the raw `.npz` arrays and saved spectra:

- every eigenvalue set (512 modes, zero mode, ordering), every heat trace, and every graph interval;
- the frozen discovery selection and the held-out difference from the avalanche arrays;
- the search minima and paired intervals from the saved per-base scores;
- the cube indicators, intervals and Holm adjustment from the saved corner sums, plus an independent scalar recomputation of complete six-bit cubes on two bases;
- the register-transport identities (`d_r = a_{r-3}`, `h_r = e_{r-3}` and their cousins) as exact array equalities on the entry-control receipts;
- classifier accuracy and AUC from the saved held-out predictions;
- the 100-pair detour replication, seed range and all five intervals;
- the exact round-one MSB observation on every base of both original partitions;
- the matched-control family: identical bases and candidate arrays to the follow-up, pool membership, per-set means and rates, intervals and counts, and the uniform-control diagnostics;
- the SHA-256 hash of every source file that a receipt records, against the file as stored.

It also rewrites `publication/verification.json`; a clean checkout shows no diff afterwards.

If any check fails, the assertion message names the receipt and the quantity. A failure would mean either the receipts were altered or the code that reads them was changed. Neither should happen on a clean clone.

## 4. Rebuild the paper (optional, about thirty seconds)

```
python publication/build_paper.py
```

This regenerates the seven figures, the four table files, `numbers.tex` and `followup_numbers.tex`, and compiles `publication/sha256_measurement.tex` twice into `output/pdf/`. The PDF bytes differ from the shipped one (timestamps, font subsets), but the text is identical; `publication/release/package_validation.json` records that the shipped PDF's extracted text matched a clean rebuild. To see that no generated file changed, run `git status` afterwards: the only modified file should be nothing, or `publication/verification.json` if you ran step 3 first and your platform prints floats differently.

`python publication/make_release.py` repeats the tests and verifier, regenerates `RESULTS.md`, and rebuilds the archives in `publication/release/`. Archive bytes will differ from the shipped ones because zip entries carry timestamps.

## 5. Re-run experiments from scratch (optional)

Every runner refuses to overwrite an existing receipt directory, so point it at a new one. Seeds are fixed in each runner's `PROTOCOL`, which the runner saves before computing.

| Runner | Command | Time | Output |
|---|---|---|---|
| Primary study | `cd publication/repro_source && python analysis/publication_study.py --output ../fresh_primary` | about 70 s | graphs, avalanche, search, cubes, classifiers |
| Graph confirmation and entry control | copy the repository, delete `publication/evidence/confirmation`, run `python analysis/publication_confirmation.py` | about 60 s | 20 fresh graph pairs; whole-word entry sums |
| Detour sensitivity | same pattern with `publication/evidence/sensitivity` and `publication_sensitivity.py` | about 30 s | 20 comparisons across four representations |
| Early-round follow-up and 100 detour pairs | `python analysis/publication_followup.py --output publication/fresh_followup` | about 40 s | search and cubes vs uniform controls; 100 detour pairs |
| Significance-matched controls | `python analysis/publication_matched.py --output publication/fresh_matched` | about 40 s | forty matched sets on the saved follow-up bases |

The primary study must be run from `publication/repro_source/`, which is the byte-exact snapshot of the source that produced the shipped receipts. The confirmation and sensitivity runners write to fixed paths and refuse to overwrite, hence the copy-and-delete pattern.

Everything except the graph spectra is integer arithmetic on seeded inputs and reproduces bit for bit. To confirm:

```python
import numpy as np
a = np.load('publication/evidence/avalanche.npz'); b = np.load('publication/fresh_primary/avalanche.npz')
print(np.array_equal(a['holdout'], b['holdout']), np.array_equal(a['discovery'], b['discovery']))
s = np.load('publication/evidence/followup/search.npz'); t = np.load('publication/fresh_followup/search.npz')
print(np.array_equal(s['pattern_scores'], t['pattern_scores']))
q = np.load('publication/evidence/followup/cubes.npz'); r = np.load('publication/fresh_followup/cubes.npz')
print(np.array_equal(q['word_sums'], r['word_sums']))
```

Graph statistics (silhouettes, spectral gaps, heat traces, detour ratios) depend on an eigensolver and on KMeans; expect agreement to many digits on the same platform and small last-digit differences elsewhere. The paper's intervals for those quantities span thousands of the differences you would see.

## 6. Where each number comes from

| In the paper | Macro or table file | Receipt | Generated by |
|---|---|---|---|
| Graph statistics table (silhouette, gap, detour; two groups) | `numbers.tex`, `\GraphRows` | `evidence/graphs.json`, `evidence/confirmation/graphs.json` | `build_paper.py` |
| Heat-trace table (16 intervals) and Figure 1 | `followup_numbers.tex`, `\HeatTraceRows`; `figures/heat_profiles.pdf` | full spectra in the two `graphs.json` files | `followup_tables.py`, `build_paper.py` |
| Cluster sizes at h = 1 and h = 8 (text) | stated from `cluster_sizes` fields | the two `graphs.json` files | read directly |
| Detour sensitivity table | `numbers.tex`, `\SensitivityRows` | `evidence/sensitivity/detour.json` | `build_paper.py` |
| Fresh 100-pair detour table | `followup_numbers.tex`, `\FreshDetourRows` | `evidence/followup/detour.json` | `followup_tables.py` |
| Round-24 holdout difference, noise floor, correlation | `\HoldoutDifference`, `\HoldoutCI`, `\PositionSD`, `\NoiseFloor`, `\RoundCorrelation` | `evidence/avalanche.json`, `evidence/avalanche.npz` | both scripts |
| Early within-W0 table and Figure 5 left | `\EarlyCoordinateRows`; `figures/early_structure.pdf` | `evidence/avalanche.npz` | `followup_tables.py` |
| Round-24 two-bit search | `\SearchDifference`, `\SearchCI` | `evidence/search.json`, `evidence/search.npz` | `build_paper.py` |
| Fresh two-bit searches vs uniform and matched controls; Figure 5 right | `\EarlySearchRows`, `\MatchedSearch*`, `\UniformSearchOffsetCorr` | `evidence/followup/search.*`, `evidence/matched/search.*`, `evidence/matched/uniform_diagnostics.json` | `followup_tables.py` |
| Original six-bit cube table | `numbers.tex`, `\CubeRows` | `evidence/cubes.json`, `evidence/cubes.npz` | `build_paper.py` |
| Fresh six-bit cube table vs both control families | `\EarlyCubeRows`, `\EarlyCube*`, `\MatchedCube*`, `\UniformCube*`, `\WholeWordA*` | `evidence/followup/cubes.*`, `evidence/matched/cubes.*` | `followup_tables.py` |
| Entry control (Figure 6 left and centre) | figure only | `evidence/confirmation/entry.json`, `entry.npz` | `build_paper.py` |
| Classifier table and Figure 7 | `\MLRows`, `\RFAccuracy`, `\SVMAccuracy` | `evidence/ml.json`, `evidence/ml_*.npz` | `build_paper.py` |

Each receipt directory also holds `protocol.json` (saved before execution), `environment.json` (versions and source hashes), `completion.json` (timestamps), and the frozen selection or control sets.

## 7. Three claims you can check by hand in a minute

The exact MSB result: flipping the most significant bit of W0 changes exactly two state bits after one round, on every base.

```python
import numpy as np
a = np.load('publication/evidence/avalanche.npz')
print(set(a['discovery'][:, 0, 0]), set(a['holdout'][:, 0, 0]))     # {2} {2}
```

The round-24 selection sits at the noise floor: the spread of the 440 per-position means equals what sampling alone produces, and discovery does not predict holdout.

```python
r24 = list(a['rounds']).index(24); h = a['holdout'][:, :, r24].astype(float); d = a['discovery'][:, :, r24]
print(h.mean(0).std(ddof=1), np.sqrt(((h - h.mean(1, keepdims=True)).var(0, ddof=1)).sum() / (439 * 128)))  # 0.697 vs 0.706
print(np.corrcoef(d.mean(0), h.mean(0))[0, 1])                                                              # 0.018
```

The candidate set is ordinary within its significance class at round 2: 19 of 40 matched control sets have a cube zero rate at or above the candidate's.

```python
import json
q = json.load(open('publication/evidence/matched/cubes.json'))['records'][0]
print(q['round'], q['candidate_zero_rate'], sum(r >= q['candidate_zero_rate'] for r in q['set_rates']), min(q['set_rates']), max(q['set_rates']))
```

## 8. Troubleshooting

- `ModuleNotFoundError: No module named 'analysis'` or `'src'`: run commands from the repository root, or from `publication/repro_source/` for the primary runner.
- `pdflatex` not found: skip step 4; the shipped PDF is `publication/release/sha256_measurement.pdf`.
- `RuntimeError: Use a fresh evidence directory`: a runner refused to overwrite; choose a new `--output`.
- The verifier fails on a source-hash check after you edited a file under `src/sha256/`, `analysis/publication_*.py`, or `publication/repro_source/`: that is by design. Those files are what the receipts hash. Restore them, or treat your change as a new experiment with its own receipt directory.
- Line-ending warnings from git on Windows: harmless, but do not enable conversions for this repository; `.gitattributes` disables them.

## 9. What is and is not being claimed

The paper reports bounded negative results for the tested procedures, one reproducible early-round position effect with an exact algebraic explanation at round 1, and the finding that a previously proposed candidate set has no advantage once controls are matched on bit significance. It does not claim an attack, a distinguisher, a security proof, intrinsic curvature, or a topological model. `review/` documents the December 2025 claims that preceded this work and why they were withdrawn.
