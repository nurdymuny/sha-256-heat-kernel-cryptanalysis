# Heat-Kernel Cryptanalysis of SHA-256

**Heat-Kernel Cryptanalysis of SHA-256: A Geometric Study of State Evolution**
Bee Rosa Davis, Davis Geometric. ORCID [0009-0009-8034-4308](https://orcid.org/0009-0009-8034-4308).
Working paper, 9 September 2026. Not yet submitted; no DOI has been assigned.

This repository is the complete record of the study: the manuscript and the scripts that build every table and figure from saved receipts, the receipts themselves with an independent verifier, the exact source snapshots that produced them, the December 2025 original code and claims that the study superseded, and the audit trail that connects the two.

- Paper: [`publication/release/sha256_measurement.pdf`](publication/release/sha256_measurement.pdf) (17 pages, 7 figures, 16 references)
- arXiv source: [`publication/release/arxiv_source.zip`](publication/release/arxiv_source.zip)
- Reproducibility archive: [`publication/release/sha256_reproducibility.zip`](publication/release/sha256_reproducibility.zip), with [`manifest.json`](publication/release/manifest.json) and [`package_validation.json`](publication/release/package_validation.json)
- Numbers digest: [`RESULTS.md`](RESULTS.md)
- Reviewer guide (run and check everything in about fifteen minutes): [`REVIEWER_GUIDE.md`](REVIEWER_GUIDE.md)
- Instructions for coding agents working in this repository: [`AGENTS.md`](AGENTS.md)
- License: MIT ([`LICENSE`](LICENSE))

## What the paper does and finds

The paper names *heat-kernel cryptanalysis* as an investigative method: equip sampled SHA-256 working states with an explicit neighborhood geometry (a 30-nearest-neighbor graph with Gaussian weights and a symmetric normalized Laplacian), probe it by spectral diffusion, and then test whether anything the geometry suggests survives fresh inputs, a matched random reference, or a specified search. The same computation is followed dynamically through single-bit interventions, equal-budget two-bit searches, and Boolean cube tests on working-state projections, with every input coordinate, register, and round labelled exactly.

What it finds, with every number regenerated from `publication/evidence/`:

- **Terminal-state graphs.** Large spectral-clustering silhouettes (about 0.90 at kernel bandwidth h = 1) occur in SHA states and in independent random bits alike, and at that bandwidth the fitted partition is one cluster of about 502 points plus four clusters of 2 to 4 points. All sixteen paired SHA-minus-random heat-trace intervals contain zero. A small negative detour-ratio contrast seen in two 20-pair groups did not reproduce in a 20-pair sensitivity sample or in 100 further pairs.
- **Avalanche and selection.** Across 112,640 one-bit interventions, every message word follows the same entry-aligned trajectory and saturates near 128 changed bits about six rounds after it enters. At round 24 the spread of per-position means equals the sampling noise floor, so a discovery-selected set of 22 positions shows no held-out advantage. Within the first message word the early response depends reproducibly on bit position (discovery-holdout correlations 0.93 to 0.98 at rounds 1 to 5), and the most-significant-bit intervention changes exactly two state bits at round 1, as modular addition predicts.
- **A previously proposed candidate set has no advantage.** Against control sets drawn uniformly from W0, the six-position candidate carried over from the December work shows smaller early search distances and a higher round-2 cube zero rate. Against forty control sets drawn from the candidate's own significance class, evaluated on the same bases, the search difference reverses and the cube rate lies inside the control range. Early response depends on bit position, not on the candidate.
- **Register transport.** Cube sums in trailing registers are exact delayed copies of leading-register sums, as the round structure requires. Surviving zero sums in trailing registers reflect entry timing, not algebraic weakness.
- **Digest classifiers** trained on actual SHA-256 digests average 50.3 % (random forest) and 49.8 % (PCA + SVM) held-out accuracy over five independent fits.

The paper does not establish a collision or preimage attack, a distinguisher, cryptographic security, intrinsic manifold curvature, or a topological model of the state space. It places its measurements against published collision attacks reaching 31 and 37 steps and biclique preimage attacks reaching 45 steps, and does not compete with them.

## Headline numbers

| Measurement | Value |
|---|---|
| Silhouette, h = 1, SHA vs random (primary group, 20 pairs) | 0.896 vs 0.907; mean sorted cluster sizes (2.0, 2.05, 2.3, 3.65, 502.0) |
| Silhouette, h = 8 | 0.161 vs 0.164 |
| Heat-trace paired differences, 2 groups × 2 bandwidths × 4 diffusion times | all 16 intervals contain zero |
| Detour ratio, SHA minus random: primary / confirmation / sensitivity / 100 fresh pairs | −0.0031 [−0.0053, −0.0008] / −0.0034 [−0.0065, −0.0006] / −0.00045 [−0.0032, +0.0025] / −0.00032 [−0.00168, +0.00098] |
| Round-24 holdout, 22 selected positions minus others | +0.0088 bits [−0.2869, +0.3140]; per-position SD 0.697 vs noise floor 0.706; discovery-holdout correlation 0.018 |
| W0 discovery-holdout correlation, rounds 1 to 5 | 0.973, 0.979, 0.962, 0.964, 0.930 |
| MSB intervention, round 1 | exactly 2.00 changed bits on every base |
| Two-bit search, round 24, candidate minus uniform controls | +0.126 bits [−0.435, +0.666] |
| Two-bit search, round 2, candidate minus uniform / minus matched controls | −1.22 [−1.65, −0.79] / +1.55 [+1.14, +1.96]; 36 of 40 matched sets below the candidate |
| Six-bit cube, rounds 8 to 24, candidate minus uniform controls | all eight intervals contain zero; no Holm-adjusted rejection |
| Six-bit cube, round 2, zero rate of bit 16 of register a | candidate 90.43 %; uniform controls 72.29 % (sets range 50.98 to 96.09 %); matched controls 86.08 % (49.6 to 94.9 %), 19 of 40 at or above the candidate; whole-word a sum zero for 0.20 % of bases |
| Digest classifiers, five fits each | RF 50.315 % (49.825 to 51.050, AUC 0.502); PCA + SVM 49.800 % (48.225 to 51.425, AUC 0.500) |

## Repository layout

```
publication/
  sha256_measurement.tex      manuscript; all numbers enter through generated macros
  numbers.tex, followup_numbers.tex, *_table.tex     generated from receipts, never edited by hand
  build_paper.py              figures, tables, macros, then pdflatex
  followup_tables.py          follow-up and matched-control tables, macros and Figure 5
  verify_evidence.py          independent cross-footing of every receipt (16 checks)
  make_release.py             tests, verifier, archives, manifest
  figures/                    seven figures, PDF and PNG
  evidence/                   frozen receipts of every experiment (see below)
  repro_source/               byte-exact snapshot of the source that produced the primary run
  release/                    paper PDF, arXiv source, reproducibility archive, manifest, validation
src/                          instrumented scalar SHA-256, vectorized raw-block compression, graph and heat-kernel code
analysis/publication_*.py     the five runners behind the paper (see Reproduce)
analysis/*.py (others)        December 2025 exploratory scripts, retained after repair; not used by the paper
tests/                        42 tests: FIPS vectors and hashlib boundaries, vectorized-versus-scalar rounds,
                              Laplacian invariants, heat-trace bounds, coordinate conventions, follow-up checks
review/                       audit trail (see Provenance)
main.py, pyproject.toml       legacy command-line entry point and package metadata
```

`publication/evidence/` holds, for every experiment, the protocol saved before execution, the environment and source hashes, frozen selection or control sets, raw arrays (`.npz`), and summaries (`.json`):

| Directory | Runner | Contents |
|---|---|---|
| `evidence/` (primary) | `analysis/publication_study.py` | 20 graph pairs with full 512-mode spectra; 128 + 128 avalanche bases × 440 positions × 27 rounds; round-24 two-bit search on 256 bases; six-bit cubes on 512 bases at 8 rounds with the entry control; five classifier replications with held-out predictions |
| `evidence/confirmation/` | `analysis/publication_confirmation.py` | 20 fresh graph pairs; uniform raw-block entry control (whole-word cube sums, five injection words) |
| `evidence/sensitivity/` | `analysis/publication_sensitivity.py` | 20 fresh detour comparisons across working state, digest, PCG64 and MT19937 |
| `evidence/followup/` | `analysis/publication_followup.py` | early-round search (256 bases, rounds 2 to 7 and 24) and cubes (512 bases, 14 rounds) against twenty uniform W0 control sets; 100 fresh detour pairs |
| `evidence/matched/` | `analysis/publication_matched.py` | forty significance-matched control sets (W0 offsets 0 to 14) evaluated on the identical follow-up bases; per-set statistics; diagnostics of the uniform controls |

## Reproduce

Python 3.12 with `numpy==2.4.0`, `scipy==1.16.3`, `scikit-learn==1.9.0`, `matplotlib==3.10.8`, `pytest==9.0.2` (see `publication/release/requirements.txt`), and a LaTeX distribution with `pdflatex`.

```
python -m pip install -r publication/release/requirements.txt
python -m pytest tests -q                 # 42 passed
python publication/verify_evidence.py     # 16 checks; recomputes every interval from raw arrays
python publication/build_paper.py         # figures, tables, macros, PDF -> output/pdf/
python publication/make_release.py        # archives, manifest, RESULTS.md
```

The verifier reads only the frozen receipts and recomputes their summaries; it does not rerun experiments. To rerun an experiment, point its runner at a new directory. Runners refuse to overwrite existing receipts.

```
cd publication/repro_source && python analysis/publication_study.py --output ../fresh_primary   # about 70 s
python analysis/publication_followup.py --output publication/fresh_followup                        # about 40 s
python analysis/publication_matched.py  --output publication/fresh_matched                         # about 40 s
```

`publication/repro_source/` has source hashes matching `publication/evidence/environment.json` byte for byte. The root `src/` differs from it only by later maintenance of unused legacy interfaces (a `state_at` helper, a removed continuum-curvature routine, annotations and comments); the measured functions are identical and the tests exercise both.

Package validation (`publication/release/package_validation.json`) was produced by extracting both archives into clean directories, compiling the arXiv source twice (zero LaTeX warnings, extracted text identical to the delivered PDF), and running the tests and the verifier inside the extracted reproducibility archive.

## Provenance and correction history

This study replaced an earlier manuscript and codebase, and this repository keeps both.

**December 2025.** The original analysis (`review/pre_correction_snapshot/`, 45 files with SHA-256 hashes in its `manifest.json`) and its results summary (`review/pre_correction_snapshot/RESULTS.md`) claimed a SHA-specific geometric fingerprint (silhouette 0.995 versus −0.574 for random), a geodesic distortion of 2.637 versus 1.0, 22 "slow bits" aligned with Σ₀ rotations, slow bit pairs at −35σ, a geometry-guided cube attack breaking registers c, d, g, h at round 17 and d, h at round 18, and a security horizon at round 20. Those claims are retained here as a record. **They are withdrawn.**

**September 2026, audit.** `review/SHA256_SUBMISSION_REVIEW.md` traced each claim to its code and found reproducible implementation errors: the graph normalization discarded positive degrees; the 0.995 silhouette was compared against random labels rather than an independently generated random dataset processed by the same pipeline; the anomaly detector rejected an exact match to its own null; two pipelines mislabelled input-message bits as working-register bits; the anisotropy figure used placeholder data; the −35σ result divided candidate distances by √2 and compared them with an unscaled baseline; cube scripts serialized bit indices big-endian, so the "trailing-register" zeros were sums evaluated before the variables had entered the state; truncated heat traces were treated as complete; the classifier experiment fitted PCA on all data and labelled working states as digests; receipts were incomplete. `review/submission_audit.py` reproduces the counterexamples.

**Corrections and measurement edition.** `review/CORRECTION_CLOSEOUT.md` records the repairs: regression gates added, normalization and full-spectrum heat support repaired, canonical coordinates for block, round, word and bit, a controlled runner with saved protocols, seeds, frozen selections and structured receipts, and a fresh execution of every experiment. `review/measurement_edition_snapshot/` preserves that edition of the manuscript. `review/RESTORED_GEOMETRIC_FRAMING.md` records the author's subsequent restoration of the geometric framing and title with unchanged numbers.

**Review and follow-ups.** A review of the measurement edition identified, from the frozen arrays alone, that the round-24 selection test sits at the sampling noise floor, that the same arrays contain reproducible early-round position dependence within W0, that the h = 1 silhouette reflects a fragmented partition, and that the heat trace itself had no tested statistic; it also replicated the detour contrast on 100 fresh pairs and found nothing. `review/FOLLOWUP_REVISION_CLOSEOUT.md` records the resulting follow-up runner, the 100-pair author replication, and the early-round experiments. A second review showed that the candidate set's early advantage against uniform W0 controls was a bit-significance effect (`review/reviewer_matched_controls_search.json`, `review/reviewer_matched_controls_cubes.json`); the significance-matched control family in `analysis/publication_matched.py` and the present wording followed.

Every step is dated in the receipts' `environment.json` and `completion.json` files.

## What is not in this repository

Separate manuscripts on the dual-torus and double-cover formulations, the December 2025 carry-field tomography outputs, and the exploratory thermocline, resonant-tunnel and algebraic-search branches are not part of this paper and are not included. Nothing in this repository establishes or claims an attack on SHA-256.

## License

The contents of this repository are released under the MIT License (see [`LICENSE`](LICENSE)). The license under which the paper itself is distributed by a preprint server or data repository is chosen at deposit.

## Citation

```
@unpublished{davis2026heatkernel,
  author = {Davis, Bee Rosa},
  title  = {Heat-Kernel Cryptanalysis of {SHA-256}: A Geometric Study of State Evolution},
  year   = {2026},
  month  = sep,
  note   = {Working paper. Code, receipts and manuscript: https://github.com/nurdymuny/sha-256-heat-kernel-cryptanalysis}
}
```
