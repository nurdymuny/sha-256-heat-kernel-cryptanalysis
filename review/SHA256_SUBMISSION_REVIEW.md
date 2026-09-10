# SHA-256 study: submission readiness and correction review

**Decision: hold submission of the present empirical claims.** Several headline findings depend on reproducible implementation errors or unmatched comparisons. Repairing prose alone would leave those findings unsupported. The SHA-256 digest implementation passed the checks run here; the substantial problems are in the measurement and inference layers.

This is an internal review of the research package, not proposed public framing. A future paper should present the final validated protocol and findings directly. It need not narrate this audit or describe itself as a correction to an unpublished paper.

## Scope and evidence

Reviewed the core SHA implementation, input generation, embeddings, graph construction, heat solver, anomaly detector, avalanche pipeline, and the analysis scripts and receipts behind the principal claims in `RESULTS.md`. The folder contains **122 Python files**; this is a dependency-focused audit, not certification of every script. The later `resonant_tunnel`, `thermocline`, and algebraic-search branches have not received an exhaustive execution audit. No cloud searches or large heavy-cube sweeps were run.

There is also a manuscript-location issue: `theory/heat kernel.tex` is titled *Spectral Geometry of Transformer Cognition: Heat Kernel Analysis Reveals Functional Organization in Language Models*. It is not the SHA-256 manuscript. The SHA heat-kernel title cited by `theory/dual_torus_sha256.tex` was not found among this folder's four TeX sources. Accordingly, this review audits the candidate paper's research foundation and `RESULTS.md`, rather than pretending to have reviewed a missing manuscript.

Artifacts from this review:

- [Executable diagnostics](C:/Users/nurdm/OneDrive/Documents/sha-256-heat_kernel/review/submission_audit.py)
- [Machine-readable results and source hashes](C:/Users/nurdm/OneDrive/Documents/sha-256-heat_kernel/review/submission_audit_results.json)

Validation completed: the existing `tests/test_core.py` suite passes **20 tests**; an additional **33 digest comparisons against `hashlib`** pass for message lengths 0, 1, 3, 55, 56, 63, 64, 65, 119, 120, and 128 bytes. The new audit also reproduces the counterexamples below. Its successful exit means the diagnostic assertions reproduced the problems; it does **not** mean the scientific pipeline passed validation. Original research code and historical receipts were not edited.

## 1. Blocking: positive graph degrees are discarded

Source: [laplacian.py:268](C:/Users/nurdm/OneDrive/Documents/sha-256-heat_kernel/src/analysis/laplacian.py:268).

The Gaussian weights are `exp(-distance**2 / (4*t))`, with default `t=1`. For 256-bit Euclidean states these weights are small but representable. Normalization treats every degree at or below `1e-10` as zero, even when the vertex has valid positive-weight edges.

For positive degrees, the stated symmetric normalized Laplacian must satisfy `L sqrt(d) = 0`. Multiplying all weights by the same positive constant must leave it unchanged. The implementation violates both requirements.

| New diagnostic: 256 vertices, k=30 | Random bits | SHA working states |
|---|---:|---:|
| Connected components in original weighted graph | 1 | 1 |
| Positive degrees discarded | 256 | 254 |
| Smallest eigenvalue, current code | 1.000000 | 0.241975 |
| Smallest eigenvalue, corrected normalization | approximately 0 | approximately 0 |
| Residual on normalized `sqrt(d)`, current code | 1.000000 | 0.973617 |

Multiplying weights by `10^12` changes entries of the current normalized matrix by up to 0.567 and 0.658 respectively; corrected normalization changes only at floating-point roundoff. The saved baseline receipt already shows the same symptom: round 32 has fifty retained eigenvalues all equal to one, and round 64 has a smallest eigenvalue around 0.222 rather than zero.

**Correction:** use numerically stable positive-degree normalization with an explicit isolated-vertex convention. Test zero modes, common-weight rescaling, permutation equivariance, connectivity, and dense-reference agreement. Recompute every result that uses this operator. Merely rescaling all weights cannot repair an overly sharp bandwidth; bandwidth needs its own controls.

## 2. Blocking: the 0.995 silhouette is not a SHA-specific fingerprint

Source: [sha256_geometric_analysis.py:689](C:/Users/nurdm/OneDrive/Documents/sha-256-heat_kernel/analysis/sha256_geometric_analysis.py:689).

The published-looking comparison fits KMeans on SHA spectral coordinates, then compares that score with **randomly assigned labels on those same coordinates**. This is not a comparison with an independently generated random-bit dataset processed through the same pipeline. The historical receipt's five clusters contain **992, 2, 2, 2, 2** points.

I ran a new control at the same 1,000-point sample size, with k=50, 20 retained modes, five fitted clusters, SHA input seed 42, and a separately seeded random-bit dataset:

| Identically processed dataset | Current normalization | Corrected normalization |
|---|---:|---:|
| SHA-256 round-64 working states | 0.995195 | 0.960185 |
| Independent random bit strings | 0.995571 | 0.963549 |

All four partitions have sizes 992/2/2/2/2. This is a current-code control, not an exact replay of the old environment or a many-seed significance test. It establishes that the headline-sized score also arises from random inputs. **Correcting the degree bug does not, by itself, remove the high silhouette.** Do not attribute the entire effect to that one bug.

The 2.637 geodesic-distortion result has a related comparison problem: the producing routine computes graph shortest-path distance divided by direct Euclidean distance for SHA states, but does not generate the claimed random-data value of 1.0. A sparse graph produces detours even on random data. These routines use a Euclidean embedding, although `RESULTS.md` describes the headline as hyperbolic.

**Correction:** withdraw the present fingerprint interpretation; fit the same pipeline on independent null datasets, match sample size and graph settings, report cluster sizes and localization, examine bandwidth sensitivity, and calibrate the chosen statistic across independent realizations. Measure random-data distortion rather than assigning it a theoretical value of one.

## 3. Blocking: the anomaly detector can reject an exact match to its null

Source: [anomaly.py:204](C:/Users/nurdm/OneDrive/Documents/sha-256-heat_kernel/src/analysis/anomaly.py:204).

Ordered eigenvalues are standardized using per-mode null means and standard deviations, then tested against an iid standard normal distribution with a KS test. Ordered modes are dependent; there is no justification that these standardized quantities are iid normal. Gap tests reuse the same eigenvalues, and Fisher's combination is applied without accounting for that dependence. `1-p` is then called confidence.

In a deterministic counterexample, the observed spectrum equals every spectrum represented by a degenerate empirical null. Every standardized difference is zero, and no mode is marked anomalous. Nevertheless, the code returns eigenvalue p-value **4.39e-12**, gap p-value **7.48e-12**, `structure_detected=True`, and `confidence=1.0`.

This counterexample is especially relevant because the graph bug can produce repeated identity spectra. It is not an estimate of the false-positive rate for every nondegenerate null; it disproves the detector's general calibration.

**Correction:** choose a graph-level statistic and compare it with the empirical distribution from independent reference graphs. Record all successful and failed reference runs. Use permutation or Monte Carlo calibration and account for selection across rounds, embeddings, and statistics. Do not express a p-value as posterior confidence. The introductory assumptions about Marchenko–Pastur spectra, uniform gaps, and constant HKS are not established for these weighted nearest-neighbor graphs.

## 4. Blocking: two pipelines misidentify what their bit indices mean

Sources: [avalanche.py:176](C:/Users/nurdm/OneDrive/Documents/sha-256-heat_kernel/src/experiments/avalanche.py:176), [slow_bit_analysis.py:21](C:/Users/nurdm/OneDrive/Documents/sha-256-heat_kernel/analysis/slow_bit_analysis.py:21), [reduced_round_analysis.py:80](C:/Users/nurdm/OneDrive/Documents/sha-256-heat_kernel/analysis/reduced_round_analysis.py:80).

The avalanche pipeline groups whole-state divergence by the **input-message bit that was flipped**. With default 55-byte inputs there are 440 possible input positions. The interpretation script maps these indices onto the eight 32-bit working registers. Input bit 5 belongs to message word W0; it is not register `a` bit 5. Its high/low-bit annotations also reverse the core's MSB-first convention.

Separately, reduced-round analysis assigns `flipped_bit = i % 512`, although its generator chooses random input bits and returns the true metadata. Replaying the stated seed and 2,000 pairs gives **1,996 incorrect bit labels**. Claimed positions above 439 cannot describe flipped message bits in this experiment. Thus the reported per-bit round-transition analysis does not measure the coordinate groups it names.

The principal avalanche routine also labels its bottom fifth percentile as slow. With 440 distinct rates this selects 22 positions by construction. A count of 22 is not independent evidence of an exceptional population. Its slope across the full round range additionally mixes injection time, growth, and saturation.

**Correction:** retain actual pair metadata; distinguish message position, message-word index, working register, bit significance, block, and round. Align comparisons by message-word entry time where appropriate. Preserve per-position sample counts and uncertainty. Select candidate positions on discovery data and test their effect on new data against matched controls. Revalidate the numerical 0.24 anisotropy claim from the appropriate complete receipt; this audit does not certify that value.

## 5. Blocking for figures: the anisotropy illustration uses placeholder data

Source: [sha256_geometric_analysis.py:856](C:/Users/nurdm/OneDrive/Documents/sha-256-heat_kernel/analysis/sha256_geometric_analysis.py:856).

The figure-producing runner assigns successive pairs' final whole-state distances to successive bit positions, divides by 64, and supplies a hard-coded slow-bit list. The source explicitly calls this a placeholder. Pair index is not bit index.

**Correction:** regenerate the heatmap and associated slow-bit annotations from the corrected per-position aggregation. A genuine measured effect elsewhere would not make this illustration valid. Figure inputs must be persisted and traceable to the exact experiment they portray.

## 6. Blocking: the roughly minus-35-sigma slow-pair result compares different scales

Source: [sha256_geometric_attacks.py:148](C:/Users/nurdm/OneDrive/Documents/sha-256-heat_kernel/analysis/sha256_geometric_attacks.py:148).

The baseline is raw distance for random single-bit flips. Candidate two-bit distances are divided by `sqrt(2)`, then compared with the **unscaled** single-bit baseline mean and standard deviation. The search also retains the best of several patterns without a matched search control.

The saved baseline is 6.636235 with standard deviation 0.059540. Merely dividing that same baseline mean by `sqrt(2)` produces **z = -32.645**, with no improvement in raw distance. For the saved pattern (5,10), undoing the scale change changes the standardized distance from -34.890 to approximately -3.174. That restored number is still a selected statistic against an unmatched one-bit baseline, not a corrected significance claim.

**Correction:** compare the same observable and perturbation size on both sides; reproduce the candidate-selection budget in the controls and validate selected patterns on fresh messages. The large negative z-scores cannot presently support exceptional geometry.

## 7. Blocking for cryptanalytic claims: cube round counts conceal late input entry

Sources: [cube_attack.py:85](C:/Users/nurdm/OneDrive/Documents/sha-256-heat_kernel/analysis/cube_attack.py:85), [cube_attack_escalation.py:84](C:/Users/nurdm/OneDrive/Documents/sha-256-heat_kernel/analysis/cube_attack_escalation.py:84).

These scripts set bits in a Python integer and serialize it as 64 big-endian bytes. Integer bit `b` therefore belongs to message word **W[15-floor(b/32)]**, not W[floor(b/32)]. The initial cube's alleged W0/W1 variables are actually in W15/W14. A flip of integer bit 2 first changes the state **after round 16**, as directly reproduced. A zero cube before that variable enters is automatic.

The seven-variable cube was rerun locally and yields zero across all eight registers at rounds 15 and 16. At round 17 only `a,e` have nonzero sums; at round 18 `a,b,e,f` do. This makes entry time and register-shift delay essential to interpreting the trailing-register pattern. It does not establish a general attack margin. The 16-variable sweep was inspected but not rerun here.

The scripts process the first raw block and return working registers before feed-forward. They do not compute the padded two-block hash of the supplied 64-byte message. A cube-sum distinguisher can be a legitimate object of study, but a zero sum is not itself a collision/preimage attack, and failure of one selected cube does not prove security.

Further corrections: the escalation script uses real-valued matrix rank for a Boolean algebra question; a 3x3 diagnostic matrix has real rank three and GF(2) rank two. The comparative script uses LSB-first bits within each byte while the original selection used MSB-first indexing, so it does not test the same physical bit set. Its ±0.02 bias-difference rule is not a significance test or an equivalence bound; its final conclusion is hard-coded.

**Correction:** declare the primitive and output projection, use canonical coordinates, report both total rounds and rounds since variable entry, and test independent backgrounds and matched variable sets. For a Boolean polynomial of total degree below **k**, its k-fold cube derivative vanishes; the companion draft's bound of `2^k` is incorrect. A zero derivative on one cube also does not prove a global degree bound. Remove “BROKEN,” “SECURE,” and “algebraic security achieved at round 19” until claims match a defined, validated cryptanalytic task.

## 8. High: truncated graph heat quantities do not establish intrinsic curvature

Sources: [heat_kernel.py:106](C:/Users/nurdm/OneDrive/Documents/sha-256-heat_kernel/src/analysis/heat_kernel.py:106), [heat_kernel.py:247](C:/Users/nurdm/OneDrive/Documents/sha-256-heat_kernel/src/analysis/heat_kernel.py:247), [sha256_geometric_analysis.py:643](C:/Users/nurdm/OneDrive/Documents/sha-256-heat_kernel/analysis/sha256_geometric_analysis.py:643).

The solver sums only retained eigenmodes. In the corrected 256-vertex diagnostic with 50 retained modes, its trace at t=0.01 is **49.725**, versus the full trace **253.454**: an **80.38% underestimate**. At t=0 the full finite-graph trace must equal 256, while the truncated trace equals 50. This is acceptable as an explicitly truncated descriptor with convergence controls, not automatically as the full heat kernel or its small-time asymptotic.

The curvature routine takes a ratio of heat-kernel diagonals at two times. Every such diagonal for a positive-semidefinite graph Laplacian is nonincreasing, so this statistic is nonpositive by construction. It drops the dimension-dependent factor from the continuum expansion. Another routine instead calls the positive raw HKS value curvature. Neither establishes scalar curvature or its sign.

The Poincare embedding also imposes negative curvature as a configuration. For binary 256-vectors, all embedded points lie on the same radius, and their distance is a function of Hamming distance alone. The audit verifies this identity numerically. This can be a useful representation, but it is not discovery of an intrinsic hyperbolic manifold. Graph diffusion time is an analysis parameter; it must not silently be identified with SHA compression-round count.

**Correction:** name graph observables directly, validate truncation error and eigenspace stability, and separate imposed geometry from inferred properties. Any continuum-curvature claim needs a justified limiting construction and estimator. Do not infer dimension from a mode-limited participation ratio without describing its mode-count dependence.

## 9. High: the negative distinguisher claim is broader than the experiment

Source: [sha256_distinguisher_test.py:45](C:/Users/nurdm/OneDrive/Documents/sha-256-heat_kernel/analysis/sha256_distinguisher_test.py:45).

The arrays called SHA outputs are round-64 **working states before feed-forward**, rather than `trajectory.final_hash`. The audit records different values for these two objects on `abc`. FIPS 180-4 includes feed-forward in producing the hash value. For the fixed-IV single-block case, feed-forward is a fixed bijection, so the working-state experiment is meaningful; however, these classifiers and bit-distance statistics need not behave identically on the two representations. [NIST specification, section 6.2.2](https://nvlpubs.nist.gov/nistpubs/fips/nist.fips.180-4.pdf).

The SVM's PCA is fit on the entire dataset before splitting the transformed data. Fit preprocessing on training observations only, including inside cross-validation. [scikit-learn guidance on leakage](https://scikit-learn.org/stable/common_pitfalls.html).

Saved RF accuracy 48.375% corresponds to 387/800 held-out classifications. Its illustrative Wilson interval is about **44.93%–51.84%**, conditional on independent test observations and a fixed classifier. This gives a sense of resolution; it is not a bound on all distinguishers. The SVM must be rerun before presenting it as a clean held-out result. Spectral and correlation-entry KS tests also require dependence-aware calibration. The random comparator is a seeded NumPy PRNG; describe it accurately rather than calling it a cryptographic generator.

**Correction:** say what was tested and what effect sizes the experiment could resolve. “No evidence of discrimination by the tested procedure” is defensible after validation. “Indistinguishable from random,” “full 64 rounds secure,” and “complete security margin” are not conclusions these tests establish.

## 10. High: reproducibility and downstream dependencies need repair

Source: [main.py:97](C:/Users/nurdm/OneDrive/Documents/sha-256-heat_kernel/main.py:97).

The serializer falls back to strings for result objects. Saved baseline spectra include dataclass representations, truncated arrays, and function memory addresses. These are not complete machine-readable observations. Some analysis summaries omit pair identities or counts; the comparative cube script prints results without saving trial-level receipts. The multi-block trajectory interface also assigns duplicate global round label 64 to a pre-feed-forward state and the next block's initial state when both boundaries are sampled.

The companion dual-torus draft currently imports stronger conclusions from this work: geometry-guided attacks, vulnerabilities, and a practical extension by 2–4 rounds. Those statements do not become established by citing this package. They must be independently justified or removed when the dependency is rebuilt. This audit does not assess all of that draft's separate carry-field evidence.

**Correction:** persist structured arrays plus configuration, seeds, message/pair identifiers, coordinates, source hashes, environment versions, trial counts, and failures. Make every table and figure build from validated receipts. Choose the actual SHA manuscript source after the evidence is repaired.

## Claim disposition before drafting

| Present claim | Disposition |
|---|---|
| Instrumented implementation produces SHA-256 digests | Retain, scoped to tested conformance cases; keep boundary tests |
| 0.995 silhouette proves a SHA-specific manifold | Withdraw; random control reproduces it |
| 2.637 distortion versus random 1.0 | Recompute the random comparator; interpret as graph detour until supported |
| 22 slow bits in register a, aligned with Sigma0 | Recompute and relabel input coordinates; validate selection on held-out data |
| Roughly -35-sigma slow pairs | Withdraw the significance claim; match normalization and search controls |
| Isotropy transition at round 16 | Recompute with true bit metadata and entry-time controls |
| Trailing-register zeros through round 18 | Retain only as precisely specified observations after coordinate/primitive checks; no general break claim |
| Intrinsic curvature explains slow diffusion | Unsupported by current estimators; retain as a hypothesis only if useful |
| RF/SVM find no full-round distinction | Revalidate exact observable and train-only preprocessing; report uncertainty |
| No slow-bit advantage | Inconclusive as a general statement; correct coordinates and test the comparison itself |
| SHA-256 is secure / security achieved at round 19 | Remove; beyond the demonstrated experiment |

## Gated correction sequence

1. **Primitive and coordinates.** Keep the passing digest checks; add block/round identity, pre/post-feed-forward distinction, every bit convention, message-word entry, and GF(2) rank tests. Nothing downstream advances with ambiguous coordinates.
2. **Graph mathematics.** Add zero-mode, scaling, connectivity, dense-reference and permutation tests; repair normalization; validate heat truncation. Use these as regression tests before rerunning experiments.
3. **Inference controls.** Require the identical-null case to remain unflagged. Measure calibration on independent random graphs. Apply fitted clustering to both datasets and match graph settings. Predefine selection and multiplicity treatment.
4. **Empirical rerun.** Preserve original receipts; generate a new, separate evidence set for corrected per-bit analysis, matched one/two-bit searches, entry-aligned cubes, and train-only ML. Use held-out data to test choices learned during exploration. Report weak or absent effects as measured outcomes.
5. **Manuscript gate.** Generate figures, tables and numerical prose from that evidence set. Audit every claim against the exact primitive, statistic, control and replication unit. Only then decide the paper's title, central contribution and submission package.

These gates do not require the original positive findings to survive. They require that whatever survives has an accurate measurement and a fair comparison. The existing work provides useful instrumentation and testable questions; the present headline evidence is not yet ready to carry the paper.

## Reproduction

From the project directory:

```powershell
python -X utf8 -m pytest tests/test_core.py -q
python -X utf8 review/submission_audit.py
```

The audit requires the project's NumPy, SciPy and scikit-learn dependencies. It writes only its own JSON receipt in `review/`, including versions and source hashes. Numerical eigensolver details can vary across environments; the algebraic counterexamples and the distinction between a matched null and random labels do not depend on an exact last decimal.
