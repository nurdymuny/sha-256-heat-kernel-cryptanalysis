# Review-driven follow-up revision

The author approved a focused evidence-led revision while retaining the geometric subject and voice. The preceding manuscript, PDF, builder, release generator and local dual-torus source are preserved in `pre_followup_snapshot/`.

## Executed work

- Added two independent scalar/algebraic regression checks before running the follow-up. The complete suite now has 40 tests.
- Saved a new protocol before execution in `publication/evidence/followup/`. Original evidence was not overwritten; all 47 prior frozen evidence and executed-source files matched the preceding release manifest.
- Ran seven-round equal-budget searches on 256 new bases with twenty fresh W0 control sets. Saved all fifteen pattern scores per set and base.
- Ran fourteen-round six-bit cube tests on 512 new bases with twenty fresh W0 control sets. Saved complete eight-word cube sums and the single-bit indicators. Independently recomputed representative target/control cubes with the scalar implementation at rounds 2, 3, 7 and 24.
- Ran 100 new detour pairs, seeds 263910-264009, with working-state, digest, PCG64 and MT19937 representations. The reviewer's separate seed range and receipt are not part of the published author-run sample.
- Added interval summaries for all sixteen saved heat-trace comparisons, W0 coordinate summaries, a covariance-aware round-24 sampling-floor estimate, fitted cluster sizes and a seventh measured figure.
- Expanded relevant citations to twelve, clarified candidate provenance and marked subsequent manuscripts as in preparation. Preserved the research sequence and first-person motivation.

## Outcomes and scope

The early search differences favor the candidate at rounds 2-5, with descriptive intervals below zero; rounds 6, 7 and 24 do not resolve a difference. Controls match entry time and budget, not bit significance. This is an early distance effect, not a demonstrated attack complexity reduction.

At round 2, the six-bit candidate zero rate is 90.4296875%, versus a mean control rate of 72.28515625%; the paired difference is 18.14453125 percentage points with interval [15.448974609375, 20.752197265625]. The candidate-versus-one-half Holm-adjusted p-value is approximately 2.03e-84 over the fourteen-round family. All thirteen later candidate-minus-control intervals contain zero. Whole-word three-bit sums and single-bit six-bit sums remain distinct observables.

The new working-state-minus-PCG64 detour estimate is -0.0003182129 with interval [-0.0016753483, 0.0009785287]. All five new detour intervals contain zero. The paper says the earlier contrast was not reproduced in these additional pairs, not that equality has been established.

The original W0 discovery/holdout correlations are 0.930-0.979 across rounds 1-5. The MSB is the minimum in both partitions across those rounds, and its round-1 response is exactly two bits. The mathematical explanation applies to that endpoint; later-round mechanisms are not inferred wholesale from it. Round-24 variation is consistent with its estimated sampling scale, not a null guaranteed by construction.

## Downstream and release

The local `theory/dual_torus_sha256.tex` citation now names this paper correctly. Its claims attributing a round-17 register break, attack extension and a round-20 security horizon to the heat-kernel measurements were replaced with the measured transport statement and an open validation question. The cube degree threshold is corrected from 2^k to k. This is a citation-consistency pass, not a validation of that draft's other clock, topology or attack claims. Other project copies and confidential documents were not edited or transmitted.

The main PDF has sixteen pages, seven figures and twelve references. All final pages were rendered; the last layout adjustment changed only page 10, which was inspected again. The final LaTeX log has no overfull/underfull boxes, undefined references or warnings. The release archives include the new runner, tests, table generator and evidence. Current archive validation and hashes are in `publication/release/package_validation.json`.

No public deposit, DOI reservation or submission was performed. The release README retains the DOI/deposit instructions.
