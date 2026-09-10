# Heat-Kernel Cryptanalysis of SHA-256

**Heat-Kernel Cryptanalysis of SHA-256: A Geometric Study of State Evolution**
Bee Rosa Davis, Davis Geometric. ORCID [0009-0009-8034-4308](https://orcid.org/0009-0009-8034-4308).
Working paper, 9 September 2026. Not yet submitted; no DOI has been assigned.

- Paper: [`publication/release/sha256_measurement.pdf`](publication/release/sha256_measurement.pdf) (18 pages, 7 figures, 16 references)
- arXiv source: [`publication/release/arxiv_source.zip`](publication/release/arxiv_source.zip)
- Reproducibility archive: [`publication/release/sha256_reproducibility.zip`](publication/release/sha256_reproducibility.zip), with [`manifest.json`](publication/release/manifest.json) and [`package_validation.json`](publication/release/package_validation.json)
- Reviewer guide, clone to full verification in about fifteen minutes: [`REVIEWER_GUIDE.md`](REVIEWER_GUIDE.md)
- Numbers digest: [`RESULTS.md`](RESULTS.md). Instructions for coding agents: [`AGENTS.md`](AGENTS.md). License: MIT ([`LICENSE`](LICENSE)).

## For the mathematical reader

This is an experimental paper in the manner of computational mathematics. The objects are defined exactly (Section 1 below), six elementary statements are stated and used, five with one-line proofs and one cited (Section 2, numbered as in the paper), and everything else is a measurement with a declared resampling unit, a frozen receipt, and an independent verifier that recomputes it from raw arrays (Sections 3 and 4). The paper claims no theorem about SHA-256 beyond those elementary statements, no attack, no distinguisher, no security proof, and no intrinsic curvature or topology of the state space. What it establishes is a fully specified way to ask geometric questions of a hash function's working state, a set of bounded negative answers, one exact positive answer at round one, and a demonstration that an earlier claimed "candidate set" was an artefact of bit significance.

The repository also preserves the December 2025 manuscript's code and claims (`review/pre_correction_snapshot/`) and the audit that withdrew them. Section 5 states each withdrawn claim and the mathematical error behind it, because a reader of the present paper is entitled to know what it replaced.

Notation: $\mathbb{W}=\mathbb{Z}/2^{32}\mathbb{Z}$; $\mathrm{wt}$ is Hamming weight; $d_H$ is Hamming distance; a working state in $\mathbb{W}^8$ is identified with $\lbrace 0,1\rbrace^{256}$ by most-significant-bit-first binary expansion.

## 1. Objects

### 1.1 The compression map

A 55-byte message $m$ is padded to one 512-bit block, parsed big-endian into $W_0,\dots,W_{15}\in\mathbb{W}$, and extended by $W_t=\sigma_1(W_{t-2})+W_{t-7}+\sigma_0(W_{t-15})+W_{t-16}$ for $16\le t\le 63$. The working state after $r$ completed rounds is $X_r(m)=(a_r,b_r,c_r,d_r,e_r,f_r,g_r,h_r)\in\mathbb{W}^8$, with $X_0$ the initial value $H_0$ and, for $1\le r\le 64$, with all additions in $\mathbb{W}$,

$$
\begin{aligned}
T_1 &= h_{r-1}+\Sigma_1(e_{r-1})+\mathrm{Ch}(e_{r-1},f_{r-1},g_{r-1})+K_{r-1}+W_{r-1},\\
T_2 &= \Sigma_0(a_{r-1})+\mathrm{Maj}(a_{r-1},b_{r-1},c_{r-1}),\\
a_r &= T_1+T_2,\qquad e_r=d_{r-1}+T_1,\\
b_r &= a_{r-1},\quad c_r=b_{r-1},\quad d_r=c_{r-1},\qquad f_r=e_{r-1},\quad g_r=f_{r-1},\quad h_r=g_{r-1}.
\end{aligned}
$$

$\Sigma_0,\Sigma_1,\sigma_0,\sigma_1,\mathrm{Ch},\mathrm{Maj},K_t$ are as in FIPS 180-4. The digest is $H_0+X_{64}$ word-wise; the paper's graph and intervention experiments use $X_r$, the classifier experiment uses the digest, and the two are never conflated. Input coordinates $i\in\lbrace 0,\dots,439\rbrace$ index message bits most-significant-first within each byte, so coordinate $i$ lies in $W_{\lfloor i/32\rfloor}$ at LSB-numbered bit position $31-(i \bmod 32)$. A variable in $W_j$ first affects the state at completed round $j+1$; the paper writes $\delta=r-(j+1)$ for rounds since entry.

Two implementations are used: a scalar instrumented SHA-256 checked against `hashlib` at padding boundaries (message lengths 0 to 128 bytes), and a vectorized raw-block compression that is checked against the scalar captures at rounds 0, 1, 2, 8, 15, 16, 17, 18, 32, 64 and produces every measurement.

### 1.2 The sampled-state graph and its heat kernel

Let $x_1,\dots,x_n\in\lbrace 0,1\rbrace^{256}$, $n=512$, be terminal working states $X_{64}$ of independent uniformly random 55-byte messages (or, for the reference population, independent uniform points of $\lbrace 0,1\rbrace^{256}$). Then $\lVert x_i-x_j\rVert_2^2=d_H(x_i,x_j)$. With $N_{30}(i)$ the 30 nearest non-self neighbours of $i$ in this metric and a bandwidth $h>0$,

$$
W^{\to}_{ij}=\mathbf{1}[j\in N_{30}(i)]\,\exp\Big(-\frac{\lVert x_i-x_j\rVert^2}{4h}\Big),\qquad
W=\tfrac12\big(W^{\to}+(W^{\to})^{\top}\big),\qquad
D=\mathrm{diag}(W\mathbf 1),\qquad
L=I-D^{-1/2}WD^{-1/2}.
$$

$L$ is symmetric positive semidefinite with spectrum $0=\lambda_0\le\lambda_1\le\dots\le\lambda_{n-1}\le 2$ and orthonormal eigenvectors $\phi_j$. The heat kernel, heat trace, diagonal, and diffusion coordinates are

$$
K_t=e^{-tL},\qquad Z(t)=\mathrm{tr}\,K_t=\sum_{j}e^{-t\lambda_j},\qquad
K_t(i,i)=\sum_j e^{-t\lambda_j}\phi_j(i)^2,\qquad
\Psi_t(i)=\big(e^{-t\lambda_j}\phi_j(i)\big)_{j=0}^{n-1},
$$

so that $\lVert \Psi_t(i)-\Psi_t(\ell)\rVert_2=\lVert K_t(i,\cdot)-K_t(\ell,\cdot)\rVert_2$. Diffusion time $t$ is a scale parameter of this fixed graph; it is not the round index $r$, and $h$ changes the graph itself. The paper computes the full spectrum ($n=512$ modes, no truncation), uses $h\in\lbrace 1,8\rbrace$, saves $Z(t)$ at $t\in\lbrace 0.01,0.1,1,10\rbrace$, and plots it on a continuous grid from the saved spectra.

The clustering statistic is the silhouette of a $k$-means partition ($k=5$, ten initialisations) fitted to the coordinates $(\phi_1,\dots,\phi_5)(i)$, evaluated in those coordinates. The reference population receives the identical fitted pipeline.

### 1.3 Detour ratio

On the undirected union of the 10-nearest-neighbour graph with edge lengths $\lVert x_i-x_j\rVert$, let $d_G$ be graph distance. The detour ratio is $\rho=\mathrm{mean}\lbrace d_G(i,\ell)/\lVert x_i-x_\ell\rVert:\ i<\ell,\ d_G(i,\ell)<\infty\rbrace$. For random points it is well above one; the paper measures it on the reference population rather than assigning it the value one.

### 1.4 Intervention functional

For a base message $m$ and coordinate $i$, with $e_i$ the corresponding unit vector,

$$
A_r(m,i)=\mathrm{wt}\big(X_r(m)\oplus X_r(m\oplus e_i)\big)\in\lbrace 0,\dots,256\rbrace.
$$

This is a whole-state Hamming response to one physical input flip, indexed by round and input coordinate. It is not a per-output-bit statistic and is distinct from the strict avalanche criterion.

### 1.5 Cube sums

For a set $S$ of $k$ input coordinates and any function $f$ of the input (a state bit, or a word with $\oplus$ taken word-wise),

$$
\Delta_S f(m)=\bigoplus_{u\in\lbrace 0,1\rbrace^k} f(m\oplus E_S u),
$$

where $E_S u$ places $u$ on the coordinates $S$: the $k$-fold Boolean derivative of $f$ in the directions $S$, evaluated at $m$. The observables are the zero rate over bases of $\Delta_S[a_r]_{16}$ (bit 16 of register $a$, LSB-numbered) and the whole-word zero fraction $\Pr_m[\Delta_S a_r(m)=0]$; the eight word sums are saved for every base, set and round.

## 2. Exact statements used in the paper

All six are numbered as in the paper. Five are proved in a line or two; the lemma is cited.

**Proposition 1 (Laplacian invariants).** $L\sqrt{d}=0$ for $\sqrt d=D^{1/2}\mathbf 1$, and $L$ is unchanged under $W\mapsto cW$, $c>0$. Both are verified numerically on every graph (zero-mode residual below $10^{-10}$).

**Proposition 2 (trace truncation).** $Z(0)=n$, and for $Z_k(t)=\sum_{j<k}e^{-t\lambda_j}$, $0\le Z(t)-Z_k(t)\le(n-k)e^{-t\lambda_{k-1}}$. The paper uses full spectra, so the bound is a control on the earlier truncated computation (Section 5), not a step in the present one.

**Proposition 3 (no curvature sign from the diagonal).** $\frac{d}{dt}K_t(i,i)=-\sum_j\lambda_je^{-t\lambda_j}\phi_j(i)^2\le0$. A ratio $K_{t_2}(i,i)/K_{t_1}(i,i)$ with $t_2>t_1$ is therefore at most one for every graph and carries no sign information about a scalar curvature; on a finite graph the continuum expansion $K_t(x,x)\sim(4\pi t)^{-d/2}(1+\tfrac{R(x)}{6}t+\dots)$ is not available without a specified limiting construction. Likewise a Poincaré-ball placement of $\pm1$ vectors at a common radius yields a distance that is a monotone function of $d_H$, so it imposes negative curvature rather than detecting it.

**Proposition 4 (round-one response to the most significant bit).** For every message $m$, $A_1(m,0)=2$.

*Proof.* In round 1, $T_1=c+W_0$ with $c=h_0+\Sigma_1(e_0)+\mathrm{Ch}(e_0,f_0,g_0)+K_0$ a constant, and $T_2$ is a constant. Coordinate 0 is bit 31 of $W_0$, so $W_0(m\oplus e_0)=W_0(m)+2^{31}$ in $\mathbb{W}$, both signs coinciding modulo $2^{32}$. Thus $a_1$ and $e_1$ each change by $2^{31}$, and $x\mapsto x+2^{31}$ toggles exactly bit 31 of $x$. The remaining six words of $X_1$ are components of $H_0$. Exactly two bits change. $\square$

The receipts record $A_1(m,0)=2$ on all 256 bases of both partitions (a verifier check). The general mechanism, that adding $\pm2^k$ to a word changes bit $k$ and a carry chain above it, so that more significant positions change fewer bits early on, is stated in the paper as the reason for the observed monotone ordering at round 1 (offsets 0, 1, 2 give mean responses 2.00, 2.48, 3.30) and is not claimed to yield exact values beyond Proposition 4.

**Lemma 5 (higher-order derivatives; Lai 1994).** If every coordinate of $f$ has algebraic degree $<k$, then $\Delta_S f\equiv0$ for every $S$ with $|S|=k$. The converse fails: a zero derivative on one cube, or at particular $m$, does not bound the degree. The cube experiments are therefore cube *tests* in the sense of Aumasson, Dinur, Meier and Shamir, and the paper reports them as bounded measurements of specified projections.

**Proposition 6 (register transport).** As functions of the input, $b_r=a_{r-1}$, $c_r=a_{r-2}$, $d_r=a_{r-3}$ and $f_r=e_{r-1}$, $g_r=e_{r-2}$, $h_r=e_{r-3}$ whenever the indices are nonnegative (with $a_0,\dots,h_0$ the constants of $H_0$). Hence $\Delta_S d_r=\Delta_S a_{r-3}$ and $\Delta_S h_r=\Delta_S e_{r-3}$, and if $S\subseteq W_j$ then $\Delta_S d_r=\Delta_S h_r=0$ for $r\le j+3$, $\Delta_S c_r=\Delta_S g_r=0$ for $r\le j+2$, and $\Delta_S b_r=\Delta_S f_r=0$ for $r\le j+1$.

*Proof.* The first line is the definition of the round map; the derivatives are of the same functions; and $a_{r'}$, $e_{r'}$ do not depend on $W_j$ for $r'\le j$. $\square$

The consequence is that a zero cube sum in a trailing register shortly after the variables enter is transport, not algebraic simplicity. The verifier checks the identities as exact array equalities on the entry-control receipts, and the entry-control figure in the paper shows precisely this staircase.

## 3. Statistical protocol

- **Resampling unit.** Independent base messages (all 440 interventions on a base are correlated and are moved together), or whole dataset pairs for the graph experiments. Intervals are percentile bootstrap intervals over that unit, 5,000 resamples, level 0.95, fixed seed. They are descriptive and unadjusted unless stated.
- **Multiplicity.** Holm's step-down adjustment over the family of candidate-versus-one-half exact binomial tests (eight in the primary cube experiment, fourteen in the follow-up). At early rounds a one-half reference is uninformative, since every six-position set is far from one half; the paper says so and reports control ranges instead.
- **Selection and holdout.** Selection rules (the bottom 22 positions by discovery mean $A_{24}$) are frozen to disk before the holdout partition is evaluated. The candidate set $S=\lbrace 1,5,7,9,10,14\rbrace$ is carried over from the December work and fixed before every evaluation.
- **Noise floor at round 24.** With $\bar A_{24}(\cdot,i)$ the holdout mean at position $i$ and base-centred residuals to account for shared bases, the observed spread $\mathrm{sd}_i\bar A_{24}(\cdot,i)=0.697$ is compared with the sampling estimate $\big[\sum_i\widehat{\mathrm{Var}}_b\big(A_{24}(b,i)-\bar A_{24}(b,\cdot)\big)/(128\cdot439)\big]^{1/2}=0.706$.
- **Control families.** Uniform: twenty 6-subsets of the 32 offsets of $W_0$. Significance-matched: forty 6-subsets of offsets $\lbrace 0,\dots,14\rbrace$ (LSB-numbered bits 17 to 31), the class containing $S$ and lying entirely above the observed bit 16, evaluated on the identical bases. For every comparison the paper reports both the paired candidate-minus-mean-control interval and the candidate's rank among the control sets, because the second is the relevant uncertainty when the question is whether the candidate is exceptional within its class.

## 4. Measured results

Every entry is regenerated from `publication/evidence/` by the build scripts and recomputed by the verifier.

| Object | Result |
|---|---|
| Silhouette of the fitted 5-partition, $h=1$, SHA vs random (primary group, 20 pairs) | 0.896 vs 0.907; mean sorted cluster sizes (2.0, 2.05, 2.3, 3.65, 502.0) in both populations |
| Same, $h=8$ | 0.161 vs 0.164; cluster sizes 81 to 121 |
| $\lambda_1$, both bandwidths, both groups | intervals for the SHA-minus-random difference contain zero |
| $Z(t)$, paired SHA-minus-random, 2 groups × 2 bandwidths × 4 times | all 16 intervals contain zero |
| Detour ratio, SHA minus random: primary / confirmation / sensitivity / 100 fresh pairs | −0.0031 [−0.0053, −0.0008] / −0.0034 [−0.0065, −0.0006] / −0.00045 [−0.0032, +0.0025] / −0.00032 [−0.00168, +0.00098]; the sign also flips against an MT19937 reference |
| Entry-aligned avalanche, all message words | common trajectory 3.8, 24.9, 55.5, 87.4, 115.6, 126.5, 127.9, 128.0 at $\delta=0,\dots,7$ |
| Round-24 holdout, 22 selected minus other positions | +0.0088 bits [−0.2869, +0.3140]; spread 0.697 vs noise floor 0.706; discovery-holdout correlation 0.018 |
| Within $W_0$, discovery-holdout correlation of position means, rounds 1 to 5 | 0.973, 0.979, 0.962, 0.964, 0.930; 0.557 at round 6; −0.017 at round 7 |
| $A_1(m,0)$ | exactly 2 on every base (Proposition 4) |
| Two-bit search minimum, round 24, candidate minus uniform controls | +0.126 bits [−0.435, +0.666] |
| Two-bit search minimum, round 2, candidate minus uniform / minus matched controls | −1.22 [−1.65, −0.79] / +1.55 [+1.14, +1.96]; 36 of 40 matched sets below the candidate; uniform-control means correlate 0.50 with mean offset |
| Six-bit cube zero rate of $[a_r]_{16}$, rounds 8 to 24, candidate minus uniform controls | all eight intervals contain zero; no Holm-adjusted rejection |
| Same at round 2 | candidate 90.43 %; uniform controls 72.29 % (sets 50.98 to 96.09 %); matched controls 86.08 % (sets 49.6 to 94.9 %), 19 of 40 at or above the candidate; whole-word $\Delta_S a_2=0$ for 0.20 % of bases |
| Transport identities on the entry control | exact equalities; staircase of zero sums as in Proposition 6 |
| Digest classifiers, five independent fits | random forest 50.315 % (49.825 to 51.050, AUC 0.502); PCA + SVM 49.800 % (48.225 to 51.425, AUC 0.500) |

Interpretation, in the paper's own terms: the terminal-state graph carries no resolved SHA-specific signal at these sample sizes; the whole state saturates about six rounds after a variable enters; there is nothing to select at round 24; there is reproducible position dependence within a word in the first five rounds, exact at round 1; a candidate set that beats uniform controls early is an ordinary member of its significance class; and cube sums in trailing registers say only what Proposition 6 says.

## 5. What the earlier version claimed, and the error behind each claim

The December 2025 manuscript (*Heat Kernel Cryptanalysis on the Davis Manifold: Geometric Phase Transitions in SHA-256*) is preserved as code and results in `review/pre_correction_snapshot/`. Its audit is `review/SHA256_SUBMISSION_REVIEW.md`, with a reproducing script. Each claim below is withdrawn.

1. *Silhouette 0.995 versus −0.574 for random.* The comparator was random labels on the SHA coordinates, not an independent random population through the same pipeline. With the correct comparator the values are 0.896 versus 0.907. The large value is partition fragmentation: at $h=1$ the neighbour Hamming distances lie in roughly $[104,115]$, so the weights $e^{-d_H/4}$ span a factor of about 25 across a node's neighbour list, the nearest neighbour carries about a fifth of each degree, and the fitted partition is one cluster of about 502 points and four of 2 to 4.
2. *Graph normalization.* Degrees at or below $10^{-10}$ were treated as zero. At $h=1$ every degree is of that order, so almost every vertex was declared isolated, the zero mode disappeared, and the smallest eigenvalue of a connected graph was reported as 0.24. The corrected normalization satisfies Proposition 1.
3. *Structure detected at every round with $p=0$.* Ordered eigenvalues were standardized by per-mode null moments and tested as i.i.d. standard normal by a Kolmogorov–Smirnov statistic; the detector rejected an exact copy of its own null at $p\approx4\times10^{-12}$. No replacement claim is made.
4. *22 slow bits in register $a$ aligned with $\Sigma_0$.* The 22 indices were input-message coordinates in $W_0$ and $W_1$, not register bits, and the selection was the bottom fifth percentile of 440 rates by construction. On fresh bases the selection has no effect at round 24 (Section 4), and the candidate's early behaviour is bit significance (Section 4).
5. *Slow pairs at −35σ.* Two-bit distances were divided by $\sqrt2$ and compared with an unscaled one-bit baseline; the scaling alone produces −32.6σ on the baseline itself. The matched, equal-budget comparison gives +0.13 bits [−0.44, +0.67].
6. *A geometry-guided cube attack breaking $c,d,g,h$ at round 17 and $d,h$ at round 18.* The cube scripts serialized integer bit $b$ big-endian, placing it in $W_{15-\lfloor b/32\rfloor}$; the "W0/W1" variables were in $W_{15}$ and $W_{14}$ and first entered the state after round 16. The trailing-register zeros are Proposition 6. The companion draft's degree threshold $2^k$ is also wrong; the correct threshold is $k$ (Lemma 5).
7. *Heat-trace and curvature statements.* Traces were computed from 50 of 256 modes, giving $Z(0.01)=49.7$ where the full value is 253.5 (Proposition 2). The "curvature" was a two-time diagonal ratio, nonpositive by Proposition 3, and a Poincaré placement imposed the sign it reported.
8. *No distinguisher, hence 64 rounds secure.* The classifier experiment fitted PCA on the whole sample before splitting and labelled pre-feed-forward working states as digests. The corrected experiment (train-only PCA, actual digests, five fits) gives the accuracies in Section 4, and the paper draws no security conclusion from them.
9. *Anisotropy figure.* Its inputs were placeholder data; pair index had been used as bit index.
10. *Receipts.* Serialized results contained object representations and memory addresses; nothing could be recomputed. Every present number has a raw-array receipt and a verifier check.

## 6. Repository layout

```
publication/
  sha256_measurement.tex          manuscript; every number enters through a generated macro
  numbers.tex, followup_numbers.tex, *_table.tex   generated from receipts
  build_paper.py, followup_tables.py               figures, tables, macros, pdflatex
  verify_evidence.py              16 checks: recomputes each receipt from its raw arrays and checks source hashes
  make_release.py                 tests, verifier, archives, manifest, RESULTS.md
  figures/                        seven figures, PDF and PNG
  evidence/                       frozen receipts (protocol saved before execution, environment, arrays, summaries)
  repro_source/                   byte-exact snapshot of the source that produced the primary run
  release/                        PDF, arXiv source, reproducibility archive, manifest, package validation
src/                              instrumented scalar SHA-256, vectorized raw-block compression, graph and heat-kernel code
analysis/publication_*.py         the five runners behind the paper
analysis/*.py (others)            December 2025 exploratory scripts, superseded; not used by the paper
tests/                            42 tests
review/                           audit, closeouts, frozen snapshots, reviewer receipts
```

| Receipts | Runner | Contents |
|---|---|---|
| `evidence/` | `analysis/publication_study.py` | 20 graph pairs with full spectra; 128 + 128 avalanche bases × 440 coordinates × 27 rounds; round-24 search on 256 bases; six-bit cubes on 512 bases at 8 rounds; five classifier fits with held-out predictions |
| `evidence/confirmation/` | `analysis/publication_confirmation.py` | 20 fresh graph pairs; raw-block entry control with all eight word sums |
| `evidence/sensitivity/` | `analysis/publication_sensitivity.py` | 20 detour comparisons across working state, digest, PCG64, MT19937 |
| `evidence/followup/` | `analysis/publication_followup.py` | early-round search (256 bases; rounds 2 to 7, 24) and cubes (512 bases; 14 rounds) against uniform controls; 100 fresh detour pairs |
| `evidence/matched/` | `analysis/publication_matched.py` | forty significance-matched control sets on the identical follow-up bases; per-set statistics; uniform-control diagnostics |

## 7. Reproduce

```
python -m pip install -r publication/release/requirements.txt
python -m pytest tests -q                 # 42 passed
python publication/verify_evidence.py     # passed, 16 checks
python publication/build_paper.py         # rebuild figures, tables, macros, PDF
python publication/make_release.py        # archives, manifest, RESULTS.md
```

Runners refuse to overwrite existing receipts; rerun any experiment into a fresh directory (about a minute each). Everything except the eigensolver-dependent graph statistics is integer arithmetic on seeded inputs and reproduces bit for bit. Details, runtimes, a table mapping every number in the paper to its receipt and script, and three claims checkable by hand are in [`REVIEWER_GUIDE.md`](REVIEWER_GUIDE.md).

The repository stores every file byte-exactly (`.gitattributes`, `* -text`), because the receipts record SHA-256 hashes of the source as stored. Package validation was performed by extracting both release archives into clean directories, compiling the arXiv source twice with no LaTeX warnings and identical extracted text, and running the tests and verifier inside the extracted archive.

## 8. Provenance

December 2025: original code and claims (Section 5). September 2026: audit (`review/SHA256_SUBMISSION_REVIEW.md`), corrections and a measurement edition (`review/CORRECTION_CLOSEOUT.md`, `review/measurement_edition_snapshot/`), restoration of the geometric framing with unchanged numbers (`review/RESTORED_GEOMETRIC_FRAMING.md`), a review that identified the round-24 noise floor, the early within-word structure, the fragmented partitions and the untested heat trace from the frozen arrays and replicated the detour contrast on 100 pairs, the follow-up runner (`review/FOLLOWUP_REVISION_CLOSEOUT.md`, `review/pre_followup_snapshot/`), a second review showing the candidate's early advantage to be bit significance (`review/reviewer_matched_controls_*.json`), and the significance-matched control family. Every receipt carries its own timestamps and source hashes.

## 9. Not in this repository

Separate manuscripts on dual-torus and double-cover formulations, the December 2025 carry-field tomography outputs, and exploratory search branches are not part of this paper and are not included. Nothing here establishes or claims an attack on SHA-256.

## 10. License and citation

The contents of this repository are released under the MIT License ([`LICENSE`](LICENSE)). The license under which the paper is distributed by a preprint server or data repository is chosen at deposit.

```
@unpublished{davis2026heatkernel,
  author = {Davis, Bee Rosa},
  title  = {Heat-Kernel Cryptanalysis of {SHA-256}: A Geometric Study of State Evolution},
  year   = {2026},
  month  = sep,
  note   = {Working paper. Code, receipts and manuscript: https://github.com/nurdymuny/sha-256-heat-kernel-cryptanalysis}
}
```
