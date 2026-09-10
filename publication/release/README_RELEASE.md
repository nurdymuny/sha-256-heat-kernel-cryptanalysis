# Heat-Kernel Cryptanalysis of SHA-256: release

Paper: Heat-Kernel Cryptanalysis of SHA-256: A Geometric Study of State Evolution
Author: Bee Rosa Davis, Davis Geometric. ORCID 0009-0009-8034-4308.

## Files to use

- `sha256_measurement.pdf`: complete paper with seven measured figures.
- `sha256_reproducibility.zip`: source, evidence, tests, builder and manuscript.
- `arxiv_source.zip`: LaTeX submission sources and the seven figure PDFs.
- `manifest.json`: checksums for the reproducibility archive's contents.

No repository upload or submission has been performed. The PDF currently names the accompanying archive; it does not assert a public DOI. If a repository DOI is obtained, add its link to the data-availability paragraph and rebuild before submitting the final PDF. The archive does not impose a new paper license; choose the deposit license explicitly.

## Reproduce from sha256_reproducibility.zip

Use Python 3.12 and install `requirements.txt`. A LaTeX distribution with pdflatex is required to build the paper.

```
python -m pip install -r requirements.txt
python -m pytest tests -q
python publication/verify_evidence.py
python publication/build_paper.py
```

The PDF is written to `output/pdf/sha256_measurement.pdf`. Verification reads the frozen receipts and recomputes their summaries. It does not rerun training or the graph experiments.

For a fresh primary run with the exact executed source snapshot:

```
cd publication/repro_source
python analysis/publication_study.py --output ../fresh_primary_evidence
```

The snapshot has byte-for-byte source hashes matching the primary receipt. The root shared code also includes subsequent maintenance of unused legacy interfaces; these changes do not affect the measured functions. Follow-up runners are `analysis/publication_confirmation.py` and `analysis/publication_sensitivity.py`; they refuse to overwrite their existing receipts. To rerun them, use a separate extracted working copy with new evidence output locations.

## Study boundaries

Graph cohorts comprise 40 independently seeded SHA/random pairs; a separate 20-dataset generator/representation sensitivity sample and an additional 100-pair sample are retained. The early-round follow-up uses 256 new search bases and 512 new cube bases, with twenty new control sets per experiment. Its runner is `analysis/publication_followup.py`; run it with `--output publication/fresh_followup` to create a new directory. A significance-matched control family (forty six-position subsets of W0 offsets 0-14, evaluated on the same follow-up bases) is produced by `analysis/publication_matched.py --output publication/fresh_matched`; its receipts are under `publication/evidence/matched/`. Follow-ups were specified after initial results and are reported separately. The unused search.discovery_bases setting in the primary protocol is reserved; the original fixed-candidate search actually evaluates 256 fresh bases, as stated in the paper and receipts.

Raw avalanche arrays preserve all 440 interventions on each base. Resampling uses bases, not their correlated within-base interventions. Cube candidate/control intervals condition on twenty sampled control sets. The five classifier replications use actual digests and train-only PCA.

The paper develops heat-kernel cryptanalysis as a geometric investigative method and connects spectral descriptions with measured perturbation transport. Its discussion explains how this inquiry motivated the author's subsequent dual-torus and double-cover work. Those later models are research directions, not results established by the present experiments. The heat-profile figure is derived from the complete frozen spectra; it adds no new experimental sample. The manuscript does not certify SHA-256 security, establish intrinsic manifold curvature, or assert an attack improvement.
