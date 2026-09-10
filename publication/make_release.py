"""Package verified local outputs; never uploads or submits."""
from pathlib import Path
import json,hashlib,zipfile,subprocess,importlib.metadata,sys
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'publication';R=P/'release';R.mkdir(exist_ok=True)
from derived import interval_counts
later=interval_counts(json.loads((P/'evidence/matched/cubes.json').read_text())['records'], min_round=3)
later_summary=f"{later['contains_zero']} of the {later['total']} intervals at sampled rounds after round 2 contain zero"
subprocess.run([sys.executable,str(P/'verify_evidence.py')],cwd=ROOT,check=True)
tests=subprocess.run([sys.executable,'-m','pytest','tests','-q'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
(P/'test_results.txt').write_text(tests.stdout,encoding='utf-8')
if tests.returncode:raise RuntimeError('Tests failed; release not created')
deps=['numpy','scipy','scikit-learn','matplotlib','pytest']
requirements='\n'.join(name+'=='+importlib.metadata.version(name) for name in deps)+'\n'
(R/'requirements.txt').write_text(requirements)
readme='''# Heat-Kernel Cryptanalysis of SHA-256: release

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
'''
(R/'README_RELEASE.md').write_text(readme,encoding='utf-8')
# README.md at the repository root is maintained by hand; only RESULTS.md is regenerated here.
(ROOT/'RESULTS.md').write_text('''# Heat-Kernel Cryptanalysis of SHA-256: measurements

All numerical tables and figures are generated from `publication/evidence/` by `publication/build_paper.py`. Read `output/pdf/sha256_measurement.pdf` for the full protocol and every interval.

- Complete graph spectra describe heat diffusion across scales; the paper plots normalized heat traces and paired SHA-minus-random contrasts for two kernel parameters and two independent groups. These profiles are derived from frozen spectra.
- Entry-aligned intervention trajectories show the organized growth of state separation. Cube measurements resolve the exact leading-to-trailing transport within the two coupled register families. This organization motivates the geometric investigation and its connection to the author's subsequent research.
- Large graph silhouettes occur in SHA and random data and accompany highly unbalanced fitted partitions at narrow bandwidth. The small primary silhouette contrast does not repeat. Small negative detour contrasts appear in two cohorts but are not reproduced in a further 100-pair author run; the prior 20-pair sensitivity sample is retained.
- W0 discovery-holdout correlations are 0.930-0.979 across rounds 1-5. The MSB intervention changes exactly two state bits at round 1; the paper gives the modular-addition explanation.
- Fresh equal-budget searches: against twenty uniform W0 control sets the candidate minimum is lower at rounds 2-5 (unadjusted paired intervals below zero); against forty significance-matched control sets (offsets 0-14, same bases) the difference reverses, +1.55 bits [+1.14, +1.96] at round 2, with 36 of 40 matched sets below the candidate. Uniform-control means correlate 0.50 with the sets' mean offset at round 2. The early distance reduction against uniform controls is largely an effect of bit significance.
- Fresh six-bit cube tests at round 2: candidate zero rate 90.43% versus 72.29% for uniform controls (twenty sets ranging 50.98% to 96.09%). Against forty significance-matched sets (49.6% to 94.9%, mean 86.08%) the candidate remains above the matched mean, +4.35 points [+1.8, +6.8], but is not exceptional among individual sets: 19 of 40 lie at or above it (base-bootstrap bounds 13 to 26 on that count). The whole-word sum of register a is zero for 0.2% of bases. Against the uniform family, rounds 3-24 resolve no difference; against the matched family __MATCHED_LATER_COUNTS__, the exception being round 7 (+4.86 [+0.4, +9.3] points), an isolated unadjusted contrast that requires confirmation. Holm adjustment covers the fourteen candidate-versus-one-half tests as a separate family and does not adjust the candidate-versus-control comparisons.
- Across 112,640 one-bit interventions, the 22 frozen discovery-selected positions have a held-out selected-minus-other difference of +0.0088 Hamming bits, 95% base-bootstrap interval [-0.2869, 0.3140].
- Equal-budget two-bit search gives a candidate-minus-mean-control difference of +0.1262 bits, interval [-0.4354, 0.6657], over 256 new bases.
- For the original six-bit cube experiment, all eight candidate-minus-control intervals include zero, and no candidate-versus-one-half test rejects after Holm adjustment. These are bounded findings for a specified output bit and candidate set.
- Input entry and exact register transport account for the timing pattern in the three-bit raw-block cube control; whole-word and single-bit zero sums are distinguished.
- Actual digest classifiers average 50.315% accuracy for RF and 49.800% for train-only PCA plus SVM across five independent fits each.

These measurements do not establish cryptographic security, general indistinguishability, intrinsic curvature, or an attack improvement. Verify the saved records with `python publication/verify_evidence.py`. Historical findings and source are retained internally in `review/pre_correction_snapshot/`.
'''.replace('__MATCHED_LATER_COUNTS__', later_summary),encoding='utf-8')

files={}
def include(path,arc=None):
    files[(arc or str(path.relative_to(ROOT))).replace('\\','/')]=path
for p in (ROOT/'src').rglob('*.py'):include(p)
for p in (ROOT/'tests').glob('*.py'):include(p)
for name in ['__init__.py','publication_study.py','publication_confirmation.py','publication_sensitivity.py','publication_followup.py','publication_matched.py']:include(ROOT/'analysis'/name)
for p in (P/'repro_source').rglob('*.py'):include(p)
for p in (P/'evidence').rglob('*'):
    if p.is_file():include(p)
for name in ['build_paper.py','followup_tables.py','derived.py','verify_evidence.py','sha256_measurement.tex','numbers.tex','followup_numbers.tex','graph_table.tex','sensitivity_table.tex','cube_table.tex','ml_table.tex','verification.json','test_results.txt']:include(P/name)
for p in (P/'figures').glob('*.pdf'):include(p)
include(ROOT/'output/pdf/sha256_measurement.pdf')
include(R/'requirements.txt','requirements.txt');include(R/'README_RELEASE.md','README.md')
manifest={arc:hashlib.sha256(p.read_bytes()).hexdigest() for arc,p in sorted(files.items())}
(R/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(R/'sha256_reproducibility.zip','w',zipfile.ZIP_DEFLATED) as z:
    for arc,p in sorted(files.items()):z.write(p,arc)
    z.write(R/'manifest.json','manifest.json')
with zipfile.ZipFile(R/'arxiv_source.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ['sha256_measurement.tex','numbers.tex','followup_numbers.tex']:z.write(P/name,name)
    for p in (P/'figures').glob('*.pdf'):z.write(p,'figures/'+p.name)
import shutil
shutil.copy2(ROOT/'output/pdf/sha256_measurement.pdf',R/'sha256_measurement.pdf')
with zipfile.ZipFile(R/'sha256_reproducibility.zip') as z:
    assert z.testzip() is None
    for arc,digest in manifest.items():assert hashlib.sha256(z.read(arc)).hexdigest()==digest,arc
print(json.dumps({'archive_files':len(files),'zip_bytes':(R/'sha256_reproducibility.zip').stat().st_size,'tests_passed':True,'archive_hashes_verified':True},indent=2))
