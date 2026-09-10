"""Package verified local outputs; never uploads or submits."""
from pathlib import Path
import json,hashlib,zipfile,subprocess,importlib.metadata,sys
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'publication';R=P/'release';R.mkdir(exist_ok=True)
subprocess.run([sys.executable,str(P/'verify_evidence.py')],cwd=ROOT,check=True)
tests=subprocess.run([sys.executable,'-m','pytest','tests','-q'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
(P/'test_results.txt').write_text(tests.stdout,encoding='utf-8')
if tests.returncode:raise RuntimeError('Tests failed; release not created')
deps=['numpy','scipy','scikit-learn','matplotlib','pytest']
requirements='\n'.join(name+'=='+importlib.metadata.version(name) for name in deps)+'\n'
(R/'requirements.txt').write_text(requirements)
readme='''# SHA-256 controlled measurement study: release

Paper: Graph Geometry and Input-Entry Effects in SHA-256: A Controlled Measurement Study
Author: Bee Rosa Davis, Davis Geometric. ORCID 0009-0009-8034-4308.

## Files to use

- `sha256_measurement.pdf`: complete paper, nine pages and five measured figures.
- `sha256_reproducibility.zip`: source, evidence, tests, builder and manuscript.
- `arxiv_source.zip`: LaTeX submission sources and the five figure PDFs.
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

Graph cohorts comprise 40 independently seeded SHA/random pairs; a separate 20-dataset generator/representation sensitivity sample is retained. Followups were specified after initial results and are reported separately. The unused search.discovery_bases setting in the primary protocol is reserved; the fixed-candidate search actually evaluates 256 fresh bases, as stated in the paper and receipts.

Raw avalanche arrays preserve all 440 interventions on each base. Resampling uses bases, not their correlated within-base interventions. Cube candidate/control intervals condition on twenty sampled control sets. The five classifier replications use actual digests and train-only PCA.

The manuscript does not certify SHA-256 security, establish intrinsic manifold curvature, or assert an attack improvement. Separate exploratory torus, thermocline and algebraic-search research is not included as evidence for this paper.
'''
(R/'README_RELEASE.md').write_text(readme,encoding='utf-8')
(ROOT/'README.md').write_text('''# SHA-256 controlled measurement study

The current paper is **Graph Geometry and Input-Entry Effects in SHA-256: A Controlled Measurement Study**, by Bee Rosa Davis.

- Manuscript: `publication/sha256_measurement.tex`
- PDF: `output/pdf/sha256_measurement.pdf`
- Reproducibility and arXiv source archives: `publication/release/`
- Verified receipts: `publication/evidence/`

Run `python -m pytest tests -q`, then `python publication/verify_evidence.py` and `python publication/build_paper.py`. To execute a fresh primary study without overwriting receipts, run `python analysis/publication_study.py --output publication/new_evidence`.

The study distinguishes working states from digests, input entry from register transport, and absolute graph statistics from matched comparisons. See the paper for the full results, including the unresolved small detour contrasts.

The exact main-run source is preserved in `publication/repro_source/`. Broader resonant_tunnel, thermocline and algebraic-search branches remain separate exploratory research. Internal historical sources and narratives are preserved under `review/pre_correction_snapshot/`; they are not the current publication claims. The placeholder geometric-figure orchestration has been replaced by the controlled publication runner.
''',encoding='utf-8')
(ROOT/'RESULTS.md').write_text('''# SHA-256 controlled measurement results

All numerical tables and figures are generated from `publication/evidence/` by `publication/build_paper.py`. Read `output/pdf/sha256_measurement.pdf` for the full protocol and every interval.

- Large graph silhouettes occur in SHA and random data. The small primary silhouette contrast does not repeat. Small negative detour contrasts appear in two cohorts, while a third sensitivity sample leaves the contrast unresolved.
- Across 112,640 one-bit interventions, the 22 frozen discovery-selected positions have a held-out selected-minus-other difference of +0.0088 Hamming bits, 95% base-bootstrap interval [-0.2869, 0.3140].
- Equal-budget two-bit search gives a candidate-minus-mean-control difference of +0.1262 bits, interval [-0.4354, 0.6657], over 256 new bases.
- For the six-bit cube experiment, all eight candidate-minus-control intervals include zero, and no candidate-versus-one-half test rejects after Holm adjustment. These are bounded findings for a specified output bit and candidate set.
- Input entry and exact register transport account for the timing pattern in the three-bit raw-block cube control; whole-word and single-bit zero sums are distinguished.
- Actual digest classifiers average 50.315% accuracy for RF and 49.800% for train-only PCA plus SVM across five independent fits each.

These measurements do not establish cryptographic security, general indistinguishability, intrinsic curvature, or an attack improvement. Verify the saved records with `python publication/verify_evidence.py`. Historical findings and source are retained internally in `review/pre_correction_snapshot/`.
''',encoding='utf-8')

files={}
def include(path,arc=None):
    files[(arc or str(path.relative_to(ROOT))).replace('\\','/')]=path
for p in (ROOT/'src').rglob('*.py'):include(p)
for p in (ROOT/'tests').glob('*.py'):include(p)
for name in ['__init__.py','publication_study.py','publication_confirmation.py','publication_sensitivity.py']:include(ROOT/'analysis'/name)
for p in (P/'repro_source').rglob('*.py'):include(p)
for p in (P/'evidence').rglob('*'):
    if p.is_file():include(p)
for name in ['build_paper.py','verify_evidence.py','sha256_measurement.tex','numbers.tex','graph_table.tex','sensitivity_table.tex','cube_table.tex','ml_table.tex','verification.json','test_results.txt']:include(P/name)
for p in (P/'figures').glob('*.pdf'):include(p)
include(ROOT/'output/pdf/sha256_measurement.pdf')
include(R/'requirements.txt','requirements.txt');include(R/'README_RELEASE.md','README.md')
manifest={arc:hashlib.sha256(p.read_bytes()).hexdigest() for arc,p in sorted(files.items())}
(R/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with zipfile.ZipFile(R/'sha256_reproducibility.zip','w',zipfile.ZIP_DEFLATED) as z:
    for arc,p in sorted(files.items()):z.write(p,arc)
    z.write(R/'manifest.json','manifest.json')
with zipfile.ZipFile(R/'arxiv_source.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ['sha256_measurement.tex','numbers.tex']:z.write(P/name,name)
    for p in (P/'figures').glob('*.pdf'):z.write(p,'figures/'+p.name)
import shutil
shutil.copy2(ROOT/'output/pdf/sha256_measurement.pdf',R/'sha256_measurement.pdf')
with zipfile.ZipFile(R/'sha256_reproducibility.zip') as z:
    assert z.testzip() is None
    for arc,digest in manifest.items():assert hashlib.sha256(z.read(arc)).hexdigest()==digest,arc
print(json.dumps({'archive_files':len(files),'zip_bytes':(R/'sha256_reproducibility.zip').stat().st_size,'tests_passed':True,'archive_hashes_verified':True},indent=2))
