# Claude Code instructions

Read `AGENTS.md` first and follow it; it is the authoritative set of rules for this repository. The short version:

- `publication/evidence/` and the three `review/*_snapshot/` directories are immutable receipts. Runners refuse to overwrite them; new measurements go into new directories with their protocol saved first.
- These files are hash-locked by the receipts and must not be edited: `publication/repro_source/**`, `analysis/publication_study.py`, `analysis/publication_followup.py`, `analysis/publication_matched.py`, `src/sha256/core.py`, `src/sha256/batch.py`.
- `.gitattributes` forces byte-exact storage (`* -text`). Never rewrite line endings or run a formatter over the tree.
- Numbers flow receipts → build scripts → macros → manuscript. Never type a result into the manuscript, `RESULTS.md`, or `README.md` by hand.
- Before claiming anything passes or reproduces, run and read `python -m pytest tests -q` (42 passed) and `python publication/verify_evidence.py` (16 checks). After touching anything that feeds the paper, run `python publication/build_paper.py` and `python publication/make_release.py`.
- No attack, vulnerability, or security-horizon language. Candidate-versus-control comparisons need significance-matched controls and the candidate's rank among sets. Nulls are reported as nulls.
- Commit messages are plain descriptions of the change. No automated co-author trailers or tool attributions.
