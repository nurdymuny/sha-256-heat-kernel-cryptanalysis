"""Reject inconsistent effect means even when generated tables agree with them.

All receipt and macro changes are virtual; frozen evidence is never written.
"""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'publication'))
import derived


def test_interval_counts_respect_round_scope_and_zero_endpoints():
    rows = [{'round': r, 'difference': {'ci95': ci}} for r, ci in
            [(2, [.01, .02]), (3, [0, .02]), (7, [.01, .02]), (24, [-.02, 0])]]
    assert derived.interval_counts(rows) == {'total': 4, 'contains_zero': 2, 'exceptions': [2, 7]}
    assert derived.interval_counts(rows, min_round=3) == {'total': 3, 'contains_zero': 2, 'exceptions': [7]}


@pytest.mark.parametrize('receipt', [
    'cubes.json', 'followup/search.json', 'followup/cubes.json',
    'matched/search.json', 'matched/cubes.json', 'followup/detour.json',
])
def test_verifier_rejects_wrong_mean_with_matching_macros(monkeypatch, receipt):
    original_read = Path.read_text
    evidence = ROOT / 'publication/evidence'
    target = evidence / receipt
    changed = json.loads(original_read(target))
    if receipt.endswith('detour.json'):
        next(iter(changed['summary'].values()))['mean'] += .01
    else:
        changed['records'][0]['difference']['mean'] += .01

    def virtual_read(path, *args, **kwargs):
        if path.resolve() == target:
            return json.dumps(changed)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', virtual_read)
    pm, pt = derived.primary_macros(evidence)
    fm, ft = derived.followup_macros(evidence)
    virtual = {
        ROOT / 'publication/numbers.tex': derived.render(pm, pt),
        ROOT / 'publication/followup_numbers.tex': derived.render(fm, ft),
        **{ROOT / 'publication' / filename: derived.table_text(pt[name])
           for name, filename in derived.TABLE_FILES},
    }

    def with_matching_macros(path, *args, **kwargs):
        return virtual[path.resolve()] if path.resolve() in virtual else virtual_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', with_matching_macros)
    monkeypatch.setattr(Path, 'write_text', lambda *args, **kwargs: 0)
    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(AssertionError, match='reported effect mean'):
            runpy.run_path(str(ROOT / 'publication/verify_evidence.py'), run_name='__main__')
