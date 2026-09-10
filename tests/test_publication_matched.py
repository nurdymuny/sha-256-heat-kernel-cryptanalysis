"""Checks for the significance-matched control family."""
import numpy as np
from analysis.publication_matched import matched_sets, POOL, PROTOCOL


def test_matched_pool_is_the_candidate_significance_class():
    assert POOL == list(range(15))
    assert set(PROTOCOL['target']) <= set(POOL)
    # Every pooled MSB-first offset is an LSB-numbered bit above the observed bit 16.
    assert all(31 - o > 16 for o in POOL)


def test_matched_sets_are_sorted_six_subsets_of_the_pool():
    sets = matched_sets(np.random.default_rng(1), count=40)
    assert len(sets) == 40
    for s in sets:
        assert s == sorted(s) and len(set(s)) == 6 and set(s) <= set(POOL)
    # A fixed seed reproduces the same sets.
    assert sets == matched_sets(np.random.default_rng(1), count=40)
