"""Hand-checkable cases for the statistical helpers shared by the runners, the table generator and the verifier."""
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'publication'))
from analysis.publication_study import interval, holm
from derived import rank_bootstrap


def test_interval_of_constant_data_is_degenerate():
    iv = interval([3.0] * 20)
    assert iv['mean'] == 3.0 and iv['ci95'] == [3.0, 3.0] and iv['n_units'] == 20


def test_interval_is_shift_equivariant_under_its_fixed_seed():
    x = np.arange(10.0)
    a, b = interval(x), interval(x + 5)
    np.testing.assert_allclose(np.array(b['ci95']) - np.array(a['ci95']), 5.0)
    assert a['ci95'][0] <= a['mean'] <= a['ci95'][1]


def test_holm_hand_computed_examples():
    # sorted p: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone max gives 0.06
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(holm([0.01, 0.02, 0.5]), [0.03, 0.04, 0.5])
    np.testing.assert_allclose(holm([0.5]), [0.5])
    assert max(holm([0.9, 0.8])) <= 1.0


def test_rank_bootstrap_degenerate_cases():
    cand = np.ones(50); same = np.ones((7, 50)); lower = np.zeros((7, 50))
    assert rank_bootstrap(cand, same, 'ge', resamples=200, chunk=50) == (7, 7)
    assert rank_bootstrap(cand, same, 'lt', resamples=200, chunk=50) == (0, 0)
    assert rank_bootstrap(cand, lower, 'ge', resamples=200, chunk=50) == (0, 0)
    assert rank_bootstrap(cand, lower, 'lt', resamples=200, chunk=50) == (7, 7)
