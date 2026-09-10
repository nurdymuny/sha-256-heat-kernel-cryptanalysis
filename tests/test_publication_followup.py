"""Independent scalar checks for the added early-round measurements."""
from itertools import combinations
import numpy as np
from src.sha256.core import InstrumentedSHA256
from src.sha256.batch import padded_blocks, compress_blocks, flip_input_bits
from analysis.publication_followup import search_pattern_scores


def test_early_search_patterns_match_scalar_interventions():
    message=bytes(range(55)); rounds=[2,3,5,7,24]
    positions=[1,5,7,9,10,14]
    actual=search_pattern_scores(padded_blocks([message]),positions,rounds)
    scalar=InstrumentedSHA256(rounds)
    original=scalar.hash_with_trajectory(message)
    expected=[]
    for p,q in combinations(positions,2):
        changed=bytearray(message)
        for bit in [p,q]: changed[bit//8]^=1<<(7-bit%8)
        trajectory=scalar.hash_with_trajectory(bytes(changed))
        expected.append([sum(int(x^y).bit_count() for x,y in zip(original.state_at(r).working_vars,trajectory.state_at(r).working_vars)) for r in rounds])
    np.testing.assert_array_equal(actual[0],expected)
    assert actual.shape==(1,15,5)


def test_round_one_msb_changes_exactly_a_and_e():
    rng=np.random.default_rng(571)
    blocks=rng.integers(0,256,(128,64),dtype=np.uint8)
    before=compress_blocks(blocks,[1])[1]
    after=compress_blocks(flip_input_bits(blocks,0),[1])[1]
    expected=np.zeros_like(before);expected[:,[0,4]]=np.uint32(1<<31)
    np.testing.assert_array_equal(before^after,expected)
