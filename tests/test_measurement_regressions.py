"""Publication gates: mathematical invariants and concrete audit failures."""
import hashlib
import numpy as np
import pytest
from scipy import sparse
from src.analysis.laplacian import LaplacianConstructor
from src.analysis.heat_kernel import HeatKernelSolver
from src.analysis.anomaly import AnomalyDetector, NullDistribution
from src.embeddings.euclidean import EuclideanEmbedding
from src.sha256.core import InstrumentedSHA256


def test_normalization_preserves_tiny_positive_degrees_and_scale():
    w = sparse.csr_matrix(np.array([[0,2,1],[2,0,3],[1,3,0]],float)*1e-14)
    d=np.asarray(w.sum(axis=1)).ravel()
    c=LaplacianConstructor(EuclideanEmbedding())
    l=c._compute_normalized_laplacian(w,d).toarray()
    np.testing.assert_allclose(l@np.sqrt(d),0,atol=1e-20)
    np.testing.assert_allclose(l,c._compute_normalized_laplacian(w*1e15,d*1e15).toarray(),atol=1e-14)


def test_full_heat_trace_and_truncation_bound():
    l=np.eye(8)-np.ones((8,8))/8
    full=HeatKernelSolver(n_eigenvalues=None).solve(l)
    partial=HeatKernelSolver(n_eigenvalues=3).solve(l)
    assert full.heat_trace(0)==pytest.approx(8)
    assert not full.truncated
    assert partial.truncated
    for t in [0,.01,1,10]:
        assert 0 <= full.heat_trace(t)-partial.heat_trace(t) <= partial.heat_trace_error_bound(t)+1e-12


def test_identical_empirical_null_not_rejected():
    hk=HeatKernelSolver(n_eigenvalues=None).solve(np.eye(8)-np.ones((8,8))/8)
    vals=np.repeat(hk.eigenvalues[None,:],39,axis=0)
    null=NullDistribution(hk.eigenvalues,np.zeros(8),hk.spectral_gaps,np.zeros(7),
        hk.heat_trace_values,np.zeros_like(hk.time_scales),39,hk.time_scales,eigenvalue_samples=vals)
    report=AnomalyDetector().analyze(hk,null)
    assert report.p_value==1
    assert not report.structure_detected
    assert report.confidence is None


def test_curvature_is_not_inferred_from_unscaled_heat_ratio():
    solver=HeatKernelSolver(n_eigenvalues=None)
    hk=solver.solve(np.eye(8))
    with pytest.raises(NotImplementedError):
        solver.spectral_curvature_estimate(hk)


@pytest.mark.parametrize('length',[0,1,3,55,56,63,64,65,119,120,128])
def test_digest_boundaries_and_unambiguous_block_coordinates(length):
    msg=np.random.default_rng(length).bytes(length)
    t=InstrumentedSHA256(sample_rounds=[0,64]).hash_with_trajectory(msg)
    assert t.final_hash==hashlib.sha256(msg).digest()
    coords=[(s.block_index,s.round_in_block) for s in t.states]
    assert len(coords)==len(set(coords))


def test_invalid_laplacian_is_not_silently_clipped():
    with pytest.raises(ValueError):
        HeatKernelSolver(n_eigenvalues=None).solve(np.diag([-1.,0.,1.]))


def test_block_boundary_selection_is_explicit():
    t=InstrumentedSHA256([0,64]).hash_with_trajectory(bytes(64))
    assert t.state_at(64,0).round_num==64
    assert t.state_at(0,1).round_num==64
    assert t.state_at(64,0).to_bytes()!=t.state_at(0,1).to_bytes()
    with pytest.raises(ValueError):t.state_at(1,0)


def test_vectorized_rounds_against_independent_scalar_core():
    from src.sha256.batch import padded_blocks, compress_blocks, state_bits, flip_input_bits, gf2_rank, cube_states
    rng=np.random.default_rng(61)
    messages=[rng.bytes(n) for n in [0,1,3,27,55]*4]
    rounds=[0,1,2,8,15,16,17,18,32,64]
    blocks=padded_blocks(messages)
    batch=compress_blocks(blocks,rounds)
    sha=InstrumentedSHA256(rounds)
    for i,m in enumerate(messages):
        for s in sha.hash_with_trajectory(m).states:
            np.testing.assert_array_equal(batch[s.round_num][i],s.working_vars)
            np.testing.assert_array_equal(state_bits(batch[s.round_num][i:i+1])[0],s.state_vector)
    pos=np.arange(len(blocks))*23
    flipped=flip_input_bits(blocks,pos)
    bits=np.unpackbits(blocks^flipped,axis=1)
    np.testing.assert_array_equal(np.argmax(bits,axis=1),pos)
    assert np.all(bits.sum(axis=1)==1)
    assert gf2_rank([[1,1,0],[1,0,1],[0,1,1]])==2
    # A variable in W15 cannot influence captures before its injection.
    cube=cube_states(blocks[:3],[480,481,482], [14,15,16])
    assert not np.any(cube[14]) and not np.any(cube[15])
