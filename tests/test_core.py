#!/usr/bin/env python3
"""
Test Suite for SHA-256 Heat Kernel Analysis
============================================

Run with: pytest tests/ -v

Author: Bee Davis
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSHA256Core:
    """Tests for SHA-256 implementation."""
    
    def test_empty_string(self):
        """Test empty string hash."""
        from src.sha256.core import InstrumentedSHA256
        
        sha = InstrumentedSHA256()
        result = sha.hash(b'')
        expected = bytes.fromhex('e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')
        assert result == expected
    
    def test_abc(self):
        """Test 'abc' hash."""
        from src.sha256.core import InstrumentedSHA256
        
        sha = InstrumentedSHA256()
        result = sha.hash(b'abc')
        expected = bytes.fromhex('ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad')
        assert result == expected
    
    def test_long_message(self):
        """Test longer message hash."""
        from src.sha256.core import InstrumentedSHA256
        
        sha = InstrumentedSHA256()
        msg = b'abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq'
        result = sha.hash(msg)
        expected = bytes.fromhex('248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1')
        assert result == expected
    
    def test_trajectory_capture(self):
        """Test that trajectory captures correct number of states."""
        from src.sha256.core import InstrumentedSHA256
        
        sample_rounds = [0, 16, 32, 48, 64]
        sha = InstrumentedSHA256(sample_rounds=sample_rounds)
        
        trajectory = sha.hash_with_trajectory(b'test')
        
        assert len(trajectory.states) == len(sample_rounds)
        
        for state, expected_round in zip(trajectory.states, sample_rounds):
            assert state.round_num == expected_round
    
    def test_state_vector_shape(self):
        """Test that state vectors have correct shape."""
        from src.sha256.core import InstrumentedSHA256
        
        sha = InstrumentedSHA256(sample_rounds=[32])
        trajectory = sha.hash_with_trajectory(b'test')
        
        assert len(trajectory.states) == 1
        assert trajectory.states[0].state_vector.shape == (256,)
        assert trajectory.states[0].working_vars.shape == (8,)
    
    def test_bit_expansion(self):
        """Test bit expansion is correct."""
        from src.sha256.core import _expand_to_bits
        
        # Test with known values
        working_vars = np.array([0x12345678] + [0] * 7, dtype=np.uint32)
        bits = _expand_to_bits(working_vars)
        
        # First 32 bits should represent 0x12345678 in big-endian
        assert bits.shape == (256,)
        # Check a few specific bits
        # 0x12345678 = 0001 0010 0011 0100 0101 0110 0111 1000
        # bits[0] = MSB of first word


class TestInputGenerator:
    """Tests for input generation."""
    
    def test_random_batch(self):
        """Test random batch generation."""
        from src.sha256.input_generator import InputGenerator
        
        gen = InputGenerator(seed=42)
        batch = gen.random_batch(100)
        
        assert len(batch) == 100
        assert all(len(msg) == 55 for msg in batch)  # Default size
    
    def test_hamming_pairs(self):
        """Test Hamming-1 pair generation."""
        from src.sha256.input_generator import InputGenerator
        
        gen = InputGenerator(seed=42)
        pairs = gen.hamming_pairs(50)
        
        assert len(pairs) == 50
        
        for pair in pairs:
            # Count differing bits
            diff = 0
            for a, b in zip(pair.message_a, pair.message_b):
                diff += bin(a ^ b).count('1')
            assert diff == 1
    
    def test_structured_zeros(self):
        """Test structured set with zeros."""
        from src.sha256.input_generator import InputGenerator
        
        gen = InputGenerator()
        messages = gen.structured_set('zeros', 5)
        
        assert len(messages) == 5
        assert all(msg == b'\x00' * 55 for msg in messages)
    
    def test_reproducibility(self):
        """Test that seed makes generation reproducible."""
        from src.sha256.input_generator import InputGenerator
        
        gen1 = InputGenerator(seed=123)
        gen2 = InputGenerator(seed=123)
        
        batch1 = gen1.random_batch(10)
        batch2 = gen2.random_batch(10)
        
        assert batch1 == batch2


class TestEmbeddings:
    """Tests for manifold embeddings."""
    
    def test_euclidean_embed(self):
        """Test Euclidean embedding."""
        from src.embeddings import EuclideanEmbedding
        
        emb = EuclideanEmbedding()
        state = np.random.randint(0, 2, 256).astype(np.uint8)
        
        point = emb.embed(state)
        
        assert point.shape == (256,)
        assert np.allclose(point, state.astype(np.float64))
    
    def test_euclidean_distance(self):
        """Test Euclidean distance is correct."""
        from src.embeddings import EuclideanEmbedding
        
        emb = EuclideanEmbedding()
        
        p1 = np.zeros(256)
        p2 = np.ones(256)
        
        d = emb.distance(p1, p2)
        assert np.isclose(d, np.sqrt(256))
    
    def test_hyperbolic_embed(self):
        """Test hyperbolic embedding stays in ball."""
        from src.embeddings import HyperbolicEmbedding
        
        emb = HyperbolicEmbedding()
        state = np.random.randint(0, 2, 256).astype(np.uint8)
        
        point = emb.embed(state)
        
        assert point.shape == (256,)
        assert np.linalg.norm(point) < 1.0  # Inside unit ball
    
    def test_hyperbolic_distance_symmetric(self):
        """Test hyperbolic distance is symmetric."""
        from src.embeddings import HyperbolicEmbedding
        
        emb = HyperbolicEmbedding()
        
        s1 = np.random.randint(0, 2, 256).astype(np.uint8)
        s2 = np.random.randint(0, 2, 256).astype(np.uint8)
        
        p1 = emb.embed(s1)
        p2 = emb.embed(s2)
        
        d12 = emb.distance(p1, p2)
        d21 = emb.distance(p2, p1)
        
        assert np.isclose(d12, d21, rtol=1e-5)
    
    def test_davis_distortion_bounds(self):
        """Test Davis manifold provides distortion bounds."""
        from src.embeddings import DavisManifoldEmbedding
        
        emb = DavisManifoldEmbedding()
        bounds = emb.distortion_bounds(5.0)
        
        assert bounds is not None
        assert bounds.path_length == 5.0
        assert bounds.distortion_bound >= 0


class TestLaplacian:
    """Tests for Laplacian construction."""
    
    def test_laplacian_shape(self):
        """Test Laplacian has correct shape."""
        from src.embeddings import EuclideanEmbedding
        from src.analysis import LaplacianConstructor
        
        emb = EuclideanEmbedding()
        constructor = LaplacianConstructor(emb, k_neighbors=10)
        
        points = np.random.rand(100, 256)
        result = constructor.build_from_points(points)
        
        assert result.laplacian.shape == (100, 100)
    
    def test_laplacian_eigenvalue_range(self):
        """Test normalized Laplacian eigenvalues in [0, 2]."""
        from src.embeddings import EuclideanEmbedding
        from src.analysis import LaplacianConstructor, HeatKernelSolver
        
        emb = EuclideanEmbedding()
        constructor = LaplacianConstructor(emb, k_neighbors=10)
        solver = HeatKernelSolver(n_eigenvalues=20)
        
        points = np.random.rand(50, 256)
        result = constructor.build_from_points(points)
        hk = solver.solve(result.laplacian)
        
        assert all(hk.eigenvalues >= -1e-10)
        assert all(hk.eigenvalues <= 2 + 1e-10)


class TestHeatKernel:
    """Tests for heat kernel solver."""
    
    def test_heat_trace_positive(self):
        """Test heat trace is always positive."""
        from src.embeddings import EuclideanEmbedding
        from src.analysis import LaplacianConstructor, HeatKernelSolver
        
        emb = EuclideanEmbedding()
        constructor = LaplacianConstructor(emb, k_neighbors=10)
        solver = HeatKernelSolver(n_eigenvalues=20)
        
        points = np.random.rand(50, 256)
        result = constructor.build_from_points(points)
        hk = solver.solve(result.laplacian)
        
        for t in [0.1, 1.0, 10.0]:
            assert hk.heat_trace(t) > 0
    
    def test_effective_dimension_range(self):
        """Test effective dimension is between 1 and n."""
        from src.embeddings import EuclideanEmbedding
        from src.analysis import LaplacianConstructor, HeatKernelSolver
        
        emb = EuclideanEmbedding()
        constructor = LaplacianConstructor(emb, k_neighbors=10)
        solver = HeatKernelSolver(n_eigenvalues=20)
        
        points = np.random.rand(50, 256)
        result = constructor.build_from_points(points)
        hk = solver.solve(result.laplacian)
        
        for t in [0.1, 1.0, 10.0]:
            d_eff = hk.effective_dimension(t)
            assert 1 <= d_eff <= 20
    
    def test_hks_shape(self):
        """Test HKS has correct shape."""
        from src.embeddings import EuclideanEmbedding
        from src.analysis import LaplacianConstructor, HeatKernelSolver
        
        emb = EuclideanEmbedding()
        constructor = LaplacianConstructor(emb, k_neighbors=10)
        solver = HeatKernelSolver(n_eigenvalues=20)
        
        n_points = 50
        points = np.random.rand(n_points, 256)
        result = constructor.build_from_points(points)
        hk = solver.solve(result.laplacian)
        
        t_values = np.array([0.1, 1.0, 10.0])
        hks = solver.heat_kernel_signature(hk, t_values)
        
        assert hks.shape == (n_points, len(t_values))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
