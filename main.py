#!/usr/bin/env python3
"""
SHA-256 Heat Kernel Analysis - Main Entry Point
================================================

Run geometric analysis experiments on SHA-256's internal state space.

Usage:
    python main.py --experiment 1 --n-samples 1000
    python main.py --experiment 2 --n-pairs 500
    python main.py --experiment 3 --quick
    python main.py --validate

Author: Bee Davis
"""

import argparse
import sys
import json
from datetime import datetime
from pathlib import Path


def validate_sha256():
    """Validate SHA-256 implementation against test vectors."""
    print("Validating SHA-256 implementation...")
    
    from src.sha256.core import validate_implementation
    
    try:
        validate_implementation()
        print("✓ SHA-256 implementation validated")
        return True
    except AssertionError as e:
        print(f"✗ Validation failed: {e}")
        return False


def run_experiment_1(args):
    """Run baseline diffusion profile experiment."""
    from src.experiments import experiment_1_baseline
    
    config = {
        'n_samples': args.n_samples,
        'sample_rounds': [0, 8, 16, 24, 32, 40, 48, 56, 64] if not args.quick else [0, 32, 64],
        'embedding': args.embedding,
        'k_neighbors': 30 if args.quick else 50,
        'n_eigenvalues': 50 if args.quick else 200,
        'null_samples': 20 if args.quick else 100,
        'seed': args.seed
    }
    
    result = experiment_1_baseline(config)
    return result


def run_experiment_2(args):
    """Run avalanche geometry experiment."""
    from src.experiments import experiment_2_avalanche
    
    config = {
        'n_pairs': args.n_pairs,
        'sample_rounds': list(range(0, 65, 4)) if not args.quick else list(range(0, 65, 8)),
        'embedding': args.embedding,
        'seed': args.seed
    }
    
    result = experiment_2_avalanche(config)
    return result


def run_experiment_3(args):
    """Run full Davis manifold comparison."""
    from src.experiments import experiment_3_davis_comparison
    
    config = {
        'n_samples': args.n_samples,
        'n_pairs': args.n_pairs,
        'sample_rounds': [0, 32, 64] if args.quick else [0, 16, 32, 48, 64],
        'k_neighbors': 20 if args.quick else 50,
        'n_eigenvalues': 30 if args.quick else 200,
        'null_samples': 10 if args.quick else 100,
        'seed': args.seed
    }
    
    result = experiment_3_davis_comparison(config)
    return result


def save_results(result: dict, experiment: str, output_dir: str = "results"):
    """Save results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"exp{experiment}_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert(obj):
        from dataclasses import is_dataclass, fields
        if is_dataclass(obj):
            return {f.name: convert(getattr(obj,f.name)) for f in fields(obj) if not callable(getattr(obj,f.name))}
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        else:
            return obj
    
    serializable = convert(result)
    
    with open(filename, 'w') as f:
        json.dump(serializable, f, indent=2, allow_nan=False)
    
    print(f"\nResults saved to {filename}")
    return filename


def demo():
    """Run a quick demo of the system."""
    print("=" * 60)
    print("SHA-256 Heat Kernel Analysis - Demo")
    print("=" * 60)
    
    # 1. Show SHA-256 trajectory capture
    print("\n1. SHA-256 Trajectory Capture")
    print("-" * 40)
    
    from src.sha256 import InstrumentedSHA256
    
    sha = InstrumentedSHA256(sample_rounds=[0, 16, 32, 48, 64])
    trajectory = sha.hash_with_trajectory(b"Hello, geometric cryptanalysis!")
    
    print(f"Input: 'Hello, geometric cryptanalysis!'")
    print(f"Hash: {trajectory.final_hash.hex()}")
    print(f"Captured {len(trajectory.states)} states")
    
    for state in trajectory.states[:3]:
        print(f"  Round {state.round_num}: first 4 vars = {[hex(v) for v in state.working_vars[:4]]}")
    
    # 2. Show embedding
    print("\n2. Manifold Embedding")
    print("-" * 40)
    
    from src.embeddings import get_embedding
    
    for emb_name in ['euclidean', 'hyperbolic', 'davis']:
        embedding = get_embedding(emb_name)
        point = embedding.embed(trajectory.states[0].state_vector)
        print(f"  {emb_name}: embedded point norm = {(point**2).sum()**0.5:.4f}")
    
    # 3. Show heat kernel analysis (minimal)
    print("\n3. Spectral Analysis (minimal demo)")
    print("-" * 40)
    
    from src.sha256 import InputGenerator
    from src.embeddings import EuclideanEmbedding
    from src.analysis import LaplacianConstructor, HeatKernelSolver
    
    gen = InputGenerator(seed=42)
    inputs = gen.random_batch(100)  # Small for demo
    
    sha = InstrumentedSHA256(sample_rounds=[0, 32, 64])
    trajectories = sha.hash_batch(inputs)
    
    embedding = EuclideanEmbedding()
    constructor = LaplacianConstructor(embedding, k_neighbors=10)
    solver = HeatKernelSolver(n_eigenvalues=20)
    
    for round_num in [0, 32, 64]:
        lap_result = constructor.build_from_trajectories(trajectories, round_num)
        hk_result = solver.solve(lap_result.laplacian)
        
        print(f"  Round {round_num}: spectral gap = {hk_result.spectral_gap:.4f}, "
              f"eff. dim (t=1) = {hk_result.effective_dimension(1.0):.2f}")
    
    print("\n" + "=" * 60)
    print("Demo complete! Run with --experiment to perform full analysis.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="SHA-256 Heat Kernel Geometric Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                     # Quick demo
  python main.py --validate                 # Validate SHA-256
  python main.py --experiment 1             # Baseline diffusion
  python main.py --experiment 2             # Avalanche geometry
  python main.py --experiment 3 --quick     # Full comparison (quick)
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                        help='Run quick demonstration')
    parser.add_argument('--validate', action='store_true',
                        help='Validate SHA-256 implementation')
    parser.add_argument('--experiment', type=str, choices=['1', '2', '3'],
                        help='Which experiment to run')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with reduced samples')
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Number of samples for experiments 1 and 3')
    parser.add_argument('--n-pairs', type=int, default=500,
                        help='Number of pairs for experiment 2')
    parser.add_argument('--embedding', type=str, default='euclidean',
                        choices=['euclidean', 'hyperbolic', 'davis'],
                        help='Embedding to use (for experiments 1 and 2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Handle flags
    if args.demo:
        demo()
        return 0
    
    if args.validate:
        success = validate_sha256()
        return 0 if success else 1
    
    if args.experiment is None:
        parser.print_help()
        return 0
    
    # Validate first
    if not validate_sha256():
        print("SHA-256 validation failed, aborting.")
        return 1
    
    # Run selected experiment
    print(f"\n{'=' * 60}")
    print(f"Running Experiment {args.experiment}")
    print(f"{'=' * 60}")
    
    if args.experiment == '1':
        result = run_experiment_1(args)
    elif args.experiment == '2':
        result = run_experiment_2(args)
    elif args.experiment == '3':
        result = run_experiment_3(args)
    
    # Save results
    save_results(result, args.experiment, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
