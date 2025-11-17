"""
Complete Infrastructure Test
Validates entire pipeline: generation -> initialization -> optimization -> logging
"""

import numpy as np
import sys
from pathlib import Path

# Add infrastructure to path
sys.path.insert(0, str(Path(__file__).parent))

from gaussian_primitives import (
    Gaussian2D, GaussianRenderer, SyntheticDataGenerator, compute_metrics
)
from initializers import EdgeInitializer, RandomInitializer
from optimizers import optimize_gaussians
from experiment_logger import ExperimentLogger


def test_case_1_straight_edge():
    """Test Case 1: Straight edge with uniform initialization"""
    print("\n" + "="*60)
    print("TEST CASE 1: Straight Sharp Edge + E1 Uniform + Adam")
    print("="*60)

    # Generate target
    target, descriptor = SyntheticDataGenerator.generate_edge(
        image_size=(100, 100),
        edge_type='straight',
        blur_sigma=0.0,  # Sharp
        contrast=0.8,
        orientation=np.pi/4
    )
    print(f"✓ Generated target: {descriptor['type']}, contrast={descriptor['contrast']}")

    # Initialize Gaussians (E1: uniform)
    gaussians_init = EdgeInitializer.uniform(
        edge_descriptor=descriptor,
        N=10,
        sigma_parallel=5.0,
        sigma_perp=1.0,
        image_size=(100, 100)
    )
    print(f"✓ Initialized {len(gaussians_init)} Gaussians with E1 (uniform)")

    # Render initial
    renderer = GaussianRenderer()
    rendered_init = renderer.render(gaussians_init, 100, 100, channels=1)
    metrics_init = compute_metrics(target, rendered_init)
    print(f"  Initial PSNR: {metrics_init['psnr']:.2f} dB")

    # Optimize
    print("  Optimizing...")
    gaussians_opt, opt_log = optimize_gaussians(
        gaussians_init=gaussians_init,
        target_image=target,
        optimizer_type='adam',
        learning_rate=0.05,
        max_iterations=300
    )

    # Render final
    rendered_final = renderer.render(gaussians_opt, 100, 100, channels=1)
    metrics_final = compute_metrics(target, rendered_final)
    print(f"✓ Optimization complete:")
    print(f"  Iterations: {len(opt_log['loss_curve'])}")
    print(f"  Final PSNR: {metrics_final['psnr']:.2f} dB (improvement: +{metrics_final['psnr'] - metrics_init['psnr']:.2f} dB)")
    print(f"  Time: {opt_log['total_time']:.2f}s")

    return metrics_final['psnr'] > 25.0  # Success if PSNR > 25 dB


def test_case_2_curved_edge():
    """Test Case 2: Curved edge with curvature-adaptive initialization"""
    print("\n" + "="*60)
    print("TEST CASE 2: Curved Blurred Edge + E2 Curvature-Adaptive + Adam")
    print("="*60)

    # Generate target
    target, descriptor = SyntheticDataGenerator.generate_edge(
        image_size=(100, 100),
        edge_type='curved',
        radius=40,
        blur_sigma=2.0,
        contrast=0.6
    )
    print(f"✓ Generated target: {descriptor['type']}, radius={descriptor['radius']}, blur={descriptor['blur_sigma']}")

    # Initialize Gaussians (E2: curvature-adaptive)
    gaussians_init = EdgeInitializer.curvature_adaptive(
        edge_descriptor=descriptor,
        N=15,
        alpha=1.0,
        sigma_parallel=4.0,
        sigma_perp=2.0,
        image_size=(100, 100)
    )
    print(f"✓ Initialized {len(gaussians_init)} Gaussians with E2 (curvature-adaptive)")

    # Render initial
    renderer = GaussianRenderer()
    rendered_init = renderer.render(gaussians_init, 100, 100, channels=1)
    metrics_init = compute_metrics(target, rendered_init)
    print(f"  Initial PSNR: {metrics_init['psnr']:.2f} dB")

    # Optimize
    print("  Optimizing...")
    gaussians_opt, opt_log = optimize_gaussians(
        gaussians_init=gaussians_init,
        target_image=target,
        optimizer_type='adam',
        learning_rate=0.05,
        max_iterations=300,
        constraints={'fix_theta': False}  # Allow orientation to adjust
    )

    # Render final
    rendered_final = renderer.render(gaussians_opt, 100, 100, channels=1)
    metrics_final = compute_metrics(target, rendered_final)
    print(f"✓ Optimization complete:")
    print(f"  Final PSNR: {metrics_final['psnr']:.2f} dB (improvement: +{metrics_final['psnr'] - metrics_init['psnr']:.2f} dB)")
    print(f"  Time: {opt_log['total_time']:.2f}s")

    return metrics_final['psnr'] > 20.0  # Success if PSNR > 20 dB (blurred edge is harder)


def test_case_3_with_logging():
    """Test Case 3: Complete pipeline with logging"""
    print("\n" + "="*60)
    print("TEST CASE 3: Complete Pipeline with Logging")
    print("="*60)

    # Setup logger
    log_dir = Path("../test_run")
    logger = ExperimentLogger(log_dir)

    # Log configuration
    config = {
        'test_case': 'straight_edge_high_contrast',
        'strategy': 'E1_uniform',
        'N': 12,
        'optimizer': 'adam',
        'learning_rate': 0.05,
        'max_iterations': 300
    }
    logger.log_config(config)
    print("✓ Configuration logged")

    # Generate target
    target, descriptor = SyntheticDataGenerator.generate_edge(
        image_size=(100, 100),
        edge_type='straight',
        blur_sigma=1.0,
        contrast=0.9,
        orientation=0.0  # Horizontal
    )
    logger.save_image("target", target)
    print("✓ Target saved")

    # Initialize
    gaussians_init = EdgeInitializer.uniform(
        edge_descriptor=descriptor,
        N=config['N'],
        sigma_parallel=5.0,
        sigma_perp=1.5,
        image_size=(100, 100)
    )
    logger.save_gaussians_init(gaussians_init)

    # Render and save initial
    renderer = GaussianRenderer()
    rendered_init = renderer.render(gaussians_init, 100, 100, channels=1)
    logger.save_image("init", rendered_init)
    logger.visualize_gaussians(gaussians_init, target, "gaussians_init_overlay")
    print("✓ Initial state saved")

    # Optimize
    print("  Optimizing with logging...")
    gaussians_opt, opt_log = optimize_gaussians(
        gaussians_init=gaussians_init,
        target_image=target,
        optimizer_type=config['optimizer'],
        learning_rate=config['learning_rate'],
        max_iterations=config['max_iterations']
    )

    # Log iterations (simulated from opt_log)
    for i, (loss, timestamp) in enumerate(zip(opt_log['loss_curve'], opt_log['timestamps'])):
        logger.log_iteration(opt_log['iterations'][i], loss)

    # Save final state
    logger.save_gaussians_final(gaussians_opt)
    rendered_final = renderer.render(gaussians_opt, 100, 100, channels=1)
    logger.save_image("final", rendered_final)

    # Compute and save residual
    residual = np.abs(target - rendered_final)
    logger.save_image("residual_final", residual)

    # Visualize final Gaussians
    logger.visualize_gaussians(gaussians_opt, target, "gaussians_final_overlay")

    # Compute metrics
    metrics = compute_metrics(target, rendered_final)
    metrics['convergence_iter'] = len(opt_log['loss_curve'])
    metrics['converged'] = opt_log['converged']
    logger.log_metrics(metrics)

    # Add notes
    logger.add_note(f"Optimization converged: {opt_log['converged']}")
    logger.add_note(f"PSNR improvement: {metrics['psnr']:.2f} dB")

    # Generate plots and report
    logger.plot_loss_curve()
    report = logger.generate_report()

    print("✓ Complete experiment logged")
    print(f"  Output directory: {log_dir}")
    print(f"  Files created: config.json, loss_curve.csv, metrics.json, images/, checkpoints/")
    print(f"  Final PSNR: {metrics['psnr']:.2f} dB")

    # Clean up
    import shutil
    shutil.rmtree(log_dir)
    print("✓ Cleanup complete")

    return True


def run_all_tests():
    """Run all infrastructure tests"""
    print("\n" + "="*70)
    print("GAUSSIAN PRIMITIVES INFRASTRUCTURE - COMPLETE VALIDATION")
    print("="*70)

    results = {}

    # Test Case 1
    try:
        results['test_1'] = test_case_1_straight_edge()
    except Exception as e:
        print(f"✗ Test Case 1 FAILED: {e}")
        results['test_1'] = False

    # Test Case 2
    try:
        results['test_2'] = test_case_2_curved_edge()
    except Exception as e:
        print(f"✗ Test Case 2 FAILED: {e}")
        results['test_2'] = False

    # Test Case 3
    try:
        results['test_3'] = test_case_3_with_logging()
    except Exception as e:
        print(f"✗ Test Case 3 FAILED: {e}")
        results['test_3'] = False

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Infrastructure is ready for experiments!")
    else:
        print("✗ SOME TESTS FAILED - Review errors above")
    print("="*70)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
