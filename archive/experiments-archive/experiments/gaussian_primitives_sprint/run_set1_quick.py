"""
Quick validation run of Set 1 - subset of experiments
"""

import numpy as np
import sys
from pathlib import Path
import json
import time

sys.path.insert(0, str(Path(__file__).parent / 'infrastructure'))

from gaussian_primitives import SyntheticDataGenerator, GaussianRenderer, compute_metrics
from initializers import EdgeInitializer
from optimizers import optimize_gaussians


# Quick test subset: 4 test cases × 2 N values = 8 experiments
TEST_CASES = [
    {'id': 'straight_sharp', 'edge_type': 'straight', 'orientation': np.pi/4, 'blur_sigma': 0.0, 'contrast': 0.8},
    {'id': 'straight_blur2', 'edge_type': 'straight', 'orientation': np.pi/4, 'blur_sigma': 2.0, 'contrast': 0.5},
    {'id': 'curved_r100_sharp', 'edge_type': 'curved', 'radius': 100, 'blur_sigma': 0.0, 'contrast': 0.5},
    {'id': 'curved_r50_blur2', 'edge_type': 'curved', 'radius': 50, 'blur_sigma': 2.0, 'contrast': 0.5},
]

N_VALUES = [10, 20]

# Faster optimization
OPT_CONFIG = {
    'optimizer_type': 'adam',
    'learning_rate': 0.1,  # Higher LR
    'max_iterations': 200,  # Fewer iterations
}


def run_quick_validation():
    """Run quick validation experiments"""

    print("\n" + "="*70)
    print("SET 1 QUICK VALIDATION (8 experiments)")
    print("="*70)

    results = []
    start_time = time.time()
    exp_id = 1

    for test_case in TEST_CASES:
        for N in N_VALUES:
            print(f"\n[{exp_id}/8] Running: {test_case['id']}, N={N}")

            try:
                # Generate target
                target, descriptor = SyntheticDataGenerator.generate_edge(
                    image_size=(100, 100),
                    edge_type=test_case['edge_type'],
                    radius=test_case.get('radius'),
                    blur_sigma=test_case['blur_sigma'],
                    contrast=test_case['contrast'],
                    orientation=test_case.get('orientation', 0.0)
                )

                # Initialize
                sigma_perp = max(1.0, test_case['blur_sigma'] * 1.5)
                gaussians_init = EdgeInitializer.uniform(
                    edge_descriptor=descriptor,
                    N=N,
                    sigma_parallel=5.0,
                    sigma_perp=sigma_perp,
                    image_size=(100, 100)
                )

                # Initial render
                renderer = GaussianRenderer()
                rendered_init = renderer.render(gaussians_init, 100, 100, channels=1)
                metrics_init = compute_metrics(target, rendered_init)

                # Optimize
                exp_start = time.time()
                gaussians_opt, opt_log = optimize_gaussians(
                    gaussians_init=gaussians_init,
                    target_image=target,
                    **OPT_CONFIG
                )
                wall_time = time.time() - exp_start

                # Final render
                rendered_final = renderer.render(gaussians_opt, 100, 100, channels=1)
                metrics_final = compute_metrics(target, rendered_final)

                success = metrics_final['psnr'] > 20.0  # Lower threshold for quick test

                print(f"  Initial PSNR: {metrics_init['psnr']:.2f} dB")
                print(f"  Final PSNR: {metrics_final['psnr']:.2f} dB (Δ={metrics_final['psnr']-metrics_init['psnr']:+.2f} dB)")
                print(f"  Time: {wall_time:.1f}s")
                print(f"  Status: {'✓ PASS' if success else '✗ FAIL'}")

                results.append({
                    'exp_id': exp_id,
                    'test_case': test_case['id'],
                    'N': N,
                    'psnr_init': metrics_init['psnr'],
                    'psnr_final': metrics_final['psnr'],
                    'improvement': metrics_final['psnr'] - metrics_init['psnr'],
                    'wall_time': wall_time,
                    'success': success
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    'exp_id': exp_id,
                    'test_case': test_case['id'],
                    'N': N,
                    'error': str(e),
                    'success': False
                })

            exp_id += 1

    total_time = time.time() - start_time

    # Summary
    print("\n" + "="*70)
    print("QUICK VALIDATION SUMMARY")
    print("="*70)

    successful = [r for r in results if r.get('success', False)]
    print(f"Total: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Success Rate: {len(successful)/len(results)*100:.1f}%")
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Avg Time/Exp: {total_time/len(results):.1f}s")

    if successful:
        improvements = [r['improvement'] for r in successful]
        final_psnrs = [r['psnr_final'] for r in successful]
        times = [r['wall_time'] for r in successful]

        print(f"\nPSNR Improvement:")
        print(f"  Mean: {np.mean(improvements):.2f} dB")
        print(f"  Min: {np.min(improvements):.2f} dB")
        print(f"  Max: {np.max(improvements):.2f} dB")

        print(f"\nFinal PSNR:")
        print(f"  Mean: {np.mean(final_psnrs):.2f} dB")
        print(f"  Min: {np.min(final_psnrs):.2f} dB")
        print(f"  Max: {np.max(final_psnrs):.2f} dB")

        print(f"\nWall Time:")
        print(f"  Mean: {np.mean(times):.1f}s")
        print(f"  Min: {np.min(times):.1f}s")
        print(f"  Max: {np.max(times):.1f}s")

    # Extrapolate to full Set 1
    if len(results) > 0:
        avg_time_per_exp = total_time / len(results)
        estimated_full_set1_time = avg_time_per_exp * 48  # 48 experiments in Set 1
        print(f"\nEstimated time for full Set 1 (48 experiments): {estimated_full_set1_time:.0f}s ({estimated_full_set1_time/60:.1f} min)")

    # Save results
    output_dir = Path(__file__).parent / "set_1_edge_baseline"
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / "quick_validation_results.json"

    with open(results_file, 'w') as f:
        json.dump({
            'total_experiments': len(results),
            'successful': len(successful),
            'success_rate': len(successful)/len(results) if results else 0,
            'total_time': total_time,
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("="*70)

    return len(successful) / len(results) >= 0.5  # At least 50% success


if __name__ == "__main__":
    success = run_quick_validation()
    print(f"\n{'✓' if success else '✗'} Quick validation {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
