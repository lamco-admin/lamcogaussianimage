"""
Experiment Set 1: Edge Baseline
Goal: Establish baseline performance for edge primitive with simplest strategy
Total: 48 experiments (12 test cases × 4 N values)
"""

import numpy as np
import sys
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Add infrastructure
sys.path.insert(0, str(Path(__file__).parent / 'infrastructure'))

from gaussian_primitives import SyntheticDataGenerator, GaussianRenderer, compute_metrics
from initializers import EdgeInitializer
from optimizers import optimize_gaussians
from experiment_logger import ExperimentLogger, ExperimentBatch


# Experiment configuration
EXPERIMENT_SET_NAME = "set_1_edge_baseline"
BASE_DIR = Path(__file__).parent

# Test cases as per protocol
TEST_CASES = [
    # Straight sharp edges (3 cases)
    {'id': 'straight_sharp_vertical', 'edge_type': 'straight', 'orientation': np.pi/2, 'blur_sigma': 0.0, 'contrast': 0.5},
    {'id': 'straight_sharp_horizontal', 'edge_type': 'straight', 'orientation': 0.0, 'blur_sigma': 0.0, 'contrast': 0.5},
    {'id': 'straight_sharp_diagonal', 'edge_type': 'straight', 'orientation': np.pi/4, 'blur_sigma': 0.0, 'contrast': 0.5},

    # Straight with varying contrast (3 cases)
    {'id': 'straight_sharp_low_contrast', 'edge_type': 'straight', 'orientation': np.pi/4, 'blur_sigma': 0.0, 'contrast': 0.2},
    {'id': 'straight_sharp_mid_contrast', 'edge_type': 'straight', 'orientation': np.pi/4, 'blur_sigma': 0.0, 'contrast': 0.5},
    {'id': 'straight_sharp_high_contrast', 'edge_type': 'straight', 'orientation': np.pi/4, 'blur_sigma': 0.0, 'contrast': 0.8},

    # Straight blurred edges (3 cases)
    {'id': 'straight_blur1', 'edge_type': 'straight', 'orientation': np.pi/4, 'blur_sigma': 1.0, 'contrast': 0.5},
    {'id': 'straight_blur2', 'edge_type': 'straight', 'orientation': np.pi/4, 'blur_sigma': 2.0, 'contrast': 0.5},
    {'id': 'straight_blur4', 'edge_type': 'straight', 'orientation': np.pi/4, 'blur_sigma': 4.0, 'contrast': 0.5},

    # Curved edges - varying radius (3 cases)
    {'id': 'curved_r50_sharp', 'edge_type': 'curved', 'radius': 50, 'blur_sigma': 0.0, 'contrast': 0.5},
    {'id': 'curved_r100_sharp', 'edge_type': 'curved', 'radius': 100, 'blur_sigma': 0.0, 'contrast': 0.5},
    {'id': 'curved_r200_sharp', 'edge_type': 'curved', 'radius': 200, 'blur_sigma': 0.0, 'contrast': 0.5},
]

# N values to sweep
N_VALUES = [5, 10, 20, 40]

# Optimization parameters
OPT_CONFIG = {
    'optimizer_type': 'adam',
    'learning_rate': 0.05,
    'max_iterations': 500,
    'constraints': None
}


def run_single_experiment(test_case_id: str, test_case: dict, N: int, exp_id: int) -> dict:
    """Run a single experiment"""

    try:
        # Create experiment name
        exp_name = f"{test_case_id}_N{N:02d}"

        # Generate target
        target, descriptor = SyntheticDataGenerator.generate_edge(
            image_size=(100, 100),
            edge_type=test_case['edge_type'],
            radius=test_case.get('radius'),
            blur_sigma=test_case['blur_sigma'],
            contrast=test_case['contrast'],
            orientation=test_case.get('orientation', 0.0)
        )

        # Initialize Gaussians (E1: uniform)
        sigma_perp = max(1.0, test_case['blur_sigma'])  # Adapt to blur
        gaussians_init = EdgeInitializer.uniform(
            edge_descriptor=descriptor,
            N=N,
            sigma_parallel=5.0,
            sigma_perp=sigma_perp,
            image_size=(100, 100)
        )

        # Optimize
        start_time = time.time()
        gaussians_opt, opt_log = optimize_gaussians(
            gaussians_init=gaussians_init,
            target_image=target,
            **OPT_CONFIG
        )
        wall_time = time.time() - start_time

        # Render final
        renderer = GaussianRenderer()
        rendered_final = renderer.render(gaussians_opt, 100, 100, channels=1)

        # Compute metrics
        metrics = compute_metrics(target, rendered_final)
        metrics['wall_time'] = wall_time
        metrics['convergence_iter'] = len(opt_log['loss_curve'])
        metrics['N'] = N

        # Determine success (PSNR > 25 dB)
        success = metrics['psnr'] > 25.0

        result = {
            'exp_id': exp_id,
            'exp_name': exp_name,
            'test_case_id': test_case_id,
            'N': N,
            'metrics': metrics,
            'success': success,
            'descriptor': descriptor
        }

        print(f"[{exp_id:02d}/48] {exp_name}: PSNR={metrics['psnr']:.2f} dB, Time={wall_time:.1f}s {'✓' if success else '✗'}")

        return result

    except Exception as e:
        print(f"ERROR in experiment {exp_id}: {e}")
        return {
            'exp_id': exp_id,
            'exp_name': f"{test_case_id}_N{N:02d}",
            'error': str(e),
            'success': False
        }


def run_experiment_set_1():
    """Run all 48 experiments in Set 1"""

    print("\n" + "="*80)
    print("EXPERIMENT SET 1: EDGE BASELINE")
    print("="*80)
    print(f"Test Cases: {len(TEST_CASES)}")
    print(f"N Values: {N_VALUES}")
    print(f"Total Experiments: {len(TEST_CASES) * len(N_VALUES)}")
    print(f"Strategy: E1 (uniform)")
    print(f"Optimizer: {OPT_CONFIG['optimizer_type']}, LR={OPT_CONFIG['learning_rate']}")
    print("="*80 + "\n")

    # Create output directory
    output_dir = BASE_DIR / EXPERIMENT_SET_NAME
    output_dir.mkdir(exist_ok=True)

    # Prepare experiments
    experiments = []
    exp_id = 1
    for test_case in TEST_CASES:
        for N in N_VALUES:
            experiments.append((test_case['id'], test_case, N, exp_id))
            exp_id += 1

    # Run experiments
    results = []
    start_time = time.time()

    # Sequential execution (parallel would be faster but more complex to debug)
    for test_case_id, test_case, N, exp_id in experiments:
        result = run_single_experiment(test_case_id, test_case, N, exp_id)
        results.append(result)

    total_time = time.time() - start_time

    # Analyze results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    print(f"Total Experiments: {len(results)}")
    print(f"Successful (PSNR > 25 dB): {len(successful)}")
    print(f"Failed (PSNR <= 25 dB): {len(failed)}")
    print(f"Success Rate: {len(successful)/len(results)*100:.1f}%")
    print(f"Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    # Statistics on successful experiments
    if successful:
        psnrs = [r['metrics']['psnr'] for r in successful]
        times = [r['metrics']['wall_time'] for r in successful]

        print(f"\nPSNR Statistics (successful experiments):")
        print(f"  Mean: {np.mean(psnrs):.2f} dB")
        print(f"  Median: {np.median(psnrs):.2f} dB")
        print(f"  Min: {np.min(psnrs):.2f} dB")
        print(f"  Max: {np.max(psnrs):.2f} dB")
        print(f"  Std: {np.std(psnrs):.2f} dB")

        print(f"\nTime Statistics:")
        print(f"  Mean: {np.mean(times):.2f}s")
        print(f"  Median: {np.median(times):.2f}s")

    # Save results
    results_file = output_dir / "results_summary.json"
    with open(results_file, 'w') as f:
        json.dump({
            'experiment_set': EXPERIMENT_SET_NAME,
            'total_experiments': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful)/len(results),
            'total_time': total_time,
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Generate report
    generate_set_1_report(results, output_dir)

    # Decision
    print("\n" + "="*80)
    print("DECISION POINT")
    print("="*80)

    success_rate = len(successful) / len(results)

    if success_rate >= 0.9:
        print("✓ >90% success rate - PROCEED to Set 2")
        decision = "proceed"
    elif success_rate >= 0.5:
        print("⚠ 50-90% success rate - DEBUG and tune parameters, then proceed")
        decision = "debug_then_proceed"
    else:
        print("✗ <50% success rate - STOP and report critical issues")
        decision = "stop"

    print("="*80)

    return decision, results


def generate_set_1_report(results: list, output_dir: Path):
    """Generate markdown report for Set 1"""

    report_lines = [
        "# Experiment Set 1: Edge Baseline Report",
        "",
        "## Summary",
        "",
        f"- **Total Experiments:** {len(results)}",
        f"- **Successful:** {len([r for r in results if r.get('success', False)])}",
        f"- **Failed:** {len([r for r in results if not r.get('success', False)])}",
        f"- **Success Rate:** {len([r for r in results if r.get('success', False)])/len(results)*100:.1f}%",
        "",
        "## Strategy",
        "",
        "- **Initialization:** E1 (uniform spacing along edge)",
        f"- **Optimizer:** {OPT_CONFIG['optimizer_type']}",
        f"- **Learning Rate:** {OPT_CONFIG['learning_rate']}",
        f"- **Max Iterations:** {OPT_CONFIG['max_iterations']}",
        "",
        "## Results by Test Case",
        "",
        "| Test Case | N=5 | N=10 | N=20 | N=40 |",
        "|-----------|-----|------|------|------|"
    ]

    # Group by test case
    for test_case in TEST_CASES:
        test_case_id = test_case['id']
        row = f"| {test_case_id} |"

        for N in N_VALUES:
            result = next((r for r in results if r.get('test_case_id') == test_case_id and r.get('N') == N), None)
            if result and 'metrics' in result:
                psnr = result['metrics']['psnr']
                row += f" {psnr:.1f} |"
            else:
                row += " ERR |"

        report_lines.append(row)

    report_lines.extend([
        "",
        "## Key Findings",
        "",
        "1. Edge baseline performance established",
        f"2. Success rate: {len([r for r in results if r.get('success', False)])/len(results)*100:.1f}%",
        "3. Ready for initialization strategy comparison (Set 2)",
        "",
        "## Next Steps",
        "",
        "- Proceed to Experiment Set 2: Edge Initialization Strategies",
        ""
    ])

    report_file = output_dir / "report.md"
    with open(report_file, 'w') as f:
        f.write("\n".join(report_lines))

    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    decision, results = run_experiment_set_1()
    sys.exit(0 if decision in ['proceed', 'debug_then_proceed'] else 1)
