"""
Experiment Set 4: LAYERED vs MONOLITHIC (CRITICAL)
Goal: Test core hypothesis - does layering improve efficiency?

This is THE MOST IMPORTANT experiment set
Allocate maximum resources here

Total: 36 experiments (3 scenes × 3 K_total × 4 approaches)
"""

import numpy as np
import sys
from pathlib import Path
import json
import time
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent / 'infrastructure'))

from gaussian_primitives import (
    Gaussian2D, GaussianRenderer, SyntheticDataGenerator, compute_metrics
)
from initializers import (
    EdgeInitializer, RegionInitializer, JunctionInitializer, BlobInitializer, RandomInitializer
)
from optimizers import optimize_gaussians


class LayeredApproach:
    """Implements layered Gaussian fitting approach"""

    @staticmethod
    def fit_monolithic(target: np.ndarray, K_total: int, seed: int = 42) -> Tuple[List[Gaussian2D], Dict]:
        """
        Approach A: MONOLITHIC
        - K_total Gaussians, random initialization
        - Single Adam optimization, max 500 iterations
        - No feature types
        """
        print(f"    Fitting monolithic (K={K_total})...")

        h, w = target.shape[:2]

        # Random initialization
        gaussians_init = RandomInitializer.random(K_total, image_size=(h, w), seed=seed)

        # Single optimization
        start_time = time.time()
        gaussians_final, opt_log = optimize_gaussians(
            gaussians_init=gaussians_init,
            target_image=target,
            optimizer_type='adam',
            learning_rate=0.05,
            max_iterations=500
        )
        wall_time = time.time() - start_time

        renderer = GaussianRenderer()
        rendered = renderer.render(gaussians_final, w, h, channels=1)
        metrics = compute_metrics(target, rendered)

        return gaussians_final, {
            'approach': 'monolithic',
            'K_total': K_total,
            'wall_time': wall_time,
            'iterations_total': len(opt_log['loss_curve']),
            'metrics': metrics,
            'converged': opt_log.get('converged', False)
        }

    @staticmethod
    def fit_layered_sequential(target: np.ndarray, K_total: int,
                               scene_type: str) -> Tuple[List[Gaussian2D], Dict]:
        """
        Approach B: LAYERED SEQUENTIAL
        - Distribute K_total across layers
        - Fit sequentially with residuals
        - Different optimizers per layer
        """
        print(f"    Fitting layered-sequential (K={K_total})...")

        h, w = target.shape[:2]
        renderer = GaussianRenderer()

        # Allocate Gaussians across layers based on scene complexity
        if scene_type == 'simple':
            allocation = {'M': 0.2, 'E': 0.6, 'R': 0.2}  # Background + edge + region
        elif scene_type == 'medium':
            allocation = {'M': 0.1, 'E': 0.3, 'R': 0.4, 'B': 0.2}
        else:  # complex
            allocation = {'M': 0.1, 'E': 0.3, 'J': 0.05, 'R': 0.4, 'B': 0.15}

        K_per_layer = {layer: max(1, int(K_total * frac)) for layer, frac in allocation.items()}

        all_gaussians = []
        residual = target.copy()
        total_time = 0
        total_iterations = 0
        layer_logs = {}

        # Layer M: Macro (large background Gaussians)
        if 'M' in K_per_layer:
            print(f"      Layer M (Macro): {K_per_layer['M']} Gaussians...")
            gaussians_m = RandomInitializer.random(K_per_layer['M'], (h, w))
            for g in gaussians_m:
                g.sigma_parallel = 20.0  # Large
                g.sigma_perp = 20.0
                g.layer_type = 'M'

            start = time.time()
            gaussians_m, log_m = optimize_gaussians(
                gaussians_init=gaussians_m,
                target_image=residual,
                optimizer_type='adam',  # Could use L-BFGS for few Gaussians
                learning_rate=0.05,
                max_iterations=200
            )
            layer_time = time.time() - start

            all_gaussians.extend(gaussians_m)
            rendered_m = renderer.render(all_gaussians, w, h, channels=1)
            residual = target - rendered_m

            total_time += layer_time
            total_iterations += len(log_m['loss_curve'])
            layer_logs['M'] = {'time': layer_time, 'iters': len(log_m['loss_curve'])}

        # Layer E: Edges (elongated Gaussians along boundaries)
        if 'E' in K_per_layer:
            print(f"      Layer E (Edges): {K_per_layer['E']} Gaussians...")

            # Simple edge detection (gradient peaks)
            from scipy import ndimage
            grad_y = ndimage.sobel(target, axis=0)
            grad_x = ndimage.sobel(target, axis=1)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)

            # Place Gaussians along high-gradient regions
            gaussians_e = []
            threshold = np.percentile(gradient_mag, 90)  # Top 10% gradients
            edge_points = np.argwhere(gradient_mag > threshold)

            if len(edge_points) > 0:
                # Sample K_E points
                indices = np.random.choice(len(edge_points), min(K_per_layer['E'], len(edge_points)), replace=False)
                sampled_points = edge_points[indices]

                for point in sampled_points:
                    y, x = point
                    # Estimate orientation from gradient
                    gx, gy = grad_x[y, x], grad_y[y, x]
                    theta = np.arctan2(gy, gx) + np.pi/2  # Perpendicular to gradient

                    gaussians_e.append(Gaussian2D(
                        x=float(x), y=float(y),
                        sigma_parallel=5.0,
                        sigma_perp=1.5,
                        theta=float(theta),
                        color=np.array([target[y, x]]),
                        alpha=1.0 / K_per_layer['E'],
                        layer_type='E'
                    ))

            if gaussians_e:
                start = time.time()
                gaussians_e, log_e = optimize_gaussians(
                    gaussians_init=gaussians_e,
                    target_image=residual,
                    optimizer_type='adam',
                    learning_rate=0.05,
                    max_iterations=300,
                    constraints={'fix_theta': True}  # Keep orientation fixed
                )
                layer_time = time.time() - start

                all_gaussians.extend(gaussians_e)
                rendered_cumulative = renderer.render(all_gaussians, w, h, channels=1)
                residual = target - rendered_cumulative

                total_time += layer_time
                total_iterations += len(log_e['loss_curve'])
                layer_logs['E'] = {'time': layer_time, 'iters': len(log_e['loss_curve'])}

        # Layer R: Regions (interior fill)
        if 'R' in K_per_layer:
            print(f"      Layer R (Regions): {K_per_layer['R']} Gaussians...")

            gaussians_r = RegionInitializer.grid({'area': h*w}, K_per_layer['R'], (h, w))

            start = time.time()
            gaussians_r, log_r = optimize_gaussians(
                gaussians_init=gaussians_r,
                target_image=residual,
                optimizer_type='adam',
                learning_rate=0.05,
                max_iterations=300
            )
            layer_time = time.time() - start

            all_gaussians.extend(gaussians_r)
            rendered_cumulative = renderer.render(all_gaussians, w, h, channels=1)
            residual = target - rendered_cumulative

            total_time += layer_time
            total_iterations += len(log_r['loss_curve'])
            layer_logs['R'] = {'time': layer_time, 'iters': len(log_r['loss_curve'])}

        # Layer B: Blobs
        if 'B' in K_per_layer:
            print(f"      Layer B (Blobs): {K_per_layer['B']} Gaussians...")

            gaussians_b = RandomInitializer.random(K_per_layer['B'], (h, w))
            for g in gaussians_b:
                g.sigma_parallel = 3.0  # Small
                g.sigma_perp = 3.0
                g.layer_type = 'B'

            start = time.time()
            gaussians_b, log_b = optimize_gaussians(
                gaussians_init=gaussians_b,
                target_image=residual,
                optimizer_type='adam',
                learning_rate=0.05,
                max_iterations=200
            )
            layer_time = time.time() - start

            all_gaussians.extend(gaussians_b)

            total_time += layer_time
            total_iterations += len(log_b['loss_curve'])
            layer_logs['B'] = {'time': layer_time, 'iters': len(log_b['loss_curve'])}

        # Final render
        rendered_final = renderer.render(all_gaussians, w, h, channels=1)
        metrics = compute_metrics(target, rendered_final)

        return all_gaussians, {
            'approach': 'layered_sequential',
            'K_total': len(all_gaussians),
            'wall_time': total_time,
            'iterations_total': total_iterations,
            'layer_logs': layer_logs,
            'metrics': metrics,
            'layers_used': list(K_per_layer.keys())
        }


def create_synthetic_scenes():
    """Create 3 synthetic test scenes with known ground truth"""

    scenes = {}

    # Scene 1: Simple (2 layers - background + edge + region)
    print("Creating Scene 1: Simple (background + rectangle with border)...")
    scene1 = np.zeros((200, 200), dtype=np.float32)

    # Background gradient
    y_coords, x_coords = np.meshgrid(np.arange(200), np.arange(200), indexing='ij')
    scene1 += 0.3 * (y_coords / 200)

    # Rectangle with border
    scene1[60:140, 60:140] = 0.7
    # Add edge blur
    from scipy.ndimage import gaussian_filter
    scene1 = gaussian_filter(scene1, sigma=1.0)

    scenes['simple'] = {
        'image': scene1,
        'description': 'Background gradient + rectangle',
        'expected_layers': ['M', 'E', 'R']
    }

    # Scene 2: Medium (4 layers)
    print("Creating Scene 2: Medium (gradient + curves + regions + blobs)...")
    scene2 = np.zeros((200, 200), dtype=np.float32)

    # Radial gradient background
    cx, cy = 100, 100
    dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    scene2 += 0.4 * (1 - dist / 150)

    # Add 2 circular regions
    mask1 = (x_coords - 60)**2 + (y_coords - 60)**2 < 30**2
    scene2[mask1] = 0.8

    mask2 = (x_coords - 140)**2 + (y_coords - 140)**2 < 25**2
    scene2[mask2] = 0.6

    # Add 5 small blobs
    for bx, by in [(50, 150), (150, 50), (100, 100), (75, 120), (125, 80)]:
        blob_mask = (x_coords - bx)**2 + (y_coords - by)**2 < 5**2
        scene2[blob_mask] = 0.9

    scene2 = gaussian_filter(scene2, sigma=0.8)

    scenes['medium'] = {
        'image': scene2,
        'description': 'Radial gradient + 2 regions + 5 blobs',
        'expected_layers': ['M', 'E', 'R', 'B']
    }

    # Scene 3: Complex (5 layers)
    print("Creating Scene 3: Complex (vignette + curve network + junctions + regions + spots)...")
    scene3 = np.zeros((200, 200), dtype=np.float32)

    # Vignette
    dist_norm = dist / 150
    scene3 += 0.5 * (1 - dist_norm**2)

    # Add grid pattern (creates junctions)
    for x in [50, 100, 150]:
        scene3[:, max(0, x-2):min(200, x+2)] = 0.7

    for y in [50, 100, 150]:
        scene3[max(0, y-2):min(200, y+2), :] = 0.7

    # Fill some grid cells
    scene3[50:100, 50:100] = 0.4
    scene3[100:150, 100:150] = 0.6

    # Add random spots
    for _ in range(15):
        bx, by = np.random.randint(10, 190, size=2)
        blob_mask = (x_coords - bx)**2 + (y_coords - by)**2 < 3**2
        scene3[blob_mask] = np.random.uniform(0.2, 0.9)

    scene3 = gaussian_filter(scene3, sigma=0.6)

    scenes['complex'] = {
        'image': scene3,
        'description': 'Vignette + grid pattern with junctions + filled regions + spots',
        'expected_layers': ['M', 'E', 'J', 'R', 'B']
    }

    return scenes


def run_experiment_set_4():
    """Run Set 4: Layered vs Monolithic comparison"""

    print("\n" + "="*80)
    print("EXPERIMENT SET 4: LAYERED vs MONOLITHIC (CRITICAL)")
    print("="*80)
    print("This is THE MOST IMPORTANT experiment set")
    print("Testing core hypothesis: Does layering improve efficiency?")
    print("="*80 + "\n")

    # Create scenes
    scenes = create_synthetic_scenes()

    # Gaussian budgets to test
    K_values = [100, 200, 400]

    # Approaches
    approaches = ['monolithic', 'layered_sequential']

    # Results storage
    all_results = []
    exp_id = 1

    total_experiments = len(scenes) * len(K_values) * len(approaches)
    print(f"Total experiments: {total_experiments}")
    print(f"Scenes: {list(scenes.keys())}")
    print(f"K values: {K_values}")
    print(f"Approaches: {approaches}\n")

    start_time_total = time.time()

    # Run experiments
    for scene_name, scene_data in scenes.items():
        print(f"\n{'='*70}")
        print(f"SCENE: {scene_name.upper()}")
        print(f"Description: {scene_data['description']}")
        print(f"{'='*70}\n")

        target = scene_data['image']

        for K_total in K_values:
            print(f"  K_total = {K_total}")

            for approach in approaches:
                print(f"    [{exp_id}/{total_experiments}] Approach: {approach}")

                try:
                    if approach == 'monolithic':
                        gaussians, result_log = LayeredApproach.fit_monolithic(
                            target, K_total, seed=42+exp_id
                        )
                    elif approach == 'layered_sequential':
                        gaussians, result_log = LayeredApproach.fit_layered_sequential(
                            target, K_total, scene_name
                        )

                    result = {
                        'exp_id': exp_id,
                        'scene': scene_name,
                        'K_total': K_total,
                        'approach': approach,
                        **result_log,
                        'success': True
                    }

                    print(f"      ✓ PSNR: {result_log['metrics']['psnr']:.2f} dB, Time: {result_log['wall_time']:.1f}s, Iters: {result_log['iterations_total']}")

                except Exception as e:
                    print(f"      ✗ ERROR: {e}")
                    result = {
                        'exp_id': exp_id,
                        'scene': scene_name,
                        'K_total': K_total,
                        'approach': approach,
                        'error': str(e),
                        'success': False
                    }

                all_results.append(result)
                exp_id += 1

            print()

    total_time = time.time() - start_time_total

    # Analysis
    print("\n" + "="*80)
    print("SET 4 ANALYSIS - CRITICAL FINDINGS")
    print("="*80)

    analyze_layered_vs_monolithic(all_results)

    # Save results
    output_dir = Path(__file__).parent / "set_4_layered_vs_monolithic"
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / "results_complete.json"
    with open(results_file, 'w') as f:
        json.dump({
            'experiment_set': 'set_4_layered_vs_monolithic_CRITICAL',
            'total_experiments': len(all_results),
            'total_time': total_time,
            'results': all_results
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Generate detailed report
    generate_set4_report(all_results, scenes, output_dir)

    print("="*80)

    return all_results


def analyze_layered_vs_monolithic(results: list):
    """Analyze and compare layered vs monolithic approaches"""

    successful = [r for r in results if r.get('success', False)]

    if not successful:
        print("⚠ No successful experiments to analyze")
        return

    print("\n## Comparison by Scene and K_total\n")

    scenes = set(r['scene'] for r in successful)
    K_values = sorted(set(r['K_total'] for r in successful))

    for scene in scenes:
        print(f"### Scene: {scene}")
        print(f"| K_total | Monolithic PSNR | Layered PSNR | Winner | Time Mono | Time Layered | Speedup |")
        print(f"|---------|----------------|--------------|--------|-----------|--------------|---------|")

        for K in K_values:
            mono = next((r for r in successful
                        if r['scene'] == scene and r['K_total'] == K and r['approach'] == 'monolithic'), None)
            layered = next((r for r in successful
                           if r['scene'] == scene and r['K_total'] == K and r['approach'] == 'layered_sequential'), None)

            if mono and layered:
                psnr_mono = mono['metrics']['psnr']
                psnr_layered = layered['metrics']['psnr']
                time_mono = mono['wall_time']
                time_layered = layered['wall_time']

                winner = "Layered" if psnr_layered > psnr_mono else "Mono" if psnr_mono > psnr_layered else "Tie"
                speedup = time_mono / time_layered if time_layered > 0 else 1.0

                print(f"| {K} | {psnr_mono:.2f} dB | {psnr_layered:.2f} dB | **{winner}** | {time_mono:.1f}s | {time_layered:.1f}s | {speedup:.2f}x |")

        print()

    # Overall verdict
    print("\n## OVERALL VERDICT\n")

    layered_wins = 0
    mono_wins = 0
    ties = 0

    for scene in scenes:
        for K in K_values:
            mono = next((r for r in successful
                        if r['scene'] == scene and r['K_total'] == K and r['approach'] == 'monolithic'), None)
            layered = next((r for r in successful
                           if r['scene'] == scene and r['K_total'] == K and r['approach'] == 'layered_sequential'), None)

            if mono and layered:
                psnr_diff = layered['metrics']['psnr'] - mono['metrics']['psnr']
                time_diff_pct = (mono['wall_time'] - layered['wall_time']) / mono['wall_time'] * 100

                # Decision criteria from protocol
                if psnr_diff > 1.0 or time_diff_pct > 20:
                    layered_wins += 1
                elif psnr_diff < -1.0 or time_diff_pct < -20:
                    mono_wins += 1
                else:
                    ties += 1

    print(f"Layered wins: {layered_wins}")
    print(f"Monolithic wins: {mono_wins}")
    print(f"Ties: {ties}")

    if layered_wins > mono_wins:
        print("\n**🎯 RESULT: LAYERED APPROACH WINS**")
        print("Layering provides measurable benefits in efficiency and/or quality.")
        print("✓ Validate core hypothesis")
        print("→ Proceed with layered architecture")
    elif mono_wins > layered_wins:
        print("\n**⚠ RESULT: MONOLITHIC APPROACH WINS**")
        print("Layering overhead not justified by benefits.")
        print("✗ Pivot to monolithic with feature-aware initialization")
    else:
        print("\n**⚙ RESULT: MIXED/TIE**")
        print("Layering helps in some scenarios but not others.")
        print("→ Analyze conditions where layering excels")


def generate_set4_report(results: list, scenes: dict, output_dir: Path):
    """Generate comprehensive report for Set 4"""

    report_lines = [
        "# Experiment Set 4: Layered vs Monolithic - CRITICAL RESULTS",
        "",
        "## Executive Summary",
        "",
        f"**Total Experiments:** {len(results)}",
        f"**Successful:** {len([r for r in results if r.get('success', False)])}",
        "",
        "**Research Question:** Does decomposing into layers (M/E/J/R/B) enable better",
        "optimization than monolithic Gaussian fitting?",
        "",
        "## Experimental Design",
        "",
        "**Scenes:**",
        ""
    ]

    for scene_name, scene_data in scenes.items():
        report_lines.append(f"- **{scene_name}:** {scene_data['description']}")

    report_lines.extend([
        "",
        "**Gaussian Budgets:** K_total ∈ {100, 200, 400}",
        "",
        "**Approaches:**",
        "1. **Monolithic:** Random init, single Adam optimization (500 iters)",
        "2. **Layered Sequential:** M→E→R→B residual fitting, specialized per layer",
        "",
        "(See detailed analysis above)",
        ""
    ])

    report_file = output_dir / "CRITICAL_REPORT.md"
    with open(report_file, 'w') as f:
        f.write("\n".join(report_lines))

    print(f"Critical report saved to: {report_file}")


if __name__ == "__main__":
    np.random.seed(42)  # Reproducibility
    results = run_experiment_set_4()
    print("\n✓ Set 4 (CRITICAL) complete!")
