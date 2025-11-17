"""
Set 4 FAST VERSION - Reduced iterations for faster results
Focus on comparison, not perfect convergence
"""

import numpy as np
import sys
from pathlib import Path
import json
import time
from typing import List, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent / 'infrastructure'))

from gaussian_primitives import Gaussian2D, GaussianRenderer, SyntheticDataGenerator, compute_metrics
from initializers import EdgeInitializer, RegionInitializer, RandomInitializer
from optimizers import optimize_gaussians


# FAST PARAMETERS
MAX_ITERS_MONO = 50  # Reduced from 500
MAX_ITERS_LAYER = 30  # Per layer
K_VALUES = [50, 100]  # Reduced from [100, 200, 400]


def fit_monolithic_fast(target: np.ndarray, K_total: int) -> Tuple[List[Gaussian2D], Dict]:
    """Fast monolithic fitting"""
    h, w = target.shape[:2]
    gaussians_init = RandomInitializer.random(K_total, (h, w), seed=42)

    start = time.time()
    gaussians_final, opt_log = optimize_gaussians(
        gaussians_init=gaussians_init,
        target_image=target,
        optimizer_type='adam',
        learning_rate=0.1,  # Higher LR
        max_iterations=MAX_ITERS_MONO
    )
    wall_time = time.time() - start

    renderer = GaussianRenderer()
    rendered = renderer.render(gaussians_final, w, h, channels=1)
    metrics = compute_metrics(target, rendered)

    return gaussians_final, {
        'approach': 'monolithic',
        'wall_time': wall_time,
        'iterations': len(opt_log['loss_curve']),
        'metrics': metrics
    }


def fit_layered_fast(target: np.ndarray, K_total: int, scene_type: str) -> Tuple[List[Gaussian2D], Dict]:
    """Fast layered fitting"""
    h, w = target.shape[:2]
    renderer = GaussianRenderer()

    # Simple allocation
    if scene_type == 'simple':
        K_M, K_E, K_R = int(K_total * 0.2), int(K_total * 0.5), int(K_total * 0.3)
    else:
        K_M, K_E, K_R = int(K_total * 0.2), int(K_total * 0.4), int(K_total * 0.4)

    all_gaussians = []
    residual = target.copy()
    total_time = 0
    layer_info = {}

    # Layer M (Macro)
    if K_M > 0:
        gaussians_m = RandomInitializer.random(K_M, (h, w))
        for g in gaussians_m:
            g.sigma_parallel = 20.0
            g.sigma_perp = 20.0
            g.layer_type = 'M'

        start = time.time()
        gaussians_m, log_m = optimize_gaussians(
            gaussians_init=gaussians_m,
            target_image=residual,
            optimizer_type='adam',
            learning_rate=0.1,
            max_iterations=MAX_ITERS_LAYER
        )
        layer_time = time.time() - start
        all_gaussians.extend(gaussians_m)
        residual = target - renderer.render(all_gaussians, w, h, channels=1)
        total_time += layer_time
        layer_info['M'] = {'time': layer_time, 'K': K_M}

    # Layer E (Edges) - simplified
    if K_E > 0:
        gaussians_e = RandomInitializer.random(K_E, (h, w))
        for g in gaussians_e:
            g.sigma_parallel = 5.0
            g.sigma_perp = 1.5
            g.layer_type = 'E'

        start = time.time()
        gaussians_e, log_e = optimize_gaussians(
            gaussians_init=gaussians_e,
            target_image=residual,
            optimizer_type='adam',
            learning_rate=0.1,
            max_iterations=MAX_ITERS_LAYER
        )
        layer_time = time.time() - start
        all_gaussians.extend(gaussians_e)
        residual = target - renderer.render(all_gaussians, w, h, channels=1)
        total_time += layer_time
        layer_info['E'] = {'time': layer_time, 'K': K_E}

    # Layer R (Regions)
    if K_R > 0:
        gaussians_r = RegionInitializer.grid({'area': h*w}, K_R, (h, w))

        start = time.time()
        gaussians_r, log_r = optimize_gaussians(
            gaussians_init=gaussians_r,
            target_image=residual,
            optimizer_type='adam',
            learning_rate=0.1,
            max_iterations=MAX_ITERS_LAYER
        )
        layer_time = time.time() - start
        all_gaussians.extend(gaussians_r)
        total_time += layer_time
        layer_info['R'] = {'time': layer_time, 'K': K_R}

    rendered_final = renderer.render(all_gaussians, w, h, channels=1)
    metrics = compute_metrics(target, rendered_final)

    return all_gaussians, {
        'approach': 'layered_sequential',
        'wall_time': total_time,
        'iterations': MAX_ITERS_LAYER * len(layer_info),  # Approximate
        'layer_info': layer_info,
        'metrics': metrics
    }


# Create simple scenes
def create_scenes_fast():
    scenes = {}

    # Simple scene
    scene1 = np.zeros((100, 100), dtype=np.float32)
    y_coords, x_coords = np.meshgrid(np.arange(100), np.arange(100), indexing='ij')
    scene1 += 0.3 * (y_coords / 100)
    scene1[30:70, 30:70] = 0.7
    from scipy.ndimage import gaussian_filter
    scene1 = gaussian_filter(scene1, sigma=1.0)
    scenes['simple'] = scene1

    # Medium scene
    scene2 = np.zeros((100, 100), dtype=np.float32)
    cx, cy = 50, 50
    dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    scene2 += 0.4 * (1 - dist / 75)
    mask1 = (x_coords - 30)**2 + (y_coords - 30)**2 < 15**2
    scene2[mask1] = 0.8
    scene2 = gaussian_filter(scene2, sigma=0.8)
    scenes['medium'] = scene2

    return scenes


def main():
    print("\n" + "="*70)
    print("SET 4 FAST: LAYERED vs MONOLITHIC (CRITICAL)")
    print("Reduced iterations for faster results")
    print("="*70 + "\n")

    scenes = create_scenes_fast()
    results = []
    exp_id = 1

    for scene_name, target in scenes.items():
        print(f"\nSCENE: {scene_name}")
        for K in K_VALUES:
            print(f"  K={K}")

            # Monolithic
            print(f"    [{exp_id}] Monolithic...")
            try:
                _, result_mono = fit_monolithic_fast(target, K)
                result_mono.update({'scene': scene_name, 'K': K, 'success': True})
                results.append(result_mono)
                print(f"        PSNR: {result_mono['metrics']['psnr']:.2f} dB, Time: {result_mono['wall_time']:.1f}s")
            except Exception as e:
                print(f"        ERROR: {e}")
                results.append({'scene': scene_name, 'K': K, 'approach': 'monolithic', 'error': str(e), 'success': False})
            exp_id += 1

            # Layered
            print(f"    [{exp_id}] Layered...")
            try:
                _, result_layered = fit_layered_fast(target, K, scene_name)
                result_layered.update({'scene': scene_name, 'K': K, 'success': True})
                results.append(result_layered)
                print(f"        PSNR: {result_layered['metrics']['psnr']:.2f} dB, Time: {result_layered['wall_time']:.1f}s")
            except Exception as e:
                print(f"        ERROR: {e}")
                results.append({'scene': scene_name, 'K': K, 'approach': 'layered_sequential', 'error': str(e), 'success': False})
            exp_id += 1

    # Analysis
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    successful = [r for r in results if r.get('success', False)]
    print(f"\nSuccessful: {len(successful)}/{len(results)}")

    print("\n| Scene | K | Mono PSNR | Layered PSNR | Mono Time | Layered Time |")
    print("|-------|---|-----------|--------------|-----------|--------------|")

    for scene in scenes.keys():
        for K in K_VALUES:
            mono = next((r for r in successful if r['scene']==scene and r['K']==K and r['approach']=='monolithic'), None)
            layered = next((r for r in successful if r['scene']==scene and r['K']==K and r['approach']=='layered_sequential'), None)

            if mono and layered:
                print(f"| {scene} | {K} | {mono['metrics']['psnr']:.2f} dB | {layered['metrics']['psnr']:.2f} dB | {mono['wall_time']:.1f}s | {layered['wall_time']:.1f}s |")

    # Save
    output_dir = Path(__file__).parent / "set_4_layered_vs_monolithic"
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "results_fast.json", 'w') as f:
        json.dump({'results': results}, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'results_fast.json'}")
    print("="*70)


if __name__ == "__main__":
    main()
