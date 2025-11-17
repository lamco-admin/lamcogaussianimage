"""
Phase 0: Corrected sweeps with better parameter ranges

Key fixes:
1. Use render_accumulate (like GaussianImage paper) instead of alpha compositing
2. Test much higher alpha values
3. Expand parameter ranges based on initial findings
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent / "experiments/gaussian_primitives_sprint/infrastructure"))
from gaussian_primitives import Gaussian2D, GaussianRenderer, compute_metrics


def load_test_cases():
    """Load the test cases from first run"""
    import matplotlib.image as mpimg

    test_images_dir = Path("phase_0_results/test_images")
    test_cases = []

    # Load metadata from CSV to get blur and contrast values (only for cases that were tested)
    df = pd.read_csv('phase_0_results/sweep_results.csv')

    case_files = sorted(test_images_dir.glob("case_*.png"))

    for img_path in case_files:
        case_id = img_path.stem
        img = mpimg.imread(img_path)

        # Ensure grayscale (H, W) shape
        if len(img.shape) == 3:
            img = img[:, :, 0]  # Take first channel

        # Try to find in CSV, otherwise parse from name
        case_rows = df[df['case_id'] == case_id]

        if len(case_rows) > 0:
            case_row = case_rows.iloc[0]
            blur_sigma = case_row['blur_sigma']
            contrast = case_row['contrast']
        else:
            # Parse from filename
            if 'blur1' in case_id:
                blur_sigma = 1.0
            elif 'blur2' in case_id:
                blur_sigma = 2.0
            elif 'blur4' in case_id:
                blur_sigma = 4.0
            else:
                blur_sigma = 0.0

            if 'contrast_low' in case_id or 'verylow' in case_id:
                contrast = 0.2 if 'contrast_low' in case_id else 0.1
            elif 'contrast_medium' in case_id:
                contrast = 0.5
            elif 'contrast_high' in case_id:
                contrast = 0.8
            else:
                contrast = 0.5

        # Determine orientation
        if 'horizontal' in case_id:
            orientation = 0.0
        elif 'diagonal' in case_id:
            orientation = np.pi/4
        else:
            orientation = np.pi/2  # vertical

        descriptor = {
            'orientation': orientation,
            'blur_sigma': blur_sigma,
            'contrast': contrast
        }

        test_cases.append({
            'id': case_id,
            'image': img,
            'descriptor': descriptor,
            'blur_sigma': blur_sigma,
            'contrast': contrast
        })

    return test_cases


def place_edge_gaussians_corrected(
    edge_descriptor: Dict,
    N: int = 10,
    sigma_perp: float = 1.0,
    sigma_parallel: float = 5.0,
    spacing: float = 2.5,
    alpha: float = 1.0,
    image_size: Tuple[int, int] = (100, 100)
) -> List[Gaussian2D]:
    """Place Gaussians with corrected color strategy"""

    h, w = image_size
    gaussians = []

    orientation = edge_descriptor.get('orientation', np.pi/2)
    contrast = edge_descriptor.get('contrast', 0.5)

    cx, cy = w / 2, h / 2
    edge_tangent = orientation

    total_length = (N - 1) * spacing if N > 1 else 0

    for i in range(N):
        if N > 1:
            t = (i - (N-1)/2) * spacing
        else:
            t = 0

        x = cx - t * np.sin(orientation)
        y = cy + t * np.cos(orientation)

        x = np.clip(x, 5, w-5)
        y = np.clip(y, 5, h-5)

        theta = edge_tangent

        # CORRECTED: Use full contrast as color value
        # The Gaussian spread creates the transition
        color_value = contrast

        gaussians.append(Gaussian2D(
            x=float(x),
            y=float(y),
            sigma_parallel=sigma_parallel,
            sigma_perp=sigma_perp,
            theta=float(theta),
            color=np.array([color_value]),
            alpha=alpha,
            layer_type='E'
        ))

    return gaussians


def run_corrected_sweep(
    sweep_name: str,
    test_cases: List[Dict],
    test_case_ids: List[str],
    param_name: str,
    param_values: List[float],
    fixed_params: Dict,
    output_dir: Path
):
    """Run a sweep with accumulative rendering"""

    sweep_dir = output_dir / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    renderer = GaussianRenderer()
    results = []

    print(f"\n{'='*60}")
    print(f"Corrected Sweep: {sweep_name}")
    print(f"{'='*60}")

    for case_id in test_case_ids:
        test_case = next(tc for tc in test_cases if tc['id'] == case_id)
        target_image = test_case['image']
        descriptor = test_case['descriptor']

        print(f"\n  {case_id}:")

        for param_value in param_values:
            params = fixed_params.copy()
            params[param_name] = param_value

            gaussians = place_edge_gaussians_corrected(
                edge_descriptor=descriptor,
                N=10,
                **params
            )

            # Use ACCUMULATIVE rendering
            rendered = renderer.render_accumulate(gaussians, 100, 100, channels=1)

            metrics = compute_metrics(target_image, rendered)

            output_prefix = f"{case_id}_{param_name}_{param_value:.3f}"

            plt.imsave(
                sweep_dir / f"{output_prefix}.png",
                rendered, cmap='gray', vmin=0, vmax=1
            )

            # Save comparison
            residual = np.abs(target_image - rendered)
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(target_image, cmap='gray', vmin=0, vmax=1)
            axes[0].set_title('Target')
            axes[0].axis('off')
            axes[1].imshow(rendered, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('Rendered')
            axes[1].axis('off')
            axes[2].imshow(residual, cmap='hot', vmin=0, vmax=0.5)
            axes[2].set_title('Residual')
            axes[2].axis('off')
            plt.suptitle(f"{case_id} | {param_name}={param_value:.3f} | PSNR={metrics['psnr']:.2f} dB")
            plt.tight_layout()
            plt.savefig(sweep_dir / f"{output_prefix}_comparison.png", dpi=100, bbox_inches='tight')
            plt.close()

            result = {
                'sweep': sweep_name,
                'case_id': case_id,
                'param_name': param_name,
                'param_value': param_value,
                **params,
                **metrics,
                'blur_sigma': test_case['blur_sigma'],
                'contrast': test_case['contrast']
            }
            results.append(result)

            print(f"    {param_name}={param_value:.3f}: PSNR={metrics['psnr']:.2f} dB")

    df = pd.DataFrame(results)
    df.to_csv(sweep_dir / 'results.csv', index=False)

    return results


def main():
    print("="*60)
    print("PHASE 0: CORRECTED PARAMETER SWEEPS")
    print("="*60)

    output_dir = Path("phase_0_results_corrected")
    output_dir.mkdir(exist_ok=True)

    # Load existing test cases
    test_cases = load_test_cases()
    print(f"✓ Loaded {len(test_cases)} test cases")

    all_results = []

    # Corrected Sweep 1: σ_perp with better alpha
    results = run_corrected_sweep(
        sweep_name='sweep_1_sigma_perp_corrected',
        test_cases=test_cases,
        test_case_ids=['case_05_blur1_contrast_medium',
                      'case_06_blur2_contrast_medium',
                      'case_07_blur4_contrast_medium'],
        param_name='sigma_perp',
        param_values=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        fixed_params={
            'sigma_parallel': 5.0,
            'spacing': 2.5,
            'alpha': 1.0  # INCREASED from 0.5
        },
        output_dir=output_dir
    )
    all_results.extend(results)

    # Corrected Sweep 4: Alpha with much higher range
    results = run_corrected_sweep(
        sweep_name='sweep_4_alpha_corrected',
        test_cases=test_cases,
        test_case_ids=['case_02_sharp_contrast_medium',
                      'case_03_sharp_contrast_high',
                      'case_10_blur1_contrast_low'],
        param_name='alpha',
        param_values=[0.5, 1.0, 1.5, 2.0, 3.0],  # MUCH HIGHER
        fixed_params={
            'sigma_perp': 2.0,
            'sigma_parallel': 5.0,
            'spacing': 2.5
        },
        output_dir=output_dir
    )
    all_results.extend(results)

    # Save all
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(output_dir / 'corrected_sweep_results.csv', index=False)

    # Analysis
    print("\n" + "="*60)
    print("CORRECTED RESULTS ANALYSIS")
    print("="*60)

    for sweep_name in df_all['sweep'].unique():
        sweep_data = df_all[df_all['sweep'] == sweep_name]
        param_name = sweep_data['param_name'].iloc[0]

        print(f"\n{sweep_name}:")

        for case_id in sweep_data['case_id'].unique():
            case_data = sweep_data[sweep_data['case_id'] == case_id]
            best_idx = case_data['psnr'].idxmax()
            best_row = case_data.loc[best_idx]

            print(f"  {case_id}:")
            print(f"    Best {param_name} = {best_row['param_value']:.3f}")
            print(f"    Best PSNR = {best_row['psnr']:.2f} dB")


if __name__ == "__main__":
    main()
