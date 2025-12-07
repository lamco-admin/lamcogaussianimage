"""
Phase 0 Sweep 5: Verification

Test empirical rules across all 12 test cases
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent / "experiments/gaussian_primitives_sprint/infrastructure"))
from gaussian_primitives import Gaussian2D, GaussianRenderer, compute_metrics


def load_test_cases():
    """Load all test cases"""
    import matplotlib.image as mpimg

    test_images_dir = Path("phase_0_results/test_images")
    test_cases = []

    case_files = sorted(test_images_dir.glob("case_*.png"))

    for img_path in case_files:
        case_id = img_path.stem
        img = mpimg.imread(img_path)

        if len(img.shape) == 3:
            img = img[:, :, 0]

        # Parse parameters from filename
        if 'blur1' in case_id:
            blur_sigma = 1.0
        elif 'blur2' in case_id:
            blur_sigma = 2.0
        elif 'blur4' in case_id:
            blur_sigma = 4.0
        else:
            blur_sigma = 0.0

        if 'verylow' in case_id:
            contrast = 0.1
        elif 'contrast_low' in case_id:
            contrast = 0.2
        elif 'contrast_medium' in case_id:
            contrast = 0.5
        elif 'contrast_high' in case_id:
            contrast = 0.8
        else:
            contrast = 0.5

        if 'horizontal' in case_id:
            orientation = 0.0
        elif 'diagonal' in case_id:
            orientation = np.pi/4
        else:
            orientation = np.pi/2

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


def place_edge_gaussians(
    edge_descriptor: Dict,
    N: int = 10,
    sigma_perp: float = 1.0,
    sigma_parallel: float = 5.0,
    spacing: float = 2.5,
    alpha: float = 1.0
) -> List[Gaussian2D]:
    """Place edge Gaussians"""

    h, w = 100, 100
    gaussians = []

    orientation = edge_descriptor['orientation']
    contrast = edge_descriptor['contrast']

    cx, cy = w / 2, h / 2
    edge_tangent = orientation

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


def empirical_rules_v1(blur_sigma: float, contrast: float) -> Dict:
    """
    Apply empirical rules discovered from Sweeps 1-4

    Based on findings:
    - σ_perp ≈ 1.0 (appears constant)
    - σ_parallel ≈ 10.0 (larger is better)
    - spacing ≈ 5.0 (larger spacing with large σ_parallel)
    - alpha ≈ 0.3 × (1.0 / contrast) (inverse relationship observed)
    """

    sigma_perp = 1.0  # Constant, works for all blur levels
    sigma_parallel = 10.0  # Large value for good coverage
    spacing = 5.0  # Large spacing to avoid over-accumulation

    # Alpha inversely related to contrast
    # Low contrast needs higher alpha, high contrast needs lower alpha
    alpha = 0.3 / contrast if contrast > 0 else 0.3

    return {
        'sigma_perp': sigma_perp,
        'sigma_parallel': sigma_parallel,
        'spacing': spacing,
        'alpha': alpha
    }


def main():
    print("="*60)
    print("PHASE 0 SWEEP 5: VERIFICATION")
    print("="*60)

    output_dir = Path("phase_0_results/sweep_5_verification")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_cases = load_test_cases()
    print(f"✓ Loaded {len(test_cases)} test cases")

    renderer = GaussianRenderer()
    results = []

    print("\nTesting empirical rules on all cases:")
    print()

    for test_case in test_cases:
        case_id = test_case['id']
        target = test_case['image']
        descriptor = test_case['descriptor']
        blur_sigma = test_case['blur_sigma']
        contrast = test_case['contrast']

        # Apply empirical rules
        params = empirical_rules_v1(blur_sigma, contrast)

        print(f"{case_id}:")
        print(f"  blur={blur_sigma:.1f}, contrast={contrast:.1f}")
        print(f"  σ_perp={params['sigma_perp']:.2f}, σ_parallel={params['sigma_parallel']:.2f}")
        print(f"  spacing={params['spacing']:.2f}, alpha={params['alpha']:.3f}")

        # Place and render
        gaussians = place_edge_gaussians(
            edge_descriptor=descriptor,
            N=10,
            **params
        )

        rendered = renderer.render_accumulate(gaussians, 100, 100, channels=1)

        # Compute metrics
        metrics = compute_metrics(target, rendered)

        print(f"  PSNR={metrics['psnr']:.2f} dB, MSE={metrics['mse']:.6f}")
        print()

        # Save images
        plt.imsave(
            output_dir / f"{case_id}_rendered.png",
            rendered, cmap='gray', vmin=0, vmax=1
        )

        # Save comparison
        residual = np.abs(target - rendered)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(target, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title('Target')
        axes[0].axis('off')
        axes[1].imshow(rendered, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Rendered')
        axes[1].axis('off')
        axes[2].imshow(residual, cmap='hot', vmin=0, vmax=0.5)
        axes[2].set_title('Residual')
        axes[2].axis('off')
        plt.suptitle(f"{case_id} | PSNR={metrics['psnr']:.2f} dB")
        plt.tight_layout()
        plt.savefig(output_dir / f"{case_id}_comparison.png", dpi=100, bbox_inches='tight')
        plt.close()

        # Record result
        result = {
            'case_id': case_id,
            'blur_sigma': blur_sigma,
            'contrast': contrast,
            **params,
            **metrics
        }
        results.append(result)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'verification_results.csv', index=False)

    print("="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    print(f"\nTotal cases: {len(results)}")
    print(f"Mean PSNR: {df['psnr'].mean():.2f} dB")
    print(f"Std PSNR: {df['psnr'].std():.2f} dB")
    print(f"Min PSNR: {df['psnr'].min():.2f} dB ({df.loc[df['psnr'].idxmin(), 'case_id']})")
    print(f"Max PSNR: {df['psnr'].max():.2f} dB ({df.loc[df['psnr'].idxmax(), 'case_id']})")

    # Analyze by blur level
    print("\nBy blur level:")
    for blur in sorted(df['blur_sigma'].unique()):
        blur_data = df[df['blur_sigma'] == blur]
        print(f"  σ_edge={blur:.1f}: PSNR={blur_data['psnr'].mean():.2f} ± {blur_data['psnr'].std():.2f} dB")

    # Analyze by contrast
    print("\nBy contrast:")
    for contrast in sorted(df['contrast'].unique()):
        contrast_data = df[df['contrast'] == contrast]
        print(f"  ΔI={contrast:.1f}: PSNR={contrast_data['psnr'].mean():.2f} ± {contrast_data['psnr'].std():.2f} dB")

    print(f"\n✓ Results saved to {output_dir}")


if __name__ == "__main__":
    main()
