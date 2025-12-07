"""
Phase 0: Edge Gaussian Function Discovery
Manual parameter exploration to discover empirical rules for f_edge

NO OPTIMIZATION - just rendering with explicit parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple
import json
import sys

# Add infrastructure to path
sys.path.insert(0, str(Path(__file__).parent / "experiments/gaussian_primitives_sprint/infrastructure"))

from gaussian_primitives import Gaussian2D, GaussianRenderer, SyntheticDataGenerator, compute_metrics


class Phase0TestCorpus:
    """Generate the 12 specific test cases for Phase 0"""

    @staticmethod
    def generate_all(output_dir: Path):
        """Generate all 12 test images and save them"""
        output_dir.mkdir(parents=True, exist_ok=True)

        test_cases = []

        # Set A: Sharp edges, varying contrast (4 images)
        print("Generating Set A: Sharp edges, varying contrast...")

        # Case 1: Vertical, sharp, ΔI=0.2
        img, desc = SyntheticDataGenerator.generate_edge(
            image_size=(100, 100),
            edge_type='straight',
            blur_sigma=0.0,
            contrast=0.2,
            orientation=np.pi/2  # vertical
        )
        case_id = "case_01_sharp_contrast_low"
        test_cases.append({
            'id': case_id,
            'image': img,
            'descriptor': desc,
            'blur_sigma': 0.0,
            'contrast': 0.2,
            'orientation': 'vertical'
        })
        plt.imsave(output_dir / f"{case_id}.png", img, cmap='gray', vmin=0, vmax=1)

        # Case 2: Vertical, sharp, ΔI=0.5
        img, desc = SyntheticDataGenerator.generate_edge(
            image_size=(100, 100),
            edge_type='straight',
            blur_sigma=0.0,
            contrast=0.5,
            orientation=np.pi/2
        )
        case_id = "case_02_sharp_contrast_medium"
        test_cases.append({
            'id': case_id,
            'image': img,
            'descriptor': desc,
            'blur_sigma': 0.0,
            'contrast': 0.5,
            'orientation': 'vertical'
        })
        plt.imsave(output_dir / f"{case_id}.png", img, cmap='gray', vmin=0, vmax=1)

        # Case 3: Vertical, sharp, ΔI=0.8
        img, desc = SyntheticDataGenerator.generate_edge(
            image_size=(100, 100),
            edge_type='straight',
            blur_sigma=0.0,
            contrast=0.8,
            orientation=np.pi/2
        )
        case_id = "case_03_sharp_contrast_high"
        test_cases.append({
            'id': case_id,
            'image': img,
            'descriptor': desc,
            'blur_sigma': 0.0,
            'contrast': 0.8,
            'orientation': 'vertical'
        })
        plt.imsave(output_dir / f"{case_id}.png", img, cmap='gray', vmin=0, vmax=1)

        # Case 4: Diagonal (45°), sharp, ΔI=0.5
        img, desc = SyntheticDataGenerator.generate_edge(
            image_size=(100, 100),
            edge_type='straight',
            blur_sigma=0.0,
            contrast=0.5,
            orientation=np.pi/4  # 45 degrees
        )
        case_id = "case_04_sharp_diagonal"
        test_cases.append({
            'id': case_id,
            'image': img,
            'descriptor': desc,
            'blur_sigma': 0.0,
            'contrast': 0.5,
            'orientation': 'diagonal'
        })
        plt.imsave(output_dir / f"{case_id}.png", img, cmap='gray', vmin=0, vmax=1)

        # Set B: Varying blur, fixed contrast (4 images)
        print("Generating Set B: Varying blur, fixed contrast...")

        # Case 5: Vertical, blur σ=1px, ΔI=0.5
        img, desc = SyntheticDataGenerator.generate_edge(
            image_size=(100, 100),
            edge_type='straight',
            blur_sigma=1.0,
            contrast=0.5,
            orientation=np.pi/2
        )
        case_id = "case_05_blur1_contrast_medium"
        test_cases.append({
            'id': case_id,
            'image': img,
            'descriptor': desc,
            'blur_sigma': 1.0,
            'contrast': 0.5,
            'orientation': 'vertical'
        })
        plt.imsave(output_dir / f"{case_id}.png", img, cmap='gray', vmin=0, vmax=1)

        # Case 6: Vertical, blur σ=2px, ΔI=0.5
        img, desc = SyntheticDataGenerator.generate_edge(
            image_size=(100, 100),
            edge_type='straight',
            blur_sigma=2.0,
            contrast=0.5,
            orientation=np.pi/2
        )
        case_id = "case_06_blur2_contrast_medium"
        test_cases.append({
            'id': case_id,
            'image': img,
            'descriptor': desc,
            'blur_sigma': 2.0,
            'contrast': 0.5,
            'orientation': 'vertical'
        })
        plt.imsave(output_dir / f"{case_id}.png", img, cmap='gray', vmin=0, vmax=1)

        # Case 7: Vertical, blur σ=4px, ΔI=0.5
        img, desc = SyntheticDataGenerator.generate_edge(
            image_size=(100, 100),
            edge_type='straight',
            blur_sigma=4.0,
            contrast=0.5,
            orientation=np.pi/2
        )
        case_id = "case_07_blur4_contrast_medium"
        test_cases.append({
            'id': case_id,
            'image': img,
            'descriptor': desc,
            'blur_sigma': 4.0,
            'contrast': 0.5,
            'orientation': 'vertical'
        })
        plt.imsave(output_dir / f"{case_id}.png", img, cmap='gray', vmin=0, vmax=1)

        # Case 8: Vertical, blur σ=2px, ΔI=0.8
        img, desc = SyntheticDataGenerator.generate_edge(
            image_size=(100, 100),
            edge_type='straight',
            blur_sigma=2.0,
            contrast=0.8,
            orientation=np.pi/2
        )
        case_id = "case_08_blur2_contrast_high"
        test_cases.append({
            'id': case_id,
            'image': img,
            'descriptor': desc,
            'blur_sigma': 2.0,
            'contrast': 0.8,
            'orientation': 'vertical'
        })
        plt.imsave(output_dir / f"{case_id}.png", img, cmap='gray', vmin=0, vmax=1)

        # Set C: Additional variations (4 images)
        print("Generating Set C: Additional variations...")

        # Case 9: Horizontal, blur σ=2px, ΔI=0.5
        img, desc = SyntheticDataGenerator.generate_edge(
            image_size=(100, 100),
            edge_type='straight',
            blur_sigma=2.0,
            contrast=0.5,
            orientation=0.0  # horizontal
        )
        case_id = "case_09_horizontal_blur2"
        test_cases.append({
            'id': case_id,
            'image': img,
            'descriptor': desc,
            'blur_sigma': 2.0,
            'contrast': 0.5,
            'orientation': 'horizontal'
        })
        plt.imsave(output_dir / f"{case_id}.png", img, cmap='gray', vmin=0, vmax=1)

        # Case 10: Vertical, blur σ=1px, ΔI=0.2
        img, desc = SyntheticDataGenerator.generate_edge(
            image_size=(100, 100),
            edge_type='straight',
            blur_sigma=1.0,
            contrast=0.2,
            orientation=np.pi/2
        )
        case_id = "case_10_blur1_contrast_low"
        test_cases.append({
            'id': case_id,
            'image': img,
            'descriptor': desc,
            'blur_sigma': 1.0,
            'contrast': 0.2,
            'orientation': 'vertical'
        })
        plt.imsave(output_dir / f"{case_id}.png", img, cmap='gray', vmin=0, vmax=1)

        # Case 11: Vertical, blur σ=4px, ΔI=0.2
        img, desc = SyntheticDataGenerator.generate_edge(
            image_size=(100, 100),
            edge_type='straight',
            blur_sigma=4.0,
            contrast=0.2,
            orientation=np.pi/2
        )
        case_id = "case_11_blur4_contrast_low"
        test_cases.append({
            'id': case_id,
            'image': img,
            'descriptor': desc,
            'blur_sigma': 4.0,
            'contrast': 0.2,
            'orientation': 'vertical'
        })
        plt.imsave(output_dir / f"{case_id}.png", img, cmap='gray', vmin=0, vmax=1)

        # Case 12: Vertical, sharp, ΔI=0.1 (challenging)
        img, desc = SyntheticDataGenerator.generate_edge(
            image_size=(100, 100),
            edge_type='straight',
            blur_sigma=0.0,
            contrast=0.1,
            orientation=np.pi/2
        )
        case_id = "case_12_sharp_contrast_verylow"
        test_cases.append({
            'id': case_id,
            'image': img,
            'descriptor': desc,
            'blur_sigma': 0.0,
            'contrast': 0.1,
            'orientation': 'vertical'
        })
        plt.imsave(output_dir / f"{case_id}.png", img, cmap='gray', vmin=0, vmax=1)

        print(f"✓ Generated {len(test_cases)} test images")
        return test_cases


class EdgeGaussianPlacer:
    """Manual placement of Gaussians along edge with explicit parameters"""

    @staticmethod
    def place_uniform(
        edge_descriptor: Dict,
        N: int = 10,
        sigma_perp: float = 1.0,
        sigma_parallel: float = 5.0,
        spacing: float = 2.5,
        alpha: float = 0.5,
        image_size: Tuple[int, int] = (100, 100)
    ) -> List[Gaussian2D]:
        """
        Place N Gaussians uniformly along edge with explicit parameters

        This is NOT the EdgeInitializer - this is manual placement for Phase 0
        where we explicitly control all parameters to discover empirical rules.
        """
        h, w = image_size
        gaussians = []

        # Extract edge properties
        orientation = edge_descriptor.get('orientation', np.pi/2)
        contrast = edge_descriptor.get('contrast', 0.5)

        # Edge is always centered in our test images
        cx, cy = w / 2, h / 2

        # Compute edge direction (tangent to edge)
        # For a vertical edge (orientation=π/2), tangent points up (π/2)
        # For a horizontal edge (orientation=0), tangent points right (0)
        edge_tangent = orientation

        # Total length covered by Gaussians
        total_length = (N - 1) * spacing if N > 1 else 0

        # Place Gaussians uniformly along edge
        for i in range(N):
            # Position along edge (centered)
            if N > 1:
                t = (i - (N-1)/2) * spacing
            else:
                t = 0

            # Position on edge (perpendicular to edge normal)
            # Edge normal is perpendicular to tangent
            # If edge orientation is θ, tangent is θ, normal is θ - π/2
            x = cx - t * np.sin(orientation)  # Move along edge
            y = cy + t * np.cos(orientation)

            # Ensure in bounds
            x = np.clip(x, 5, w-5)
            y = np.clip(y, 5, h-5)

            # Gaussian orientation: aligned with edge (major axis along edge)
            theta = edge_tangent

            # Color: average of both sides (for now, use half of contrast)
            # In reality, we'd sample both sides of the edge
            color_value = contrast / 2

            gaussians.append(Gaussian2D(
                x=float(x),
                y=float(y),
                sigma_parallel=sigma_parallel,  # along edge
                sigma_perp=sigma_perp,  # across edge
                theta=float(theta),
                color=np.array([color_value]),
                alpha=alpha,
                layer_type='E'
            ))

        return gaussians


class Phase0Sweeper:
    """Execute parameter sweeps and collect results"""

    def __init__(self, output_dir: Path, test_cases: List[Dict]):
        self.output_dir = output_dir
        self.test_cases = test_cases
        self.renderer = GaussianRenderer()
        self.results = []

    def run_sweep(
        self,
        sweep_name: str,
        test_case_ids: List[str],
        param_name: str,
        param_values: List[float],
        fixed_params: Dict
    ):
        """Execute a parameter sweep"""

        sweep_dir = self.output_dir / sweep_name
        sweep_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Executing {sweep_name}: {param_name}")
        print(f"{'='*60}")
        print(f"Test cases: {test_case_ids}")
        print(f"Parameter values: {param_values}")
        print(f"Fixed parameters: {fixed_params}")

        sweep_results = []

        for case_id in test_case_ids:
            # Find test case
            test_case = next(tc for tc in self.test_cases if tc['id'] == case_id)
            target_image = test_case['image']
            descriptor = test_case['descriptor']

            print(f"\n  {case_id}:")

            for param_value in param_values:
                # Combine fixed params with current param value
                params = fixed_params.copy()
                params[param_name] = param_value

                # Place Gaussians
                gaussians = EdgeGaussianPlacer.place_uniform(
                    edge_descriptor=descriptor,
                    N=10,
                    **params
                )

                # Render
                rendered = self.renderer.render(gaussians, 100, 100, channels=1)

                # Compute metrics
                metrics = compute_metrics(target_image, rendered)

                # Save images
                output_prefix = f"{case_id}_{param_name}_{param_value:.3f}"

                # Save rendered image
                plt.imsave(
                    sweep_dir / f"{output_prefix}.png",
                    rendered, cmap='gray', vmin=0, vmax=1
                )

                # Save comparison (target | rendered | residual)
                residual = np.abs(target_image - rendered)
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(target_image, cmap='gray', vmin=0, vmax=1)
                axes[0].set_title('Target')
                axes[0].axis('off')
                axes[1].imshow(rendered, cmap='gray', vmin=0, vmax=1)
                axes[1].set_title('Rendered')
                axes[1].axis('off')
                axes[2].imshow(residual, cmap='hot', vmin=0, vmax=0.5)
                axes[2].set_title('Residual (abs)')
                axes[2].axis('off')
                plt.suptitle(f"{case_id} | {param_name}={param_value:.3f} | PSNR={metrics['psnr']:.2f} dB")
                plt.tight_layout()
                plt.savefig(sweep_dir / f"{output_prefix}_comparison.png", dpi=100, bbox_inches='tight')
                plt.close()

                # Record result
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
                sweep_results.append(result)
                self.results.append(result)

                print(f"    {param_name}={param_value:.3f}: PSNR={metrics['psnr']:.2f} dB, MSE={metrics['mse']:.6f}")

        # Save sweep results to CSV
        df = pd.DataFrame(sweep_results)
        df.to_csv(sweep_dir / 'results.csv', index=False)

        # Generate sweep plot
        self._generate_sweep_plot(sweep_results, sweep_dir, param_name)

        # Generate atlas
        self._generate_atlas(sweep_results, sweep_dir, param_name, param_values, test_case_ids)

        print(f"\n✓ Sweep completed: {len(sweep_results)} renders")

        return sweep_results

    def _generate_sweep_plot(self, sweep_results: List[Dict], sweep_dir: Path, param_name: str):
        """Generate PSNR vs parameter plot"""

        df = pd.DataFrame(sweep_results)

        fig, ax = plt.subplots(figsize=(10, 6))

        for case_id in df['case_id'].unique():
            case_data = df[df['case_id'] == case_id]
            ax.plot(
                case_data['param_value'],
                case_data['psnr'],
                marker='o',
                label=case_id,
                linewidth=2
            )

        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel('PSNR (dB)', fontsize=12)
        ax.set_title(f'Sweep: {param_name}', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(sweep_dir / 'psnr_plot.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _generate_atlas(
        self,
        sweep_results: List[Dict],
        sweep_dir: Path,
        param_name: str,
        param_values: List[float],
        test_case_ids: List[str]
    ):
        """Generate visual atlas (grid of thumbnails)"""

        n_cases = len(test_case_ids)
        n_params = len(param_values)

        fig, axes = plt.subplots(n_cases, n_params, figsize=(n_params * 2, n_cases * 2))

        if n_cases == 1:
            axes = axes.reshape(1, -1)
        if n_params == 1:
            axes = axes.reshape(-1, 1)

        for i, case_id in enumerate(test_case_ids):
            for j, param_value in enumerate(param_values):
                # Load rendered image
                img_path = sweep_dir / f"{case_id}_{param_name}_{param_value:.3f}.png"

                if img_path.exists():
                    img = plt.imread(img_path)
                    axes[i, j].imshow(img, cmap='gray')

                    # Find PSNR
                    result = next(
                        (r for r in sweep_results
                         if r['case_id'] == case_id and r['param_value'] == param_value),
                        None
                    )

                    if result:
                        psnr = result['psnr']
                        axes[i, j].set_title(f"{param_value:.2f}\nPSNR={psnr:.1f}", fontsize=8)

                axes[i, j].axis('off')

            # Add row label
            axes[i, 0].set_ylabel(case_id, fontsize=8, rotation=0, ha='right', va='center')

        plt.suptitle(f'Atlas: {param_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(sweep_dir / 'atlas.png', dpi=150, bbox_inches='tight')
        plt.close()

    def save_all_results(self):
        """Save all results to master CSV"""
        df = pd.DataFrame(self.results)
        df.to_csv(self.output_dir / 'sweep_results.csv', index=False)
        print(f"\n✓ All results saved to sweep_results.csv ({len(self.results)} renders)")


def main():
    """Execute Phase 0 protocol"""

    print("="*60)
    print("PHASE 0: EDGE GAUSSIAN FUNCTION DISCOVERY")
    print("="*60)
    print("Manual parameter exploration (NO optimization)")
    print("Goal: Discover empirical rules for f_edge")
    print()

    # Create output directory
    output_dir = Path("phase_0_results")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Generate test corpus
    print("\n" + "="*60)
    print("STEP 1: GENERATE TEST CORPUS")
    print("="*60)

    test_images_dir = output_dir / "test_images"
    test_cases = Phase0TestCorpus.generate_all(test_images_dir)

    # Initialize sweeper
    sweeper = Phase0Sweeper(output_dir, test_cases)

    # Step 2: Execute sweeps
    print("\n" + "="*60)
    print("STEP 2: EXECUTE PARAMETER SWEEPS")
    print("="*60)

    # Sweep 1: σ_perp (cross-edge width)
    sweep1_results = sweeper.run_sweep(
        sweep_name='sweep_1_sigma_perp',
        test_case_ids=['case_05_blur1_contrast_medium',
                      'case_06_blur2_contrast_medium',
                      'case_07_blur4_contrast_medium'],
        param_name='sigma_perp',
        param_values=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        fixed_params={
            'sigma_parallel': 5.0,
            'spacing': 2.5,
            'alpha': 0.5
        }
    )

    # Sweep 2: σ_parallel (along-edge spread)
    sweep2_results = sweeper.run_sweep(
        sweep_name='sweep_2_sigma_parallel',
        test_case_ids=['case_06_blur2_contrast_medium'],
        param_name='sigma_parallel',
        param_values=[3.0, 4.0, 5.0, 7.0, 10.0],
        fixed_params={
            'sigma_perp': 2.0,
            'spacing': 2.5,  # Will be updated per iteration
            'alpha': 0.5
        }
    )

    # Sweep 3: Spacing
    sweep3_results = sweeper.run_sweep(
        sweep_name='sweep_3_spacing',
        test_case_ids=['case_06_blur2_contrast_medium'],
        param_name='spacing',
        param_values=[1.25, 1.67, 2.5, 3.33, 5.0],  # σ_parallel × [1/4, 1/3, 1/2, 2/3, 1]
        fixed_params={
            'sigma_perp': 2.0,
            'sigma_parallel': 5.0,
            'alpha': 0.5
        }
    )

    # Sweep 4: Alpha (opacity)
    sweep4_results = sweeper.run_sweep(
        sweep_name='sweep_4_alpha',
        test_case_ids=['case_02_sharp_contrast_medium',
                      'case_03_sharp_contrast_high',
                      'case_10_blur1_contrast_low'],
        param_name='alpha',
        param_values=[0.1, 0.15, 0.2, 0.25, 0.3],  # Will relate to ΔI
        fixed_params={
            'sigma_perp': 1.0,
            'sigma_parallel': 5.0,
            'spacing': 2.5
        }
    )

    # Save all results
    sweeper.save_all_results()

    print("\n" + "="*60)
    print("PHASE 0 INFRASTRUCTURE COMPLETE")
    print("="*60)
    print(f"Total renders: {len(sweeper.results)}")
    print(f"Output directory: {output_dir.absolute()}")
    print()
    print("Next steps:")
    print("1. Run Sweep 5 (verification)")
    print("2. Analyze results and identify patterns")
    print("3. Write empirical_rules_v1.md")
    print("4. Write PHASE_0_REPORT.md")


if __name__ == "__main__":
    main()
