"""
Phase 0.5: Edge Region Isolated Testing
Key change: Measure PSNR on edge strip ONLY (ignore background)

Experiments:
1. N Sweep (18 renders) - find optimal N
2. Parameter Refinement (13 renders) - re-sweep parameters at N_optimal
3. Coverage Analysis (5 measurements) - vary edge strip width
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


# ============================================================================
# MASKED METRICS FUNCTIONS (NEW FOR PHASE 0.5)
# ============================================================================

def create_edge_mask(image_size: Tuple[int, int],
                    edge_position: float,
                    edge_orientation: str,
                    strip_width: int = 20) -> np.ndarray:
    """
    Create binary mask for edge region

    Args:
        image_size: (height, width)
        edge_position: center position of edge (pixel coordinate)
        edge_orientation: 'vertical', 'horizontal', or 'diagonal'
        strip_width: width of strip to evaluate (pixels)

    Returns:
        Binary mask (True = edge region, False = ignore)
    """
    h, w = image_size
    mask = np.zeros((h, w), dtype=bool)

    half_width = strip_width // 2

    if edge_orientation == 'vertical':
        # Vertical edge: mask is horizontal strip around edge
        x_center = int(edge_position)
        mask[:, max(0, x_center-half_width):min(w, x_center+half_width)] = True
    elif edge_orientation == 'horizontal':
        # Horizontal edge: mask is vertical strip
        y_center = int(edge_position)
        mask[max(0, y_center-half_width):min(h, y_center+half_width), :] = True
    elif edge_orientation == 'diagonal':
        # Diagonal edge: create diagonal strip (approximate as rotated rectangle)
        # For 45-degree diagonal through center
        for i in range(h):
            for j in range(w):
                # Distance from point to diagonal line (y = x + offset)
                # For diagonal through center: y = x
                dist = abs(i - j) / np.sqrt(2)
                if dist <= half_width:
                    mask[i, j] = True

    return mask


def compute_metrics_masked(target: np.ndarray,
                          rendered: np.ndarray,
                          mask: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics only on masked region

    Args:
        target: Target image
        rendered: Rendered image
        mask: Binary mask (True = compute, False = ignore)

    Returns:
        Dict with mse, psnr, mae on masked region only
    """
    # Extract pixels in mask
    target_masked = target[mask]
    rendered_masked = rendered[mask]

    # Compute metrics
    mse = np.mean((target_masked - rendered_masked) ** 2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    mae = np.mean(np.abs(target_masked - rendered_masked))

    return {
        'mse': mse,
        'psnr': psnr,
        'mae': mae,
        'num_pixels': int(np.sum(mask))
    }


# ============================================================================
# GAUSSIAN PLACEMENT WITH VARIABLE N
# ============================================================================

def place_edge_gaussians_variable_N(edge_descriptor: Dict,
                                   N: int,
                                   sigma_perp: float,
                                   sigma_parallel: float,
                                   alpha: float,
                                   image_size: Tuple[int, int] = (100, 100)) -> List[Gaussian2D]:
    """
    Place N Gaussians uniformly along edge

    Args:
        edge_descriptor: Dict with 'position', 'orientation', 'contrast'
        N: Number of Gaussians to place
        sigma_perp: Cross-edge width
        sigma_parallel: Along-edge spread
        alpha: Opacity
        image_size: (height, width)

    Returns:
        List of Gaussian2D objects
    """
    gaussians = []

    edge_length = min(image_size)  # 100 pixels

    # Uniform spacing based on N
    # Avoid exact edges (5 to 95)
    positions = np.linspace(5, edge_length - 5, N)

    orientation = edge_descriptor['orientation']
    edge_pos = edge_descriptor['position']
    contrast = edge_descriptor['contrast']

    for i, pos in enumerate(positions):
        if orientation == 'vertical':
            x, y = edge_pos, pos
            theta = 0  # horizontal Gaussians (perpendicular to vertical edge)
        elif orientation == 'horizontal':
            x, y = pos, edge_pos
            theta = np.pi/2
        elif orientation == 'diagonal':
            # For diagonal, position along the diagonal line
            # Diagonal goes from (0,0) to (100,100)
            t = pos / edge_length
            x = t * edge_length
            y = t * edge_length
            theta = np.pi/4  # 45 degrees
        else:
            raise ValueError(f"Unknown orientation: {orientation}")

        g = Gaussian2D(
            x=x, y=y,
            sigma_parallel=sigma_parallel,
            sigma_perp=sigma_perp,
            theta=theta,
            color=np.array([contrast]),
            alpha=alpha,
            layer_type='E'
        )
        gaussians.append(g)

    return gaussians


# ============================================================================
# TEST CORPUS (SAME AS PHASE 0, BUT WITH EDGE MASKS)
# ============================================================================

class Phase05TestCorpus:
    """Generate test cases with edge masks"""

    @staticmethod
    def generate_test_case(case_id: str,
                          blur_sigma: float,
                          contrast: float,
                          orientation_str: str,
                          image_size: Tuple[int, int] = (100, 100)) -> Tuple[np.ndarray, Dict, np.ndarray]:
        """
        Generate a test case with edge mask

        Returns:
            (target_image, edge_descriptor, edge_mask)
        """
        # Map orientation string to angle
        orientation_map = {
            'vertical': np.pi/2,
            'horizontal': 0,
            'diagonal': np.pi/4
        }
        orientation_angle = orientation_map[orientation_str]

        # Generate edge image
        img, desc = SyntheticDataGenerator.generate_edge(
            image_size=image_size,
            edge_type='straight',
            blur_sigma=blur_sigma,
            contrast=contrast,
            orientation=orientation_angle
        )

        # Edge position is center for straight edges (50 for 100x100 image)
        edge_pos = image_size[1] / 2 if orientation_str == 'vertical' else image_size[0] / 2

        # Create edge mask (20px wide strip)
        mask = create_edge_mask(
            image_size=image_size,
            edge_position=edge_pos,
            edge_orientation=orientation_str,
            strip_width=20
        )

        # Add orientation string and position to descriptor
        desc['orientation'] = orientation_str
        desc['position'] = edge_pos

        return img, desc, mask


# ============================================================================
# EXPERIMENT 1: N SWEEP
# ============================================================================

def experiment_1_n_sweep(output_dir: Path):
    """
    Experiment 1: Find minimum N for edge strip PSNR > 30 dB

    Test cases: 3 representative (case_02, case_06, case_12)
    N values: [5, 10, 20, 50, 100, 200]
    Total: 18 renders
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: N SWEEP")
    print("="*80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Test cases (representative from Phase 0)
    test_cases = [
        {'id': 'case_02', 'blur': 0.0, 'contrast': 0.5, 'orientation': 'vertical'},
        {'id': 'case_06', 'blur': 2.0, 'contrast': 0.5, 'orientation': 'vertical'},
        {'id': 'case_12', 'blur': 0.0, 'contrast': 0.1, 'orientation': 'vertical'}
    ]

    # N values to test
    N_values = [5, 10, 20, 50, 100, 200]

    # Fixed parameters (from Phase 0 empirical rules)
    sigma_perp = 1.0
    sigma_parallel = 10.0

    results = []

    for tc in test_cases:
        print(f"\nTest case: {tc['id']} (blur={tc['blur']}, contrast={tc['contrast']})")

        # Generate test case
        target, desc, mask = Phase05TestCorpus.generate_test_case(
            case_id=tc['id'],
            blur_sigma=tc['blur'],
            contrast=tc['contrast'],
            orientation_str=tc['orientation']
        )

        for N in N_values:
            print(f"  N={N}...", end=' ')

            # Alpha from empirical rules: alpha = 0.3 / ΔI
            # BUT scale by N to avoid over-accumulation
            # Base alpha was calibrated for N=10
            alpha_base = 0.3 / tc['contrast'] if tc['contrast'] > 0 else 0.3
            alpha = alpha_base * (10.0 / N)  # Scale inversely with N

            # Place Gaussians
            gaussians = place_edge_gaussians_variable_N(
                edge_descriptor=desc,
                N=N,
                sigma_perp=sigma_perp,
                sigma_parallel=sigma_parallel,
                alpha=alpha
            )

            # Render
            rendered = GaussianRenderer.render_accumulate(
                gaussians,
                width=100,
                height=100,
                channels=1
            )

            # Compute metrics on edge strip only
            metrics_masked = compute_metrics_masked(target, rendered, mask)

            # Also compute full-image metrics for comparison
            metrics_full = compute_metrics(target, rendered)

            print(f"PSNR_strip={metrics_masked['psnr']:.2f} dB, PSNR_full={metrics_full['psnr']:.2f} dB")

            # Save result
            result = {
                'case_id': tc['id'],
                'blur_sigma': tc['blur'],
                'contrast': tc['contrast'],
                'orientation': tc['orientation'],
                'N': N,
                'sigma_perp': sigma_perp,
                'sigma_parallel': sigma_parallel,
                'alpha': alpha,
                'psnr_strip': metrics_masked['psnr'],
                'mse_strip': metrics_masked['mse'],
                'mae_strip': metrics_masked['mae'],
                'num_pixels_strip': metrics_masked['num_pixels'],
                'psnr_full': metrics_full['psnr'],
                'mse_full': metrics_full['mse'],
                'mae_full': metrics_full['mae']
            }
            results.append(result)

            # Save render for N=50, 100, 200 for visual inspection
            if N in [50, 100, 200]:
                vis_dir = output_dir / "renders"
                vis_dir.mkdir(exist_ok=True)
                plt.imsave(
                    vis_dir / f"{tc['id']}_N{N}.png",
                    rendered,
                    cmap='gray',
                    vmin=0,
                    vmax=1
                )

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "experiment_1_n_sweep_results.csv", index=False)

    print(f"\n✓ Experiment 1 complete. Results saved to {output_dir}")
    print(f"  Total renders: {len(results)}")

    return df


def analyze_experiment_1(df: pd.DataFrame, output_dir: Path) -> int:
    """
    Analyze Experiment 1 results and determine N_optimal

    Returns:
        N_optimal: Minimum N where PSNR > 30 dB
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1 ANALYSIS: Determine N_optimal")
    print("="*80)

    # Plot PSNR vs N for each test case
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: PSNR on edge strip vs N
    ax = axes[0]
    for case_id in df['case_id'].unique():
        case_data = df[df['case_id'] == case_id]
        ax.plot(case_data['N'], case_data['psnr_strip'], 'o-', label=case_id, linewidth=2, markersize=8)

    ax.axhline(30, color='red', linestyle='--', linewidth=2, label='Target (30 dB)')
    ax.set_xlabel('Number of Gaussians (N)', fontsize=12)
    ax.set_ylabel('PSNR on Edge Strip (dB)', fontsize=12)
    ax.set_title('Experiment 1: Edge Strip Quality vs N', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Plot 2: Comparison of strip vs full-image PSNR
    ax = axes[1]
    for case_id in df['case_id'].unique():
        case_data = df[df['case_id'] == case_id]
        ax.plot(case_data['N'], case_data['psnr_strip'], 'o-', label=f'{case_id} (strip)', linewidth=2)
        ax.plot(case_data['N'], case_data['psnr_full'], 's--', label=f'{case_id} (full)', alpha=0.6)

    ax.set_xlabel('Number of Gaussians (N)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Strip vs Full-Image PSNR', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / "experiment_1_n_sweep_analysis.png", dpi=150)
    print(f"✓ Saved plot: {output_dir / 'experiment_1_n_sweep_analysis.png'}")

    # Determine N_optimal
    # Find minimum N where mean PSNR across test cases > 30 dB
    mean_psnr_by_N = df.groupby('N')['psnr_strip'].mean()

    print("\nMean PSNR on edge strip by N:")
    for N, psnr in mean_psnr_by_N.items():
        print(f"  N={N:3d}: PSNR = {psnr:5.2f} dB")

    # Find N where PSNR > 30 dB
    candidates = mean_psnr_by_N[mean_psnr_by_N >= 30.0]

    if len(candidates) > 0:
        N_optimal = int(candidates.index[0])
        print(f"\n✓ N_optimal = {N_optimal} (first N with mean PSNR > 30 dB)")
    else:
        # If no N achieves 30 dB, use N where PSNR is highest
        N_optimal = int(mean_psnr_by_N.idxmax())
        max_psnr = mean_psnr_by_N.max()
        print(f"\n⚠ No N achieved 30 dB target")
        print(f"  Using N_optimal = {N_optimal} (highest mean PSNR = {max_psnr:.2f} dB)")

    return N_optimal


# ============================================================================
# EXPERIMENT 2: PARAMETER REFINEMENT
# ============================================================================

def experiment_2_parameter_refinement(N_optimal: int, output_dir: Path):
    """
    Experiment 2: Re-sweep parameters with sufficient N

    Use N_optimal from Experiment 1
    Test case: case_06 (blur=2px, contrast=0.5)

    Sweeps:
    A. σ_perp: [0.5, 1.0, 2.0, 3.0, 4.0] (5 renders)
    B. σ_parallel: [5, 10, 15, 20] (4 renders)
    C. Spacing: [σ_parallel/4, σ_parallel/3, σ_parallel/2, σ_parallel] (4 renders)

    Total: 13 renders
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT 2: PARAMETER REFINEMENT (N={N_optimal})")
    print("="*80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Test case
    target, desc, mask = Phase05TestCorpus.generate_test_case(
        case_id='case_06',
        blur_sigma=2.0,
        contrast=0.5,
        orientation_str='vertical'
    )

    results = []

    # Sweep A: σ_perp
    print("\nSweep A: σ_perp")
    sigma_perp_values = [0.5, 1.0, 2.0, 3.0, 4.0]

    for sigma_perp in sigma_perp_values:
        print(f"  σ_perp={sigma_perp}...", end=' ')

        gaussians = place_edge_gaussians_variable_N(
            edge_descriptor=desc,
            N=N_optimal,
            sigma_perp=sigma_perp,
            sigma_parallel=10.0,
            alpha=0.6
        )

        rendered = GaussianRenderer.render_accumulate(gaussians, 100, 100, 1)
        metrics = compute_metrics_masked(target, rendered, mask)

        print(f"PSNR={metrics['psnr']:.2f} dB")

        results.append({
            'sweep': 'A_sigma_perp',
            'N': N_optimal,
            'sigma_perp': sigma_perp,
            'sigma_parallel': 10.0,
            'spacing': 5.0,
            'alpha': 0.6,
            'psnr': metrics['psnr'],
            'mse': metrics['mse'],
            'mae': metrics['mae']
        })

    # Sweep B: σ_parallel
    print("\nSweep B: σ_parallel")
    sigma_parallel_values = [5, 10, 15, 20]

    for sigma_parallel in sigma_parallel_values:
        print(f"  σ_parallel={sigma_parallel}...", end=' ')

        gaussians = place_edge_gaussians_variable_N(
            edge_descriptor=desc,
            N=N_optimal,
            sigma_perp=1.0,
            sigma_parallel=sigma_parallel,
            alpha=0.6
        )

        rendered = GaussianRenderer.render_accumulate(gaussians, 100, 100, 1)
        metrics = compute_metrics_masked(target, rendered, mask)

        print(f"PSNR={metrics['psnr']:.2f} dB")

        results.append({
            'sweep': 'B_sigma_parallel',
            'N': N_optimal,
            'sigma_perp': 1.0,
            'sigma_parallel': sigma_parallel,
            'spacing': sigma_parallel / 2,
            'alpha': 0.6,
            'psnr': metrics['psnr'],
            'mse': metrics['mse'],
            'mae': metrics['mae']
        })

    # Sweep C: Spacing (vary placement density)
    print("\nSweep C: Spacing")
    sigma_parallel = 10.0
    spacing_factors = [0.25, 0.33, 0.5, 1.0]

    for factor in spacing_factors:
        spacing = sigma_parallel * factor
        print(f"  spacing={spacing:.2f} (σ_parallel × {factor})...", end=' ')

        # NOTE: Spacing is implicit in N (for uniform placement)
        # We approximate by varying N to achieve desired spacing
        # spacing ≈ edge_length / N
        # N ≈ edge_length / spacing
        N_adjusted = int(100 / spacing)

        gaussians = place_edge_gaussians_variable_N(
            edge_descriptor=desc,
            N=N_adjusted,
            sigma_perp=1.0,
            sigma_parallel=sigma_parallel,
            alpha=0.6
        )

        rendered = GaussianRenderer.render_accumulate(gaussians, 100, 100, 1)
        metrics = compute_metrics_masked(target, rendered, mask)

        print(f"PSNR={metrics['psnr']:.2f} dB (N={N_adjusted})")

        results.append({
            'sweep': 'C_spacing',
            'N': N_adjusted,
            'sigma_perp': 1.0,
            'sigma_parallel': sigma_parallel,
            'spacing': spacing,
            'alpha': 0.6,
            'psnr': metrics['psnr'],
            'mse': metrics['mse'],
            'mae': metrics['mae']
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "experiment_2_parameter_refinement_results.csv", index=False)

    print(f"\n✓ Experiment 2 complete. Results saved to {output_dir}")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Sweep A plot
    sweep_a = df[df['sweep'] == 'A_sigma_perp']
    axes[0].plot(sweep_a['sigma_perp'], sweep_a['psnr'], 'o-', linewidth=2, markersize=10)
    axes[0].set_xlabel('σ_perp (pixels)', fontsize=12)
    axes[0].set_ylabel('PSNR on Edge Strip (dB)', fontsize=12)
    axes[0].set_title(f'Sweep A: σ_perp (N={N_optimal})', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Sweep B plot
    sweep_b = df[df['sweep'] == 'B_sigma_parallel']
    axes[1].plot(sweep_b['sigma_parallel'], sweep_b['psnr'], 'o-', linewidth=2, markersize=10)
    axes[1].set_xlabel('σ_parallel (pixels)', fontsize=12)
    axes[1].set_ylabel('PSNR on Edge Strip (dB)', fontsize=12)
    axes[1].set_title(f'Sweep B: σ_parallel (N={N_optimal})', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Sweep C plot
    sweep_c = df[df['sweep'] == 'C_spacing']
    axes[2].plot(sweep_c['spacing'], sweep_c['psnr'], 'o-', linewidth=2, markersize=10)
    axes[2].set_xlabel('Spacing (pixels)', fontsize=12)
    axes[2].set_ylabel('PSNR on Edge Strip (dB)', fontsize=12)
    axes[2].set_title(f'Sweep C: Spacing (σ_parallel=10)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "experiment_2_parameter_sweeps.png", dpi=150)
    print(f"✓ Saved plot: {output_dir / 'experiment_2_parameter_sweeps.png'}")

    return df


# ============================================================================
# EXPERIMENT 3: COVERAGE ANALYSIS
# ============================================================================

def experiment_3_coverage_analysis(N_optimal: int, output_dir: Path):
    """
    Experiment 3: Edge strip width analysis

    Use best configuration from Experiment 2
    Vary strip width: [5px, 10px, 20px, 40px, full]
    Same render, different measurement masks

    Total: 5 measurements (0 new renders)
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT 3: COVERAGE ANALYSIS (N={N_optimal})")
    print("="*80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Test case
    target, desc, _ = Phase05TestCorpus.generate_test_case(
        case_id='case_06',
        blur_sigma=2.0,
        contrast=0.5,
        orientation_str='vertical'
    )

    # Render once with best parameters
    print("\nRendering with best parameters...")
    gaussians = place_edge_gaussians_variable_N(
        edge_descriptor=desc,
        N=N_optimal,
        sigma_perp=1.0,
        sigma_parallel=10.0,
        alpha=0.6
    )

    rendered = GaussianRenderer.render_accumulate(gaussians, 100, 100, 1)

    # Vary strip width
    strip_widths = [5, 10, 20, 40, 100]  # 100 = full image

    results = []

    print("\nMeasuring PSNR at different strip widths:")
    for width in strip_widths:
        if width == 100:
            # Full image
            metrics = compute_metrics(target, rendered)
            num_pixels = 100 * 100
            print(f"  strip_width=FULL: PSNR={metrics['psnr']:.2f} dB (all {num_pixels} pixels)")
        else:
            # Edge strip
            mask = create_edge_mask(
                image_size=(100, 100),
                edge_position=desc['position'],
                edge_orientation=desc['orientation'],
                strip_width=width
            )
            metrics = compute_metrics_masked(target, rendered, mask)
            print(f"  strip_width={width:2d}px: PSNR={metrics['psnr']:.2f} dB ({metrics['num_pixels']} pixels)")

        results.append({
            'strip_width': width,
            'psnr': metrics['psnr'],
            'mse': metrics['mse'],
            'mae': metrics['mae'],
            'num_pixels': metrics['num_pixels'] if 'num_pixels' in metrics else num_pixels
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "experiment_3_coverage_analysis_results.csv", index=False)

    print(f"\n✓ Experiment 3 complete. Results saved to {output_dir}")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: PSNR vs strip width
    axes[0].plot(df['strip_width'], df['psnr'], 'o-', linewidth=2, markersize=10, color='green')
    axes[0].set_xlabel('Edge Strip Width (pixels)', fontsize=12)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_title('Coverage Analysis: PSNR vs Strip Width', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(20, color='red', linestyle='--', alpha=0.5, label='Standard strip (20px)')
    axes[0].legend()

    # Plot 2: PSNR vs number of pixels
    axes[1].plot(df['num_pixels'], df['psnr'], 'o-', linewidth=2, markersize=10, color='purple')
    axes[1].set_xlabel('Number of Pixels Evaluated', fontsize=12)
    axes[1].set_ylabel('PSNR (dB)', fontsize=12)
    axes[1].set_title('PSNR vs Evaluation Area', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_dir / "experiment_3_coverage_analysis.png", dpi=150)
    print(f"✓ Saved plot: {output_dir / 'experiment_3_coverage_analysis.png'}")

    return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute all Phase 0.5 experiments"""

    print("="*80)
    print("PHASE 0.5: EDGE REGION ISOLATED TESTING")
    print("="*80)
    print("\nKey change: Measure PSNR on edge strip ONLY (ignore background)")
    print("\nExperiments:")
    print("  1. N Sweep (18 renders) - find optimal N")
    print("  2. Parameter Refinement (13 renders) - re-sweep at N_optimal")
    print("  3. Coverage Analysis (5 measurements) - vary strip width")
    print("="*80)

    # Create output directory
    output_dir = Path("phase_0.5_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Experiment 1: N Sweep
    df_exp1 = experiment_1_n_sweep(output_dir / "experiment_1")

    # Analyze and determine N_optimal
    N_optimal = analyze_experiment_1(df_exp1, output_dir / "experiment_1")

    # Experiment 2: Parameter Refinement
    df_exp2 = experiment_2_parameter_refinement(N_optimal, output_dir / "experiment_2")

    # Experiment 3: Coverage Analysis
    df_exp3 = experiment_3_coverage_analysis(N_optimal, output_dir / "experiment_3")

    print("\n" + "="*80)
    print("PHASE 0.5 EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"\nN_optimal: {N_optimal}")
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Review empirical_rules_v2.md (to be generated)")
    print("  2. Review PHASE_0.5_REPORT.md (to be generated)")
    print("  3. Compare to Phase 0 results")
    print("="*80)


if __name__ == "__main__":
    main()
