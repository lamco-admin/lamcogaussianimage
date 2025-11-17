"""
Phase 1: Multi-Primitive Composition Test

KEY HYPOTHESIS: Primitives don't need perfect individual quality if they compose well.
Edge primitive alone = 10-15 dB, but edge + background + interior together might achieve 25-30 dB.

Test Scene: Rectangle with border (200×200 pixels, grayscale)
- Background (exterior): gray value 0.3
- Border (edge): white 1.0, width 5-10 pixels
- Interior (fill): black 0.0
- Rectangle size: ~100×100 centered

Three Primitive Layers:
1. Background Gaussians (exterior gray)
2. Interior Gaussians (interior black)
3. Edge Gaussians (border white)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple, Optional
import json
import sys

# Add infrastructure to path
sys.path.insert(0, str(Path(__file__).parent / "experiments/gaussian_primitives_sprint/infrastructure"))

from gaussian_primitives import Gaussian2D, GaussianRenderer, compute_metrics


# ============================================================================
# TEST SCENE GENERATION
# ============================================================================

def generate_rectangle_with_border(
    image_size: Tuple[int, int] = (200, 200),
    rect_size: Tuple[int, int] = (100, 100),
    border_width: int = 8,
    background_value: float = 0.3,
    border_value: float = 1.0,
    interior_value: float = 0.0
) -> Tuple[np.ndarray, Dict]:
    """
    Generate rectangle with border test scene

    Args:
        image_size: (height, width)
        rect_size: (height, width) of rectangle
        border_width: width of border in pixels
        background_value: gray value for exterior
        border_value: value for border
        interior_value: value for interior

    Returns:
        (image, descriptor)
    """
    h, w = image_size
    rh, rw = rect_size

    # Create image filled with background
    image = np.full((h, w), background_value, dtype=np.float32)

    # Center rectangle
    cx, cy = w // 2, h // 2
    x1 = cx - rw // 2
    y1 = cy - rh // 2
    x2 = x1 + rw
    y2 = y1 + rh

    # Fill entire rectangle with border value
    image[y1:y2, x1:x2] = border_value

    # Fill interior with interior value
    interior_x1 = x1 + border_width
    interior_y1 = y1 + border_width
    interior_x2 = x2 - border_width
    interior_y2 = y2 - border_width
    image[interior_y1:interior_y2, interior_x1:interior_x2] = interior_value

    descriptor = {
        'image_size': image_size,
        'rect_size': rect_size,
        'rect_position': (x1, y1, x2, y2),
        'border_width': border_width,
        'background_value': background_value,
        'border_value': border_value,
        'interior_value': interior_value,
        'interior_region': (interior_x1, interior_y1, interior_x2, interior_y2)
    }

    return image, descriptor


# ============================================================================
# PRIMITIVE GENERATORS
# ============================================================================

def generate_background_gaussians(
    descriptor: Dict,
    N: int = 20,
    sigma: float = 25.0,
    alpha: float = 0.3
) -> List[Gaussian2D]:
    """
    Generate background Gaussians (exterior gray region)

    Args:
        descriptor: Scene descriptor
        N: Number of Gaussians
        sigma: Isotropic sigma
        alpha: Opacity

    Returns:
        List of Gaussian2D objects
    """
    gaussians = []

    h, w = descriptor['image_size']
    x1, y1, x2, y2 = descriptor['rect_position']
    background_value = descriptor['background_value']

    # Create grid of positions in exterior region
    # We'll place Gaussians in a grid and skip those inside rectangle
    grid_size = int(np.sqrt(N * 2))  # Overestimate to account for skipping
    x_positions = np.linspace(10, w - 10, grid_size)
    y_positions = np.linspace(10, h - 10, grid_size)

    positions = []
    for x in x_positions:
        for y in y_positions:
            # Skip if inside rectangle
            if x1 <= x <= x2 and y1 <= y <= y2:
                continue
            positions.append((x, y))

    # Sample N positions
    if len(positions) > N:
        indices = np.random.choice(len(positions), N, replace=False)
        positions = [positions[i] for i in indices]

    for x, y in positions:
        g = Gaussian2D(
            x=x, y=y,
            sigma_parallel=sigma,
            sigma_perp=sigma,
            theta=0.0,
            color=np.array([background_value]),
            alpha=alpha,
            layer_type='B'  # Background
        )
        gaussians.append(g)

    return gaussians


def generate_interior_gaussians(
    descriptor: Dict,
    N: int = 20,
    sigma: float = 20.0,
    alpha: float = 0.3
) -> List[Gaussian2D]:
    """
    Generate interior Gaussians (interior black region)

    Args:
        descriptor: Scene descriptor
        N: Number of Gaussians
        sigma: Isotropic sigma
        alpha: Opacity

    Returns:
        List of Gaussian2D objects
    """
    gaussians = []

    ix1, iy1, ix2, iy2 = descriptor['interior_region']
    interior_value = descriptor['interior_value']

    # Grid placement inside interior
    grid_size = int(np.ceil(np.sqrt(N)))
    x_positions = np.linspace(ix1 + 10, ix2 - 10, grid_size)
    y_positions = np.linspace(iy1 + 10, iy2 - 10, grid_size)

    positions = []
    for x in x_positions:
        for y in y_positions:
            positions.append((x, y))

    # Sample N positions
    if len(positions) > N:
        indices = np.random.choice(len(positions), N, replace=False)
        positions = [positions[i] for i in indices]

    for x, y in positions[:N]:
        g = Gaussian2D(
            x=x, y=y,
            sigma_parallel=sigma,
            sigma_perp=sigma,
            theta=0.0,
            color=np.array([interior_value]),
            alpha=alpha,
            layer_type='I'  # Interior
        )
        gaussians.append(g)

    return gaussians


def generate_edge_gaussians(
    descriptor: Dict,
    N: int = 50,
    sigma_perp: float = 0.5,
    sigma_parallel: float = 10.0,
    contrast: float = 0.7,
    alpha_override: Optional[float] = None
) -> List[Gaussian2D]:
    """
    Generate edge Gaussians (border) using empirical_rules_v2.md

    Args:
        descriptor: Scene descriptor
        N: Total number of edge Gaussians (distributed across 4 sides)
        sigma_perp: Cross-edge width
        sigma_parallel: Along-edge spread
        contrast: Edge contrast (ΔI)
        alpha_override: Override alpha calculation

    Returns:
        List of Gaussian2D objects
    """
    gaussians = []

    x1, y1, x2, y2 = descriptor['rect_position']
    border_value = descriptor['border_value']

    # Calculate alpha from empirical rules v2.0
    # alpha = (0.3/ΔI) × (10/N_per_edge)
    N_per_edge = N // 4

    if alpha_override is not None:
        alpha = alpha_override
    else:
        alpha_base = 0.3 / contrast if contrast > 0 else 0.3
        alpha = alpha_base * (10.0 / N_per_edge)

    # Generate Gaussians for each of the 4 sides

    # Top edge (y=y1, x from x1 to x2)
    edge_length = x2 - x1
    positions = np.linspace(x1 + 5, x2 - 5, N_per_edge)
    for x in positions:
        g = Gaussian2D(
            x=x, y=y1,
            sigma_parallel=sigma_parallel,
            sigma_perp=sigma_perp,
            theta=0.0,  # Horizontal
            color=np.array([border_value]),
            alpha=alpha,
            layer_type='E'  # Edge
        )
        gaussians.append(g)

    # Bottom edge (y=y2, x from x1 to x2)
    for x in positions:
        g = Gaussian2D(
            x=x, y=y2,
            sigma_parallel=sigma_parallel,
            sigma_perp=sigma_perp,
            theta=0.0,  # Horizontal
            color=np.array([border_value]),
            alpha=alpha,
            layer_type='E'
        )
        gaussians.append(g)

    # Left edge (x=x1, y from y1 to y2)
    edge_length = y2 - y1
    positions = np.linspace(y1 + 5, y2 - 5, N_per_edge)
    for y in positions:
        g = Gaussian2D(
            x=x1, y=y,
            sigma_parallel=sigma_parallel,
            sigma_perp=sigma_perp,
            theta=np.pi/2,  # Vertical
            color=np.array([border_value]),
            alpha=alpha,
            layer_type='E'
        )
        gaussians.append(g)

    # Right edge (x=x2, y from y1 to y2)
    for y in positions:
        g = Gaussian2D(
            x=x2, y=y,
            sigma_parallel=sigma_parallel,
            sigma_perp=sigma_perp,
            theta=np.pi/2,  # Vertical
            color=np.array([border_value]),
            alpha=alpha,
            layer_type='E'
        )
        gaussians.append(g)

    return gaussians


# ============================================================================
# LAYER-BY-LAYER BUILDUP EXPERIMENT
# ============================================================================

def experiment_layer_buildup(output_dir: Path):
    """
    Experiment 1: Layer-by-layer buildup

    Renders:
    1. Background only
    2. Background + Interior
    3. Background + Interior + Edges (full)
    4. Interior + Edges (no background)
    5. Edges only (baseline)

    Measures PSNR for each to understand composition.
    """
    print("\n" + "="*80)
    print("EXPERIMENT: LAYER-BY-LAYER BUILDUP")
    print("="*80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate test scene
    print("\nGenerating test scene...")
    target, descriptor = generate_rectangle_with_border(
        image_size=(200, 200),
        rect_size=(100, 100),
        border_width=8,
        background_value=0.3,
        border_value=1.0,
        interior_value=0.0
    )

    # Save target
    plt.imsave(
        output_dir / "target.png",
        target,
        cmap='gray',
        vmin=0,
        vmax=1
    )
    print(f"✓ Target scene saved: {output_dir / 'target.png'}")

    # Generate primitive layers
    print("\nGenerating primitive layers...")

    # Layer parameters
    N_background = 20
    N_interior = 20
    N_edge = 50

    background_gaussians = generate_background_gaussians(
        descriptor,
        N=N_background,
        sigma=25.0,
        alpha=0.3
    )
    print(f"  Background: {len(background_gaussians)} Gaussians")

    interior_gaussians = generate_interior_gaussians(
        descriptor,
        N=N_interior,
        sigma=20.0,
        alpha=0.3
    )
    print(f"  Interior: {len(interior_gaussians)} Gaussians")

    # Calculate edge contrast
    edge_contrast = descriptor['border_value'] - descriptor['background_value']  # 0.7

    edge_gaussians = generate_edge_gaussians(
        descriptor,
        N=N_edge,
        sigma_perp=0.5,
        sigma_parallel=10.0,
        contrast=edge_contrast
    )
    print(f"  Edge: {len(edge_gaussians)} Gaussians (ΔI={edge_contrast})")

    print(f"\n  Total Gaussians: {len(background_gaussians) + len(interior_gaussians) + len(edge_gaussians)}")

    # Render 5 compositions
    print("\nRendering layer compositions...")

    compositions = [
        ("1_background_only", background_gaussians),
        ("2_background_interior", background_gaussians + interior_gaussians),
        ("3_full_composition", background_gaussians + interior_gaussians + edge_gaussians),
        ("4_interior_edges", interior_gaussians + edge_gaussians),
        ("5_edges_only", edge_gaussians)
    ]

    results = []

    for name, gaussians in compositions:
        print(f"  {name} ({len(gaussians)} Gaussians)...", end=' ')

        rendered = GaussianRenderer.render_accumulate(
            gaussians,
            width=200,
            height=200,
            channels=1
        )

        # Save render
        plt.imsave(
            output_dir / f"{name}.png",
            rendered,
            cmap='gray',
            vmin=0,
            vmax=1
        )

        # Compute metrics
        metrics = compute_metrics(target, rendered)

        print(f"PSNR={metrics['psnr']:.2f} dB")

        results.append({
            'composition': name,
            'num_gaussians': len(gaussians),
            'psnr': metrics['psnr'],
            'mse': metrics['mse'],
            'mae': metrics['mae']
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "layer_buildup_results.csv", index=False)

    print(f"\n✓ Layer buildup experiment complete")
    print(f"  Results saved to {output_dir}")

    return df, target, descriptor


# ============================================================================
# REGIONAL ANALYSIS
# ============================================================================

def create_region_masks(descriptor: Dict) -> Dict[str, np.ndarray]:
    """
    Create masks for different regions of the scene

    Returns:
        Dict with 'background', 'border', 'interior' masks
    """
    h, w = descriptor['image_size']
    x1, y1, x2, y2 = descriptor['rect_position']
    ix1, iy1, ix2, iy2 = descriptor['interior_region']

    # Background: everything outside rectangle
    background_mask = np.ones((h, w), dtype=bool)
    background_mask[y1:y2, x1:x2] = False

    # Interior: interior region
    interior_mask = np.zeros((h, w), dtype=bool)
    interior_mask[iy1:iy2, ix1:ix2] = True

    # Border: rectangle minus interior
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[y1:y2, x1:x2] = True
    border_mask[iy1:iy2, ix1:ix2] = False

    return {
        'background': background_mask,
        'border': border_mask,
        'interior': interior_mask
    }


def compute_regional_metrics(
    target: np.ndarray,
    rendered: np.ndarray,
    masks: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each region separately

    Returns:
        Dict with region names as keys, metrics dicts as values
    """
    regional_metrics = {}

    for region_name, mask in masks.items():
        target_region = target[mask]
        rendered_region = rendered[mask]

        mse = np.mean((target_region - rendered_region) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        mae = np.mean(np.abs(target_region - rendered_region))

        regional_metrics[region_name] = {
            'mse': mse,
            'psnr': psnr,
            'mae': mae,
            'num_pixels': int(np.sum(mask))
        }

    return regional_metrics


def experiment_regional_analysis(
    target: np.ndarray,
    descriptor: Dict,
    output_dir: Path
):
    """
    Experiment 2: Regional analysis of full composition

    Break down PSNR by region (background, border, interior)
    """
    print("\n" + "="*80)
    print("EXPERIMENT: REGIONAL ANALYSIS")
    print("="*80)

    # Generate full composition
    print("\nGenerating full composition...")

    background_gaussians = generate_background_gaussians(descriptor, N=20, sigma=25.0, alpha=0.3)
    interior_gaussians = generate_interior_gaussians(descriptor, N=20, sigma=20.0, alpha=0.3)

    edge_contrast = descriptor['border_value'] - descriptor['background_value']
    edge_gaussians = generate_edge_gaussians(
        descriptor,
        N=50,
        sigma_perp=0.5,
        sigma_parallel=10.0,
        contrast=edge_contrast
    )

    all_gaussians = background_gaussians + interior_gaussians + edge_gaussians

    rendered = GaussianRenderer.render_accumulate(
        all_gaussians,
        width=200,
        height=200,
        channels=1
    )

    # Create region masks
    masks = create_region_masks(descriptor)

    # Compute regional metrics
    print("\nRegional PSNR breakdown:")
    regional_metrics = compute_regional_metrics(target, rendered, masks)

    results = []
    for region_name, metrics in regional_metrics.items():
        print(f"  {region_name:12s}: PSNR = {metrics['psnr']:6.2f} dB ({metrics['num_pixels']:6d} pixels)")
        results.append({
            'region': region_name,
            'psnr': metrics['psnr'],
            'mse': metrics['mse'],
            'mae': metrics['mae'],
            'num_pixels': metrics['num_pixels']
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "regional_analysis_results.csv", index=False)

    # Create residual visualization
    residual = np.abs(target - rendered)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(target, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Target', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(rendered, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Rendered (Full)', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(residual, cmap='hot', vmin=0, vmax=0.5)
    axes[2].set_title('Residual (|error|)', fontsize=12)
    axes[2].axis('off')

    # Residual with region overlay
    residual_overlay = axes[3].imshow(residual, cmap='hot', vmin=0, vmax=0.5)
    axes[3].set_title('Residual with Regions', fontsize=12)
    axes[3].axis('off')

    # Add region boundaries
    x1, y1, x2, y2 = descriptor['rect_position']
    ix1, iy1, ix2, iy2 = descriptor['interior_region']

    from matplotlib.patches import Rectangle
    rect_outer = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='cyan', facecolor='none', label='Border')
    rect_inner = Rectangle((ix1, iy1), ix2-ix1, iy2-iy1, linewidth=2, edgecolor='lime', facecolor='none', label='Interior')
    axes[3].add_patch(rect_outer)
    axes[3].add_patch(rect_inner)
    axes[3].legend(fontsize=8)

    plt.colorbar(residual_overlay, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    plt.savefig(output_dir / "residual_analysis.png", dpi=150)
    print(f"\n✓ Residual visualization saved: {output_dir / 'residual_analysis.png'}")

    print(f"✓ Regional analysis complete")

    return df


# ============================================================================
# PARAMETER SENSITIVITY
# ============================================================================

def experiment_parameter_sensitivity(
    target: np.ndarray,
    descriptor: Dict,
    output_dir: Path
):
    """
    Experiment 3: Parameter sensitivity

    Vary Gaussian counts per layer to find optimal allocation
    """
    print("\n" + "="*80)
    print("EXPERIMENT: PARAMETER SENSITIVITY")
    print("="*80)

    print("\nTesting different Gaussian allocations...")

    # Test different allocations (N_background, N_interior, N_edge)
    allocations = [
        (10, 10, 50),
        (15, 15, 50),
        (20, 20, 50),  # Default
        (30, 10, 50),
        (10, 30, 50),
        (20, 20, 100),
        (30, 30, 100)
    ]

    edge_contrast = descriptor['border_value'] - descriptor['background_value']

    results = []

    for N_bg, N_int, N_edge in allocations:
        print(f"  N=({N_bg:2d}, {N_int:2d}, {N_edge:3d})...", end=' ')

        background_gaussians = generate_background_gaussians(descriptor, N=N_bg, sigma=25.0, alpha=0.3)
        interior_gaussians = generate_interior_gaussians(descriptor, N=N_int, sigma=20.0, alpha=0.3)
        edge_gaussians = generate_edge_gaussians(
            descriptor,
            N=N_edge,
            sigma_perp=0.5,
            sigma_parallel=10.0,
            contrast=edge_contrast
        )

        all_gaussians = background_gaussians + interior_gaussians + edge_gaussians

        rendered = GaussianRenderer.render_accumulate(
            all_gaussians,
            width=200,
            height=200,
            channels=1
        )

        metrics = compute_metrics(target, rendered)

        print(f"PSNR={metrics['psnr']:.2f} dB")

        results.append({
            'N_background': N_bg,
            'N_interior': N_int,
            'N_edge': N_edge,
            'N_total': len(all_gaussians),
            'psnr': metrics['psnr'],
            'mse': metrics['mse'],
            'mae': metrics['mae']
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "parameter_sensitivity_results.csv", index=False)

    # Plot results
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    x_labels = [f"({r['N_background']},{r['N_interior']},{r['N_edge']})"
                for _, r in df.iterrows()]

    ax.bar(range(len(df)), df['psnr'], color='steelblue', alpha=0.7)
    ax.set_xlabel('Allocation (N_bg, N_int, N_edge)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.set_title('Parameter Sensitivity: Gaussian Allocation', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.axhline(25, color='red', linestyle='--', linewidth=2, label='Target (25 dB)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "parameter_sensitivity.png", dpi=150)
    print(f"\n✓ Parameter sensitivity plot saved: {output_dir / 'parameter_sensitivity.png'}")

    print(f"✓ Parameter sensitivity experiment complete")

    return df


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualization_summary(output_dir: Path, df_buildup: pd.DataFrame):
    """
    Create comprehensive visualization summary
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATION SUMMARY")
    print("="*80)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Layer buildup renders
    renders = [
        ("target.png", "Target"),
        ("1_background_only.png", "Background Only"),
        ("2_background_interior.png", "Background + Interior")
    ]

    for idx, (filename, title) in enumerate(renders):
        img = plt.imread(output_dir / filename)
        axes[0, idx].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[0, idx].set_title(title, fontsize=11, fontweight='bold')
        axes[0, idx].axis('off')

    # Row 2: More renders
    renders2 = [
        ("3_full_composition.png", "Full Composition"),
        ("4_interior_edges.png", "Interior + Edges"),
        ("5_edges_only.png", "Edges Only")
    ]

    for idx, (filename, title) in enumerate(renders2):
        img = plt.imread(output_dir / filename)
        axes[1, idx].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[1, idx].set_title(title, fontsize=11, fontweight='bold')
        axes[1, idx].axis('off')

        # Add PSNR annotation
        comp_name = filename.replace('.png', '')
        psnr = df_buildup[df_buildup['composition'] == comp_name]['psnr'].values[0]
        axes[1, idx].text(
            0.5, -0.1,
            f"PSNR: {psnr:.2f} dB",
            transform=axes[1, idx].transAxes,
            ha='center',
            fontsize=10,
            color='blue',
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(output_dir / "layer_buildup_visualization.png", dpi=150)
    print(f"✓ Visualization summary saved: {output_dir / 'layer_buildup_visualization.png'}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute Phase 1 experiments"""

    print("="*80)
    print("PHASE 1: MULTI-PRIMITIVE COMPOSITION TEST")
    print("="*80)
    print("\nHYPOTHESIS: Primitives compose well even if individually limited")
    print("Edge primitive alone: ~10-15 dB")
    print("Full composition target: 25-30 dB")
    print("="*80)

    # Create output directory
    output_dir = Path("phase_1_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Experiment 1: Layer-by-layer buildup
    df_buildup, target, descriptor = experiment_layer_buildup(output_dir)

    # Experiment 2: Regional analysis
    df_regional = experiment_regional_analysis(target, descriptor, output_dir)

    # Experiment 3: Parameter sensitivity
    df_sensitivity = experiment_parameter_sensitivity(target, descriptor, output_dir)

    # Create visualization summary
    create_visualization_summary(output_dir, df_buildup)

    # Print summary
    print("\n" + "="*80)
    print("PHASE 1 EXPERIMENTS COMPLETE")
    print("="*80)

    print("\nKey Results:")
    print("\n1. Layer Buildup:")
    for _, row in df_buildup.iterrows():
        print(f"   {row['composition']:30s}: PSNR = {row['psnr']:6.2f} dB ({row['num_gaussians']:3d} Gaussians)")

    print("\n2. Regional Analysis:")
    for _, row in df_regional.iterrows():
        print(f"   {row['region']:12s}: PSNR = {row['psnr']:6.2f} dB")

    print("\n3. Best Allocation:")
    best_idx = df_sensitivity['psnr'].idxmax()
    best = df_sensitivity.iloc[best_idx]
    print(f"   N = ({best['N_background']:.0f}, {best['N_interior']:.0f}, {best['N_edge']:.0f})")
    print(f"   PSNR = {best['psnr']:.2f} dB")

    # Check success criteria
    full_psnr = df_buildup[df_buildup['composition'] == '3_full_composition']['psnr'].values[0]

    print("\n" + "="*80)
    print("SUCCESS CRITERIA EVALUATION:")
    print("="*80)

    if full_psnr >= 25.0:
        print(f"✓ Full composition PSNR = {full_psnr:.2f} dB >= 25 dB (SUCCESS)")
    else:
        print(f"✗ Full composition PSNR = {full_psnr:.2f} dB < 25 dB (PARTIAL)")

    # Check layer contributions
    bg_psnr = df_buildup[df_buildup['composition'] == '1_background_only']['psnr'].values[0]
    bg_int_psnr = df_buildup[df_buildup['composition'] == '2_background_interior']['psnr'].values[0]

    if bg_int_psnr > bg_psnr and full_psnr > bg_int_psnr:
        print(f"✓ Each layer adds positive PSNR contribution (SUCCESS)")
    else:
        print(f"? Layer contributions need analysis")

    print("\n" + "="*80)
    print(f"Results saved to: {output_dir}")
    print("\nNext step: Generate compositional_primitive_report.md")
    print("="*80)

    return {
        'buildup': df_buildup,
        'regional': df_regional,
        'sensitivity': df_sensitivity,
        'target': target,
        'descriptor': descriptor
    }


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    results = main()
