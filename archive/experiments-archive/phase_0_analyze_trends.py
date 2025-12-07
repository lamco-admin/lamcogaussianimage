"""Quick analysis of Phase 0 sweep trends"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
df = pd.read_csv('phase_0_results/sweep_results.csv')

print("="*60)
print("PHASE 0 PRELIMINARY ANALYSIS")
print("="*60)

# Analyze each sweep
sweeps = df['sweep'].unique()

for sweep_name in sweeps:
    sweep_data = df[df['sweep'] == sweep_name]
    param_name = sweep_data['param_name'].iloc[0]

    print(f"\n{sweep_name}:")
    print(f"  Parameter: {param_name}")

    # Find best value for each test case
    for case_id in sweep_data['case_id'].unique():
        case_data = sweep_data[sweep_data['case_id'] == case_id]
        best_idx = case_data['psnr'].idxmax()
        best_row = case_data.loc[best_idx]

        print(f"  {case_id}:")
        print(f"    Best {param_name} = {best_row['param_value']:.3f}")
        print(f"    Best PSNR = {best_row['psnr']:.2f} dB")
        print(f"    Edge blur = {best_row['blur_sigma']:.1f}, Contrast = {best_row['contrast']:.1f}")

# Sweep 1 analysis: sigma_perp vs edge blur
print("\n" + "="*60)
print("SWEEP 1 ANALYSIS: σ_perp vs Edge Blur")
print("="*60)

sweep1 = df[df['sweep'] == 'sweep_1_sigma_perp']

# Group by blur level and find best sigma_perp
for blur in sweep1['blur_sigma'].unique():
    blur_data = sweep1[sweep1['blur_sigma'] == blur]
    best_idx = blur_data['psnr'].idxmax()
    best_row = blur_data.loc[best_idx]

    ratio = best_row['sigma_perp'] / blur if blur > 0 else None

    print(f"\nEdge blur σ = {blur:.1f}:")
    print(f"  Best σ_perp = {best_row['sigma_perp']:.1f}")
    if ratio:
        print(f"  Ratio σ_perp/σ_edge = {ratio:.2f}")
    print(f"  PSNR = {best_row['psnr']:.2f} dB")

# Sweep 4 analysis: alpha vs contrast
print("\n" + "="*60)
print("SWEEP 4 ANALYSIS: Alpha vs Contrast")
print("="*60)

sweep4 = df[df['sweep'] == 'sweep_4_alpha']

# Group by contrast and find best alpha
for contrast in sweep4['contrast'].unique():
    contrast_data = sweep4[sweep4['contrast'] == contrast]
    best_idx = contrast_data['psnr'].idxmax()
    best_row = contrast_data.loc[best_idx]

    ratio = best_row['alpha'] / contrast

    print(f"\nContrast ΔI = {contrast:.1f}:")
    print(f"  Best alpha = {best_row['alpha']:.2f}")
    print(f"  Ratio alpha/ΔI = {ratio:.2f}")
    print(f"  PSNR = {best_row['psnr']:.2f} dB")

# Overall trends
print("\n" + "="*60)
print("OBSERVED TRENDS")
print("="*60)

print("\n1. σ_perp (cross-edge width):")
print("   - PSNR increases with σ_perp (larger is better in tested range)")
print("   - Suggests σ_perp should be LARGE relative to edge blur")

print("\n2. σ_parallel (along-edge spread):")
print("   - PSNR increases with σ_parallel (larger is better)")
print("   - Suggests wider Gaussians give better coverage")

print("\n3. Spacing:")
print("   - PSNR increases with spacing (less overlap seems better?)")
print("   - This is counterintuitive - needs investigation")

print("\n4. Alpha (opacity):")
print("   - PSNR increases slightly with alpha")
print("   - Suggests alpha values tested are too low")

print("\n" + "="*60)
print("ISSUES IDENTIFIED")
print("="*60)
print("\n1. PSNR values are very low (5-17 dB)")
print("   - Good reconstruction should be > 30 dB")
print("   - Suggests parameter ranges are not optimal")
print("\n2. Alpha values appear too low")
print("   - Tested range: 0.1-0.5")
print("   - Should test higher values (0.5-2.0)")
print("\n3. Need to verify Gaussian placement strategy")
print("   - May need different approach for edge representation")
