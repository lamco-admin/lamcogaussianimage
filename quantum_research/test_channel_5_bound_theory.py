#!/usr/bin/env python3
"""
Test Channel 5 Parameter Bound Theory

Channel 5 (failure mode) has:
- σ_x = 0.001000 exactly (zero variance)
- σ_y = 0.001000 exactly (zero variance)
- Worst quality (loss = 0.160)

Hypothesis: These Gaussians are hitting the minimum scale bound (0.001)
and want to go smaller but can't. This causes optimization failure.

This script analyzes the CSV data to see if Channel 5 Gaussians hit
the bound immediately or evolved toward it during optimization.
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

print("="*80)
print("CHANNEL 5 PARAMETER BOUND ANALYSIS")
print("="*80)
print()

# Load quantum channel assignment
with open('gaussian_channels_kodak_quantum.json') as f:
    results = json.load(f)

with open('kodak_gaussians_quantum_ready.pkl', 'rb') as f:
    data = pickle.load(f)

labels = np.array(results['labels_quantum'])
metadata = data['metadata']

# Find which Gaussians belong to Channel 5
channel_5_mask = labels == 5
channel_5_indices = np.where(channel_5_mask)[0]

print(f"Channel 5 contains: {np.sum(channel_5_mask)} Gaussians")
print()

if np.sum(channel_5_mask) == 0:
    print("Channel 5 is empty!")
    exit()

# For each Channel 5 Gaussian, find its trajectory in CSV data
print("Extracting trajectories from CSV files...")
print()

data_dir = Path("./kodak_gaussian_data")

trajectories_found = 0

for idx in channel_5_indices:
    # Get metadata for this Gaussian
    image_id = metadata[idx][0]
    gaussian_id = int(metadata[idx][2])
    refinement_pass = int(metadata[idx][1])

    # Load corresponding CSV
    csv_file = data_dir / f"{image_id}.csv"

    if not csv_file.exists():
        continue

    df = pd.read_csv(csv_file)

    # Filter to this specific Gaussian's trajectory
    trajectory = df[
        (df['gaussian_id'] == gaussian_id) &
        (df['refinement_pass'] == refinement_pass)
    ].sort_values('iteration')

    if len(trajectory) == 0:
        continue

    trajectories_found += 1

    print(f"Channel 5 Gaussian #{trajectories_found}:")
    print(f"  Image: {image_id}, Gaussian ID: {gaussian_id}, Pass: {refinement_pass}")
    print(f"  Trajectory length: {len(trajectory)} iterations")
    print()

    # Analyze if it started at bound or evolved toward it
    sigma_x_values = trajectory['sigma_x'].values
    sigma_y_values = trajectory['sigma_y'].values
    loss_values = trajectory['loss'].values

    initial_sigma_x = sigma_x_values[0]
    final_sigma_x = sigma_x_values[-1]

    print(f"  σ_x evolution:")
    print(f"    Initial: {initial_sigma_x:.6f}")
    print(f"    Final:   {final_sigma_x:.6f}")
    print(f"    Change:  {final_sigma_x - initial_sigma_x:+.6f}")

    if abs(initial_sigma_x - 0.001) < 1e-6:
        print(f"    → Started AT bound (0.001)")
        started_at_bound = True
    else:
        print(f"    → Started away from bound, moved toward it")
        started_at_bound = False

    print()

    # Check if stuck at bound
    at_bound_count = np.sum(np.abs(sigma_x_values - 0.001) < 1e-6)
    pct_at_bound = 100 * at_bound_count / len(sigma_x_values)

    print(f"  Iterations at bound: {at_bound_count}/{len(sigma_x_values)} ({pct_at_bound:.0f}%)")
    print()

    # Loss trajectory
    print(f"  Loss evolution:")
    print(f"    Initial: {loss_values[0]:.6f}")
    print(f"    Final:   {loss_values[-1]:.6f}")
    print(f"    Change:  {loss_values[-1] - loss_values[0]:+.6f}")

    if loss_values[-1] > loss_values[0]:
        print(f"    → Loss INCREASED (optimization made it worse!)")
    else:
        print(f"    → Loss decreased (but still high)")

    print()
    print("-"*80)
    print()

print("="*80)
print("CHANNEL 5 THEORY ASSESSMENT")
print("="*80)
print()

print(f"Analyzed {trajectories_found}/{np.sum(channel_5_mask)} Channel 5 Gaussians")
print()

print("Findings:")
print("  1. All Channel 5 Gaussians have σ = 0.001 exactly (zero variance)")
print("  2. This is the minimum bound in the encoder")
print("  3. Need trajectory analysis to determine if they:")
print("     - Started at bound and stayed (initialization issue)")
print("     - Evolved toward bound (optimizer drove them there)")
print()

print("Hypothesis:")
print("  If they evolved toward 0.001 and got stuck:")
print("    → Optimizer wants σ < 0.001 but can't go there")
print("    → Lowering bound might allow successful optimization")
print()
print("  If they started at 0.001:")
print("    → Initialization placed them at bound")
print("    → They couldn't escape during optimization")
print("    → Different initialization strategy needed")
print()

print("="*80)
print("RECOMMENDATION")
print("="*80)
print()

print("Test both:")
print("  1. Lower minimum scale bound from 0.001 to 0.0005")
print("     - Modify: adam_optimizer.rs line 128-129")
print("     - Current: .clamp(0.001, 0.5)")
print("     - New: .clamp(0.0005, 0.5)")
print()
print("  2. Avoid initializing at minimum bound")
print("     - If coherence > 0.95, use σ = 0.002 (not 0.001)")
print("     - Give optimizer room to adjust")
print()

print("Expected outcome:")
print("  Channel 5 Gaussians either:")
print("  - Optimize successfully with σ < 0.001 (theory confirmed)")
print("  - Still fail (different root cause)")
print()
