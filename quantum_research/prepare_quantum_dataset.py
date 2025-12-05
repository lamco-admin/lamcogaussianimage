#!/usr/bin/env python3
"""
Prepare quantum-ready dataset from Kodak Gaussian data.

Loads all 24 CSV files, filters to representative samples,
normalizes features, and prepares for quantum kernel computation.

Input: kodak_gaussian_data/*.csv (682,059 snapshots)
Output: kodak_gaussians_quantum_ready.pkl (~10,000 samples)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import pickle
import time

print("="*80)
print("QUANTUM DATASET PREPARATION")
print("Kodak Gaussian Configurations → Quantum-Ready Format")
print("="*80)
print()

# Load all CSV files
data_dir = Path("./kodak_gaussian_data")
csv_files = sorted(data_dir.glob("kodim*.csv"))

print(f"Found {len(csv_files)} CSV files")
print()

# Load and combine
print("Loading CSV files...")
dfs = []
for i, csv_file in enumerate(csv_files, 1):
    df = pd.read_csv(csv_file)
    dfs.append(df)
    print(f"  [{i:2d}/24] {csv_file.name}: {len(df):,} snapshots")

print()
combined = pd.concat(dfs, ignore_index=True)
print(f"✓ Total snapshots loaded: {len(combined):,}")
print()

# Data quality check
print("="*80)
print("DATA QUALITY CHECK")
print("="*80)
print()

# Check for NaN
nan_mask = combined.isna().any(axis=1)
nan_count = nan_mask.sum()

if nan_count > 0:
    print(f"⚠️  Found {nan_count:,} rows with NaN values ({100*nan_count/len(combined):.2f}%)")
    print(f"   Removing contaminated rows...")
    combined = combined[~nan_mask]
    print(f"   ✓ Clean dataset: {len(combined):,} rows")
else:
    print(f"✓ No NaN values detected - data is completely clean!")

print()

# Statistical summary
print("="*80)
print("DATASET STATISTICS")
print("="*80)
print()

print("Parameter ranges:")
print(f"  sigma_x:     [{combined['sigma_x'].min():.6f}, {combined['sigma_x'].max():.6f}]")
print(f"  sigma_y:     [{combined['sigma_y'].min():.6f}, {combined['sigma_y'].max():.6f}]")
print(f"  rotation:    [{combined['rotation'].min():.3f}, {combined['rotation'].max():.3f}] rad")
print(f"  alpha:       [{combined['alpha'].min():.3f}, {combined['alpha'].max():.3f}]")
print(f"  loss:        [{combined['loss'].min():.6f}, {combined['loss'].max():.6f}]")
print(f"  coherence:   [{combined['edge_coherence'].min():.3f}, {combined['edge_coherence'].max():.3f}]")
print(f"  gradient:    [{combined['local_gradient'].min():.6f}, {combined['local_gradient'].max():.6f}]")
print()

# Filter to representative samples
print("="*80)
print("FILTERING TO REPRESENTATIVE SAMPLES")
print("="*80)
print()

target_samples = 1000  # Conservative for 76GB RAM (actual peak: ~57GB with 19GB headroom)
print(f"Target: {target_samples:,} diverse samples for quantum analysis")
print(f"  (Conservative configuration - prioritizes stability)")
print(f"  (Can scale to 1,500+ with future memory optimizations)")
print()

# Strategy 1: Keep only final iterations (best quality per pass)
print("Strategy 1: Final iterations per pass...")
final_iterations = combined[combined['iteration'] >= 90]
print(f"  Final iterations (≥90): {len(final_iterations):,}")

# Strategy 2: Stratified by image
print("Strategy 2: Stratified sampling across images...")
stratified = []
samples_per_image = target_samples // 24
for image_id in combined['image_id'].unique():
    image_data = combined[combined['image_id'] == image_id]
    n_samples = min(samples_per_image, len(image_data))
    sample = image_data.sample(n=n_samples, random_state=42)
    stratified.append(sample)
stratified_df = pd.concat(stratified)
print(f"  Stratified ({samples_per_image}/image): {len(stratified_df):,}")

# Strategy 3: Diversity sampling in parameter space
print("Strategy 3: Diversity sampling in parameter space...")

# Use all unique configurations
combined_strategies = pd.concat([final_iterations, stratified_df]).drop_duplicates()
print(f"  Combined strategies: {len(combined_strategies):,}")

# If still too many, use k-means clustering for diversity
if len(combined_strategies) > target_samples:
    print(f"  Subsampling to {target_samples:,} using k-means diversity...")

    # Extract features for clustering
    X_temp = combined_strategies[['sigma_x', 'sigma_y', 'alpha', 'loss',
                                   'edge_coherence', 'local_gradient']].values

    # Cluster into target_samples clusters
    kmeans = MiniBatchKMeans(n_clusters=target_samples, random_state=42, batch_size=1000)
    labels = kmeans.fit_predict(X_temp)

    # Select samples closest to centroids (most representative)
    selected_indices = []
    for i in range(target_samples):
        cluster_points = np.where(labels == i)[0]
        if len(cluster_points) > 0:
            centroid = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(X_temp[cluster_points] - centroid, axis=1)
            closest_idx = cluster_points[np.argmin(distances)]
            selected_indices.append(closest_idx)

    filtered = combined_strategies.iloc[selected_indices].copy()
else:
    filtered = combined_strategies.copy()

print(f"✓ Final filtered dataset: {len(filtered):,} samples")
print()

# Extract features for quantum kernel
print("="*80)
print("FEATURE EXTRACTION")
print("="*80)
print()

features = ['sigma_x', 'sigma_y', 'alpha', 'loss', 'edge_coherence', 'local_gradient']
X = filtered[features].values

print(f"Feature matrix: {X.shape}")
print(f"Features: {', '.join(features)}")
print()

print("Feature statistics:")
for i, feat in enumerate(features):
    print(f"  {feat:15s}: mean={X[:,i].mean():8.4f}, std={X[:,i].std():8.4f}")
print()

# Normalize features
print("Normalizing features for quantum kernel...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✓ Scaled features (mean≈0, std≈1)")
print()

# Save quantum-ready dataset
output_file = "kodak_gaussians_quantum_ready.pkl"

print("="*80)
print("SAVING QUANTUM-READY DATASET")
print("="*80)
print()

dataset = {
    'X': X,
    'X_scaled': X_scaled,
    'scaler': scaler,
    'metadata': filtered[['image_id', 'refinement_pass', 'iteration', 'gaussian_id']].values,
    'n_samples': len(X),
    'n_features': len(features),
    'features': features,
    'source': 'Kodak PhotoCD (24 images, real encoding)',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'statistics': {
        'total_raw_snapshots': len(combined),
        'filtered_snapshots': len(filtered),
        'filter_rate': len(filtered) / len(combined),
    }
}

with open(output_file, 'wb') as f:
    pickle.dump(dataset, f, protocol=4)

file_size = Path(output_file).stat().st_size

print(f"✓ Dataset saved: {output_file}")
print(f"  Samples: {len(X):,}")
print(f"  Features: {len(features)}")
print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
print()

# Summary
print("="*80)
print("SUMMARY")
print("="*80)
print()
print(f"Raw data: {len(combined):,} Gaussian snapshots from 24 Kodak images")
print(f"Filtered: {len(filtered):,} representative samples")
print(f"Filter ratio: {100*len(filtered)/len(combined):.1f}%")
print()
print("Features prepared for quantum kernel:")
for feat in features:
    print(f"  • {feat}")
print()
print("Ready for quantum clustering!")
print()

print("="*80)
print("NEXT STEP: QUANTUM ANALYSIS")
print("="*80)
print()
print("Run quantum clustering:")
print("  python3 Q1_production_real_data.py")
print()
print("This will:")
print("  • Load quantum-ready dataset")
print("  • Subsample to 1,500 diverse configurations")
print("  • Compute quantum kernel (1,500 × 1,500 matrix)")
print("  • Discover 4-6 fundamental Gaussian channels")
print("  • Save results to gaussian_channels_kodak_quantum.json")
print()
print("Requirements:")
print("  • 70GB RAM (you have 76GB ✓)")
print("  • 22-37 minute runtime")
print("  • Peak memory: ~64GB")
print()
print("="*80)
