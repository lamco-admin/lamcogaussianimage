#!/usr/bin/env python3
"""
Analyze Channel 1 Substructure

Channel 1 contains 71% of all Gaussians with enormous variance (CV=2.81).
This suggests it might actually be 3-4 sub-channels that couldn't be
distinguished with current 6D features.

This script applies hierarchical clustering to Channel 1 alone to discover
if meaningful sub-groups exist.
"""

import json
import pickle
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("="*80)
print("CHANNEL 1 SUBSTRUCTURE ANALYSIS")
print("="*80)
print()

# Load data
with open('gaussian_channels_kodak_quantum.json') as f:
    results = json.load(f)

with open('kodak_gaussians_quantum_ready.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
X_scaled = data['X_scaled']
features = data['features']
labels = np.array(results['labels_quantum'])

# Extract Channel 1
channel_1_mask = labels == 1
channel_1_data = X[channel_1_mask]
channel_1_scaled = X_scaled[channel_1_mask]

print(f"Channel 1 size: {np.sum(channel_1_mask)} Gaussians (71.1% of total)")
print()

# Statistics
print("Channel 1 characteristics:")
print(f"  σ_x: {channel_1_data[:,0].mean():.4f} ± {channel_1_data[:,0].std():.4f}")
print(f"  Range: [{channel_1_data[:,0].min():.4f}, {channel_1_data[:,0].max():.4f}]")
print(f"  CV: {channel_1_data[:,0].std() / channel_1_data[:,0].mean():.2f} (very high variance)")
print()
print(f"  loss: {channel_1_data[:,3].mean():.4f} ± {channel_1_data[:,3].std():.4f}")
print(f"  Range: [{channel_1_data[:,3].min():.4f}, {channel_1_data[:,3].max():.4f}]")
print()

# Try different numbers of sub-clusters
print("="*80)
print("TESTING SUB-CLUSTER COUNTS")
print("="*80)
print()

best_k = None
best_score = -1

for k in range(2, 8):
    clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
    sub_labels = clustering.fit_predict(channel_1_scaled)

    score = silhouette_score(channel_1_scaled, sub_labels)

    print(f"  {k} sub-clusters: silhouette = {score:.3f}")

    if score > best_score:
        best_score = score
        best_k = k

print()
print(f"Optimal: {best_k} sub-clusters (silhouette = {best_score:.3f})")
print()

# Cluster with optimal k
clustering = AgglomerativeClustering(n_clusters=best_k, linkage='ward')
sub_labels = clustering.fit_predict(channel_1_scaled)

# Analyze sub-clusters
print("="*80)
print(f"CHANNEL 1 SUB-CLUSTERS (k={best_k})")
print("="*80)
print()

sub_channel_chars = []

for sub_id in range(best_k):
    mask = sub_labels == sub_id
    n = np.sum(mask)

    if n == 0:
        continue

    sub_data = channel_1_data[mask]

    print(f"Sub-Channel 1.{sub_id}: {n} Gaussians ({100*n/len(channel_1_data):.1f}% of Channel 1)")
    print(f"  σ_x:        {sub_data[:,0].mean():.6f} ± {sub_data[:,0].std():.6f}")
    print(f"  σ_y:        {sub_data[:,1].mean():.6f} ± {sub_data[:,1].std():.6f}")
    print(f"  loss:       {sub_data[:,3].mean():.6f} ± {sub_data[:,3].std():.6f}")
    print(f"  coherence:  {sub_data[:,4].mean():.3f} ± {sub_data[:,4].std():.3f}")

    # Characterize this sub-cluster
    mean_scale = np.sqrt(sub_data[:,0].mean() * sub_data[:,1].mean())
    mean_loss = sub_data[:,3].mean()

    if mean_loss < 0.05:
        quality = "HIGH"
    elif mean_loss < 0.1:
        quality = "MEDIUM-HIGH"
    elif mean_loss < 0.15:
        quality = "MEDIUM"
    else:
        quality = "LOW"

    if mean_scale < 0.005:
        size = "Tiny"
    elif mean_scale < 0.02:
        size = "Small"
    elif mean_scale < 0.05:
        size = "Medium"
    else:
        size = "Large"

    char = f"{size}, {quality}"
    sub_channel_chars.append(char)

    print(f"  Characterization: {char}")
    print()

# Test: Do sub-clusters have meaningful differences?
print("="*80)
print("SUB-CLUSTER VALIDATION")
print("="*80)
print()

from scipy.stats import kruskal

# Kruskal-Wallis test on loss distributions
sub_losses = [channel_1_data[sub_labels == i, 3] for i in range(best_k) if np.sum(sub_labels == i) > 0]

if len(sub_losses) > 2:
    h_stat, p_val = kruskal(*sub_losses)
    print(f"Kruskal-Wallis test (loss distributions):")
    print(f"  H-statistic: {h_stat:.2f}")
    print(f"  p-value: {p_val:.6f}")
    print()

    if p_val < 0.001:
        print("  → Sub-clusters have SIGNIFICANTLY different quality")
        print("  → Channel 1 contains distinct quality classes")
    else:
        print("  → Sub-clusters have similar quality")
        print("  → Splitting doesn't reveal meaningful structure")

print()

# Dendrogram
print("Generating hierarchical clustering dendrogram...")
print("  (Saved to channel_1_dendrogram.png)")
print()

# Use subset for dendrogram visualization (max 100 samples)
if len(channel_1_scaled) > 100:
    indices = np.random.choice(len(channel_1_scaled), 100, replace=False)
    sample_data = channel_1_scaled[indices]
else:
    sample_data = channel_1_scaled

linkage_matrix = linkage(sample_data, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix)
plt.title('Channel 1 Hierarchical Clustering Dendrogram (100 sample)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('channel_1_dendrogram.png', dpi=150)
print("✓ Dendrogram saved")
print()

# Comparison to quantum sub-division
print("="*80)
print("COMPARISON TO QUANTUM CLUSTERING")
print("="*80)
print()

print("Quantum discovered 8 total channels:")
print("  - Channel 1 (71%): Dominant 'medium quality' class")
print("  - Channels 0,2,6 (24.6%): Various intermediate")
print("  - Channels 3,4,7 (3.8%): High quality")
print("  - Channel 5 (0.5%): Failure mode")
print()

print(f"Hierarchical sub-clustering of Channel 1 found {best_k} sub-groups:")
for i, char in enumerate(sub_channel_chars):
    print(f"  - Sub-Channel 1.{i}: {char}")
print()

# Final assessment
print("="*80)
print("CONCLUSIONS")
print("="*80)
print()

if best_score > 0.15:
    print("Channel 1 has STRONG sub-structure (silhouette > 0.15)")
    print()
    print("Finding: The quantum 8-channel clustering might be under-clustering.")
    print("Channel 1 actually contains multiple quality classes that should")
    print("be separated for per-channel optimization strategies.")
    print()
    print("Recommendation:")
    print("  1. Re-run quantum with more clusters (try k=10-12)")
    print("  2. OR use hierarchical sub-division of Channel 1")
    print("  3. Test if treating sub-channels separately improves optimization")
elif best_score > 0.05:
    print("Channel 1 has MODERATE sub-structure (silhouette 0.05-0.15)")
    print()
    print("Finding: Some internal heterogeneity exists but not strongly separated.")
    print("May benefit from optimization feature extraction to reveal structure.")
else:
    print("Channel 1 has WEAK sub-structure (silhouette < 0.05)")
    print()
    print("Finding: Channel 1 appears to be genuinely heterogeneous.")
    print("It's the 'miscellaneous' category for medium-quality Gaussians")
    print("that don't fit other channels.")

print()

# Save results
output = {
    'channel_1_size': int(np.sum(channel_1_mask)),
    'optimal_sub_clusters': int(best_k),
    'silhouette_score': float(best_score),
    'sub_clusters': [
        {
            'sub_id': int(i),
            'n_gaussians': int(np.sum(sub_labels == i)),
            'characterization': char
        }
        for i, char in enumerate(sub_channel_chars)
    ],
    'interpretation': 'See stdout for detailed analysis'
}

with open('channel_1_substructure_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)

print("Results saved: channel_1_substructure_analysis.json")
print()
