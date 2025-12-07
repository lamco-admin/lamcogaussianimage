#!/usr/bin/env python3
"""
Xanadu CV Quantum: Gaussian Fidelity Clustering

Tests if natural Gaussian similarity metric (from CV quantum theory)
separates channels better than arbitrary ZZFeatureMap.

Mathematical alignment: Both CV quantum states and 2D Gaussian primitives
are characterized by covariance matrices. Gaussian fidelity is the natural
similarity metric.

Comparison: Silhouette score (Gaussian fidelity) vs (ZZFeatureMap)
"""

import numpy as np
import pickle
from scipy.linalg import det
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
import json
import time

print("="*80)
print("XANADU CV QUANTUM: GAUSSIAN FIDELITY CLUSTERING")
print("="*80)
print()
print("Framework: Compositional layers via natural Gaussian metric")
print("Theory: CV quantum Gaussian states ≈ 2D Gaussian primitives")
print()

# Load enhanced dataset
with open('kodak_gaussians_quantum_ready_enhanced.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
X_scaled = data['X_scaled']
features = data['features']

print(f"Loaded {len(X)} Gaussian configurations")
print(f"Features: {features}")
print()

def gaussian_covariance_from_features(sigma_x, sigma_y):
    """
    Convert Gaussian primitive parameters to covariance matrix.

    For CV quantum state representation:
    Σ = diag([σ_x², σ_y²]) in position-position space
    """
    eps = 1e-8
    Sigma = np.diag([sigma_x**2 + eps, sigma_y**2 + eps])
    return Sigma

def gaussian_fidelity(Sigma1, Sigma2):
    """
    Quantum fidelity between two Gaussian states.

    For Gaussian states with covariances Σ₁, Σ₂:
    F² = 4·det(Σ₁)·det(Σ₂) / (det(Σ₁+Σ₂) + 2·√(det(Σ₁)·det(Σ₂)))

    This is the NATURAL similarity metric for Gaussian states.
    """
    eps = 1e-8

    det1 = det(Sigma1)
    det2 = det(Sigma2)

    if det1 <= 0 or det2 <= 0:
        return 0.0  # Degenerate covariance

    Sigma_sum = Sigma1 + Sigma2
    det_sum = det(Sigma_sum)

    if det_sum <= 0:
        return 0.0

    numerator = 4 * det1 * det2
    denominator = det_sum + 2 * np.sqrt(det1 * det2)

    F_squared = numerator / denominator
    F = np.sqrt(max(0.0, F_squared))

    return F

def build_gaussian_fidelity_kernel(gaussians_features):
    """
    Build similarity kernel using Gaussian fidelity.

    This is the natural metric from CV quantum theory.
    """
    n = len(gaussians_features)
    K = np.zeros((n, n))

    print("Computing Gaussian fidelity kernel...")
    print(f"  Size: {n} × {n}")
    print(f"  Evaluations: {n * (n-1) // 2:,}")
    print()

    # Convert to covariance matrices
    covariances = []
    for i in range(n):
        sigma_x = gaussians_features[i, 0]  # First feature
        sigma_y = gaussians_features[i, 1]  # Second feature
        Sigma = gaussian_covariance_from_features(sigma_x, sigma_y)
        covariances.append(Sigma)

    start_time = time.time()
    computed = 0
    total = n * (n - 1) // 2

    for i in range(n):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = computed / elapsed if elapsed > 0 else 0
            remaining = (total - computed) / rate if rate > 0 else 0
            print(f"  Progress: {i+1}/{n} rows | ETA: {remaining/60:.1f} min")

        for j in range(i, n):
            fid = gaussian_fidelity(covariances[i], covariances[j])
            K[i, j] = fid
            K[j, i] = fid
            computed += 1

    elapsed_total = time.time() - start_time

    print(f"✓ Kernel computed in {elapsed_total/60:.1f} minutes")
    print()

    return K

# Build Gaussian fidelity kernel
K_fidelity = build_gaussian_fidelity_kernel(X)

# Test different cluster counts
print("="*80)
print("TESTING CLUSTER COUNTS (Gaussian Fidelity Kernel)")
print("="*80)
print()

silhouette_scores_fidelity = []

for k in range(3, 13):
    clustering = SpectralClustering(
        n_clusters=k,
        affinity='precomputed',
        random_state=42,
        n_init=10
    )
    labels = clustering.fit_predict(K_fidelity)

    # Convert to distance for silhouette
    K_dist = 1 - K_fidelity
    np.fill_diagonal(K_dist, 0)

    score = silhouette_score(K_dist, labels, metric='precomputed')
    silhouette_scores_fidelity.append((k, score))

    print(f"  {k:2d} clusters: silhouette = {score:.3f}")

optimal_k_fidelity = max(silhouette_scores_fidelity, key=lambda x: x[1])[0]
optimal_score_fidelity = max(silhouette_scores_fidelity, key=lambda x: x[1])[1]

print()
print(f"✓ Optimal: {optimal_k_fidelity} channels (silhouette = {optimal_score_fidelity:.3f})")
print()

# Cluster with optimal k
final_clustering = SpectralClustering(
    n_clusters=optimal_k_fidelity,
    affinity='precomputed',
    random_state=42,
    n_init=10
)
labels_fidelity = final_clustering.fit_predict(K_fidelity)

# Analyze channels
print("="*80)
print(f"GAUSSIAN FIDELITY CHANNELS (k={optimal_k_fidelity})")
print("="*80)
print()

for ch_id in range(optimal_k_fidelity):
    mask = labels_fidelity == ch_id
    n = np.sum(mask)

    if n == 0:
        continue

    ch_data = X[mask]

    print(f"Channel {ch_id}: {n} Gaussians ({100*n/len(X):.1f}%)")
    print(f"  σ_x: {ch_data[:,0].mean():.5f} ± {ch_data[:,0].std():.5f}")
    print(f"  σ_y: {ch_data[:,1].mean():.5f} ± {ch_data[:,1].std():.5f}")
    print(f"  loss: {ch_data[:,3].mean():.5f} ± {ch_data[:,3].std():.5f}")

    # Check if optimization features helped
    if len(features) > 4:
        print(f"  convergence_speed: {ch_data[:,4].mean():.3f}")
        print(f"  sigma_stability: {ch_data[:,7].mean():.3f}")

    print()

# Save results
results = {
    'method': 'gaussian_fidelity_cv_quantum',
    'optimal_k': int(optimal_k_fidelity),
    'optimal_silhouette': float(optimal_score_fidelity),
    'silhouette_scores': [(int(k), float(s)) for k, s in silhouette_scores_fidelity],
    'labels': labels_fidelity.tolist(),
}

with open('gaussian_fidelity_clustering_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Results saved: gaussian_fidelity_clustering_results.json")
print()

# Comparison to gate-based quantum (if results exist)
print("="*80)
print("COMPARISON: Gaussian Fidelity vs ZZFeatureMap")
print("="*80)
print()

try:
    with open('gaussian_channels_kodak_quantum.json') as f:
        gate_based = json.load(f)

    print("Gate-based quantum (ZZFeatureMap):")
    print(f"  Optimal k: {gate_based['n_clusters']}")
    print(f"  Best silhouette: {max([s[1] for s in gate_based['silhouette_scores']]):.3f}")
    print()

    print("Gaussian fidelity (CV quantum theory):")
    print(f"  Optimal k: {optimal_k_fidelity}")
    print(f"  Best silhouette: {optimal_score_fidelity:.3f}")
    print()

    improvement = optimal_score_fidelity - max([s[1] for s in gate_based['silhouette_scores']])

    if improvement > 0.05:
        print(f"CV Quantum metric is BETTER (+{improvement:.3f} silhouette)")
        print("  → Natural Gaussian similarity metric superior to arbitrary feature map")
    elif improvement < -0.05:
        print(f"Gate-based metric is BETTER ({improvement:.3f} silhouette)")
        print("  → ZZFeatureMap works better than natural metric (surprising!)")
    else:
        print(f"Both metrics perform similarly (Δ={improvement:.3f})")
        print("  → Choice of quantum kernel doesn't significantly matter")

except FileNotFoundError:
    print("(Gate-based results not available for comparison)")

print()
print("="*80)
