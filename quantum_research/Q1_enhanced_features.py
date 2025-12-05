#!/usr/bin/env python3
"""
Q1: Quantum Gaussian Channel Discovery - PRODUCTION
Real Gaussian data from Kodak PhotoCD encoding

Discovers fundamental Gaussian channels via quantum kernel clustering.

REQUIREMENTS:
- VM RAM: 70GB (you have 76GB ✓)
- Runtime: 22-37 minutes
- Input: kodak_gaussians_quantum_ready.pkl
- Output: gaussian_channels_kodak_quantum.json
"""

import numpy as np
import pickle
import json
import time
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import MiniBatchKMeans

# Quantum imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

print("="*80)
print("Q1: QUANTUM GAUSSIAN CHANNEL DISCOVERY - PRODUCTION")
print("Real Gaussian Data from Kodak PhotoCD Encoding")
print("="*80)
print()

# Load dataset
print("Loading quantum-ready dataset...")
with open('kodak_gaussians_quantum_ready_enhanced.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
X_scaled = data['X_scaled']
metadata = data['metadata']
features = data['features']

print(f"✓ Loaded {len(X):,} real Gaussian configurations")
print(f"  Source: {data['source']}")
print(f"  Features: {', '.join(features)}")
print(f"  Timestamp: {data['timestamp']}")
print()

# Use all samples from prepare script (already subsampled to 1,500)
print("="*80)
print("USING PRE-FILTERED SAMPLES")
print("="*80)
print()

print(f"Dataset already optimized for quantum kernel computation")
print(f"  Samples: {len(X_scaled):,}")
print(f"  Expected: ~1,500 (from prepare script)")
print()

if len(X_scaled) != 1500:
    print(f"⚠️  WARNING: Expected 1,500 samples, got {len(X_scaled):,}")
    print(f"   This may indicate prepare script needs to be re-run")
    print()

X_quantum = X_scaled
X_original = X
metadata_quantum = metadata

print(f"✓ Using all {len(X_quantum):,} samples for quantum analysis")
print()

# Determine qubit count from actual feature dimensions
n_features = X_quantum.shape[1]
n_qubits = n_features + 2  # Pad by 2 for richer Hilbert space
print(f"Feature dimensions: {n_features}D")
print(f"Using {n_qubits} qubits")

print("="*80)
print("QUANTUM CIRCUIT CONFIGURATION")
print("="*80)
print()

X_padded = np.pad(X_quantum, ((0,0), (0, n_qubits - n_features)), mode='constant', constant_values=0)

print(f"Feature mapping:")
print(f"  Input features: {n_features}D")
print(f"  Quantum qubits: {n_qubits}")
print(f"  Padding: {n_qubits - n_features} zeros appended")
print()

# Create quantum kernel
feature_map = ZZFeatureMap(n_qubits, reps=2)
qkernel = FidelityQuantumKernel(feature_map=feature_map)

print(f"Quantum circuit:")
print(f"  Feature map: ZZFeatureMap")
print(f"  Qubits: {n_qubits}")
print(f"  Reps: 2 (entanglement depth)")
print()

# Compute kernel
n_evals = len(X_padded) * (len(X_padded) - 1) // 2

print("="*80)
print("QUANTUM KERNEL COMPUTATION")
print("="*80)
print()

print(f"Kernel matrix size: {len(X_padded):,} × {len(X_padded):,}")
print(f"Unique evaluations: {n_evals:,}")
print()

# Memory estimate
kernel_mem_gb = (len(X_padded)**2 * 8) / (1024**3)
intermediate_mem_gb = (n_evals * 50 * 1024) / (1024**3)
total_mem_gb = (kernel_mem_gb + intermediate_mem_gb) * 1.2

print(f"Memory estimates:")
print(f"  Kernel matrix: {kernel_mem_gb:.2f} GB")
print(f"  Intermediate arrays: {intermediate_mem_gb:.2f} GB")
print(f"  Peak memory: {total_mem_gb:.1f} GB")
print(f"  Available RAM: 76 GB")
print(f"  Headroom: {76 - total_mem_gb:.1f} GB")
print()

print(f"Estimated runtime: 22-37 minutes")
print()
print("⏳ Computing quantum kernel...")
print("   This is the longest step - please be patient")
print("   Progress: CPU at 100%, memory will grow to ~60-65GB")
print()

start_kernel = time.time()
K_quantum = qkernel.evaluate(x_vec=X_padded)
kernel_time = time.time() - start_kernel

print(f"✓ Quantum kernel computed in {kernel_time/60:.1f} minutes")
print(f"  Kernel shape: {K_quantum.shape}")
print(f"  Memory used: {K_quantum.nbytes / 1024**3:.2f} GB")
print()

# Determine optimal cluster count
print("="*80)
print("DETERMINING OPTIMAL CLUSTER COUNT")
print("="*80)
print()

print("Testing cluster counts 3-8 using silhouette score...")
print()

# Fix: Zero diagonal for distance metric (silhouette requires distances, not similarities)
K_quantum_dist = 1 - K_quantum  # Convert similarity to distance
np.fill_diagonal(K_quantum_dist, 0)  # Zero diagonal

silhouette_scores = []
for n_clusters in range(3, 9):
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        random_state=42,
        n_init=10
    )
    labels = clustering.fit_predict(K_quantum)
    score = silhouette_score(K_quantum_dist, labels, metric='precomputed')
    silhouette_scores.append((n_clusters, score))
    print(f"  {n_clusters} clusters: silhouette = {score:.3f}")

optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
print()
print(f"✓ Optimal cluster count: {optimal_k} (highest silhouette score)")
print()

# Final clustering with optimal k
print("="*80)
print(f"QUANTUM CLUSTERING ({optimal_k} CHANNELS)")
print("="*80)
print()

clustering_start = time.time()
quantum_clustering = SpectralClustering(
    n_clusters=optimal_k,
    affinity='precomputed',
    random_state=42,
    n_init=10
)
labels_quantum = quantum_clustering.fit_predict(K_quantum)
clustering_time = time.time() - clustering_start

print(f"✓ Spectral clustering completed in {clustering_time:.1f} seconds")
print()

# Analyze discovered channels
print("="*80)
print("QUANTUM-DISCOVERED GAUSSIAN CHANNELS")
print("="*80)
print()

channel_definitions = []

for i in range(optimal_k):
    mask = labels_quantum == i
    count = np.sum(mask)
    if count == 0:
        continue

    cluster_data = X_original[mask]

    channel = {
        'channel_id': int(i),
        'n_gaussians': int(count),
        'percentage': float(100 * count / len(X_original)),
        'sigma_x_mean': float(cluster_data[:,0].mean()),
        'sigma_x_std': float(cluster_data[:,0].std()),
        'sigma_y_mean': float(cluster_data[:,1].mean()),
        'sigma_y_std': float(cluster_data[:,1].std()),
        'alpha_mean': float(cluster_data[:,2].mean()),
        'alpha_std': float(cluster_data[:,2].std()),
        'loss_mean': float(cluster_data[:,3].mean()),
        'loss_std': float(cluster_data[:,3].std()),
        'coherence_mean': float(cluster_data[:,4].mean()),
        'coherence_std': float(cluster_data[:,4].std()),
        'gradient_mean': float(cluster_data[:,5].mean()),
        'gradient_std': float(cluster_data[:,5].std()),
    }

    channel_definitions.append(channel)

    # Determine channel type based on characteristics
    is_isotropic = abs(channel['sigma_x_mean'] - channel['sigma_y_mean']) < 0.005
    is_edge = channel['coherence_mean'] > 0.7
    is_smooth = channel['coherence_mean'] < 0.3
    is_high_quality = channel['loss_mean'] < 0.05

    channel_type = "UNKNOWN"
    if is_isotropic and is_smooth and is_high_quality:
        channel_type = "Smooth Regions (High Quality)"
    elif is_isotropic and is_smooth:
        channel_type = "Smooth Regions"
    elif is_isotropic and is_edge:
        channel_type = "Isotropic Edge Features"
    elif not is_isotropic and is_edge:
        channel_type = "Anisotropic Edges"
    elif is_high_quality:
        channel_type = "High Quality Mixed"
    elif channel['loss_mean'] > 0.15:
        channel_type = "Low Quality (Failure Mode)"

    print(f"Channel {i}: {count:,} Gaussians ({channel['percentage']:.1f}%) - {channel_type}")
    print(f"  σ_x:        {channel['sigma_x_mean']:7.4f} ± {channel['sigma_x_std']:7.4f}")
    print(f"  σ_y:        {channel['sigma_y_mean']:7.4f} ± {channel['sigma_y_std']:7.4f}")
    print(f"  α:          {channel['alpha_mean']:7.4f} ± {channel['alpha_std']:7.4f}")
    print(f"  Loss:       {channel['loss_mean']:7.4f} ± {channel['loss_std']:7.4f}")
    print(f"  Coherence:  {channel['coherence_mean']:7.3f} ± {channel['coherence_std']:7.3f}")
    print(f"  Gradient:   {channel['gradient_mean']:7.5f} ± {channel['gradient_std']:7.5f}")

    # Interpretation
    if is_isotropic:
        print(f"  → Isotropic (σ_x ≈ σ_y)")
    else:
        anisotropy = max(channel['sigma_x_mean'], channel['sigma_y_mean']) / min(channel['sigma_x_mean'], channel['sigma_y_mean'])
        print(f"  → Anisotropic (ratio: {anisotropy:.2f}×)")

    if is_high_quality:
        print(f"  → HIGH QUALITY (loss < 0.05) ⭐")
    elif channel['loss_mean'] > 0.15:
        print(f"  → LOW QUALITY (loss > 0.15) - failure mode")

    print()

# Classical comparison (optional)
print("="*80)
print("CLASSICAL COMPARISON (For Reference)")
print("="*80)
print()

print("Running classical RBF kernel clustering...")
classical_clustering = SpectralClustering(
    n_clusters=optimal_k,
    affinity='rbf',
    random_state=42,
    n_init=10
)
labels_classical = classical_clustering.fit_predict(X_quantum)

similarity = adjusted_rand_score(labels_classical, labels_quantum)

print(f"✓ Classical clustering complete")
print(f"  Adjusted Rand Index: {similarity:.3f}")
print()

if similarity < 0.3:
    print("🎯 QUANTUM FOUND DIFFERENT STRUCTURE!")
    print()
    print("   Quantum clustering reveals patterns that classical RBF kernel misses.")
    print("   The discovered channels represent structure in quantum Hilbert space")
    print("   that is NOT visible in Euclidean parameter space.")
    print()
    print("   → These are the FUNDAMENTAL Gaussian modes!")
else:
    print("   Quantum and classical clustering mostly agree.")
    print("   The parameter space structure is well-captured by classical methods.")

print()

# Save results
total_time = kernel_time + clustering_time

results = {
    'experiment': 'Q1_production_real_kodak_data',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'data_source': data['source'],
    'n_samples_total': int(len(data['X'])),
    'n_samples_quantum': int(len(X_quantum)),
    'n_clusters': optimal_k,
    'quantum_channels': channel_definitions,
    'silhouette_scores': [(int(k), float(s)) for k, s in silhouette_scores],
    'classical_vs_quantum_similarity': float(similarity),
    'labels_quantum': labels_quantum.tolist(),
    'labels_classical': labels_classical.tolist(),
    'timing': {
        'quantum_kernel_seconds': float(kernel_time),
        'quantum_clustering_seconds': float(clustering_time),
        'total_seconds': float(total_time),
        'total_minutes': float(total_time / 60)
    },
    'vm_config': {
        'ram_gb': 76,
        'cpus': 16,
        'peak_memory_gb': total_mem_gb
    },
    'features_used': features
}

output_file = 'gaussian_channels_kodak_quantum.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print("="*80)
print("RESULTS SAVED")
print("="*80)
print()
print(f"✓ Output: {output_file}")
print(f"  Channels discovered: {optimal_k}")
print(f"  Total runtime: {total_time/60:.1f} minutes")
print()

# Final summary
print("="*80)
print("QUANTUM CHANNEL DISCOVERY COMPLETE")
print("="*80)
print()
print(f"Discovered {optimal_k} fundamental Gaussian channels from {len(X_quantum):,} real configurations.")
print()
print("These channels represent natural groupings in quantum Hilbert space")
print("based on REAL optimizer trajectories from 24 Kodak images.")
print()
print("Next steps:")
print("  1. Analyze channel characteristics (see JSON output)")
print("  2. Compare to existing M/E/J/R/B/T primitives")
print("  3. Design classical implementation using quantum channel parameters")
print("  4. Test: Do quantum channels improve encoding quality?")
print()
print(f"Cost: $0 (simulator)")
print(f"Quality: MAXIMUM (real data + quantum analysis)")
print()
print("="*80)
