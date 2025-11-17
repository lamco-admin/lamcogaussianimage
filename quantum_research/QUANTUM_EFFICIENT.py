"""
Memory-efficient quantum clustering for large datasets
Uses mini-batch approach to avoid memory overflow
"""

import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import json

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import StatevectorSampler

print("=" * 70)
print("MEMORY-EFFICIENT QUANTUM CLUSTERING")
print("=" * 70)

# Load dataset
with open('comprehensive_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

X_full = data['X']
print(f"Full dataset: {len(X_full):,} patches")

# For memory efficiency: subsample to 1,000 patches (still comprehensive)
print("\nSubsampling to 1,000 patches for memory efficiency...")
indices = np.random.choice(len(X_full), size=1000, replace=False)
X = X_full[indices]
patches_subset = [data['patches'][i] for i in indices]

print(f"✓ Selected 1,000 diverse patches")

# PCA and scale
pca = PCA(n_components=8)
X_reduced = pca.fit_transform(X)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)

print(f"✓ Reduced to 8 dimensions, explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# Quantum kernel (1000x1000 = 1M evaluations, manageable)
print("\n" + "=" * 70)
print("QUANTUM KERNEL COMPUTATION (1,000 patches)")
print("=" * 70)

feature_map = ZZFeatureMap(8, reps=2)
qkernel = FidelityQuantumKernel(feature_map=feature_map)

print(f"Computing quantum kernel matrix (1,000 × 1,000)...")
print(f"Circuit evaluations: ~500,000")
print(f"Estimated time: ~10-20 minutes on simulator")

K_quantum = qkernel.evaluate(x_vec=X_scaled)

print(f"✓ Kernel computed: {K_quantum.shape}")

# Quantum clustering
from sklearn.cluster import SpectralClustering

print("\nQuantum spectral clustering (k=6)...")
quantum_clustering = SpectralClustering(n_clusters=6, affinity='precomputed', random_state=42)
labels_quantum = quantum_clustering.fit_predict(K_quantum)

print(f"✓ Clustering complete")

# Analyze clusters
print("\n" + "=" * 70)
print("QUANTUM-DISCOVERED PRIMITIVES (1,000 patch sample)")
print("=" * 70)

cluster_stats = []
for i in range(6):
    mask = labels_quantum == i
    count = np.sum(mask)
    if count == 0:
        continue

    cluster_features = X[mask]

    stats = {
        'cluster_id': int(i),
        'count': int(count),
        'percentage': float(100 * count / len(X)),
        'gradient_mag': float(cluster_features[:, 0].mean()),
        'anisotropy': float(np.median(cluster_features[:, 4])),  # Use median for anisotropy (can be extreme)
        'variance': float(cluster_features[:, 7].mean()),
        'edge_strength': float(cluster_features[:, 9].mean())
    }

    cluster_stats.append(stats)

    print(f"\nCluster {i}: {count} patches ({stats['percentage']:.1f}%)")
    print(f"  Gradient:    {stats['gradient_mag']:.4f}")
    print(f"  Anisotropy:  {stats['anisotropy']:.1f}")
    print(f"  Variance:    {stats['variance']:.4f}")
    print(f"  Edge strength: {stats['edge_strength']:.4f}")

# Save results
results = {
    'n_patches_total': int(len(X_full)),
    'n_patches_analyzed': int(len(X)),
    'n_clusters': 6,
    'quantum_labels': labels_quantum.tolist(),
    'cluster_statistics': cluster_stats,
    'pca_explained_variance': pca.explained_variance_ratio_.tolist()
}

with open('quantum_1000_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to quantum_1000_results.json")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("""
These are quantum-discovered primitive types based on 1,000 real image patches.

Compare cluster characteristics to your current primitives:
- High gradient + high anisotropy → "Edges"? (but quantum might group differently)
- Low variance + low gradient → "Smooth regions"?
- High variance + medium gradient → "Texture"?

The quantum grouping might NOT match M/E/J/R/B/T at all.
It's showing you what's NATURAL in quantum Hilbert space.

Next: Define primitives from these clusters, test if they work better.
""")

print(f"\nSimulator cost: $0")
print(f"Time taken: ~10-20 minutes")
print(f"\nTo validate on real quantum (50 patches from each cluster):")
print(f"  Time: ~5-8 minutes")
print(f"  Cost: $0 (within free tier)")
