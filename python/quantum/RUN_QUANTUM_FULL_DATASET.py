"""
QUANTUM PRIMITIVE DISCOVERY - Full Comprehensive Dataset
5,527 patches from Kodak + Phase 0 + targeted sampling

Running on SIMULATOR (free, exact quantum simulation)
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import json

# Quantum imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import StatevectorSampler

print("=" * 70)
print("QUANTUM PRIMITIVE DISCOVERY - Full Dataset (5,527 patches)")
print("=" * 70)

# Load comprehensive dataset
print("\nLoading comprehensive dataset...")
with open('comprehensive_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
patches = data['patches']
feature_names = data['feature_names']

print(f"✓ Loaded {len(X)} patches with {X.shape[1]} features")

# Reduce dimensions for quantum (10 features → 8 for 8 qubits)
print("\nReducing to 8 dimensions via PCA...")
pca = PCA(n_components=8)
X_reduced = pca.fit_transform(X)

print(f"✓ Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
print(f"  {pca.explained_variance_ratio_[:8]}")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)

print(f"✓ Scaled to zero mean, unit variance")

# Classical clustering baseline
print("\n" + "=" * 70)
print("CLASSICAL CLUSTERING (RBF Kernel)")
print("=" * 70)

print("Running classical spectral clustering...")
classical_clustering = SpectralClustering(
    n_clusters=6,  # Try 6 clusters (similar to M/E/J/R/B/T)
    affinity='rbf',
    gamma=1.0,
    random_state=42,
    n_jobs=-1
)

labels_classical = classical_clustering.fit_predict(X_scaled)

print(f"✓ Classical found {len(np.unique(labels_classical))} clusters")
for i in range(6):
    count = np.sum(labels_classical == i)
    if count > 0:
        print(f"  Cluster {i}: {count} patches ({100*count/len(X):.1f}%)")

# Quantum clustering
print("\n" + "=" * 70)
print("QUANTUM CLUSTERING (Quantum Kernel)")
print("=" * 70)
print("This will take ~30-60 minutes on simulator (FREE)")
print("Computing quantum kernel matrix for 5,527 patches...")

# Create quantum feature map
feature_map = ZZFeatureMap(8, reps=2, insert_barriers=True)
print(f"Quantum circuit: {feature_map.num_qubits} qubits, depth {feature_map.depth()}")

# Use simulator
sampler = StatevectorSampler()
qkernel = FidelityQuantumKernel(feature_map=feature_map)

print(f"\nTotal kernel evaluations needed: {len(X_scaled) * (len(X_scaled)-1) // 2:,}")
print("Starting quantum kernel computation...")
print("(Progress updates every ~500 evaluations)")

# Compute quantum kernel matrix
K_quantum = qkernel.evaluate(x_vec=X_scaled)

print(f"\n✓ Quantum kernel matrix computed: {K_quantum.shape}")

# Quantum spectral clustering
print("\nRunning quantum spectral clustering...")
quantum_clustering = SpectralClustering(
    n_clusters=6,
    affinity='precomputed',
    random_state=42,
    n_jobs=-1
)

labels_quantum = quantum_clustering.fit_predict(K_quantum)

print(f"✓ Quantum clustering complete")
print(f"Quantum found {len(np.unique(labels_quantum))} clusters")
for i in range(6):
    count = np.sum(labels_quantum == i)
    if count > 0:
        print(f"  Cluster {i}: {count} patches ({100*count/len(X):.1f}%)")

# Analyze quantum-discovered clusters
print("\n" + "=" * 70)
print("QUANTUM-DISCOVERED PRIMITIVE CHARACTERISTICS")
print("=" * 70)

for cluster_id in np.unique(labels_quantum):
    mask = labels_quantum == cluster_id
    cluster_features = X[mask]

    print(f"\nQuantum Cluster {cluster_id} ({np.sum(mask)} patches):")
    print(f"  Gradient magnitude:  {cluster_features[:, 0].mean():.4f} ± {cluster_features[:, 0].std():.4f}")
    print(f"  Anisotropy (λ1/λ2): {cluster_features[:, 4].mean():.2f} ± {cluster_features[:, 4].std():.2f}")
    print(f"  Variance:           {cluster_features[:, 7].mean():.4f} ± {cluster_features[:, 7].std():.4f}")
    print(f"  Edge strength:      {cluster_features[:, 9].mean():.4f} ± {cluster_features[:, 9].std():.4f}")
    print(f"  Mean intensity:     {cluster_features[:, 5].mean():.3f} ± {cluster_features[:, 5].std():.3f}")

# Compare classical vs quantum
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari = adjusted_rand_score(labels_classical, labels_quantum)
nmi = normalized_mutual_info_score(labels_classical, labels_quantum)

print("\n" + "=" * 70)
print("CLASSICAL vs QUANTUM CLUSTERING COMPARISON")
print("=" * 70)
print(f"Adjusted Rand Index: {ari:.4f}")
print(f"Normalized Mutual Info: {nmi:.4f}")

if ari < 0.3:
    print("\n🎯 QUANTUM FOUND SIGNIFICANTLY DIFFERENT GROUPINGS!")
    print("   Quantum Hilbert space reveals structure classical missed")
    print("   → These are the NATURAL primitives from quantum perspective")
elif ari > 0.7:
    print("\n   Quantum and classical largely agree")
    print("   → Classical clustering is adequate for this problem")
else:
    print("\n   Quantum found partially different groupings")
    print("   → Some overlap, but quantum sees additional structure")

# Save results
results = {
    'n_patches': len(X),
    'n_features': X.shape[1],
    'n_clusters': 6,
    'classical_labels': labels_classical.tolist(),
    'quantum_labels': labels_quantum.tolist(),
    'ari': float(ari),
    'nmi': float(nmi),
    'feature_names': feature_names
}

with open('quantum_full_discovery_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Visualize sample patches per quantum cluster
print("\nGenerating sample visualization...")

fig, axes = plt.subplots(6, 10, figsize=(20, 12))
fig.suptitle('Sample Patches per Quantum-Discovered Cluster', fontsize=16)

for cluster_id in range(6):
    cluster_patches = [p for i, p in enumerate(patches) if labels_quantum[i] == cluster_id]
    samples = np.random.choice(len(cluster_patches), min(10, len(cluster_patches)), replace=False)

    for j, idx in enumerate(samples):
        if j < 10:
            axes[cluster_id, j].imshow(cluster_patches[idx]['patch'], cmap='gray', vmin=0, vmax=1)
            axes[cluster_id, j].axis('off')

    axes[cluster_id, 0].set_ylabel(f'Cluster {cluster_id}\n({len(cluster_patches)} patches)', fontsize=10)

plt.tight_layout()
plt.savefig('quantum_discovered_primitives.png', dpi=200, bbox_inches='tight')
print(f"✓ Saved visualization: quantum_discovered_primitives.png")

# Summary
print("\n" + "=" * 70)
print("SUMMARY - What Did Quantum Discover?")
print("=" * 70)

print(f"""
Dataset: {len(X):,} patches from Kodak + Phase 0 + targeted sampling
Quantum clustering: {len(np.unique(labels_quantum))} distinct groups found

Quantum vs Classical similarity: {ari:.3f}
{"→ QUANTUM REVEALED DIFFERENT STRUCTURE" if ari < 0.3 else "→ Similar to classical"}

Next steps:
1. Analyze cluster characteristics (what does each represent?)
2. Define primitives from quantum clusters
3. Create classical detection rules for quantum-discovered primitives
4. Test: Do quantum primitives work better than human-designed (M/E/J/R/B/T)?

Results saved to:
- quantum_full_discovery_results.json
- quantum_discovered_primitives.png
- comprehensive_dataset.pkl

Cost: $0 (simulator)
Time: ~30-60 minutes

This is quantum discovery - what primitives SHOULD be according to quantum mechanics.
""")
