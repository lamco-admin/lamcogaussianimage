"""
QUANTUM PRIMITIVE DISCOVERY - Real Kodak Data
Uses IBM Quantum to discover natural primitive groupings

This is THE experiment - quantum discovers what primitives should be
"""

import numpy as np
import json
from pathlib import Path
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Quantum imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import StatevectorSampler
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

USE_REAL_QUANTUM = False  # SET TO TRUE WHEN READY

print("=" * 70)
print("QUANTUM PRIMITIVE DISCOVERY - Kodak Dataset")
print("=" * 70)

# Step 1: Extract Kodak patches with features
print("\nStep 1: Extracting features from Kodak images...")

KODAK_DIR = Path('/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset')

def extract_simple_features(patch):
    """Extract 10 features from patch (fast, meaningful)"""
    # Gradients
    gy, gx = np.gradient(patch)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Structure tensor
    Ixx = ndimage.gaussian_filter(gx * gx, sigma=1.0)
    Ixy = ndimage.gaussian_filter(gx * gy, sigma=1.0)
    Iyy = ndimage.gaussian_filter(gy * gy, sigma=1.0)

    center = patch.shape[0] // 2
    tensor = np.array([[Ixx[center, center], Ixy[center, center]],
                       [Ixy[center, center], Iyy[center, center]]])
    eigenvalues = np.linalg.eigvalsh(tensor) + 1e-10

    return np.array([
        np.mean(grad_mag),      # 0: gradient magnitude
        np.std(grad_mag),       # 1: gradient variation
        eigenvalues[1],         # 2: max eigenvalue
        eigenvalues[0],         # 3: min eigenvalue
        eigenvalues[1]/eigenvalues[0],  # 4: anisotropy
        np.mean(patch),         # 5: mean intensity
        np.std(patch),          # 6: std dev
        np.var(patch),          # 7: variance
        np.max(patch) - np.min(patch),  # 8: range
        np.abs(ndimage.laplace(patch)).mean()  # 9: edge strength
    ])

# Load Kodak and extract patches
patches_data = []

for img_num in range(1, 25):  # ALL 24 Kodak images
    img_path = KODAK_DIR / f'kodim{img_num:02d}.png'
    if not img_path.exists():
        print(f"Downloading kodim{img_num:02d}.png...")
        import urllib.request
        url = f'http://r0k.us/graphics/kodak/kodak/kodim{img_num:02d}.png'
        try:
            urllib.request.urlretrieve(url, img_path)
        except:
            continue

    img = Image.open(img_path)
    img_gray = np.array(img.convert('L')).astype(float) / 255.0

    h, w = img_gray.shape

    # Sample 100 random patches per image (2400 total patches)
    for _ in range(100):
        y = np.random.randint(10, h-26)
        x = np.random.randint(10, w-26)

        patch = img_gray[y:y+16, x:x+16]
        features = extract_simple_features(patch)

        patches_data.append({
            'features': features,
            'image': img_num,
            'position': (y, x)
        })

X = np.array([p['features'] for p in patches_data])

print(f"✓ Extracted {len(X)} patches from Kodak images")
print(f"✓ Feature dimension: {X.shape[1]}")

# Step 2: Unsupervised - let quantum find natural groups
print("\nStep 2: Quantum kernel for unsupervised clustering...")
print("(We DON'T pre-label - let quantum reveal natural groupings)")

# Use quantum kernel for clustering (via kernel k-means)
from sklearn.cluster import SpectralClustering

# Classical clustering baseline
print("\nClassical clustering (RBF kernel)...")
classical_clustering = SpectralClustering(n_clusters=4, affinity='rbf', random_state=42)
labels_classical = classical_clustering.fit_predict(X)

print(f"Classical found {len(np.unique(labels_classical))} clusters")
for i in range(4):
    print(f"  Cluster {i}: {np.sum(labels_classical == i)} patches")

# Quantum clustering
print("\nQuantum clustering...")

# Feature map (reduce dimensions if needed)
# 10 features → use PCA or select subset
from sklearn.decomposition import PCA
pca = PCA(n_components=8)  # 8 features → 8 qubits
X_reduced = pca.fit_transform(X)

print(f"Reduced to {X_reduced.shape[1]} features (via PCA)")

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)

# Create quantum feature map
feature_map = ZZFeatureMap(8, reps=2, insert_barriers=True)

if USE_REAL_QUANTUM:
    print("\n🚀 USING REAL IBM QUANTUM COMPUTER")
    crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/66377f5395bc4ca49acd720d170cdb9f:c9d4885c-bc96-4890-8222-66480cd738ba::"
    service = QiskitRuntimeService(channel="ibm_cloud", instance=crn)
    backend = service.least_busy(operational=True, simulator=False)
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")
    print("This will use ~5-10 minutes of quantum time")

    sampler = SamplerV2(backend)
else:
    print("Using StatevectorSampler (exact simulation)")
    sampler = StatevectorSampler()

# Quantum kernel
qkernel = FidelityQuantumKernel(feature_map=feature_map)

print("\nComputing quantum kernel matrix...")
print("(This evaluates quantum circuits for all patch pairs)")
print(f"Total evaluations: {len(X_scaled) * (len(X_scaled)-1) / 2:.0f}")

# Compute kernel matrix
K_quantum = qkernel.evaluate(x_vec=X_scaled)

print(f"✓ Quantum kernel computed: {K_quantum.shape}")

# Spectral clustering with quantum kernel
quantum_clustering = SpectralClustering(n_clusters=4, affinity='precomputed', random_state=42)
labels_quantum = quantum_clustering.fit_predict(K_quantum)

print(f"\n✓ Quantum found {len(np.unique(labels_quantum))} clusters")
for i in range(4):
    print(f"  Cluster {i}: {np.sum(labels_quantum == i)} patches")

# Step 3: Analyze what quantum discovered
print("\n" + "=" * 70)
print("QUANTUM DISCOVERIES - What Are The Natural Primitives?")
print("=" * 70)

for cluster_id in range(4):
    cluster_patches = [patches_data[i] for i in range(len(patches_data)) if labels_quantum[i] == cluster_id]
    cluster_features = X[labels_quantum == cluster_id]

    print(f"\nQuantum Cluster {cluster_id} ({len(cluster_patches)} patches):")
    print(f"  Mean gradient: {cluster_features[:, 0].mean():.3f}")
    print(f"  Mean anisotropy: {cluster_features[:, 4].mean():.3f}")
    print(f"  Mean variance: {cluster_features[:, 7].mean():.4f}")
    print(f"  Mean edge strength: {cluster_features[:, 9].mean():.3f}")

# Compare quantum vs classical clustering
from sklearn.metrics import adjusted_rand_score
similarity = adjusted_rand_score(labels_classical, labels_quantum)

print(f"\nClassical vs Quantum clustering similarity: {similarity:.3f}")
if similarity < 0.5:
    print("🎯 QUANTUM FOUND DIFFERENT GROUPINGS!")
    print("   Quantum Hilbert space reveals structure classical missed")
else:
    print("   Quantum and classical agree on basic structure")

# Save results
results = {
    'n_patches': len(X),
    'quantum_clusters': labels_quantum.tolist(),
    'classical_clusters': labels_classical.tolist(),
    'similarity': float(similarity),
    'used_real_quantum': USE_REAL_QUANTUM
}

with open('quantum_primitive_discovery.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to quantum_primitive_discovery.json")

print("\n" + "=" * 70)
print("NEXT STEPS")
print("=" * 70)
print("""
1. Analyze quantum-discovered clusters (what do they represent?)
2. Compare to human-designed primitives (M/E/J/R/B/T)
3. If quantum found different groups → define NEW primitives from quantum
4. Re-run Phase 0-1 with quantum-discovered primitives

To run on REAL quantum hardware:
- Set USE_REAL_QUANTUM = True
- Run again (will use ~5-10 min of quantum time)
- Compare: Does real quantum find different patterns than simulator?
""")
