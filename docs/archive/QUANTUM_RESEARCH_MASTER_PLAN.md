# Quantum Research Master Plan - Real Gaussian Data Collection & Analysis

**Created**: 2025-12-04
**Status**: Ready for VM resize and execution
**VM Current**: 22GB RAM, 8 CPUs
**VM Target**: 70GB RAM, 8 CPUs

---

## Executive Summary

This plan generates **REAL Gaussian configuration data** from actual image encoding, then uses **quantum clustering** to discover fundamental Gaussian "channels" - the natural primitives for Gaussian image representation.

**Key Insight**: Instead of generating synthetic random Gaussians, we extract actual configurations from the optimizer as it encodes real photos. This gives us:
- Real parameter combinations that actually work
- Quality scores from real usage
- Context about what image features each Gaussian represents
- Failure modes (what configurations don't work)

**Timeline**: ~3-4 hours total (mostly automated)
**Data Generated**: 24,000-50,000 real Gaussian configurations
**Quantum Analysis**: 1,500 samples, 22-37 minutes
**Output**: 4-6 discovered Gaussian channels with statistical validation

---

## Phase 1: Instrument Encoder for Data Collection

### 1.1 Current State Analysis

**Encoder Location**: `packages/lgi-rs/lgi-encoder-v2/src/lib.rs`

**Key Method**: `encode_error_driven_adam()` (line 427)
- Uses Adam optimizer
- 100 iterations per pass
- Up to 10 refinement passes (adaptive Gaussian placement)
- Achieves +9.69 dB improvement (validated October 2025)

**Optimizer Location**: `packages/lgi-rs/lgi-encoder-v2/src/optimizer_v2.rs`
- Main optimization loop: lines 140-200
- Currently NO intermediate logging
- Only final results returned

### 1.2 What Data to Collect

For each Gaussian during each optimization iteration:

**Core Parameters:**
```rust
struct GaussianSnapshot {
    // Gaussian parameters
    position_x: f32,      // Normalized [0,1]
    position_y: f32,      // Normalized [0,1]
    sigma_perp: f32,      // Perpendicular scale
    sigma_parallel: f32,  // Parallel scale
    rotation: f32,        // Rotation angle (radians)
    alpha: f32,           // Opacity [0,1]
    color_r: f32,
    color_g: f32,
    color_b: f32,

    // Quality metrics
    local_psnr: f32,      // PSNR contribution of this Gaussian
    global_psnr: f32,     // Total PSNR of all Gaussians
    loss_contribution: f32, // This Gaussian's loss contribution

    // Context
    iteration: u32,       // Which optimization iteration
    refinement_pass: u32, // Which refinement pass (0-9)
    image_id: String,     // Which Kodak image (kodim01-24)
    gaussian_id: usize,   // Index in Gaussian array

    // Image context
    edge_coherence: f32,  // Structure tensor coherence at position
    local_gradient: f32,  // Gradient magnitude at position
    geodesic_dist: f32,   // Geodesic distance at position
}
```

**Why This Data:**
- `sigma_perp, sigma_parallel, alpha` → Core Gaussian parameters for quantum clustering
- `rotation, position` → Contextual information
- `local_psnr, loss_contribution` → Quality achieved (what works vs what fails)
- `edge_coherence, local_gradient` → Image context (what this Gaussian was trying to represent)

### 1.3 Implementation Strategy

**Approach**: Modify `OptimizerV2::optimize()` to log snapshots

**File to Modify**: `packages/lgi-rs/lgi-encoder-v2/src/optimizer_v2.rs`

**Changes Required:**

1. Add logging callback parameter to `optimize()`:
```rust
pub fn optimize(
    &mut self,
    gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
    target: &ImageBuffer<f32>,
    logger: Option<&mut GaussianDataLogger>,  // NEW
) -> f32
```

2. Inside optimization loop (after each iteration), call logger:
```rust
for iter in 0..self.max_iterations {
    // ... existing rendering and gradient computation ...

    // NEW: Log Gaussian states
    if let Some(ref mut log) = logger {
        for (idx, gaussian) in gaussians.iter().enumerate() {
            log.record_snapshot(gaussian, idx, iter, &target, &rendered);
        }
    }

    // ... existing parameter updates ...
}
```

3. Create `GaussianDataLogger` struct:
```rust
pub struct GaussianDataLogger {
    snapshots: Vec<GaussianSnapshot>,
    image_id: String,
    refinement_pass: u32,
}

impl GaussianDataLogger {
    pub fn new(image_id: String, pass: u32) -> Self { ... }

    pub fn record_snapshot(
        &mut self,
        gaussian: &Gaussian2D<f32, Euler<f32>>,
        gaussian_id: usize,
        iteration: u32,
        target: &ImageBuffer<f32>,
        rendered: &ImageBuffer<f32>,
    ) { ... }

    pub fn save_to_file(&self, path: &str) -> Result<()> { ... }
}
```

**Storage Format**: CSV or Binary (npz for Python compatibility)

**Estimated Size**:
- 24 images × 10 passes × 100 iterations × 200 Gaussians avg × 100 bytes/record
- = 4.8GB uncompressed
- = ~500MB compressed

### 1.4 Alternative: Python Wrapper Approach

**If modifying Rust is complex**, create Python wrapper that:
1. Calls encoder via FFI/CLI
2. Intercepts results after each pass
3. Saves Gaussian states
4. Re-initializes encoder for next pass with logging

This is SLOWER but requires no Rust modification.

---

## Phase 2: Data Collection from Kodak Dataset

### 2.1 Kodak Dataset Specifications

**Location**: `test-data/kodak-dataset/`
**Images**: kodim01.png through kodim24.png
**Count**: 24 images
**Size**: 768×512 pixels (393,216 pixels)
**Format**: PNG, 8-bit sRGB

**Why Kodak Over 4K:**
- **21× faster encoding** (393K pixels vs 8.3M pixels)
- **24 hours saved** (2 hours vs 24+ hours for 4K)
- **Industry standard** for image quality benchmarking
- **Content diversity**: All image types represented (smooth, edges, texture, detail)
- **Gaussian density**: Produces 100-500 Gaussians/image (manageable for analysis)

### 2.2 Encoding Configuration

**Method**: `encode_error_driven_adam()`

**Parameters**:
```rust
initial_gaussians: 25   // Start with 5×5 grid
max_gaussians: 500      // Allow up to 500 Gaussians
```

**Expected Behavior**:
- Pass 0: Initialize 25 Gaussians (grid)
- Pass 0: Optimize 100 iterations → ~14-18 dB
- Pass 1-9: Add Gaussians at high-error regions, optimize
- Final: 200-500 Gaussians, 24-28 dB PSNR

**Per Image**:
- ~10 refinement passes
- ~100 iterations per pass
- ~200-300 Gaussians on average
- = **~200,000 Gaussian snapshots per image**

**All 24 Images**:
- 24 × 200,000 = **~4.8 million Gaussian snapshots**
- After filtering (keep every 10th iteration): **~480,000 snapshots**

### 2.3 Collection Script

**File**: `quantum_research/collect_kodak_gaussian_data.py`

```python
"""
Collect real Gaussian configuration data from Kodak encoding.

Runs encoder on all 24 Kodak images, logs every Gaussian configuration
during optimization, saves to dataset for quantum analysis.
"""

import subprocess
import json
import numpy as np
from pathlib import Path
import time

KODAK_DIR = Path("../test-data/kodak-dataset")
OUTPUT_DIR = Path("./kodak_gaussian_data")
OUTPUT_DIR.mkdir(exist_ok=True)

def encode_single_image(image_path, image_id):
    """
    Encode a single Kodak image and collect Gaussian data.

    Calls Rust encoder with logging enabled.
    """
    print(f"\n{'='*80}")
    print(f"Encoding {image_id}: {image_path.name}")
    print(f"{'='*80}")

    start_time = time.time()

    # Call Rust encoder with data logging
    # (Assumes we've built a CLI tool that accepts --log-gaussians flag)
    result = subprocess.run([
        "../packages/lgi-rs/target/release/lgi-encode",
        "--input", str(image_path),
        "--method", "adam",
        "--initial-gaussians", "25",
        "--max-gaussians", "500",
        "--log-gaussians", str(OUTPUT_DIR / f"{image_id}_gaussians.csv"),
        "--log-interval", "10",  # Log every 10th iteration
    ], capture_output=True, text=True)

    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return None

    print(f"✓ Completed in {elapsed/60:.1f} minutes")

    # Parse output for final PSNR
    for line in result.stdout.split('\n'):
        if 'Final PSNR' in line:
            print(f"  {line.strip()}")

    return OUTPUT_DIR / f"{image_id}_gaussians.csv"

def main():
    print("="*80)
    print("KODAK GAUSSIAN DATA COLLECTION")
    print("="*80)
    print()
    print("Configuration:")
    print("  Images: 24 (kodim01-24)")
    print("  Method: encode_error_driven_adam")
    print("  Initial Gaussians: 25")
    print("  Max Gaussians: 500")
    print("  Logging: Every 10th iteration")
    print()
    print("Expected:")
    print("  Time per image: 3-5 minutes")
    print("  Total time: 1-2 hours")
    print("  Snapshots collected: ~480,000")
    print("  Disk space: ~500MB compressed")
    print()

    # Find all Kodak images
    kodak_images = sorted(KODAK_DIR.glob("kodim*.png"))[:24]

    print(f"Found {len(kodak_images)} Kodak images")
    print()

    collected_files = []

    for i, image_path in enumerate(kodak_images, 1):
        image_id = image_path.stem  # kodim01, kodim02, etc.

        print(f"[{i}/24] Processing {image_id}...")

        result_file = encode_single_image(image_path, image_id)

        if result_file:
            collected_files.append(result_file)
            print(f"  ✓ Data saved to {result_file}")

        # Progress update
        if i % 5 == 0:
            elapsed_total = time.time() - start_time_total
            avg_time = elapsed_total / i
            remaining = (24 - i) * avg_time
            print()
            print(f"Progress: {i}/24 ({100*i/24:.0f}%)")
            print(f"Average time/image: {avg_time/60:.1f} min")
            print(f"Estimated remaining: {remaining/60:.0f} min")
            print()

    print()
    print("="*80)
    print("COLLECTION COMPLETE")
    print("="*80)
    print(f"Files collected: {len(collected_files)}/24")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    print("Next step: Run prepare_quantum_dataset.py")

if __name__ == "__main__":
    import time
    start_time_total = time.time()
    main()
```

### 2.4 Expected Runtime

**Conservative Estimate**:
- 3-5 minutes per image (768×512 encoding)
- 24 images × 4 min avg = **96 minutes (~1.5 hours)**

**Optimistic Estimate**:
- 2-3 minutes per image (if encoder is fast)
- 24 images × 2.5 min = **60 minutes (~1 hour)**

**Disk Space**:
- Raw CSV: ~4.8GB
- Compressed (gzip): ~500MB
- Processed for quantum: ~50MB

---

## Phase 3: Prepare Dataset for Quantum Analysis

### 3.1 Data Processing Pipeline

**Input**: 24 CSV files (one per Kodak image)
**Output**: Single numpy array ready for quantum kernel

**File**: `quantum_research/prepare_quantum_dataset.py`

```python
"""
Process raw Gaussian data into quantum-ready format.

Loads all Kodak Gaussian snapshots, filters, normalizes, and
prepares for quantum kernel computation.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle

def load_all_gaussian_data():
    """Load all CSV files and combine."""
    data_dir = Path("./kodak_gaussian_data")
    csv_files = sorted(data_dir.glob("kodim*_gaussians.csv"))

    print(f"Loading {len(csv_files)} CSV files...")

    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
        print(f"  {csv_file.name}: {len(df):,} snapshots")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal snapshots: {len(combined):,}")

    return combined

def filter_representative_samples(df, target_samples=10000):
    """
    Select diverse representative samples.

    Strategy:
    1. Keep final iteration from each pass (best quality)
    2. Keep snapshots that achieved high PSNR (>20 dB)
    3. Stratified sampling across images
    4. Diversity sampling in parameter space
    """
    print(f"\nFiltering to {target_samples:,} representative samples...")

    # Strategy 1: Keep final iterations (highest quality)
    final_iters = df[df['iteration'] >= 90]  # Last 10 iterations
    print(f"  Final iterations: {len(final_iters):,}")

    # Strategy 2: Keep high-quality Gaussians
    high_quality = df[df['local_psnr'] > 20.0]
    print(f"  High quality (>20dB): {len(high_quality):,}")

    # Strategy 3: Stratified by image
    stratified = []
    samples_per_image = target_samples // 24
    for image_id in df['image_id'].unique():
        image_data = df[df['image_id'] == image_id]
        sample = image_data.sample(n=min(samples_per_image, len(image_data)), random_state=42)
        stratified.append(sample)

    stratified_df = pd.concat(stratified)
    print(f"  Stratified sampling: {len(stratified_df):,}")

    # Combine strategies
    combined = pd.concat([final_iters, high_quality, stratified_df]).drop_duplicates()

    # If still too many, random sample
    if len(combined) > target_samples:
        combined = combined.sample(n=target_samples, random_state=42)

    print(f"  Final count: {len(combined):,}")

    return combined

def extract_features(df):
    """
    Extract feature vectors for quantum kernel.

    Features:
    - sigma_perp (Gaussian perpendicular scale)
    - sigma_parallel (Gaussian parallel scale)
    - alpha (opacity)
    - local_psnr (quality achieved)
    - edge_coherence (what it was representing)
    - local_gradient (image complexity)
    """
    features = df[[
        'sigma_perp',
        'sigma_parallel',
        'alpha',
        'local_psnr',
        'edge_coherence',
        'local_gradient'
    ]].values

    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Feature ranges:")
    print(f"  sigma_perp: [{features[:,0].min():.4f}, {features[:,0].max():.4f}]")
    print(f"  sigma_parallel: [{features[:,1].min():.4f}, {features[:,1].max():.4f}]")
    print(f"  alpha: [{features[:,2].min():.4f}, {features[:,2].max():.4f}]")
    print(f"  local_psnr: [{features[:,3].min():.2f}, {features[:,3].max():.2f}]")

    return features

def main():
    print("="*80)
    print("PREPARING QUANTUM DATASET FROM KODAK GAUSSIAN DATA")
    print("="*80)
    print()

    # Load all data
    df = load_all_gaussian_data()

    # Filter to representative samples
    filtered = filter_representative_samples(df, target_samples=10000)

    # Extract features
    X = extract_features(filtered)

    # Normalize
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save for quantum analysis
    output_file = "kodak_gaussians_quantum_ready.pkl"

    with open(output_file, 'wb') as f:
        pickle.dump({
            'X': X,
            'X_scaled': X_scaled,
            'scaler': scaler,
            'metadata': filtered[['image_id', 'gaussian_id', 'iteration', 'refinement_pass']].values,
            'n_samples': len(X),
            'features': ['sigma_perp', 'sigma_parallel', 'alpha', 'local_psnr', 'edge_coherence', 'local_gradient']
        }, f)

    print(f"\n✓ Dataset saved to {output_file}")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: 6")
    print(f"  Memory: {X_scaled.nbytes / 1024**2:.1f} MB")
    print()
    print("Ready for quantum clustering!")
    print("Next step: Run Q1_production_real_data.py")

if __name__ == "__main__":
    main()
```

### 3.2 Subsample for Quantum (If Needed)

If we collect 10,000+ samples but want 1,500 for quantum:

```python
def subsample_for_quantum(X_scaled, target_samples=1500, method='diverse'):
    """
    Subsample to target size using diversity-preserving strategy.

    Methods:
    - 'diverse': K-means clustering, select centroids + boundary points
    - 'random': Simple random sampling
    - 'stratified': Stratified by parameter ranges
    """
    if method == 'diverse':
        from sklearn.cluster import MiniBatchKMeans

        # Cluster into target_samples clusters
        kmeans = MiniBatchKMeans(n_clusters=target_samples, random_state=42)
        labels = kmeans.fit_predict(X_scaled)

        # Select samples closest to centroids
        selected_indices = []
        for i in range(target_samples):
            cluster_points = np.where(labels == i)[0]
            centroid = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(X_scaled[cluster_points] - centroid, axis=1)
            closest = cluster_points[np.argmin(distances)]
            selected_indices.append(closest)

        return X_scaled[selected_indices]

    elif method == 'random':
        indices = np.random.choice(len(X_scaled), target_samples, replace=False)
        return X_scaled[indices]

    else:
        raise ValueError(f"Unknown method: {method}")
```

---

## Phase 4: Quantum Clustering on Real Data

### 4.1 VM Resource Requirements

**CRITICAL**: VM must be resized BEFORE running quantum analysis

**Current VM**: 22GB RAM, 8 CPUs
**Required VM**: 70GB RAM, 8 CPUs

**Why 70GB:**
- 1,500 samples requires 64.4GB peak memory
- 5.6GB headroom for safety
- See `RESOURCE_REQUIREMENTS.md` for detailed calculations

**CPU Count**: 8 is optimal
- Quantum kernel doesn't parallelize (6 qubits < 14 qubit threshold)
- NumPy operations use multi-threading automatically
- More CPUs won't help significantly

### 4.2 Quantum Script

**File**: `quantum_research/Q1_production_real_data.py`

```python
"""
Q1: Quantum Gaussian Channel Discovery - PRODUCTION (Real Kodak Data)

Uses REAL Gaussian configurations extracted from Kodak encoding.
Discovers fundamental Gaussian channels via quantum kernel clustering.

REQUIREMENTS:
- VM RAM: 70GB
- VM CPUs: 8
- Runtime: 22-37 minutes
- Input: kodak_gaussians_quantum_ready.pkl
"""

import numpy as np
import pickle
import json
import time
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

# Quantum imports
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

print("="*80)
print("Q1: QUANTUM GAUSSIAN CHANNEL DISCOVERY - PRODUCTION")
print("Using REAL Gaussian data from Kodak encoding")
print("="*80)
print()

# Load dataset
print("Loading Kodak Gaussian dataset...")
with open('kodak_gaussians_quantum_ready.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
X_scaled = data['X_scaled']
metadata = data['metadata']
features = data['features']

print(f"✓ Loaded {len(X):,} real Gaussian configurations")
print(f"  Features: {', '.join(features)}")
print()

# Subsample to 1,500 for quantum (if needed)
if len(X_scaled) > 1500:
    print(f"Subsampling to 1,500 diverse samples...")
    from sklearn.cluster import MiniBatchKMeans

    kmeans = MiniBatchKMeans(n_clusters=1500, random_state=42, batch_size=1000)
    labels_kmeans = kmeans.fit_predict(X_scaled)

    selected_indices = []
    for i in range(1500):
        cluster_points = np.where(labels_kmeans == i)[0]
        if len(cluster_points) > 0:
            centroid = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(X_scaled[cluster_points] - centroid, axis=1)
            closest = cluster_points[np.argmin(distances)]
            selected_indices.append(closest)

    X_quantum = X_scaled[selected_indices]
    X_original = X[selected_indices]
    metadata_quantum = metadata[selected_indices]

    print(f"✓ Selected 1,500 diverse samples")
else:
    X_quantum = X_scaled
    X_original = X
    metadata_quantum = metadata

print()
print("="*80)
print("QUANTUM KERNEL COMPUTATION")
print("="*80)
print()

# Map 6D features to 8 qubits (pad with zeros)
n_features = X_quantum.shape[1]
n_qubits = 8
X_padded = np.pad(X_quantum, ((0,0), (0, n_qubits - n_features)), mode='constant')

print(f"Quantum circuit configuration:")
print(f"  Features: {n_features} → {n_qubits} qubits (zero-padded)")
print(f"  Samples: {len(X_padded):,}")
print(f"  Reps: 2")
print()

feature_map = ZZFeatureMap(n_qubits, reps=2)
qkernel = FidelityQuantumKernel(feature_map=feature_map)

n_evals = len(X_padded) * (len(X_padded) - 1) // 2
print(f"Computing quantum kernel matrix...")
print(f"  Kernel size: {len(X_padded)} × {len(X_padded)}")
print(f"  Unique evaluations: {n_evals:,}")
print(f"  Estimated time: 22-37 minutes")
print(f"  Peak memory: ~64.4 GB")
print()
print("⏳ This will take a while. Progress indicators:")
print("   - Python process will use ~50-60GB RAM")
print("   - CPU usage will be ~100% on one core")
print("   - No output until completion (output is buffered)")
print()

start_kernel = time.time()
K_quantum = qkernel.evaluate(x_vec=X_padded)
kernel_time = time.time() - start_kernel

print(f"✓ Quantum kernel computed in {kernel_time/60:.1f} minutes")
print(f"  Kernel shape: {K_quantum.shape}")
print()

# Determine optimal number of clusters
print("="*80)
print("DETERMINING OPTIMAL CLUSTER COUNT")
print("="*80)
print()

# Try 3-8 clusters, use silhouette score
silhouette_scores = []
for n_clusters in range(3, 9):
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    labels = clustering.fit_predict(K_quantum)
    score = silhouette_score(K_quantum, labels, metric='precomputed')
    silhouette_scores.append((n_clusters, score))
    print(f"  {n_clusters} clusters: silhouette = {score:.3f}")

optimal_k = max(silhouette_scores, key=lambda x: x[1])[0]
print()
print(f"✓ Optimal cluster count: {optimal_k} (highest silhouette score)")
print()

# Final clustering with optimal k
print("="*80)
print(f"QUANTUM CLUSTERING ({optimal_k} channels)")
print("="*80)
print()

clustering_start = time.time()
quantum_clustering = SpectralClustering(n_clusters=optimal_k, affinity='precomputed', random_state=42)
labels_quantum = quantum_clustering.fit_predict(K_quantum)
clustering_time = time.time() - clustering_start

print(f"✓ Clustering completed in {clustering_time:.1f} seconds")
print()

# Analyze channels
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
        'sigma_perp_mean': float(cluster_data[:,0].mean()),
        'sigma_perp_std': float(cluster_data[:,0].std()),
        'sigma_parallel_mean': float(cluster_data[:,1].mean()),
        'sigma_parallel_std': float(cluster_data[:,1].std()),
        'alpha_mean': float(cluster_data[:,2].mean()),
        'alpha_std': float(cluster_data[:,2].std()),
        'quality_mean': float(cluster_data[:,3].mean()),
        'quality_std': float(cluster_data[:,3].std()),
    }

    channel_definitions.append(channel)

    print(f"Channel {i}: {count} Gaussians ({channel['percentage']:.1f}%)")
    print(f"  σ_perp:     {channel['sigma_perp_mean']:6.4f} ± {channel['sigma_perp_std']:6.4f}")
    print(f"  σ_parallel: {channel['sigma_parallel_mean']:6.4f} ± {channel['sigma_parallel_std']:6.4f}")
    print(f"  α:          {channel['alpha_mean']:6.4f} ± {channel['alpha_std']:6.4f}")
    print(f"  Quality:    {channel['quality_mean']:6.2f} ± {channel['quality_std']:5.2f} dB")
    print()

# Save results
results = {
    'experiment': 'Q1_production_real_kodak_data',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'data_source': 'Kodak PhotoCD (24 images, real encoding)',
    'n_samples_total': int(len(X)),
    'n_samples_quantum': int(len(X_quantum)),
    'n_clusters': optimal_k,
    'quantum_channels': channel_definitions,
    'silhouette_scores': [(int(k), float(s)) for k, s in silhouette_scores],
    'labels_quantum': labels_quantum.tolist(),
    'timing': {
        'quantum_kernel_seconds': kernel_time,
        'quantum_clustering_seconds': clustering_time,
        'total_seconds': kernel_time + clustering_time
    },
    'vm_config': {
        'ram_gb': 70,
        'cpus': 8
    }
}

output_file = 'gaussian_channels_kodak_quantum.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to {output_file}")
print()

print("="*80)
print("QUANTUM-DISCOVERED GAUSSIAN CHANNELS (FROM REAL DATA)")
print("="*80)
print()
print("These channels represent NATURAL groupings discovered in quantum")
print("Hilbert space from REAL Gaussian configurations used in actual encoding.")
print()
print("Next steps:")
print("1. Analyze what each channel represents (edge, smooth, texture, etc.)")
print("2. Compare to existing M/E/J/R/B/T primitives")
print("3. Test classical implementation with discovered parameters")
print("4. Validate: Do quantum channels improve encoding quality?")
print()
print(f"Total runtime: {(kernel_time + clustering_time)/60:.1f} minutes")
print(f"Cost: $0 (simulator)")
print(f"Quality: MAXIMUM (real data + quantum analysis)")
print()
print("="*80)
```

### 4.3 Expected Output

**Console Output**:
```
================================================================================
Q1: QUANTUM GAUSSIAN CHANNEL DISCOVERY - PRODUCTION
Using REAL Gaussian data from Kodak encoding
================================================================================

✓ Loaded 10,000 real Gaussian configurations
  Features: sigma_perp, sigma_parallel, alpha, local_psnr, edge_coherence, local_gradient

✓ Selected 1,500 diverse samples

================================================================================
QUANTUM KERNEL COMPUTATION
================================================================================

Quantum circuit configuration:
  Features: 6 → 8 qubits (zero-padded)
  Samples: 1,500
  Reps: 2

Computing quantum kernel matrix...
  Kernel size: 1,500 × 1,500
  Unique evaluations: 1,124,250
  Estimated time: 22-37 minutes
  Peak memory: ~64.4 GB

⏳ This will take a while...

✓ Quantum kernel computed in 28.3 minutes
  Kernel shape: (1500, 1500)

================================================================================
DETERMINING OPTIMAL CLUSTER COUNT
================================================================================

  3 clusters: silhouette = 0.342
  4 clusters: silhouette = 0.418
  5 clusters: silhouette = 0.451
  6 clusters: silhouette = 0.389
  7 clusters: silhouette = 0.312
  8 clusters: silhouette = 0.287

✓ Optimal cluster count: 5 (highest silhouette score)

================================================================================
QUANTUM CLUSTERING (5 channels)
================================================================================

✓ Clustering completed in 3.2 seconds

Channel 0: 312 Gaussians (20.8%)
  σ_perp:     0.0143 ± 0.0082
  σ_parallel: 0.0856 ± 0.0234
  α:          0.1243 ± 0.0521
  Quality:    12.34 ± 3.21 dB

Channel 1: 287 Gaussians (19.1%)
  σ_perp:     0.0234 ± 0.0091
  σ_parallel: 0.0234 ± 0.0088
  α:          0.3421 ± 0.1123
  Quality:    24.12 ± 2.87 dB

... [3 more channels]

✓ Results saved to gaussian_channels_kodak_quantum.json

Total runtime: 31.5 minutes
Cost: $0 (simulator)
Quality: MAXIMUM (real data + quantum analysis)
```

**JSON Output** (`gaussian_channels_kodak_quantum.json`):
```json
{
  "experiment": "Q1_production_real_kodak_data",
  "timestamp": "2025-12-04 14:23:15",
  "data_source": "Kodak PhotoCD (24 images, real encoding)",
  "n_samples_total": 10000,
  "n_samples_quantum": 1500,
  "n_clusters": 5,
  "quantum_channels": [
    {
      "channel_id": 0,
      "n_gaussians": 312,
      "percentage": 20.8,
      "sigma_perp_mean": 0.0143,
      "sigma_perp_std": 0.0082,
      "sigma_parallel_mean": 0.0856,
      "sigma_parallel_std": 0.0234,
      "alpha_mean": 0.1243,
      "alpha_std": 0.0521,
      "quality_mean": 12.34,
      "quality_std": 3.21
    },
    ...
  ],
  "silhouette_scores": [
    [3, 0.342],
    [4, 0.418],
    [5, 0.451],
    ...
  ],
  "timing": {
    "quantum_kernel_seconds": 1698.2,
    "quantum_clustering_seconds": 3.2,
    "total_seconds": 1701.4
  }
}
```

---

## Phase 5: Analysis & Validation

### 5.1 Interpreting Quantum Channels

**Questions to Answer**:

1. **What does each channel represent?**
   - Channel with high σ_parallel, low σ_perp, high coherence → Edges?
   - Channel with similar σ_perp ≈ σ_parallel → Smooth regions?
   - Channel with high variance, medium quality → Texture?

2. **Do quantum channels match classical primitives?**
   - Compare to M/E/J/R/B/T system
   - Are quantum groupings fundamentally different?

3. **Which channels achieve high quality?**
   - Channel with quality_mean > 25 dB → successful configurations
   - Channel with quality_mean < 15 dB → failure modes

4. **What are the cluster boundaries?**
   - Where does "edge" channel transition to "smooth" channel?
   - Are boundaries sharp or fuzzy?

### 5.2 Classical Validation

**Next Experiment**: Test if quantum channels improve encoding

**Approach**:
1. Implement classical Gaussian placement using quantum channel rules
2. For each pixel, determine which channel applies (based on local coherence, gradient, etc.)
3. Initialize Gaussians with parameters from that channel
4. Compare quality vs current method

**Expected Outcome**:
- If quantum channels are good: +2-5 dB improvement
- If quantum channels are neutral: ~same quality
- If quantum channels are worse: Something went wrong in analysis

### 5.3 Validation Script Outline

```python
def validate_quantum_channels(channels):
    """
    Test quantum-discovered channels on fresh images.

    1. Load validation image (not in Kodak training set)
    2. For each pixel, compute local properties
    3. Assign to nearest quantum channel
    4. Initialize Gaussians with channel parameters
    5. Optimize and measure quality
    6. Compare vs baseline (current method)
    """
    pass
```

---

## Implementation Checklist

### Before VM Restart:
- [x] Document complete plan (this file)
- [x] Document resource requirements (RESOURCE_REQUIREMENTS.md)
- [x] Identify encoder architecture
- [x] Confirm Kodak dataset available
- [ ] Create implementation scripts (below)

### After VM Restart (70GB RAM):
- [ ] Verify VM has 70GB RAM: `free -h`
- [ ] Verify 8 CPUs: `nproc`
- [ ] Update quantum packages: `cd quantum_research && source venv/bin/activate && pip install --upgrade qiskit qiskit-machine-learning`

### Phase 1 Implementation:
- [ ] Create `GaussianDataLogger` in Rust (or Python wrapper)
- [ ] Modify `OptimizerV2::optimize()` to accept logger
- [ ] Test logging on single image
- [ ] Verify CSV format is correct

### Phase 2 Execution:
- [ ] Run `collect_kodak_gaussian_data.py`
- [ ] Monitor progress (1-2 hours)
- [ ] Verify 24 CSV files created
- [ ] Check disk space (~500MB)

### Phase 3 Execution:
- [ ] Run `prepare_quantum_dataset.py`
- [ ] Verify `kodak_gaussians_quantum_ready.pkl` created
- [ ] Check dataset size (~10,000 samples)

### Phase 4 Execution:
- [ ] Run `Q1_production_real_data.py`
- [ ] Monitor memory usage: `watch -n 5 free -h`
- [ ] Wait 22-37 minutes
- [ ] Verify `gaussian_channels_kodak_quantum.json` created

### Phase 5 Analysis:
- [ ] Load and examine channel definitions
- [ ] Interpret what each channel represents
- [ ] Design classical validation experiment
- [ ] Run validation on test images

---

## Troubleshooting Guide

### Problem: VM Runs Out of Memory During Quantum

**Symptoms**:
- Python process killed
- `free -h` shows 0 available RAM
- Swap usage at 100%

**Solutions**:
1. Verify VM has 70GB: `free -h` (should show ~70G total)
2. Reduce sample count: Change 1,500 → 1,200 in quantum script
3. Close other processes: `htop` and kill unnecessary programs
4. Check for memory leaks: Monitor `top` during execution

### Problem: Quantum Kernel Takes Too Long (>1 hour)

**Symptoms**:
- No output after 45+ minutes
- CPU still at 100%

**Solutions**:
- This is NORMAL for 1,500 samples
- Expected: 22-37 minutes, can be up to 60 minutes on slower systems
- Monitor with: `ps aux | grep python` to confirm still running
- If >90 minutes: Something wrong, kill and restart

### Problem: Encoding Collection Fails

**Symptoms**:
- Rust encoder crashes
- CSV files empty or corrupted

**Solutions**:
1. Test single image first: Manually run encoder on kodim01.png
2. Check Rust compilation: `cd packages/lgi-rs && cargo build --release`
3. Verify logging code works
4. Reduce logging frequency: Log every 20 iterations instead of 10

### Problem: Quantum Clustering Produces 1 Cluster

**Symptoms**:
- All samples assigned to single cluster
- Silhouette scores all negative

**Solutions**:
- Data may not have clear structure
- Try different feature combinations
- Check for feature scaling issues
- Verify quantum kernel computed correctly (non-zero off-diagonal)

---

## File Locations Reference

### Documentation:
- **Master Plan**: `/home/greg/gaussian-image-projects/lgi-project/QUANTUM_RESEARCH_MASTER_PLAN.md` (this file)
- **Resource Requirements**: `/home/greg/gaussian-image-projects/lgi-project/quantum_research/RESOURCE_REQUIREMENTS.md`

### Code to Modify:
- **Encoder**: `/home/greg/gaussian-image-projects/lgi-project/packages/lgi-rs/lgi-encoder-v2/src/lib.rs`
- **Optimizer**: `/home/greg/gaussian-image-projects/lgi-project/packages/lgi-rs/lgi-encoder-v2/src/optimizer_v2.rs`

### Scripts to Create:
- **Data Collection**: `/home/greg/gaussian-image-projects/lgi-project/quantum_research/collect_kodak_gaussian_data.py`
- **Data Preparation**: `/home/greg/gaussian-image-projects/lgi-project/quantum_research/prepare_quantum_dataset.py`
- **Quantum Analysis**: `/home/greg/gaussian-image-projects/lgi-project/quantum_research/Q1_production_real_data.py`

### Data Files:
- **Input**: `/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim*.png`
- **Collection Output**: `/home/greg/gaussian-image-projects/lgi-project/quantum_research/kodak_gaussian_data/*.csv`
- **Quantum Input**: `/home/greg/gaussian-image-projects/lgi-project/quantum_research/kodak_gaussians_quantum_ready.pkl`
- **Final Results**: `/home/greg/gaussian-image-projects/lgi-project/quantum_research/gaussian_channels_kodak_quantum.json`

---

## Timeline Summary

### Human Work:
- **Phase 1 (Implementation)**: 30-60 minutes
  - Modify Rust optimizer for logging
  - OR create Python wrapper
  - Test on single image

### Compute Work:
- **Phase 2 (Data Collection)**: 1-2 hours (automated)
  - Encode 24 Kodak images
  - Collect ~480,000 Gaussian snapshots

- **Phase 3 (Preparation)**: 2-5 minutes (automated)
  - Load and process CSV files
  - Create quantum-ready dataset

- **Phase 4 (Quantum)**: 22-37 minutes (automated)
  - Compute quantum kernel: ~25-30 min
  - Spectral clustering: ~2-5 min

### Analysis Work:
- **Phase 5 (Interpretation)**: Variable
  - Examine channel definitions: 15-30 min
  - Design validation: 30-60 min
  - Run validation: 1-2 hours

**Total Compute Time**: ~2-3 hours (mostly automated)
**Total Human Time**: ~2-3 hours (implementation + analysis)

---

## Success Criteria

### Phase 2 Success:
- ✓ 24 CSV files generated (one per Kodak image)
- ✓ Total snapshots: 400,000-600,000
- ✓ Disk usage: ~500MB compressed
- ✓ Files contain expected columns (sigma_perp, sigma_parallel, etc.)

### Phase 3 Success:
- ✓ `kodak_gaussians_quantum_ready.pkl` created
- ✓ Dataset contains 8,000-12,000 samples
- ✓ Features properly normalized (mean≈0, std≈1)
- ✓ No NaN or infinite values

### Phase 4 Success:
- ✓ Quantum kernel computation completes without crash
- ✓ Runtime: 20-40 minutes
- ✓ Peak memory: 50-65GB (within 70GB limit)
- ✓ 4-6 distinct clusters discovered
- ✓ Silhouette score > 0.3

### Phase 5 Success:
- ✓ Channels are interpretable (can explain what each represents)
- ✓ Channel parameters match expected ranges
- ✓ Quality metrics make sense (successful channels have higher PSNR)
- ✓ Validation shows improvement or neutral (not regression)

---

## Next Session Action Items

**When you resume after VM restart:**

1. **Verify VM Configuration**:
   ```bash
   free -h          # Should show ~70G total RAM
   nproc            # Should show 8 CPUs
   df -h            # Check disk space (need ~5GB free)
   ```

2. **Navigate to Project**:
   ```bash
   cd /home/greg/gaussian-image-projects/lgi-project
   ```

3. **Read This Document**:
   ```bash
   cat QUANTUM_RESEARCH_MASTER_PLAN.md
   ```

4. **Ask AI Assistant**:
   - "I've restarted the VM with 70GB RAM. Ready to implement Phase 1."
   - AI will guide you through implementation

5. **Stay Focused**:
   - This is a systematic, well-planned approach
   - Each phase builds on previous
   - Document everything as you go
   - Run benchmarks to validate results

---

## Alternative Approaches (If Issues Arise)

### Alternative 1: Python-Only Data Collection

**If Rust modification is too complex**:
- Create Python script that calls encoder via FFI
- Intercept results between refinement passes
- Slower but requires no Rust changes

### Alternative 2: Synthetic Data (Fallback)

**If real data collection fails**:
- Use existing `Q1_moderate_500_samples.py`
- Run with synthetic Gaussians
- Still provides value, but less realistic

### Alternative 3: Smaller Quantum Run

**If 1,500 samples OOM**:
- Reduce to 1,200 samples (56GB peak)
- Or 1,000 samples (48GB peak)
- Still scientifically valid, lower confidence

---

## References

- **Qiskit Documentation**: https://qiskit.org/documentation/
- **Quantum Machine Learning**: https://qiskit-community.github.io/qiskit-machine-learning/
- **Resource Requirements**: See `RESOURCE_REQUIREMENTS.md` in quantum_research/
- **Session Handoff**: See `docs/SESSION_HANDOFF.md` for project context
- **Development Standards**: See `docs/DEVELOPMENT_STANDARDS.md` for quality guidelines

---

**END OF MASTER PLAN**

*This document is comprehensive and self-contained. After VM restart, resume from Phase 1 implementation.*
