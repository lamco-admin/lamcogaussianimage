# Comprehensive Quantum Research Proposal for Gaussian Image Primitives

**Date:** November 23, 2025
**Project:** LGI (Lamco Gaussian Image) Codec
**Purpose:** Quantum-assisted discovery of optimal Gaussian primitive parameterization, placement, and iteration methods

---

## Executive Summary

This document presents a comprehensive research plan leveraging **quantum computing** to solve fundamental representation theory problems in Gaussian-based image encoding. The goal is to discover optimal strategies for three critical aspects:

1. **Parameterization**: What are the natural Gaussian primitive "modes" (like RGB channels for color)?
2. **Placement**: Where should Gaussians be positioned to faithfully represent images?
3. **Iteration Methods**: How should different Gaussian types be optimized?

**Key Innovation**: Use quantum computing for **one-time discovery** of classical rules, then deploy those rules classically forever (no runtime quantum dependency).

**Cost Strategy**:
- Primary research on FREE quantum simulators (exact, unlimited)
- Validation on FREE tier real quantum hardware (10 min/month IBM Quantum)
- Total research cost: **$0-$50** (potentially entirely free)

---

## Part 1: Project Overview & Current State

### 1.1 What is LGI (Lamco Gaussian Image)?

LGI is a novel image codec that represents images as **collections of 2D Gaussian primitives** instead of pixel grids. Think of it as painting with smooth, semi-transparent elliptical blobs that blend together to form the final image.

**Traditional approach:**
```
Image = Grid of pixels [128×128 = 16,384 values]
```

**Gaussian approach:**
```
Image = Collection of N Gaussians [~100-200 Gaussians, 9 params each = ~1,800 values]
Compression ratio: ~9-18× fewer parameters
```

### 1.2 What are Gaussian Primitives?

A **2D Gaussian primitive** is a smooth elliptical blob defined by 9 parameters:

```python
Gaussian = {
    # Position (where is it?)
    'x': float,  # center x-coordinate
    'y': float,  # center y-coordinate

    # Shape (what shape is it?)
    'σ_perp': float,      # width perpendicular to orientation
    'σ_parallel': float,  # width parallel to orientation
    'θ': float,           # rotation angle

    # Appearance (what does it look like?)
    'color': (r, g, b),   # RGB color
    'α': float            # opacity/transparency
}
```

**Mathematical definition:**
```
G(x, y) = α × exp(-½ × d²) × color

where d² = ((x-x₀)cos(θ) + (y-y₀)sin(θ))² / σ_parallel²
          + (-(x-x₀)sin(θ) + (y-y₀)cos(θ))² / σ_perp²
```

**Rendering:**
```
For each pixel (x,y):
    W = Σᵢ weight_i(x,y)           # total weight from all Gaussians
    C = Σᵢ weight_i(x,y) × color_i # weighted color sum
    Image(x,y) = C / max(W, ε)      # normalized final color
```

### 1.3 Why Gaussian Primitives?

**Advantages:**
- ✅ **Resolution independent**: Render at any resolution from single file
- ✅ **Smooth interpolation**: Natural anti-aliasing, no pixelation
- ✅ **Adaptive density**: More primitives in complex areas, fewer in smooth regions
- ✅ **Compact**: ~10-20× fewer parameters than pixels
- ✅ **GPU-friendly**: Fast rendering (1000+ FPS achieved)
- ✅ **Differentiable**: Can optimize with gradient descent

**Challenges:**
- ❌ **High-frequency content**: Smooth Gaussians struggle with sharp edges
- ❌ **Non-convex optimization**: Can get stuck in local minima
- ❌ **Initialization-sensitive**: Starting point matters
- ❌ **Representation theory unknown**: What are the "right" Gaussian types?

### 1.4 Current Research State

**Experimental findings (Phase 0, 0.5, 1):**

| Gaussian Type | Use Case | PSNR Quality | Status |
|---------------|----------|--------------|---------|
| **Large isotropic** (σ_perp ≈ σ_parallel ≈ 20px) | Uniform regions | **21.95 dB** | ✅ **Works well** |
| **Medium isotropic** (σ ≈ 15px) | Background fills | **15.06 dB** | ✅ Acceptable |
| **Small elongated** (σ_perp=0.5, σ_parallel=10) | Sharp edges (ΔI≥0.5) | **1.56-10 dB** | ❌ **Fails catastrophically** |

**Critical discovery:** Current edge primitive (elongated 2D Gaussian) has **fundamental capacity limit** of ~10-15 dB for high-contrast edges, even with optimal parameters (N=50-100, σ_perp=0.5).

**Empirical rules discovered (v2.0):**
```python
# For edge primitives:
N = edge_length / 2              # ~1 Gaussian per 2 pixels
σ_perp = 0.5 - 1.0              # constant, small (sharp perpendicular)
σ_parallel = 0.10 × edge_length  # 10% of edge length
α = (0.3 / ΔI) × (10 / N)       # scales with contrast and density
```

**Key insight:** Rules work for **low contrast** (ΔI<0.2, ~25 dB) but **fail for high contrast** (ΔI≥0.5, ~10 dB).

### 1.5 The Fundamental Questions

**Q1: What are the natural Gaussian "channels"?**
- Like RGB has 3 color channels, what are the fundamental Gaussian modes?
- How many? (3? 6? 12?)
- What parameter configurations define each?
- How do they compose?

**Q2: How should Gaussians be placed?**
- Grid? Gradient-based? Error-driven? Quantum-optimized?
- Different strategies for different image regions?

**Q3: How should each Gaussian type be optimized?**
- Large isotropic: Adam with lr=0.01?
- Small elongated: L-BFGS?
- Different iteration counts?

**Q4: Are 2D Gaussians even the right primitive?**
- Should we use Gabor functions (Gaussian × sinusoid)?
- Separable 1D×1D Gaussians?
- Something quantum mechanics reveals?

---

## Part 2: Quantum Computing Capabilities & Resources

### 2.1 Available Quantum Hardware

**IBM Quantum (via Qiskit):**
- **Access**: Cloud-based, API available
- **Free tier**: 10 minutes/month quantum processing unit (QPU) time
- **Commercial**: $1.60/second ($96/minute)
- **Backends available**:
  - `ibm_fez`: 156 qubits
  - `ibm_marrakesh`: 156 qubits
  - `ibm_torino`: 133 qubits
- **Simulator**: StatevectorSampler (exact, unlimited, FREE)

**D-Wave Quantum Annealer:**
- **Access**: Cloud-based (D-Wave Leap)
- **Free tier**: 1 minute/month QPU time
- **Qubits**: ~5000 (quantum annealer, different architecture)
- **Best for**: QUBO optimization problems (placement, strategy search)

### 2.2 Quantum Algorithms Applicable to Our Problem

**2.2.1 Quantum Kernel Methods (Classification & Clustering)**

**What it does:**
- Maps data to quantum Hilbert space (exponentially higher dimensional)
- Computes similarity in quantum space
- Reveals patterns classical Euclidean space misses

**Applications for LGI:**
- Cluster Gaussian parameter configurations → discover natural modes
- Classify image patches → discover primitive types
- Pattern recognition in high-dimensional feature spaces

**Evidence it works:**
- **Haiqu (Nov 2025)**: Quantum kernels outperformed classical for anomaly detection
- **Our test**: Quantum found different clusters than classical (ARI=0.011)

**Cost:** FREE on simulator, ~5-10 min on real quantum

---

**2.2.2 Quantum Annealing (QUBO Optimization)**

**What it does:**
- Solves combinatorial optimization problems
- Explores solution space in quantum superposition
- Quantum tunneling escapes local minima

**Formulation:**
```
Minimize: E(x) = Σᵢ hᵢxᵢ + Σᵢⱼ Jᵢⱼxᵢxⱼ
where xᵢ ∈ {0,1} (binary variables)
```

**Applications for LGI:**
- **Gaussian placement**: Which grid locations should have Gaussians?
- **Strategy selection**: Which optimizer/hyperparameters for each primitive?
- **Influence masking**: Which Gaussians affect which pixels?

**Example - Placement QUBO:**
```python
# Discretize image to 64×64 grid (4096 locations)
# Binary variables: x_i = 1 if Gaussian placed at location i

# Objective:
Q(x) = -Σᵢ w_i·x_i                    # benefit of placing Gaussian at i
       + λ₁·Σᵢⱼ overlap_ij·x_i·x_j    # penalty for overlap
       + λ₂·(Σᵢ x_i - N)²              # constraint: exactly N Gaussians

where:
    w_i = gradient magnitude at location i (benefit)
    overlap_ij = 1 if locations too close, else 0
```

**Cost:** FREE on D-Wave simulator, ~1-5 min on real quantum

---

**2.2.3 Variational Quantum Algorithms (VQA/VQE)**

**What it does:**
- Hybrid quantum-classical optimization
- Quantum circuit learns complex functions
- Parameters optimized classically

**Applications for LGI:**
- Learn f_edge: (blur, contrast, N) → (σ_perp, σ_parallel, α)
- Function approximation for parameter prediction
- Meta-learning optimization strategies

**Architecture:**
```
Classical input → Quantum encoding → Parameterized circuit →
    → Measurement → Classical output → Loss computation →
    → Classical optimizer updates quantum parameters
```

**Cost:** Expensive on real quantum (hours), but FREE on simulator

---

### 2.3 Quantum vs Classical: When Does Quantum Help?

**Quantum has advantage when:**
- ✅ High-dimensional data (100+ features) → quantum Hilbert space
- ✅ Non-convex landscape → quantum tunneling
- ✅ Combinatorial explosion → quantum superposition
- ✅ Pattern recognition in complex spaces

**Classical is sufficient when:**
- ❌ Low dimensions (<10 features)
- ❌ Convex problems (linear/quadratic)
- ❌ Small search spaces (<100 options)
- ❌ Simple patterns (linear separability)

**For our Gaussian problem:**
- Parameter space: 4-10 dimensions (**quantum might help**)
- Placement: combinatorial (4096 locations, choose 100) (**quantum definitely helps**)
- Function learning: non-convex (**quantum might help**)
- Pattern discovery: unknown structure (**quantum exploration valuable**)

---

## Part 3: Comprehensive Quantum Experiment Proposals

### Experiment 1: Quantum Discovery of Gaussian Channel Modes

**Objective:** Discover the natural "channels" (like RGB for color) in Gaussian parameter space.

**Hypothesis:** Quantum clustering in parameter space will reveal fundamental Gaussian modes that classical Euclidean clustering misses.

---

**E1.1 Data Preparation (Classical, FREE)**

```python
# Extract all Gaussian configurations from Phase 0, 0.5, 1 experiments
# Each Gaussian that was placed during experiments

gaussians_data = []
for experiment in [phase_0, phase_0_5, phase_1]:
    for render in experiment.renders:
        for gaussian in render.gaussians:
            gaussians_data.append({
                'σ_perp': gaussian.sigma_perp,
                'σ_parallel': gaussian.sigma_parallel,
                'α': gaussian.alpha,
                'θ': gaussian.theta,
                'quality_achieved': render.psnr,  # how well did it work?
                'contrast': render.edge_contrast,  # what context?
                'blur': render.edge_blur,
                'image_type': render.scene_type
            })

# Expected: ~1000-3000 Gaussian instances
# Feature space: 4D primary (σ_perp, σ_parallel, α, quality)
#                + 3D context (contrast, blur, image_type)
```

**E1.2 Classical Baseline (FREE, 5 minutes)**

```python
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler

X = extract_features(gaussians_data)  # 4D: σ_perp, σ_parallel, α, quality
X_scaled = StandardScaler().fit_transform(X)

# Classical clustering
classical_clustering = SpectralClustering(n_clusters=6, affinity='rbf')
labels_classical = classical_clustering.fit_predict(X_scaled)

# Analyze clusters
for i in range(6):
    cluster_gaussians = X[labels_classical == i]
    print(f"Classical Cluster {i}:")
    print(f"  σ_perp: {cluster_gaussians[:,0].mean():.2f}")
    print(f"  σ_parallel: {cluster_gaussians[:,1].mean():.2f}")
    print(f"  Quality: {cluster_gaussians[:,3].mean():.1f} dB")
```

**E1.3 Quantum Clustering (Simulator: FREE, 10-20 minutes)**

```python
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.primitives import StatevectorSampler

# Quantum feature map (4D → 6 qubits for richer Hilbert space)
X_padded = np.pad(X_scaled, ((0,0), (0,2)), mode='wrap')  # 4D → 6D
feature_map = ZZFeatureMap(6, reps=2)
qkernel = FidelityQuantumKernel(feature_map=feature_map)

# Compute quantum kernel matrix
K_quantum = qkernel.evaluate(x_vec=X_padded)

# Spectral clustering with quantum kernel
quantum_clustering = SpectralClustering(n_clusters=6, affinity='precomputed')
labels_quantum = quantum_clustering.fit_predict(K_quantum)

# Analyze quantum-discovered channels
for i in range(6):
    cluster_gaussians = X[labels_quantum == i]
    print(f"Quantum Channel {i}:")
    print(f"  σ_perp: {cluster_gaussians[:,0].mean():.2f} ± {cluster_gaussians[:,0].std():.2f}")
    print(f"  σ_parallel: {cluster_gaussians[:,1].mean():.2f} ± {cluster_gaussians[:,1].std():.2f}")
    print(f"  α: {cluster_gaussians[:,2].mean():.3f} ± {cluster_gaussians[:,2].std():.3f}")
    print(f"  Typical quality: {cluster_gaussians[:,3].mean():.1f} dB")
```

**E1.4 Validation on Real Quantum (FREE tier, 5-8 minutes)**

```python
# Subsample to 200 representative Gaussians (reduce quantum time)
X_subsample = stratified_sample(X_padded, n=200, labels=labels_quantum)

# Real quantum backend
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
service = QiskitRuntimeService(channel="ibm_cloud", instance=YOUR_CRN)
backend = service.least_busy(operational=True, simulator=False)

sampler = SamplerV2(backend)
qkernel_real = FidelityQuantumKernel(feature_map=feature_map, sampler=sampler)

K_quantum_real = qkernel_real.evaluate(x_vec=X_subsample)
labels_real = SpectralClustering(n_clusters=6, affinity='precomputed').fit_predict(K_quantum_real)

# Compare: Do simulator and real quantum agree?
from sklearn.metrics import adjusted_rand_score
similarity = adjusted_rand_score(labels_quantum[:200], labels_real)
print(f"Simulator vs Real Quantum similarity: {similarity:.3f}")
```

**E1.5 Channel Definition Extraction (Classical, FREE)**

```python
# Define channels based on quantum clustering
channels = []
for i in range(6):
    cluster_mask = labels_quantum == i
    cluster_data = X[cluster_mask]

    channel = {
        'channel_id': i,
        'name': infer_semantic_name(cluster_data),  # e.g., "large_isotropic"
        'σ_perp_range': (cluster_data[:,0].quantile(0.1),
                         cluster_data[:,0].quantile(0.9)),
        'σ_parallel_range': (cluster_data[:,1].quantile(0.1),
                             cluster_data[:,1].quantile(0.9)),
        'α_typical': cluster_data[:,2].mean(),
        'quality_profile': cluster_data[:,3].mean(),
        'use_cases': analyze_contexts(gaussians_data[cluster_mask])
    }
    channels.append(channel)

# Example output:
# Channel 0: "large_isotropic"
#   σ_perp: [15, 25], σ_parallel: [15, 25]
#   α: 0.25, quality: 20 dB
#   Use: Uniform regions, background fills

# Channel 1: "small_elongated"
#   σ_perp: [0.5, 1.5], σ_parallel: [8, 12]
#   α: 0.08, quality: 10 dB
#   Use: Low-contrast edges (fails on high-contrast)
```

---

**E1.6 Expected Outcomes**

**Success criteria:**
1. ✅ Quantum finds ≥3 distinct clusters with clear parameter separation
2. ✅ Clusters correlate with quality profiles (some channels work better than others)
3. ✅ Quantum-discovered channels differ from classical (ARI < 0.5)
4. ✅ Channels are interpretable (can assign semantic meaning)

**Deliverables:**
- **Channel definitions** (6 Gaussian modes with parameter ranges)
- **Quality profiles** (which channels work for what quality levels)
- **Classical classification rules** (how to assign Gaussians to channels)

**Cost:** $0 (entirely on simulator and free tier)

**Time:** 1-2 days (mostly classical data prep and analysis)

**Next steps:** Use discovered channels to redesign Phase 1 experiments (compositional representation with quantum-discovered channels instead of human-designed M/E/J/R/B/T primitives)

---

### Experiment 2: Quantum Optimization of Gaussian Placement

**Objective:** Use quantum annealing to find optimal Gaussian placement for representing edges/regions.

**Hypothesis:** Quantum annealing will find better placements than classical heuristics (grid, gradient-based, k-means) by exploring the combinatorial space in superposition.

---

**E2.1 Problem Formulation (Classical, FREE)**

```python
# Test case: 64×64 edge image, place N=50 Gaussians optimally

# Discretize to grid
GRID_SIZE = 32  # 32×32 = 1024 candidate locations
N_GAUSSIANS = 50  # target count

# Compute placement benefit at each location
target_image = generate_test_edge(size=64, blur=2, contrast=0.5)
gradient_map = compute_gradient_magnitude(target_image)
benefit = downsample(gradient_map, 32)  # 1024 benefits

# Compute overlap penalties
overlap_matrix = np.zeros((1024, 1024))
for i in range(1024):
    for j in range(i+1, 1024):
        dist = distance(grid_location[i], grid_location[j])
        if dist < 5:  # too close
            overlap_matrix[i,j] = 10.0  # penalty
```

**E2.2 Classical Baselines (FREE, 5 minutes each)**

```python
# Baseline 1: Gradient-based placement
locations_gradient = top_k_gradient_locations(benefit, k=50)
psnr_gradient = evaluate_placement(locations_gradient, target_image)

# Baseline 2: K-means clustering
locations_kmeans = kmeans_on_gradient(gradient_map, k=50)
psnr_kmeans = evaluate_placement(locations_kmeans, target_image)

# Baseline 3: Uniform grid
locations_grid = uniform_grid(64, 64, n=50)
psnr_grid = evaluate_placement(locations_grid, target_image)

# Baseline 4: Random
locations_random = random_sample(1024, k=50)
psnr_random = evaluate_placement(locations_random, target_image)

print(f"Gradient: {psnr_gradient:.2f} dB")
print(f"K-means: {psnr_kmeans:.2f} dB")
print(f"Grid: {psnr_grid:.2f} dB")
print(f"Random: {psnr_random:.2f} dB")
```

**E2.3 Quantum Annealing (D-Wave Simulator: FREE)**

```python
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite

# Formulate QUBO
Q = {}

# Linear terms (benefit of placing Gaussian at location i)
for i in range(1024):
    Q[(i,i)] = -benefit[i]  # negative because we minimize

# Quadratic terms (penalty for overlap)
for i in range(1024):
    for j in range(i+1, 1024):
        if overlap_matrix[i,j] > 0:
            Q[(i,j)] = overlap_matrix[i,j]

# Constraint: exactly N=50 Gaussians (penalty method)
# Add: λ·(Σxᵢ - N)² = λ·(Σxᵢ² - 2N·Σxᵢ + N²)
#    = λ·(Σxᵢ - 2N·Σxᵢ) + const  (since xᵢ² = xᵢ for binary)
lambda_constraint = 5.0
for i in range(1024):
    Q[(i,i)] += lambda_constraint * (1 - 2*N_GAUSSIANS)

# Solve on simulator first
sampler_sim = dimod.SimulatedAnnealingSampler()
response_sim = sampler_sim.sample_qubo(Q, num_reads=100)

best_solution_sim = response_sim.first.sample
locations_quantum_sim = [i for i, val in best_solution_sim.items() if val == 1]
psnr_quantum_sim = evaluate_placement(locations_quantum_sim, target_image)

print(f"Quantum (simulator): {psnr_quantum_sim:.2f} dB, N={len(locations_quantum_sim)}")
```

**E2.4 Real Quantum Annealing (D-Wave FREE tier, 1-3 minutes)**

```python
# Real D-Wave quantum annealer
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_qubo(Q, num_reads=100,
                               annealing_time=20)  # microseconds

best_solution = response.first.sample
locations_quantum = [i for i, val in best_solution.items() if val == 1]
psnr_quantum = evaluate_placement(locations_quantum, target_image)

print(f"Quantum (D-Wave): {psnr_quantum:.2f} dB, N={len(locations_quantum)}")
```

**E2.5 Analysis & Extraction (Classical, FREE)**

```python
# Compare all methods
results = {
    'Random': psnr_random,
    'Grid': psnr_grid,
    'Gradient': psnr_gradient,
    'K-means': psnr_kmeans,
    'Quantum (sim)': psnr_quantum_sim,
    'Quantum (real)': psnr_quantum
}

best_method = max(results, key=results.get)
improvement = results[best_method] - results['Gradient']  # vs best classical

print(f"Best method: {best_method} ({results[best_method]:.2f} dB)")
print(f"Improvement over gradient: {improvement:+.2f} dB")

# If quantum wins, analyze why
if 'Quantum' in best_method:
    # Compare quantum placement vs gradient placement
    quantum_locs = set(locations_quantum)
    gradient_locs = set(locations_gradient)

    unique_quantum = quantum_locs - gradient_locs
    print(f"Quantum found {len(unique_quantum)} locations gradient missed")

    # Visualize differences
    plot_placement_comparison(locations_gradient, locations_quantum, target_image)

    # Extract rules: where did quantum place Gaussians that gradient didn't?
    analyze_quantum_placement_strategy(unique_quantum, target_image)
```

---

**E2.6 Expected Outcomes**

**Success criteria:**
1. ✅ Quantum placement achieves PSNR ≥ best classical + 1 dB
2. ✅ Quantum finds non-obvious locations (not just gradient peaks)
3. ✅ Results are reproducible (multiple runs agree)

**Possible outcomes:**

**Scenario A: Quantum wins by >1 dB**
- Extract quantum placement strategy
- Analyze: What heuristic approximates quantum placement?
- Deploy classically using discovered heuristic

**Scenario B: Quantum ≈ Classical (±0.5 dB)**
- Quantum provides no advantage for this problem
- Classical gradient-based placement is sufficient
- Negative result (but publishable!)

**Scenario C: Quantum worse**
- QUBO formulation may need refinement
- Constraint penalties may be too high/low
- Iterate on problem encoding

**Cost:** $0 (simulator + D-Wave free tier)

**Time:** 2-3 days (formulation, testing, analysis)

---

### Experiment 3: Quantum Meta-Optimization of Iteration Strategies

**Objective:** Discover optimal optimization strategy (optimizer, learning rate, iterations) for each Gaussian channel.

**Hypothesis:** Different Gaussian types benefit from different optimization strategies, and quantum can search this meta-space efficiently.

---

**E3.1 Strategy Space Definition (Classical, FREE)**

```python
# Define meta-parameters to optimize
strategy_space = {
    'optimizer': ['adam', 'lbfgs', 'sgd', 'rmsprop'],           # 4 options
    'learning_rate': [0.001, 0.003, 0.01, 0.03, 0.1],          # 5 options
    'iterations': [50, 100, 200, 500, 1000],                    # 5 options
    'constraints': ['free', 'fix_theta', 'fix_shape']          # 3 options
}

# Total combinations: 4 × 5 × 5 × 3 = 300 strategies

# For each channel discovered in E1
channels = ['large_isotropic', 'small_elongated', 'medium_isotropic', ...]
```

**E3.2 Strategy Evaluation Function (Classical, time-intensive)**

```python
def evaluate_strategy(channel_type, optimizer, lr, iterations, constraints):
    """
    Test a strategy on 10 test cases for this channel type
    Returns: (mean_psnr, mean_convergence_time, mean_final_loss)
    """
    test_cases = get_test_cases_for_channel(channel_type, n=10)
    results = []

    for test_case in test_cases:
        # Initialize Gaussians for this channel type
        gaussians = initialize_channel_gaussians(channel_type, test_case)

        # Optimize with given strategy
        final_gaussians, metrics = optimize(
            gaussians, test_case.target,
            optimizer=optimizer, lr=lr,
            max_iter=iterations, constraints=constraints
        )

        psnr = compute_psnr(render(final_gaussians), test_case.target)
        results.append({
            'psnr': psnr,
            'iterations_used': metrics['iterations'],
            'final_loss': metrics['final_loss']
        })

    return {
        'mean_psnr': np.mean([r['psnr'] for r in results]),
        'std_psnr': np.std([r['psnr'] for r in results]),
        'mean_time': np.mean([r['iterations_used'] for r in results])
    }

# This function is EXPENSIVE: ~60 seconds per evaluation
# Total for exhaustive search: 300 strategies × 60s ≈ 5 hours per channel
```

**E3.3 Classical Baseline: Bayesian Optimization (FREE, 3-4 hours per channel)**

```python
from skopt import gp_minimize
from skopt.space import Categorical, Real, Integer

def objective(params):
    """Objective function for Bayesian optimization"""
    optimizer, lr_idx, iter_idx, constraints = params
    lr = [0.001, 0.003, 0.01, 0.03, 0.1][lr_idx]
    iterations = [50, 100, 200, 500, 1000][iter_idx]

    result = evaluate_strategy('large_isotropic', optimizer, lr,
                              iterations, constraints)
    return -result['mean_psnr']  # negative because we minimize

space = [
    Categorical(['adam', 'lbfgs', 'sgd', 'rmsprop']),
    Integer(0, 4),  # learning rate index
    Integer(0, 4),  # iterations index
    Categorical(['free', 'fix_theta', 'fix_shape'])
]

# Run Bayesian optimization (100 evaluations)
result = gp_minimize(objective, space, n_calls=100, random_state=42)

best_strategy_classical = {
    'optimizer': result.x[0],
    'lr': [0.001, 0.003, 0.01, 0.03, 0.1][result.x[1]],
    'iterations': [50, 100, 200, 500, 1000][result.x[2]],
    'constraints': result.x[3],
    'psnr': -result.fun
}
```

**E3.4 Quantum Strategy Search (QUBO formulation)**

```python
# Encode discrete strategy space as binary variables
# optimizer: 2 bits (4 options → 00, 01, 10, 11)
# lr: 3 bits (5 options → 000 to 100)
# iterations: 3 bits (5 options)
# constraints: 2 bits (3 options)
# Total: 10 binary variables

# However, evaluating objective requires classical simulation (expensive)
# Instead: Use quantum for DISCRETE search, classical for evaluation

# Pre-compute performance matrix (one-time cost: expensive but offline)
performance_matrix = np.zeros((4, 5, 5, 3))  # [optimizer, lr, iter, constraints]

# Sample intelligently: don't test all 300, use adaptive sampling
# Test 50-100 strategic combinations
sample_strategies = strategic_sample(strategy_space, n=50)

for s in sample_strategies:
    result = evaluate_strategy('large_isotropic', **s)
    idx = encode_strategy_to_index(s)
    performance_matrix[idx] = result['mean_psnr']

# Now formulate QUBO to find maximum in pre-computed matrix
# This is a MAX-SAT problem, solvable with quantum annealing

Q = formulate_max_sat_qubo(performance_matrix)
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_qubo(Q, num_reads=100)

best_strategy_quantum = decode_strategy(response.first.sample)
```

**Alternative: Hybrid Quantum-Classical**

```python
# Use quantum to suggest next strategy to test (exploration)
# Classical Bayesian optimization for exploitation

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms import QAOA

for iteration in range(50):
    # Quantum suggests exploration points
    next_strategies = quantum_suggest_exploration(
        tested_strategies, performance_seen,
        n_suggestions=5
    )

    # Classical evaluates
    for strategy in next_strategies:
        performance = evaluate_strategy('large_isotropic', **strategy)
        tested_strategies.append(strategy)
        performance_seen.append(performance)

    # Update model
    bayesian_model.update(tested_strategies, performance_seen)

best_strategy = tested_strategies[np.argmax(performance_seen)]
```

---

**E3.5 Expected Outcomes**

**Success criteria:**
1. ✅ Find strategy that achieves best PSNR for each channel
2. ✅ Quantum exploration reduces total evaluations needed (vs exhaustive)
3. ✅ Discovered strategies generalize to new test cases

**Deliverables:**
```python
optimization_recipes = {
    'large_isotropic': {
        'optimizer': 'lbfgs',
        'learning_rate': 0.1,
        'iterations': 200,
        'constraints': 'free',
        'expected_psnr': 22.5
    },
    'small_elongated': {
        'optimizer': 'adam',
        'learning_rate': 0.003,
        'iterations': 500,
        'constraints': 'fix_theta',  # don't optimize angle
        'expected_psnr': 12.0  # limited by primitive capacity
    },
    # ... for each channel
}
```

**Cost:**
- Offline evaluation: ~5-10 hours classical compute (one-time, can parallelize)
- Quantum search: FREE (simulator) or ~2-5 min (D-Wave)
- Total: $0

**Time:** 3-5 days (mostly waiting for evaluations)

---

### Experiment 4: Quantum Discovery of Optimal Basis Function

**Objective:** Test if 2D Gaussians are the optimal primitive, or if quantum reveals better alternatives.

**Hypothesis:** Quantum search over parameterized basis functions will discover optimal representation (might be Gabor, separable 1D×1D, or novel).

---

**E4.1 Basis Function Space Parameterization**

```python
# General 2D basis function family
def basis_function(x, y, params):
    """
    Parameterized family of basis functions

    params = {
        'envelope_type': ['gaussian', 'cauchy', 'laplace'],
        'modulation': ['none', 'sinusoid', 'chirp'],
        'separable': [True, False],
        'frequency': float,
        'phase': float,
        'anisotropy': float
    }
    """

    # Envelope
    if params['envelope_type'] == 'gaussian':
        envelope = exp(-0.5 * distance_squared)
    elif params['envelope_type'] == 'cauchy':
        envelope = 1 / (1 + distance_squared)
    elif params['envelope_type'] == 'laplace':
        envelope = exp(-abs(distance))

    # Modulation (for texture/frequency)
    if params['modulation'] == 'none':
        carrier = 1.0
    elif params['modulation'] == 'sinusoid':
        carrier = cos(2*pi*params['frequency']*x_rotated + params['phase'])
    elif params['modulation'] == 'chirp':
        carrier = cos(2*pi*params['frequency']*(x_rotated**2) + params['phase'])

    # Combine
    return envelope * carrier

# Examples:
# Pure Gaussian: envelope='gaussian', modulation='none'
# Gabor: envelope='gaussian', modulation='sinusoid'
# Novel: envelope='cauchy', modulation='chirp'
```

**E4.2 Classical Baseline: Test Known Alternatives (FREE, 1-2 days)**

```python
# Test several known basis functions
basis_functions_to_test = [
    {'name': 'gaussian', 'envelope': 'gaussian', 'modulation': 'none'},
    {'name': 'gabor', 'envelope': 'gaussian', 'modulation': 'sinusoid'},
    {'name': 'cauchy', 'envelope': 'cauchy', 'modulation': 'none'},
    {'name': 'separable_gaussian', 'separable': True}
]

results = {}
for bf in basis_functions_to_test:
    # Test on Phase 0.5 edge cases
    psnr_results = []
    for test_case in phase_05_test_cases:
        gaussians = initialize_with_basis_function(test_case, bf)
        optimized = optimize_gaussians(gaussians, test_case.target)
        psnr = compute_psnr(render(optimized), test_case.target)
        psnr_results.append(psnr)

    results[bf['name']] = {
        'mean_psnr': np.mean(psnr_results),
        'std_psnr': np.std(psnr_results)
    }

# Compare
print("Basis Function Performance:")
for name, result in sorted(results.items(), key=lambda x: x[1]['mean_psnr'], reverse=True):
    print(f"{name:20s}: {result['mean_psnr']:.2f} ± {result['std_psnr']:.2f} dB")
```

**E4.3 Quantum Basis Function Search (VQE, Simulator: FREE)**

```python
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator

# Parameterize basis function with quantum circuit
def create_basis_function_circuit(n_params=8):
    """
    Quantum circuit that encodes basis function parameters

    Circuit output (measured) → basis function parameters:
      - envelope_decay_rate
      - modulation_frequency
      - anisotropy_ratio
      - phase_offset
      - etc.
    """
    qc = QuantumCircuit(n_params)

    # Variational parameters
    theta = [Parameter(f'θ{i}') for i in range(n_params)]

    # Encoding (example - would need careful design)
    for i in range(n_params):
        qc.rx(theta[i], i)
        if i < n_params - 1:
            qc.cx(i, i+1)

    # Additional layers...

    return qc

# Objective function
def objective_function(params):
    """
    Given quantum circuit parameters, decode to basis function params,
    test on images, return quality
    """
    bf_params = decode_quantum_params(params)

    # Test basis function
    psnr_results = []
    for test_case in sampled_test_cases:  # use subset for speed
        gaussians = initialize_with_basis_function(test_case, bf_params)
        optimized = optimize_gaussians(gaussians, test_case.target, max_iter=50)
        psnr = compute_psnr(render(optimized), test_case.target)
        psnr_results.append(psnr)

    return -np.mean(psnr_results)  # negative for minimization

# Quantum optimization (VQE on simulator)
circuit = create_basis_function_circuit(n_params=8)
optimizer = COBYLA(maxiter=100)

# This is expensive even on simulator (each iteration calls objective_function)
# Use small test set and aggressive caching
vqe = VQE(estimator=Estimator(),
          ansatz=circuit,
          optimizer=optimizer)

# Note: This needs careful implementation - objective evaluation is classical
# VQE typically for quantum hamiltonians, we're using it for classical optimization
# May need hybrid approach
```

**Practical Hybrid Approach:**

```python
# More realistic: Use quantum to explore discrete choices,
# classical to optimize continuous parameters

# Discrete quantum search
quantum_basis_types = quantum_search_discrete_basis_functions(
    search_space=['gaussian×none', 'gaussian×sin', 'cauchy×none',
                  'cauchy×sin', 'laplace×sin', 'separable_gauss'],
    n_top=3  # return top 3
)

# Classical refinement
for bf_type in quantum_basis_types:
    best_params = classical_parameter_optimization(bf_type, test_cases)

    final_results[bf_type] = {
        'params': best_params,
        'psnr': evaluate(bf_type, best_params, test_cases)
    }

optimal_basis_function = max(final_results, key=lambda k: final_results[k]['psnr'])
```

---

**E4.4 Expected Outcomes**

**Scenario A: Gaussian wins**
- Pure 2D Gaussian is optimal
- Continue with current approach
- Quantum provides validation

**Scenario B: Gabor wins**
- Gabor functions (Gaussian × sinusoid) achieve +5-10 dB on edges
- Redesign primitive library to use Gabor for edges
- Major improvement, validates quantum exploration

**Scenario C: Separable 1D×1D wins**
- Product of 1D Gaussians works better for edges
- 8× computational speedup + better quality
- Huge win, implement immediately

**Scenario D: Novel function wins**
- Quantum discovers unexpected combination (e.g., Cauchy × chirp)
- Test extensively, understand why it works
- Potentially publishable discovery

**Cost:** $0 (simulator-based exploration)

**Time:** 1-2 weeks (complex implementation + testing)

**Note:** This is the most speculative experiment - may not yield clear results

---

## Part 4: Implementation Roadmap

### Phase 1: Immediate (Week 1-2) - FREE

**Goal:** Run Experiment 1 (Channel Discovery) on simulator

**Tasks:**
1. ✅ Extract all Gaussian configurations from Phase 0/0.5/1 data
2. ✅ Run classical clustering baseline
3. ✅ Run quantum clustering on simulator
4. ✅ Analyze and define channels
5. ✅ Document quantum vs classical differences

**Deliverable:** `quantum_channels_discovered.json` with 4-6 channel definitions

**Cost:** $0
**Time:** 3-5 days

---

### Phase 2: Validation (Week 3) - FREE TIER

**Goal:** Validate E1 on real quantum hardware

**Tasks:**
1. ✅ Subsample data to 200 Gaussians (reduce quantum time)
2. ✅ Run on IBM Quantum real hardware
3. ✅ Compare simulator vs real quantum results
4. ✅ If agrees: channels are validated
5. ✅ If differs: investigate NISQ noise effects

**Deliverable:** Validation report

**Cost:** $0 (free tier: 5-8 minutes)
**Time:** 2-3 days

---

### Phase 3: Classical Alternatives (Week 4-5) - FREE

**Goal:** Test separable 1D×1D and Gabor functions (Experiment 4, classical part)

**Tasks:**
1. ✅ Implement separable Gaussian rendering
2. ✅ Implement Gabor function rendering
3. ✅ Re-run Phase 0.5 edge tests with both
4. ✅ Compare PSNR: Gaussian vs Separable vs Gabor
5. ✅ Identify winner for edge representation

**Deliverable:** Basis function comparison report

**Cost:** $0 (classical compute)
**Time:** 1 week

---

### Phase 4: Quantum Placement (Week 6) - FREE TIER

**Goal:** Run Experiment 2 (Placement Optimization)

**Tasks:**
1. ✅ Formulate QUBO for edge case
2. ✅ Test on D-Wave simulator
3. ✅ Run on D-Wave free tier (1-3 min)
4. ✅ Compare vs classical baselines
5. ✅ Extract placement rules if quantum wins

**Deliverable:** Placement strategy comparison

**Cost:** $0 (simulator + free tier)
**Time:** 3-5 days

---

### Phase 5: Meta-Optimization (Week 7-8) - FREE

**Goal:** Run Experiment 3 (Iteration Strategy Discovery)

**Tasks:**
1. ✅ Pre-compute performance matrix (offline, parallelizable)
2. ✅ Quantum strategy search
3. ✅ Compare vs Bayesian optimization
4. ✅ Define optimization recipes per channel

**Deliverable:** `optimization_recipes.json`

**Cost:** $0 (compute-intensive but parallelizable)
**Time:** 1-2 weeks (mostly CPU time)

---

### Phase 6: Integration & Publication (Week 9-12)

**Goal:** Integrate quantum discoveries into classical codec

**Tasks:**
1. ✅ Implement quantum-discovered channels in Rust codec
2. ✅ Use quantum-discovered placement heuristics
3. ✅ Apply optimization recipes
4. ✅ Benchmark: Before vs After quantum discoveries
5. ✅ Write paper: "Quantum-Discovered Primitives for Gaussian Image Compression"

**Deliverable:**
- Enhanced codec (classical, no runtime quantum)
- Research paper

**Cost:** $0 (all discoveries already made)
**Time:** 4 weeks

---

## Part 5: Expected Impact & Publications

### 5.1 Technical Impact

**If successful, quantum discoveries will provide:**

1. **Channel Definitions** (E1)
   - Replace human-designed M/E/J/R/B/T primitives
   - Data-driven, fundamentally motivated
   - Better compositional properties

2. **Placement Strategies** (E2)
   - Heuristic that approximates quantum-optimal placement
   - 1-3 dB PSNR improvement
   - Classical deployment, no runtime cost

3. **Optimization Recipes** (E3)
   - Per-channel optimization strategies
   - Faster convergence, better quality
   - 20-50% iteration reduction

4. **Basis Function** (E4)
   - If Gabor/Separable wins: +5-10 dB for edges
   - If novel discovered: potential breakthrough
   - Complete redesign of primitive library

**Conservative estimate:** +3-5 dB overall PSNR improvement
**Optimistic estimate:** +8-12 dB (if Gabor fixes edges)

### 5.2 Publication Opportunities

**Paper 1: "Quantum Discovery of Representation Primitives for Gaussian Image Compression"**
- **Venue:** CVPR, ICCV, or ECCV (top-tier computer vision)
- **Contribution:** First use of quantum clustering for primitive discovery
- **Novel:** Quantum reveals structure classical misses
- **Impact:** Hybrid quantum-classical methodology for codec design

**Paper 2: "Quantum Annealing for Optimal Gaussian Placement in Image Representation"**
- **Venue:** Quantum computing conference (QIP, TQC) or vision venue
- **Contribution:** QUBO formulation for image primitive placement
- **Novel:** Combinatorial image optimization with quantum
- **Impact:** Demonstrates quantum advantage for practical vision problem

**Paper 3: "From Quantum to Classical: Discovering Image Compression Primitives"**
- **Venue:** Interdisciplinary (Nature Communications, PNAS if big result)
- **Contribution:** Complete pipeline: quantum discovery → classical deployment
- **Novel:** Amortized quantum cost (one-time discovery)
- **Impact:** Template for other domains to use quantum similarly

### 5.3 Open Source Release

**Repository:** `quantum-gaussian-primitives`

**Contents:**
- Quantum clustering code (Qiskit)
- QUBO formulations for placement
- Discovered channel definitions (JSON)
- Classical codec using quantum discoveries
- Benchmark comparisons
- Jupyter notebooks with tutorials

**Impact:** Enable others to use quantum for codec/representation research

---

## Part 6: Risk Analysis & Mitigation

### Risk 1: Quantum shows no advantage

**Probability:** Medium (30-40%)

**Impact:** Quantum and classical clustering agree, placement is similar, no improvement

**Mitigation:**
- Still publishable as **negative result**
- "We tested quantum for X, found classical sufficient"
- Validates classical approach
- Informs future: when quantum helps vs doesn't

**Outcome:** Paper on when quantum doesn't help (valuable)

---

### Risk 2: NISQ noise corrupts results

**Probability:** Low (10-20%)

**Impact:** Real quantum hardware too noisy, results unstable

**Mitigation:**
- Use simulator for discovery (exact, no noise)
- Real quantum only for validation
- If noise problematic: wait for better hardware (2026+)
- All discoveries work on simulator, which is FREE and exact

**Outcome:** Publish with simulator results, note "validated on quantum when available"

---

### Risk 3: Problem formulation incorrect

**Probability:** Medium-Low (20%)

**Impact:** QUBO for placement doesn't capture problem correctly

**Mitigation:**
- Test on small examples first
- Compare quantum solution to ground truth (exhaustive search on tiny problems)
- Iterate on formulation
- Literature review: similar problems in quantum optimization

**Outcome:** Refine QUBO, learn correct formulation (still valuable)

---

### Risk 4: Quantum discoveries don't generalize

**Probability:** Low (10%)

**Impact:** Channels discovered on test set don't work on new images

**Mitigation:**
- Use diverse dataset (24 Kodak + synthetic + Phase 0/0.5/1)
- Cross-validation: train on subset, test on held-out
- Compare generalization: classical vs quantum discoveries

**Outcome:** Understand limitations, publish with caveats

---

### Risk 5: Implementation complexity

**Probability:** Medium (25%)

**Impact:** Quantum APIs change, bugs, integration issues

**Mitigation:**
- Start simple (E1 on simulator)
- Use stable Qiskit releases
- Ask community (Qiskit Slack, Stack Exchange)
- Fallback: classical-only if quantum too hard

**Outcome:** Learning curve, but manageable with time

---

## Part 7: Resource Requirements

### 7.1 Computational Resources

**Classical:**
- 16-core CPU (or cloud equivalent)
- 32-64 GB RAM
- Storage: 50-100 GB for datasets/results
- **Cost:** $50-100/month cloud credits (or use local)

**Quantum:**
- IBM Quantum free tier (10 min/month) - **FREE**
- D-Wave Leap free tier (1 min/month) - **FREE**
- Simulator: unlimited, runs on laptop/cloud - **FREE**
- **Cost:** $0 if staying within free tiers

### 7.2 Human Resources

**Primary researcher:** 1 person (you)
- Background in ML/optimization helpful
- Learning curve for quantum: 1-2 weeks
- Full-time: 2-3 months for all experiments
- Part-time: 4-6 months

**Optional collaborators:**
- Quantum computing expert (for E4, complex VQE)
- Image compression expert (for validation/benchmarking)
- Not required, but helpful

### 7.3 Learning Resources

**Quantum computing basics:**
- Qiskit Textbook (free online)
- "Quantum Machine Learning" course (edX, free)
- Time: 1-2 weeks of study

**Quantum annealing:**
- D-Wave tutorials (free)
- "QUBO formulation guide" (D-Wave docs)
- Time: 3-5 days

**Total learning time:** 2-3 weeks before starting experiments

---

## Part 8: Alternative Approaches (If Quantum Not Viable)

If quantum proves too difficult or shows no advantage, **classical alternatives:**

### A8.1 Advanced Classical Clustering

```python
# Instead of quantum kernel, try:
- Density-based (HDBSCAN)
- Hierarchical with correlation distance
- Gaussian Mixture Models (EM algorithm)
- Spectral clustering with learned kernel

# These might find similar channels without quantum
```

### A8.2 Neural Architecture Search (NAS)

```python
# Learn basis function with neural network
def learned_basis_function(x, y, params):
    # Small MLP: (x, y, params) → weight
    return MLP([x, y, *params])

# Train on image dataset
# Extract learned function
```

### A8.3 Evolutionary Algorithms

```python
# Genetic programming for:
- Placement strategies
- Basis function search
- Optimization recipe discovery

# Slower than quantum but more accessible
```

---

## Part 9: Success Criteria & Go/No-Go Decisions

### After E1 (Channel Discovery):

**GO if:**
- ✅ Quantum finds ≥3 distinct channels
- ✅ ARI (classical vs quantum) < 0.5 (different structure)
- ✅ Channels correlate with quality profiles
- → **Proceed to E2 (Placement)**

**NO-GO if:**
- ❌ Quantum ≈ classical (ARI > 0.8)
- ❌ Only 1-2 clusters (no structure)
- → **Stop quantum path, use classical clustering**

---

### After E2 (Placement):

**GO if:**
- ✅ Quantum PSNR > classical + 1 dB
- ✅ Improvement is consistent (low variance)
- → **Extract placement heuristic, proceed to E3**

**NO-GO if:**
- ❌ Quantum ≈ classical (±0.5 dB)
- → **Use gradient-based placement, skip E3 quantum part**

---

### After E4 (Basis Function):

**GO if:**
- ✅ Non-Gaussian wins by >2 dB
- ✅ Results reproducible
- → **Redesign codec with new basis**

**NO-GO if:**
- ❌ Gaussian is optimal
- → **Validate current approach, proceed with Gaussians**

---

## Part 10: Conclusion & Next Steps

### 10.1 Summary

This research proposal outlines a comprehensive plan to use **quantum computing** to solve fundamental representation theory problems in Gaussian image compression:

1. **What primitives?** → Quantum clustering discovers natural channels
2. **Where to place?** → Quantum annealing optimizes placement
3. **How to optimize?** → Quantum meta-search finds strategies
4. **Right primitive?** → Quantum explores basis function space

**Key advantages:**
- ✅ **One-time quantum cost** → classical deployment forever
- ✅ **Free research** (simulators + free tiers)
- ✅ **Novel methodology** (hybrid quantum-classical)
- ✅ **Multiple publications** (4-5 papers potential)
- ✅ **Practical impact** (+3-12 dB quality improvement)

### 10.2 Immediate Next Steps

**This week:**
1. ✅ Extract Gaussian configurations from Phase 0/0.5/1 experiments
2. ✅ Set up Qiskit environment (already done based on git history)
3. ✅ Run Experiment 1 (E1) on simulator
4. ✅ Analyze results, define channels

**Next 2 weeks:**
5. ✅ Validate E1 on real quantum (free tier)
6. ✅ Implement separable 1D×1D Gaussians (classical)
7. ✅ Implement Gabor functions (classical)
8. ✅ Compare basis functions

**Month 2:**
9. ✅ Run E2 (quantum placement) if E1 successful
10. ✅ Run E3 (meta-optimization)

**Month 3:**
11. ✅ Integrate discoveries into codec
12. ✅ Benchmark improvements
13. ✅ Write first paper draft

### 10.3 Decision Point: Start Now?

**Recommendation: YES, start with E1 immediately**

**Reasons:**
- ✅ Data already exists (Phase 0/0.5/1 results)
- ✅ Qiskit already set up
- ✅ Free (simulator-based)
- ✅ 3-5 days to first results
- ✅ Low risk, high potential reward

**If E1 shows promise → continue to E2/E3**
**If E1 shows no advantage → classical alternatives, still publishable**

---

## Appendices

### Appendix A: Code Templates

**(Available in separate files: see `/quantum_research/` directory)**

- `E1_channel_discovery.py` - Quantum clustering
- `E2_placement_qubo.py` - QUBO formulation
- `E3_meta_optimization.py` - Strategy search
- `E4_basis_function_search.py` - Basis exploration

### Appendix B: Quantum Computing Primer

**(For readers unfamiliar with quantum)**

**Key concepts:**
- **Qubit**: Quantum bit, superposition of 0 and 1
- **Quantum kernel**: Similarity in quantum Hilbert space
- **QUBO**: Quadratic Unconstrained Binary Optimization
- **VQE**: Variational Quantum Eigensolver
- **Quantum annealing**: Quantum optimization for QUBO

**Resources:**
- Qiskit Textbook: https://qiskit.org/learn
- D-Wave tutorials: https://docs.ocean.dwavesys.com

### Appendix C: Glossary

- **Gaussian primitive**: 2D elliptical blob with position, shape, color, opacity
- **Channel**: Fundamental mode of Gaussian representation (like RGB for color)
- **PSNR**: Peak Signal-to-Noise Ratio (quality metric, higher = better)
- **QUBO**: Binary optimization problem solvable by quantum annealing
- **VQE**: Hybrid quantum-classical optimization algorithm
- **Quantum kernel**: Similarity metric computed in quantum Hilbert space
- **Gabor function**: Gaussian modulated by sinusoid (better for textures/edges)

---

**Document Status:** Comprehensive research proposal complete
**Ready for:** Executive review and immediate implementation of E1
**Estimated total research time:** 2-3 months full-time
**Estimated total cost:** $0-$50 (potentially entirely free)
**Expected impact:** +3-12 dB quality improvement, 3-5 publications

**Author:** AI Research Assistant
**Date:** November 23, 2025
**Version:** 1.0
