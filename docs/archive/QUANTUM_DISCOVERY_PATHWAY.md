# Quantum Computing Discovery Pathway
## Using Quantum Computers to Discover Classical Rules for Gaussian Image Primitives

**Date:** 2025-11-17
**Status:** Future Research Direction
**Approach:** Quantum for one-time discovery, classical for deployment

---

## Core Concept

**NOT:** Use quantum computing at runtime (expensive, impractical)

**YES:** Use quantum computing to DISCOVER rules/formulas/primitives ONCE, then apply classically forever

### The Model

```
DISCOVERY PHASE (ONE TIME - Expensive but amortized):
  Dataset → Quantum Computer → Optimization/Learning → Extract Rules → Classical Formula
            ^expensive quantum time                                      ^free forever

DEPLOYMENT PHASE (FOREVER - Fast and free):
  Image → Apply Formula (classical) → Gaussian Parameters → Render → Output
          ^no quantum needed
```

**Key insight:** Amortize expensive quantum discovery over infinite classical uses.

---

## Three Discovery Problems

### 1. PRIMITIVE DISCOVERY
**Question:** What are the natural primitive types?

**Current approach:**
- Designed M/E/J/R/B/T based on intuition
- May not be optimal or natural groupings

**Quantum approach:**
- Let quantum clustering discover primitives from data
- Data-driven, not human-designed

### 2. FORMULA DISCOVERY
**Question:** What is f_edge(blur, contrast) → (σ_perp, σ_parallel, spacing, alpha)?

**Current approach:**
- Manual parameter sweeps (Phase 0, 0.5)
- Polynomial regression on results

**Quantum approach:**
- Quantum symbolic regression
- Quantum function learning (VQA)
- Find optimal functional forms

### 3. STRATEGY DISCOVERY
**Question:** What optimizer, constraints, iterations for each primitive?

**Current approach:**
- Try Adam, L-BFGS, SGD manually
- Grid search hyperparameters

**Quantum approach:**
- Quantum meta-optimization
- Search strategy space in superposition
- Extract optimal strategies per primitive

---

## Problem 1: Primitive Discovery via Quantum Clustering

### Objective
Discover natural groupings of image features → define primitives from data, not intuition.

### Classical Baseline (Current)

**Approach:**
- Design primitives: M/E/J/R/B/T (human intuition)
- Detect features: Canny edges, SLIC regions, etc.
- Classification: Rule-based or learned CNN

**Limitations:**
- Primitives may not be natural groupings
- Detection methods chosen a priori
- May miss non-obvious patterns

---

### Quantum Approach: Unsupervised Quantum Clustering

**Based on: Haiqu demonstration (November 2025) - 500 features, 128 qubits**

#### Step 1: Data Collection

**Collect diverse image patches:**
```
Source: 100-1000 images (natural photos, diagrams, textures)
Extract: 10,000 patches (10×10 or 20×20 pixels each)
Features per patch:
  - Gradient magnitude, direction
  - Curvature (structure tensor eigenvalues)
  - Entropy, variance
  - Texture descriptors (Gabor responses, LBP)
  - Color distribution
  - Frequency content
  → 500-dimensional feature vector per patch
```

**Dataset:** 10,000 patches × 500 features

#### Step 2: Quantum Feature Encoding

**Method:** Amplitude encoding (Haiqu's approach)
```python
# Encode 10,000 patches in quantum state
|ψ⟩ = ∑_{i=1}^{10000} α_i |feature_vector_i⟩

# Where feature_vector_i is 500-dimensional
# Requires ~log2(10000) + log2(500) ≈ 13 + 9 = 22 qubits for addressing
# Plus qubits for computation: ~100-128 qubits total

# IBM Quantum Eagle (127 qubits): Sufficient
```

**Platform:** IBM Quantum (free tier or research access)

#### Step 3: Quantum Clustering Algorithm

**Approaches:**

**Option A: Quantum K-Means**
- Quantum version of K-means clustering
- Explores cluster assignments in superposition
- Faster convergence than classical (proven for some cases)

**Option B: Quantum Hierarchical Clustering**
- Build dendrogram using quantum distance computation
- Quantum advantage in distance matrix calculation (high-dimensional)

**Option C: Quantum Spectral Clustering**
- Construct affinity matrix
- Quantum eigenvalue decomposition (potential speedup)
- Extract clusters from eigenvectors

**Implementation:** Use Qiskit quantum machine learning library (has clustering primitives)

#### Step 4: Extract Primitive Definitions

**Quantum outputs:** K clusters (K might be discovered too)

**Analysis:**
```python
# For each cluster, analyze characteristics
Cluster 1: {patches with high |∇I|, low κ, σ_edge < 2px}
  → Primitive: "Sharp Edges"

Cluster 2: {patches with high κ, junction-like features}
  → Primitive: "Corners/Junctions"

Cluster 3: {patches with low variance, low |∇I|}
  → Primitive: "Flat Regions"

Cluster 4: {patches with high frequency, directional structure}
  → Primitive: "Oriented Texture"

Cluster 5: {patches with high frequency, isotropic}
  → Primitive: "Stochastic Texture"

Etc.
```

**Output:** Data-driven primitive definitions

#### Step 5: Classical Deployment

**Create classical classifiers:**
```python
def classify_patch(patch_features):
    """
    Classical implementation of quantum-discovered classification

    Based on quantum clustering results:
    - Cluster 1 boundaries: |∇I| > 0.3, κ < 0.1, σ_edge < 2px
    - Cluster 2 boundaries: κ > 0.3
    - Etc.
    """
    if features['gradient_magnitude'] > 0.3 and features['curvature'] < 0.1:
        return 'sharp_edge'
    elif features['curvature'] > 0.3:
        return 'junction'
    # etc.
```

**Or:** Train small classical NN on quantum cluster labels

**Deployment:** Pure classical, no quantum needed

---

### Cost-Benefit Analysis

**Cost:**
- Data collection: 1-2 days classical compute (free)
- Feature extraction: 1 day classical compute (free)
- Quantum clustering: ~1-10 hours quantum time
  - IBM Quantum free tier: 10 min/month
  - Researcher access: hours/week
  - Commercial: ~$100-1000 for this job
- Analysis: 1-2 days classical (free)

**Benefit:**
- Discover primitives from data (not intuition)
- May find non-obvious groupings
- Primitives used forever (amortized value)

**Verdict:** Worth trying if you get quantum access

---

## Problem 2: Formula Discovery via Quantum Symbolic Regression

### Objective
Discover f_edge functional form: (blur, contrast, N) → (σ_perp, σ_parallel, spacing, alpha)

### Classical Baseline (Do This First)

**Phase 0.5 + Classical Regression:**

```python
# Collect data from Phase 0.5
data = [
    (blur=0, contrast=0.5, N=50, σ_perp=1.0, PSNR=32.5),
    (blur=2, contrast=0.5, N=50, σ_perp=2.0, PSNR=35.2),
    # ... 100-200 data points
]

# Try polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

X = [[blur, contrast, N], ...]
y_sigma_perp = [σ_perp, ...]

poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
model = Ridge().fit(X_poly, y_sigma_perp)

# Extract formula
σ_perp = c0 + c1·blur + c2·contrast + c3·N + c4·blur² + ...

# Evaluate
R² = model.score(X_poly, y_sigma_perp)

If R² > 0.9: Formula works, use it (classical sufficient)
If R² < 0.7: Function is complex, try quantum
```

**Also try:**
- Neural network (MLP: 3 → 4)
- Symbolic regression (PySR library - genetic programming)
- Decision trees, random forests

**Timeline:** 1-2 days

**Cost:** Free

---

### Quantum Approach (If Classical Fails)

**Only if:** Classical methods fail to find predictive formulas (R² < 0.7)

#### Option A: Variational Quantum Regressor (VQA)

**Concept:** Quantum circuit learns function f_edge

```
Architecture:

1. Input encoding (classical → quantum):
   feature_encode(blur, contrast, N) → |ψ_in⟩

2. Parameterized Quantum Circuit (PQC):
   U(θ) |ψ_in⟩ = |ψ_out⟩

   Where θ = variational parameters (trained)

3. Measurement (quantum → classical):
   Measure |ψ_out⟩ → (σ_perp, σ_parallel, spacing, alpha)

4. Classical evaluation:
   Render with parameters, compute PSNR
   Loss = |PSNR_predicted - PSNR_target|²

5. Optimization (hybrid quantum-classical):
   Classical optimizer adjusts θ to minimize loss
   Quantum circuit evaluated at each step

6. Extraction:
   After training, probe quantum circuit:
   - Test on grid of (blur, contrast, N) values
   - Record outputs
   - Fit classical function to quantum outputs
   OR: Use quantum circuit as oracle (requires quantum access)
```

**Implementation:** Qiskit VQRegressor

**Advantage:** Quantum circuit might capture complex patterns

**Disadvantage:**
- Classical evaluation (PSNR) is bottleneck
- Requires many quantum circuit evaluations (expensive)
- NISQ noise limits precision

**Feasibility:** Medium (VQAs exist, but this specific problem is novel)

**Cost:** IBM Quantum time (hours of circuit evaluations)

**Timeline:** 2-4 weeks (implement, train, extract)

---

#### Option B: Quantum-Enhanced Symbolic Regression

**Concept:** Quantum algorithm searches space of mathematical expressions

**Classical symbolic regression (PySR):**
- Genetic programming explores expressions
- Try: σ_perp = a·blur + b, σ_perp = a·exp(b·blur), etc.
- Evaluates ~10^6 candidates (hours-days)
- Finds best-fit formula

**Quantum symbolic regression (theoretical):**
- Encode expression space in quantum superposition
- Quantum search explores expressions simultaneously
- Quantum tunneling escapes local optima in expression space
- Finds globally optimal formula

**Challenge:** This is RESEARCH (no off-the-shelf quantum symbolic regression)

**Would require:**
- Custom quantum algorithm development
- Encoding mathematical expressions in quantum states
- Quantum evaluation of fitness
- Collaboration with quantum computing researchers

**Timeline:** Months (research project)

**Feasibility:** Low (cutting edge, no existing tools)

**Verdict:** Too ambitious unless you're doing quantum computing PhD

---

### Practical Quantum Path for Formula Discovery

**Realistic approach:** Use quantum for SAMPLING optimal solutions, classical for fitting

```
Phase A: Generate Quantum-Optimized Examples (expensive)

For 100 test cases (blur, contrast) combinations:
  1. Use quantum annealing to find optimal Gaussian placement
  2. Classically refine parameters
  3. Record: (blur, contrast) → optimal_params

This generates high-quality training data (globally optimal solutions)

Phase B: Classical Learning (cheap)

Fit classical model to quantum-generated data:
  - Input: (blur, contrast)
  - Output: optimal_params (from quantum solutions)
  - Model: Polynomial, NN, or symbolic regression

Extract formula that approximates quantum solutions

Phase C: Classical Deployment (free)

Use formula discovered from quantum data
No quantum needed at runtime
```

**Cost:** 100 quantum annealing runs (~1-10 hours total on D-Wave)

**Benefit:** Training data is globally optimal (quantum-discovered)

**Output:** Classical formula based on quantum-optimized examples

**Feasibility:** HIGH (quantum annealing is proven for placement problems)

---

## Problem 3: Strategy Discovery via Quantum Meta-Optimization

### Objective
Find optimal optimization strategy per primitive type

**Search space:**
- Optimizer: {Adam, L-BFGS, SGD, RMSprop, ...}
- Learning rate: [0.0001, 0.001, 0.01, 0.1]
- Constraints: {free, fix_theta, fix_shape, ...}
- Iterations: [100, 500, 1000, 2000]
- → Combinatorial explosion (~1000s of combinations)

---

### Classical Baseline

**Grid search or Bayesian optimization:**
```python
from skopt import gp_minimize

def evaluate_strategy(optimizer, lr, constraints, iterations):
    # Run optimization on 10 test cases
    # Return: average PSNR, average convergence speed
    pass

# Search space
space = [
    Categorical(['adam', 'lbfgs', 'sgd']),
    Real(1e-4, 1e-1, prior='log-uniform'),
    Categorical(['free', 'fix_theta', 'fix_shape']),
    Integer(100, 2000)
]

# Bayesian optimization
result = gp_minimize(evaluate_strategy, space, n_calls=100)
best_strategy = result.x
```

**Cost:** 100 evaluations × 10 test cases × ~60s = ~17 hours

**Works, but slow and potentially stuck in local optimum (Bayesian GP has local minima)**

---

### Quantum Approach: QUBO for Strategy Search

**Formulation:**

```
Encode strategy as binary variables:

optimizer_type:
  - adam: [1, 0, 0]
  - lbfgs: [0, 1, 0]
  - sgd: [0, 0, 1]

learning_rate (discretized to 8 levels):
  - [b0, b1, b2] binary encoding

constraints:
  - free: [1, 0, 0]
  - fix_theta: [0, 1, 0]
  - fix_shape: [0, 0, 1]

iterations (discretized):
  - [b0, b1, b2, b3] binary (16 levels)

Total: ~15 binary variables

QUBO objective:
Q(x) = performance_score(strategy)
     + penalty for invalid combinations

Solve with D-Wave quantum annealer
```

**Evaluation:**
- Still need to run classical optimizations to measure performance
- But quantum explores strategy combinations in superposition
- Potentially finds global optimum faster

**Advantage:** Quantum tunneling escapes local optima in strategy space

**Feasibility:** HIGH (small QUBO, D-Wave can solve)

**Cost:** Multiple D-Wave runs (~10-50) with classical evaluations

---

### Practical Hybrid Approach

**Phase 1: Coarse quantum search**
- Use quantum annealing to identify top 10 promising strategies
- Quantum explores discrete strategy space globally

**Phase 2: Classical fine-tuning**
- Take top 10 from quantum
- Classical Bayesian optimization to refine continuous parameters (exact LR, exact iterations)

**Phase 3: Validation**
- Test refined strategies on held-out test cases
- Extract: "For edges, use Adam lr=0.023, fix_theta, 743 iterations"

**Phase 4: Classical deployment**
- Apply discovered strategy (no quantum needed)

---

## Problem-Specific Quantum Algorithms

### For Gaussian Placement: Quantum Annealing (QUBO)

**Problem:** Place N Gaussians on image to minimize reconstruction error

**QUBO Formulation:**

```
Discretize image to grid: W×H locations (e.g., 64×64 = 4096)

Binary variables:
  x_i ∈ {0,1} for i ∈ [1, W×H]
  x_i = 1 means "place Gaussian at grid location i"

Constraint:
  ∑ x_i = N (exactly N Gaussians)

Objective:
  Q(x) = -∑_i w_i·x_i + λ₁·∑_ij overlap_ij·x_i·x_j + λ₂·(∑x_i - N)²

Where:
  w_i = error reduction (from placing Gaussian at location i)
       = |∇I(i)| or local variance or residual energy

  overlap_ij = penalty if Gaussians i and j too close
             = 1 if distance(i,j) < threshold, else 0

  λ₁, λ₂ = penalty weights

D-Wave Constraint: ~5000 variables (qubits)
→ Can handle up to ~5000 candidate locations
```

**Workflow:**

```
1. Classical preprocessing:
   - Compute error reduction map w_i for all grid points
   - Compute overlap penalty matrix overlap_ij
   - Construct QUBO matrix Q

2. Quantum annealing (D-Wave):
   - Submit QUBO to quantum annealer
   - Annealing time: ~0.01-1 seconds (actual quantum time)
   - Returns: Binary solution x* (which locations selected)

3. Classical post-processing:
   - Extract selected locations: {i : x_i = 1}
   - Place Gaussians at those (x,y) coordinates
   - Set initial parameters (σ, θ, color, alpha) from local image properties
   - Optionally: classical refinement (Adam on continuous params)

4. Render and evaluate
```

**Comparison:**
- vs Random placement
- vs Gradient-based placement
- vs K-means clustering placement
- vs Classical simulated annealing

**Expected advantage:** Better global placement (escaped local minima)

**Proven similar problem:** Stereo matching (2024 paper), point cloud compression (2020 paper)

---

### For Feature Classification: Quantum CNN / Quantum Kernels

**Problem:** Classify image regions into primitive types

**Quantum CNN Approach (based on MicroCloud Hologram, 2025):**

```
1. Input: Image patch features (500-dimensional)

2. Quantum encoding:
   |ψ_in⟩ = encode(features)

3. Quantum Convolutional Layers:
   - Quantum convolution (local feature extraction)
   - Quantum pooling (downsampling)
   - Hierarchical feature learning

4. Quantum classification layer:
   - Output qubits measured
   - Collapse to class label: {edge, region, texture, junction, blob}

5. Training (hybrid):
   - Classical optimizer adjusts quantum circuit parameters
   - Trained on labeled dataset (patches with known primitives)

6. Extraction:
   - Use trained quantum circuit for classification
   - OR: Distill to classical CNN (teacher-student)

7. Classical deployment:
   - If distilled: use classical CNN (no quantum)
   - If oracle: need quantum access (not ideal)

Recommendation: Distill to classical for deployment
```

**Advantage:** Quantum Hilbert space might capture complex patterns better than classical kernels

**Evidence:** Haiqu Nov 2025 showed quantum > classical for pattern recognition

**Feasibility:** MEDIUM-HIGH (quantum CNNs exist in Qiskit, demonstrated recently)

**Cost:** Training time on quantum computer (hours-days)

---

## Realistic Near-Term Experiment

### Experiment: Quantum Annealing for Gaussian Placement

**Goal:** Test if quantum finds better placement than classical heuristics

**Setup:**

**Problem:** Place N=50 Gaussians on 32×32 grid (1024 locations) to represent edge image

**Step 1: Classical (prepare QUBO)**
```python
# Compute error reduction map
edge_image = generate_straight_edge(blur=2, contrast=0.5)
gradient_map = compute_gradient_magnitude(edge_image)

# Error reduction: placing Gaussian at location i reduces error by w_i
w = gradient_map.flatten()  # 1024 values

# Overlap penalty (Gaussians too close)
overlap_matrix = compute_overlap_penalties(threshold=5)  # 1024×1024

# Construct QUBO
from dwave.system import DWaveSampler, EmbeddingComposite

Q = {}
for i in range(1024):
    Q[(i,i)] = -w[i]  # Linear terms (benefit of placing here)

for i in range(1024):
    for j in range(i+1, 1024):
        if overlap_matrix[i,j] > 0:
            Q[(i,j)] = overlap_matrix[i,j]  # Quadratic terms (overlap penalty)

# Constraint: exactly N=50 Gaussians
# Use penalty method or constrained QUBO formulation
```

**Step 2: Quantum annealing (D-Wave)**
```python
sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_qubo(Q, num_reads=100)

# Get best solution
best_solution = response.first.sample
selected_locations = [i for i, val in best_solution.items() if val == 1]

print(f"Quantum selected {len(selected_locations)} locations")
```

**Step 3: Classical evaluation**
```python
# Place Gaussians at quantum-selected locations
gaussians = []
for idx in selected_locations:
    x, y = idx % 32, idx // 32
    g = Gaussian2D(x=x, y=y, sigma_perp=1.0, sigma_parallel=5.0, ...)
    gaussians.append(g)

# Render
rendered = render(gaussians)

# Measure quality
psnr_quantum = compute_psnr(edge_image, rendered)

# Compare vs classical baselines
psnr_random = baseline_random(N=50)
psnr_gradient = baseline_gradient(N=50)
psnr_kmeans = baseline_kmeans(N=50)

print(f"Quantum: {psnr_quantum:.2f} dB")
print(f"Gradient: {psnr_gradient:.2f} dB")
print(f"K-means: {psnr_kmeans:.2f} dB")
```

**Decision:**
- If quantum > classical baselines by >1 dB → advantage demonstrated
- If quantum ≈ classical → no advantage (but still publishable negative result)

---

### Cost & Feasibility

**D-Wave Leap Free Tier:**
- 1 minute/month of QPU time
- ~10-100 QUBO solves (depending on problem size)
- Sufficient for initial experiments

**D-Wave Leap Commercial:**
- $2000/month
- Unlimited access
- For serious research

**IBM Quantum Free Tier:**
- 10 minutes/month
- For VQA experiments
- Queuing (may wait hours)

**Feasibility for You:**
- ✓ D-Wave free tier: Try quantum placement now (sign up, free)
- ✓ IBM free tier: Try quantum clustering/VQA (sign up, free)
- ✗ Commercial: Not worth cost until you validate approach

**Timeline:**
- Week 1: Learn D-Wave Ocean SDK, formulate QUBO
- Week 2: Test on D-Wave free tier
- Week 3-4: Analyze results, compare vs classical
- → 1 month total

---

## Quantum Discovery Workflow

### Research Pipeline

**Phase 1: Classical Foundation (Current - Months 1-3)**
- Phase 0.5: Edge primitive understanding (happening now)
- Classical regression: Try polynomial, NN, symbolic regression
- Establish: What classical methods CAN discover
- Build: Comprehensive dataset (parameters → PSNR)

**Phase 2: Quantum Primitive Discovery (Months 3-4)**
- Collect: 10,000 image patch features
- Run: Quantum clustering (IBM Quantum)
- Extract: Natural primitive groupings
- Validate: Do quantum-discovered primitives make sense?
- Document: Classical classification rules

**Phase 3: Quantum Placement Testing (Months 4-5)**
- Formulate: Gaussian placement as QUBO
- Run: D-Wave quantum annealing (free tier)
- Compare: Quantum vs gradient-based vs K-means
- Evaluate: Advantage or not?
- Extract: If advantageous, use quantum-discovered placement as training data

**Phase 4: Classical Formula Extraction (Months 5-6)**
- Compile: All quantum-discovered solutions
- Fit: Classical formulas to quantum results
- Validate: On held-out test cases
- Deploy: Pure classical rules (derived from quantum)

**Phase 5: Publication (Months 6-7)**
- Paper: "Quantum-Discovered Primitives and Rules for Gaussian Image Representation"
- Contributions:
  - First quantum clustering for primitive discovery
  - Quantum annealing for Gaussian placement
  - Classical deployment of quantum-discovered rules
- Venues: Quantum computing OR computer vision (interdisciplinary)

---

## Expected Outputs (Quantum-Discovered, Classically-Deployed)

### 1. Primitive Definitions
```python
# Discovered via quantum clustering on 10K patches

Primitive 1: "Sharp Directional Structures"
  Detection rule: |∇I| > 0.3 AND eigenvalue_ratio > 5
  Gaussian config: elongated, θ = gradient direction, σ_perp < 2px

Primitive 2: "Smooth Isotropic Regions"
  Detection rule: variance < 0.01 AND eigenvalue_ratio < 1.5
  Gaussian config: large, isotropic, σ = 10-20px

# Etc. - data-driven, not designed
```

**Usage:** Classical feature detection + rule-based classification

---

### 2. Placement Formula
```python
# Discovered via quantum annealing on 100 test cases + classical regression

def f_edge_quantum_discovered(blur, contrast, N):
    """
    Formula learned from quantum-optimized placements
    """
    sigma_perp = 0.73 * blur + 0.41  # From fitting quantum solutions
    sigma_parallel = 8.2 + 1.3 * log(N)  # Non-obvious form found by quantum
    spacing = sigma_parallel / (2.1 + 0.3 * contrast)  # Complex relationship
    alpha = 0.28 / contrast  # Consistent with Phase 0 finding

    return {
        'sigma_perp': sigma_perp,
        'sigma_parallel': sigma_parallel,
        'spacing': spacing,
        'alpha': alpha
    }
```

**Usage:** Pure classical formula (evaluated in microseconds)

---

### 3. Optimization Strategy
```python
# Discovered via quantum meta-optimization

EDGE_STRATEGY = {
    'optimizer': 'adam',
    'learning_rate': 0.0234,  # Quantum-discovered optimal
    'constraints': ['fix_theta'],  # Don't optimize orientation
    'iterations': 743  # Surprisingly specific optimal value
}

REGION_STRATEGY = {
    'optimizer': 'lbfgs',
    'learning_rate': 0.1,
    'constraints': ['mask_to_region'],
    'iterations': 156
}

# Different per primitive - quantum found specialized strategies
```

**Usage:** Classical optimizer with quantum-discovered hyperparameters

---

## Advantages of Quantum Discovery

### 1. **Global Search** (vs local classical search)
- Quantum tunneling explores entire space
- Classical grid search: tests 10^3 points
- Quantum superposition: explores 2^100 configurations simultaneously (theoretical)
- Escapes local optima

### 2. **Pattern Recognition in High-Dimensional Data**
- 500-dimensional feature spaces
- Quantum Hilbert space naturally high-dimensional
- November 2025: Haiqu demonstrated quantum > classical for anomaly detection
- Could find non-obvious primitive groupings

### 3. **Non-Convex Function Learning**
- f_edge might be complex (non-polynomial)
- Quantum VQA can represent complex functions
- Classical regression might miss functional form

### 4. **One-Time Cost, Infinite Value**
- Expensive quantum discovery
- But rules used forever (no recurring cost)
- Amortization makes it economical

---

## When to Use Quantum Discovery

### Green Light (Quantum Worth Trying):
✓ Classical methods fail (R² < 0.7, no clear patterns)
✓ High-dimensional feature space (100+ features)
✓ Non-convex, many local minima
✓ Large search space (combinatorial)
✓ One-time discovery acceptable (not real-time)
✓ Can extract classical rules from quantum results

### Red Light (Stick with Classical):
✗ Classical works well (R² > 0.9)
✗ Simple patterns (linear, quadratic)
✗ Small search space (<100 combinations)
✗ Low-dimensional features (<10 dimensions)
✗ Need real-time discovery (quantum too slow)
✗ Can't extract classical rules (quantum circuit required forever)

---

## Realistic Timeline & Costs

### Near-Term (Next 3 Months)

**Focus:** Classical baseline
- Phase 0.5 results (masked PSNR, N sweep)
- Classical regression on collected data
- Evaluate if classical sufficient

**Cost:** $0 (classical compute)

---

### Mid-Term (Months 3-6, if classical insufficient)

**Action:** Quantum discovery experiments

**Experiment 1: Quantum Clustering (Primitive Discovery)**
- Platform: IBM Quantum free tier (10 min/month)
- Time needed: ~1-5 hours quantum circuit evaluation (spread over months with free tier)
- Alternative: Apply for IBM Quantum Researcher access (more time)
- Cost: $0 (free tier) or research access (free for academics)

**Experiment 2: Quantum Annealing (Placement)**
- Platform: D-Wave Leap free tier (1 min/month)
- Time needed: ~10-50 QUBO solves (~1-10 minutes total)
- Cost: $0 (free tier) or $100-500 if commercial needed

**Total cost: $0-500** (free tiers likely sufficient for proof-of-concept)

---

### Long-Term (Months 6-12, if quantum shows promise)

**Publication & Refinement:**
- Write paper on quantum-discovered rules
- Validate on larger datasets
- Compare quantum vs classical discovery rigorously
- Extract final classical formulas for deployment

**Cost:** Time (not money if using free tiers)

---

## Getting Started (Practical Steps)

### Step 1: Finish Phase 0.5 (Classical Baseline)
- Get masked PSNR data
- Try classical regression
- See if patterns emerge

### Step 2: If Classical Works
→ You have your formulas (free, fast)
→ No quantum needed
→ Deploy classically

### Step 3: If Classical Fails (Complex Patterns)
→ Learn quantum computing (1-2 weeks)
→ Sign up for IBM Quantum + D-Wave Leap free tiers
→ Run quantum discovery experiments (2-4 weeks)
→ Extract classical rules from quantum results

### Step 4: Deploy Classically
→ Apply quantum-discovered formulas
→ No quantum at runtime
→ Pure classical deployment

---

## Resources for Quantum Discovery

### Free Platforms (Research Access)

**IBM Quantum:**
- Free tier: 10 min/month
- 127-qubit Eagle processor
- Qiskit library (Python)
- Use for: VQA, quantum clustering, quantum ML

**D-Wave Leap:**
- Free tier: 1 min/month QPU time
- ~5000-qubit quantum annealer
- Ocean SDK (Python)
- Use for: QUBO placement problems

**Google Quantum AI:**
- Limited researcher access
- Cirq library
- Use for: VQA, quantum optimization

---

### Learning Resources

**For QUBO formulation:**
- D-Wave tutorials: "Problem Formulation Guide"
- Paper: "Quantum Annealing for Computer Vision" (2024)
- Book: "Quantum Computing: An Applied Approach" (Chapter on QUBO)

**For VQA:**
- Qiskit textbook: "Variational Quantum Algorithms"
- Tutorial: "Quantum Machine Learning with Qiskit"
- Paper: "Variational Quantum Optimization" (2024)

**For Quantum ML:**
- Qiskit Machine Learning documentation
- Haiqu blog posts (Nov 2025 breakthrough)
- Course: "Quantum Machine Learning" (edX, free)

**Timeline:** 1-2 weeks to learn basics, 1 month to become proficient

---

## Success Criteria

### Quantum Discovery Succeeds If:

**For Primitive Discovery:**
- ✓ Quantum clustering finds groupings with R² > 0.9 (high separability)
- ✓ Primitives are interpretable (can explain what each cluster represents)
- ✓ Classical classifier based on quantum clusters outperforms human-designed primitives

**For Formula Discovery:**
- ✓ Quantum-discovered formula has R² > 0.9 on test set
- ✓ Formula is extractable (analytic or small NN)
- ✓ Outperforms classical regression (>5% improvement in prediction accuracy)

**For Placement Discovery:**
- ✓ Quantum annealing finds placements with >1 dB PSNR improvement vs classical heuristics
- ✓ Advantage is consistent across multiple test cases
- ✓ Can extract classical placement rules from quantum solutions

---

## Failure Modes & Alternatives

### If Quantum Shows No Advantage:

**Still valuable:**
- Negative result is publishable
- "We tested quantum for Gaussian image optimization, no advantage found"
- Validates that classical methods are sufficient
- Rules out quantum for this problem class

**Alternatives:**
- Advanced classical methods (symbolic regression, neural architecture search)
- Larger datasets (maybe quantum needs more data)
- Different quantum algorithms (try VQA if QUBO failed, or vice versa)

---

## The Vision: Quantum-Classical Hybrid Pipeline

### Discovery Phase (Quantum - One Time)

```
Image Dataset
  ↓
Feature Extraction (classical)
  ↓
Quantum Clustering → Primitive Definitions
  ↓
Quantum Annealing → Optimal Placements (100 examples)
  ↓
Classical Regression → f_placement formula
  ↓
Quantum Meta-Optimization → Optimization Strategies
  ↓
EXTRACT ALL TO CLASSICAL RULES
```

**Output:**
- Primitive definitions (classical rules)
- Placement formulas (classical functions)
- Optimization strategies (classical recipes)

---

### Deployment Phase (Classical - Forever)

```
New Image
  ↓
Feature Extraction (classical)
  ↓
Primitive Classification (classical, using quantum-discovered primitives)
  ↓
Apply Placement Formulas (classical, quantum-discovered)
  ↓
Optimize Parameters (classical, quantum-discovered strategies)
  ↓
Render (classical)
  ↓
Output
```

**No quantum computer needed.**
**Fast, cheap, scalable.**
**All knowledge extracted from quantum discovery.**

---

## Recommendation

**Short-term (Now):**
- Finish Phase 0.5 classically
- Try classical regression
- See if you need quantum

**Mid-term (3-6 months):**
- If classical insufficient, try quantum
- Start with quantum clustering (easiest, proven Nov 2025)
- Then try quantum annealing (placement)

**Long-term (6-12 months):**
- Publish quantum-discovered rules
- Deploy classically
- Quantum impact without quantum runtime cost

**Your idea is solid:** Quantum for discovery, classical for deployment.

**Just need:** Classical baseline first to know if quantum is needed.

---

**Document complete. Ready for future quantum exploration when classical baseline is established.**
