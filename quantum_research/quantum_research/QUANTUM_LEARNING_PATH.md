# Quantum Computing Learning & Experimentation Path
## Discovering Gaussian Image Rules with IBM Quantum

**Date:** 2025-11-17
**Status:** Active Learning & Environment Setup
**Goal:** Learn Qiskit capabilities, build test environment, run first quantum experiments

---

## What We're Learning About

### IBM Quantum Free Tier (Current Offering - 2025)

**Access:**
- 10 minutes/month of quantum computer runtime
- 100+ qubit systems (Eagle, Heron processors)
- Cloud-based access via IBM Quantum Platform

**What 10 minutes can do:**
- Run QAOA on small graphs (~9 minutes for one experiment)
- Multiple small VQE experiments (~10-20 circuits)
- Many short circuit evaluations for learning
- Algorithm testing and development

**Limitation:** Not enough for extensive training, but sufficient for:
- Proof-of-concept experiments
- Algorithm validation
- Learning and development
- Initial testing before scaling up

---

## Qiskit Stack (What's Available)

### Core Libraries (as of 2025)

**1. Qiskit (Core)**
- Circuit construction and transpilation
- Quantum gates, measurements, statevector simulation
- Backend management (local simulators + IBM quantum computers)

**2. Qiskit Runtime**
- **Primitives (V2 - required as of Jan 2025):**
  - `Sampler`: Computes probabilities of measurement outcomes
  - `Estimator`: Computes expectation values of observables
- Error mitigation and resilience options
- Optimized circuit execution

**3. Qiskit Machine Learning**
- **Quantum Neural Networks (QNN):**
  - `EstimatorQNN`: Uses expectation values
  - `SamplerQNN`: Uses measurement outcomes
  - Can be used for regression and classification
- **Quantum Kernels:**
  - `FidelityQuantumKernel`: For SVM and clustering
  - Projects data into quantum Hilbert space
- **Classifiers/Regressors:**
  - `NeuralNetworkClassifier`
  - `NeuralNetworkRegressor`

**4. Qiskit Algorithms**
- `VQE` (Variational Quantum Eigensolver)
- `QAOA` (Quantum Approximate Optimization Algorithm)
- Various quantum algorithms for specific problems

**5. Qiskit Optimization**
- Quadratic Program formulation
- QUBO conversion
- Integration with QAOA

---

## Key Capabilities for Our Problem

### 1. **Amplitude Encoding** (High-Dimensional Data)

**What it does:**
- Encode N classical features in log2(N) qubits
- Example: 256 features → 8 qubits
- Your 500-dimensional feature vectors → 9 qubits

**Implementation:**
```python
from qiskit.circuit.library import RawFeatureVector

# Encode 256-dimensional feature vector
features = np.array([...])  # 256 values
feature_map = RawFeatureVector(feature_dimension=256)

# Creates quantum circuit that encodes features in amplitudes
# |ψ⟩ = ∑_i α_i |i⟩ where α_i = features[i] (normalized)
```

**Advantage:** Extremely efficient encoding of high-dimensional data

**Your use case:** Encode image patch features (gradient, curvature, entropy, texture → 500 dims)

---

### 2. **Quantum Kernels** (Pattern Recognition)

**What it does:**
- Maps classical data to quantum Hilbert space
- Computes kernel function using quantum state overlap (fidelity)
- Can reveal patterns invisible to classical kernels

**Implementation:**
```python
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC

# Create quantum feature map
feature_map = ZZFeatureMap(feature_dimension=10, reps=2)

# Quantum kernel
qkernel = FidelityQuantumKernel(feature_map=feature_map)

# Use with classical SVM
svc = SVC(kernel=qkernel.evaluate)
svc.fit(X_train, y_train)
```

**Advantage:** Quantum Hilbert space might capture complex patterns better than RBF/polynomial kernels

**Your use case:**
- Classify patches into primitives (edge vs region vs texture)
- High-dimensional feature space (500 features)
- Non-linear decision boundaries

---

### 3. **Quantum Neural Networks** (Function Learning)

**What it does:**
- Parameterized quantum circuits (PQC) as function approximators
- Can learn mappings: input → output
- Trained with classical optimizers (hybrid)

**Implementation:**
```python
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
from qiskit.circuit.library import RealAmplitudes

# Feature map (encode inputs)
feature_map = ZZFeatureMap(3)  # 3 input features

# Ansatz (trainable circuit)
ansatz = RealAmplitudes(3, reps=2)

# Combine into QNN
qnn = EstimatorQNN(
    feature_map=feature_map,
    ansatz=ansatz,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters
)

# Regressor (for learning functions)
regressor = NeuralNetworkRegressor(qnn, optimizer='L_BFGS_B')
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)
```

**Advantage:** Might learn complex functional relationships

**Your use case:**
- Learn f_edge: (blur, contrast, N) → (σ_perp, σ_parallel, spacing, alpha)
- Non-linear function approximation
- Could discover non-obvious functional forms

---

### 4. **QAOA** (Combinatorial Optimization)

**What it does:**
- Quantum Approximate Optimization Algorithm
- Solves combinatorial problems (Max-Cut, graph coloring, etc.)
- Hybrid: quantum circuit + classical optimization

**Implementation:**
```python
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_optimization import QuadraticProgram

# Define problem
qp = QuadraticProgram()
qp.binary_var_list(n)  # n binary variables
qp.minimize(linear=..., quadratic=...)  # Objective

# Convert to Ising Hamiltonian
operator, offset = qp.to_ising()

# Solve with QAOA
qaoa = QAOA(optimizer=COBYLA(), reps=3)
result = qaoa.compute_minimum_eigenvalue(operator)

# Extract solution
solution = result.eigenstate
```

**Advantage:** Can escape local minima better than classical annealers

**Your use case:**
- Placement: which grid points to select for Gaussians?
- Budget allocation: how many Gaussians per primitive type?
- Combinatorial optimization problems

---

## What We Can Actually Build

### Experiment 1: Quantum Kernel for Primitive Classification

**Setup:**
```python
# Dataset: Image patches with features
patches = [
    {'features': [grad_mag, curvature, entropy, ...], 'label': 'edge'},
    {'features': [...], 'label': 'region'},
    # ... 1000 patches
]

# Extract features (500-dimensional)
X = np.array([p['features'] for p in patches])
y = np.array([p['label'] for p in patches])

# Quantum kernel SVM
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Feature map for high dimensions (use subset or PCA if >20 dims)
feature_map = ZZFeatureMap(feature_dimension=20, reps=2)

qkernel = FidelityQuantumKernel(feature_map=feature_map)

# Classical SVM with quantum kernel
from sklearn.svm import SVC
qsvm = SVC(kernel=qkernel.evaluate)
qsvm.fit(X_train, y_train)

# Compare vs classical SVM
from sklearn.svm import SVC as ClassicalSVC
classical_svm = ClassicalSVC(kernel='rbf')
classical_svm.fit(X_train, y_train)

# Evaluate
print(f"Quantum Kernel Accuracy: {qsvm.score(X_test, y_test)}")
print(f"Classical RBF Accuracy: {classical_svm.score(X_test, y_test)}")
```

**Tests:** Does quantum kernel find better decision boundaries for primitive classification?

**Time:** ~5-10 kernel evaluations per minute on free tier → ~50-100 evaluations total

**Feasibility:** HIGH (documented, proven)

---

### Experiment 2: Quantum Regressor for Formula Learning

**Setup:**
```python
# Dataset: Parameter configurations → PSNR
# From Phase 0.5 results
data = [
    {'inputs': [blur, contrast, N], 'output': [sigma_perp]},
    # ... 100 data points
]

X = np.array([d['inputs'] for d in data])  # 3 features
y = np.array([d['output'] for d in data])  # 1 output

# Quantum Neural Network Regressor
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor

# Feature map + ansatz
feature_map = ZZFeatureMap(3, reps=2)
ansatz = RealAmplitudes(3, reps=3)

qnn = EstimatorQNN(
    feature_map=feature_map,
    ansatz=ansatz,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters
)

# Train quantum regressor
qregressor = NeuralNetworkRegressor(qnn, optimizer='L_BFGS_B')
qregressor.fit(X_train, y_train)

# Compare vs classical
from sklearn.neural_network import MLPRegressor
classical_nn = MLPRegressor(hidden_layers=(10, 10))
classical_nn.fit(X_train, y_train)

# Evaluate on test set
quantum_r2 = qregressor.score(X_test, y_test)
classical_r2 = classical_nn.score(X_test, y_test)

print(f"Quantum R²: {quantum_r2}")
print(f"Classical R²: {classical_r2}")

# If quantum is better, extract learned function
# Test on grid of inputs to understand what QNN learned
```

**Tests:** Does quantum regressor learn f_edge better than classical NN?

**Time:** Training might take full 10 minutes (many circuit evaluations)

**Feasibility:** MEDIUM (QNN training is slow, might hit time limit)

---

### Experiment 3: QAOA for Placement Optimization

**Setup:**
```python
# Problem: Place N=50 Gaussians on 32×32 grid (1024 locations)

from qiskit_optimization import QuadraticProgram

# Define binary optimization problem
qp = QuadraticProgram('gaussian_placement')

# 1024 binary variables (one per grid location)
qp.binary_var_list(1024, name='x')

# Objective: maximize weighted placement quality
# linear[i] = error_reduction at location i (from gradient map)
linear_coefficients = gradient_magnitude.flatten()  # 1024 values

# Quadratic: penalize nearby placements (overlap)
quadratic = {}
for i in range(1024):
    for j in range(i+1, 1024):
        dist = distance(i, j)
        if dist < 5:  # too close
            quadratic[(i,j)] = 10.0  # penalty

qp.maximize(linear=linear_coefficients, quadratic=quadratic)

# Constraint: exactly N=50 selected
qp.linear_constraint([1]*1024, '==', 50)

# Solve with QAOA
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA

qaoa = QAOA(optimizer=COBYLA(), reps=3)
result = qaoa.compute_minimum_eigenvalue(qp.to_ising()[0])

# Extract selected locations
solution = result.eigenstate
selected = [i for i in range(1024) if solution[i] > 0.5]

print(f"Quantum selected {len(selected)} locations")
```

**Tests:** Does QAOA find better placement than gradient-based?

**Time:** ~2-9 minutes per QAOA run (depends on circuit depth)

**Feasibility:** MEDIUM (QAOA documented but for smaller problems ~100 variables, not 1024)

---

## What I'm Starting to Understand (Your Vision)

Through researching the tools, I think you're seeing:

**Quantum computing doesn't just optimize better - it explores DIFFERENT search spaces:**

### The Quantum Hilbert Space Advantage

Classical machine learning:
- Projects data into feature space (finite dimensions)
- Learns in that space

Quantum machine learning:
- Projects data into quantum Hilbert space (exponentially large)
- Patterns that are non-linear/complex in classical space might be linear/simple in quantum space

**Example:**
- Classical: 500 features → 500-dimensional space (hard to find patterns)
- Quantum: 500 features → 2^9 = 512-dimensional Hilbert space (might reveal structure)

### Your Primitives in Quantum Space

**Classical approach:**
- Design primitives (M/E/J/R/B/T)
- OR cluster in classical feature space (K-means, hierarchical)

**Quantum approach:**
- Encode patches in quantum state
- Let quantum interference/entanglement reveal natural groupings
- Groupings might be INVISIBLE in classical space but obvious in quantum space

**This is what Haiqu demonstrated (Nov 2025) - quantum found patterns classical missed.**

---

## Realistic First Experiments (With Free Tier)

### Week 1: Environment Setup & Basics

**Install Qiskit:**
```bash
pip install qiskit qiskit-ibm-runtime qiskit-machine-learning
```

**IBM Quantum account:**
- Sign up: https://quantum.cloud.ibm.com
- Get API token
- 10 minutes/month free

**Hello World:**
```python
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService

# Save credentials (one time)
QiskitRuntimeService.save_account(token="YOUR_TOKEN")

# Simple circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Run on real quantum computer
service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False)

# Use Sampler primitive
from qiskit_ibm_runtime import Sampler
sampler = Sampler(backend)
job = sampler.run([qc])
result = job.result()
```

**Goal:** Verify quantum access works

**Time:** <1 minute

---

### Week 2: Quantum Kernel Classification Test

**Experiment:** Can quantum kernel classify edge vs region patches better than RBF kernel?

**Dataset:**
- Extract 200 patches from Kodak images
- 100 edge patches (high gradient)
- 100 region patches (low gradient)
- Features: 10-20 dimensional (gradient, curvature, variance, etc.)

**Code:**
```python
# Extract features (classical)
patches = extract_patches_from_kodak(n=200)
X = compute_features(patches)  # (200, 20) shape
y = label_patches(patches)  # 'edge' or 'region'

# Quantum kernel SVM
feature_map = ZZFeatureMap(20, reps=2)
qkernel = FidelityQuantumKernel(feature_map=feature_map)
qsvm = SVC(kernel=qkernel.evaluate)

# Classical baseline
classical_svm = SVC(kernel='rbf', gamma='scale')

# Train both
qsvm.fit(X_train, y_train)
classical_svm.fit(X_train, y_train)

# Compare accuracy
print(f"Quantum: {qsvm.score(X_test, y_test)}")
print(f"Classical: {classical_svm.score(X_test, y_test)}")
```

**Expected time:** ~5-10 minutes (kernel evaluations on quantum hardware)

**Within free tier:** YES (barely)

**Decision:** If quantum > classical → worth exploring more

---

### Week 3: Quantum Regressor for f_edge

**Experiment:** Can quantum NN learn f_edge better than classical NN?

**Dataset:**
- Phase 0.5 results (blur, contrast, N) → optimal parameters
- ~100-200 data points

**Code:**
```python
# From Phase 0.5
X = phase05_data[['blur', 'contrast', 'N']].values  # (200, 3)
y = phase05_data['sigma_perp'].values  # (200,)

# Quantum regressor
feature_map = ZZFeatureMap(3, reps=2)
ansatz = RealAmplitudes(3, reps=3)
qnn = EstimatorQNN(feature_map=feature_map, ansatz=ansatz, ...)
qregressor = NeuralNetworkRegressor(qnn, optimizer='L_BFGS_B')

# Train (this will use quantum circuits)
qregressor.fit(X_train, y_train)

# Classical baseline
from sklearn.neural_network import MLPRegressor
classical = MLPRegressor((10, 10)).fit(X_train, y_train)

# Compare
quantum_r2 = qregressor.score(X_test, y_test)
classical_r2 = classical.score(X_test, y_test)
```

**Expected time:** ~10 minutes (circuit training)

**Within free tier:** MAYBE (tight fit, might need multiple months or simulator first)

**Decision:** If quantum learns complex function better → extract as formula

---

## Learning Resources (Practical Path)

### Path 1: Official IBM Quantum Learning (Start Here)

**1. IBM Quantum Learning Platform**
- URL: https://quantum.cloud.ibm.com/learning
- Free courses with hands-on exercises
- Runs on real quantum computers

**Key courses:**
- "Quantum Computing in Practice" (fundamentals)
- "Variational Quantum Algorithms" (VQA, QAOA)
- "Quantum Machine Learning" (QNN, kernels)

**Timeline:** 2-4 weeks (few hours per week)

---

### Path 2: Qiskit Machine Learning Tutorials

**Official tutorials:**
- https://qiskit-community.github.io/qiskit-machine-learning/tutorials/

**Key tutorials for us:**
1. **Quantum Neural Networks** - How QNN works
2. **Classification and Regression** - Using QNN for prediction
3. **Quantum Kernels** - For SVM/clustering
4. **Quantum Autoencoder** - For dimensionality reduction

**Timeline:** 1-2 weeks (hands-on coding)

---

### Path 3: Practical Examples (GitHub)

**Search for:**
- "qiskit machine learning examples"
- "qiskit quantum kernel svm"
- "qiskit vqe optimization"

**Study implementations, adapt to our problem**

**Timeline:** Ongoing (reference as needed)

---

## Setting Up Environment

### Installation

```bash
# Create virtual environment
python3 -m venv quantum_env
source quantum_env/bin/activate

# Install Qiskit stack
pip install qiskit>=1.0
pip install qiskit-ibm-runtime
pip install qiskit-machine-learning
pip install qiskit-algorithms
pip install qiskit-optimization

# Supporting libraries
pip install numpy scipy matplotlib
pip install scikit-learn pandas

# For visualization
pip install pylatexenc  # circuit visualization
```

### IBM Quantum Account Setup

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Save credentials (one time)
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="YOUR_IBM_QUANTUM_TOKEN",  # Get from quantum.cloud.ibm.com
    overwrite=True
)

# Verify access
service = QiskitRuntimeService()
backends = service.backends()
print(f"Available backends: {[b.name for b in backends]}")
```

### Testing

```python
# Test 1: Can we run circuits?
from qiskit import QuantumCircuit
qc = QuantumCircuit(2)
qc.h(0)
qc.measure_all()

# Use simulator first (free, unlimited)
from qiskit_aer import Aer
simulator = Aer.get_backend('qasm_simulator')
result = simulator.run(qc).result()
print(result.get_counts())

# Test 2: Can we use machine learning?
from qiskit_machine_learning.kernels import FidelityQuantumKernel
qkernel = FidelityQuantumKernel(feature_map=ZZFeatureMap(2))
matrix = qkernel.evaluate(X_small)  # Small test
print(f"Kernel matrix shape: {matrix.shape}")
```

---

## Understanding Quantum Advantage (What I'm Learning)

### It's Not Just "Faster Optimization"

**Classical optimization:**
- Local search (gradient descent, Adam)
- Gets stuck in local minima
- Multi-start helps but expensive

**Quantum optimization:**
- Global search (superposition explores many solutions simultaneously)
- Tunneling through barriers (escapes local minima via quantum effects)
- BUT: Measurement collapses to ONE solution (still need multiple runs)

**The advantage:** Quantum explores different parts of landscape via quantum mechanics, not random sampling

---

### It's About the Search Space Structure

**Classical function learning:**
- Fit polynomial, NN, symbolic regression
- These are TEMPLATES (polynomial of degree k, NN with m layers, etc.)
- Search within template space

**Quantum function learning:**
- PQC (Parameterized Quantum Circuit) represents functions
- Quantum Hilbert space structure might match problem structure better
- Could discover functional forms that aren't in classical template space

**Example:**
- Classical tries: σ = a·x + b, σ = a·x² + b·x + c, σ = a·exp(b·x)
- Quantum PQC might learn: σ = weird_quantum_function(x) that doesn't have classical analog
- Then you probe it, fit classical approximation

---

### It's About Feature Space Geometry

**Classical feature space:**
- 500 features → 500-dimensional Euclidean space
- K-means finds clusters based on Euclidean distance
- Might miss non-Euclidean structure

**Quantum Hilbert space:**
- 500 features → encoded in ~9 qubits → 2^9 = 512 dimensional Hilbert space
- But it's a DIFFERENT 512D space (complex amplitudes, quantum interference)
- Structure might reveal different patterns

**Haiqu (Nov 2025) found:** Quantum clustering found anomalies classical missed

**For primitives:** Quantum clustering might reveal "natural primitives" that aren't obvious classically

---

## What's Clicking For Me (Your Vision)

I think you're seeing quantum computing as:

**NOT:** Better runtime performance (faster optimization)

**BUT:** Access to a DIFFERENT SEARCH SPACE that might contain the answers

**The quantum Hilbert space might naturally encode image structure in a way classical feature spaces don't.**

**Example:**
- Maybe "edge primitive" and "junction primitive" are NOT separate in quantum space
- Maybe they're one primitive with continuous parameter (revealed by quantum clustering)
- Classical sees: two clusters (edges, junctions)
- Quantum sees: one cluster with curvature parameter
- **The quantum view might be more fundamental**

**Is this closer to what you're thinking?**

Let me continue building the framework...