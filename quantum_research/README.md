# Quantum Computing Research for Gaussian Image Primitives

**Purpose:** Explore quantum computing for discovering primitives, formulas, and optimization strategies

**Status:** Learning Phase - Environment Setup and Initial Experiments

**Key Insight:** Use quantum computing for ONE-TIME discovery, deploy classically forever

---

## The Vision

### Quantum as Discovery Tool

**Problem:** Classical approaches are limited:
- Manual parameter sweeps (slow, coarse)
- Gradient-based optimization (local minima)
- Pre-designed primitives (M/E/J/R/B/T might not be optimal)
- Grid search (exponential cost)

**Quantum possibility:**
- Explore ENTIRE spaces via superposition
- Escape local minima via tunneling
- Discover patterns in quantum Hilbert space invisible classically
- Find global optima more efficiently

**Deployment:** Extract classical rules from quantum discoveries, use forever

---

## Three Discovery Targets

### 1. **What ARE the Primitives?**

**Classical:** Design based on computer vision (edges, regions, textures)

**Quantum:** Cluster 10K image patches in quantum Hilbert space
- Encode 500-dimensional features in ~9 qubits (amplitude encoding)
- Quantum clustering finds natural groupings
- Groups might NOT match human intuition (data-driven discovery)

**Output:** Primitive definitions extracted from quantum clustering

**Status:** Haiqu (Nov 2025) demonstrated this works - quantum found patterns classical missed

---

### 2. **What are the Placement Formulas?**

**Classical:** Polynomial regression, NN, symbolic regression

**Quantum:** Quantum regressor learns f_edge(blur, contrast, N) → parameters
- Parameterized quantum circuit (PQC) as function approximator
- Might learn complex relationships classical models miss
- Quantum Hilbert space structure might match problem structure

**Output:** Probe quantum circuit, extract classical formula

**Status:** QNN regressors exist in Qiskit, not yet tested for this problem

---

### 3. **What are Optimal Strategies?**

**Classical:** Grid search, Bayesian optimization (local search)

**Quantum:** QAOA meta-optimization over strategy space
- Discrete: optimizer type, constraints (QUBO formulation)
- Global search via quantum annealing
- Find specialized strategies per primitive

**Output:** Classical recipes (optimizer + hyperparameters)

**Status:** QAOA proven for combinatorial optimization, need to formulate our problem

---

## Current Capabilities (IBM Quantum Free Tier)

### What You Get (2025)

**Access:**
- 10 minutes/month of quantum processor time
- 100+ qubit systems (Eagle r3, Heron processors)
- Cloud-based via IBM Quantum Platform
- Qiskit Runtime primitives (Sampler, Estimator)

**Realistic usage:**
- Quantum kernel classification: ~50-100 kernel evaluations (~5-10 min)
- QNN training: ~1 full training run (~10 min) OR simulator + final validation
- QAOA: ~1-3 optimization runs (~3-9 min each)
- Many small experiments (circuit testing, algorithm development)

**Strategy:** Develop on simulator (free, unlimited), validate on quantum (10 min/month)

---

## Qiskit Stack (Available Tools)

### Core Libraries

**Qiskit 1.0+**
- Circuit construction
- Transpilation (optimization for quantum hardware)
- Simulation (local, unlimited)

**Qiskit Runtime**
- Sampler primitive (measurement probabilities)
- Estimator primitive (expectation values)
- Error mitigation built-in

**Qiskit Machine Learning**
- Quantum Neural Networks (EstimatorQNN, SamplerQNN)
- Quantum Kernels (FidelityQuantumKernel)
- Classifiers/Regressors (NeuralNetworkClassifier, NeuralNetworkRegressor)

**Qiskit Algorithms**
- VQE (Variational Quantum Eigensolver)
- QAOA (Quantum Approximate Optimization Algorithm)
- Grover, Shor, etc.

**Qiskit Optimization**
- QuadraticProgram (define optimization problems)
- QUBO conversion
- Integration with QAOA

---

## Key Techniques for Our Problems

### 1. **Amplitude Encoding** (High-Dimensional Features)

**Capability:**
- Encode N classical features in log2(N) qubits
- 256 features → 8 qubits
- 500 features → 9 qubits

**Efficiency:**
- Exponential compression of classical data
- Access to exponentially large Hilbert space

**Implementation:**
```python
from qiskit.circuit.library import RawFeatureVector

features = np.array([...])  # Your 500-dimensional feature vector
normalized = features / np.linalg.norm(features)

feature_map = RawFeatureVector(feature_dimension=len(features))
qc = feature_map.assign_parameters(normalized)

# Now |ψ⟩ = ∑_i features[i] |i⟩
```

**Your use:** Encode image patch features (gradient, curvature, entropy, texture...)

---

### 2. **Quantum Kernels** (Pattern Recognition)

**Capability:**
- Classical data → quantum feature space → compute similarity
- Fidelity-based kernel: K(x,y) = |⟨ψ(x)|ψ(y)⟩|²
- Can capture non-linear patterns

**Advantage:**
- Quantum Hilbert space might reveal structure
- Nov 2025: Haiqu showed quantum kernels find anomalies classical missed

**Implementation:**
```python
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel

feature_map = ZZFeatureMap(feature_dimension=D, reps=2)
qkernel = FidelityQuantumKernel(feature_map=feature_map)

# Use with sklearn
from sklearn.svm import SVC
qsvm = SVC(kernel=qkernel.evaluate)
qsvm.fit(X, y)
```

**Your use:** Classify patches → primitive types

---

### 3. **Quantum Neural Networks** (Function Learning)

**Capability:**
- Parameterized quantum circuit as function approximator
- Learn: inputs → outputs
- Trained with classical optimizer (hybrid)

**Potential advantage:**
- PQC might represent complex functions efficiently
- Quantum circuit structure could match problem structure

**Implementation:**
```python
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms import NeuralNetworkRegressor

feature_map = ZZFeatureMap(input_dim, reps=2)
ansatz = RealAmplitudes(input_dim, reps=3)

qnn = EstimatorQNN(
    circuit=feature_map.compose(ansatz),
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters
)

regressor = NeuralNetworkRegressor(qnn, optimizer='L_BFGS_B')
regressor.fit(X_train, y_train)
```

**Your use:** Learn f_edge, f_region, etc.

---

### 4. **QAOA** (Combinatorial Optimization)

**Capability:**
- Solve QUBO / Ising problems
- Approximate solutions to NP-hard problems
- Hybrid quantum-classical algorithm

**Implementation:**
```python
from qiskit_optimization import QuadraticProgram
from qiskit_algorithms import QAOA

qp = QuadraticProgram()
qp.binary_var_list(n)
qp.minimize(linear=..., quadratic=...)

qaoa = QAOA(optimizer='COBYLA', reps=3)
result = qaoa.compute_minimum_eigenvalue(qp.to_ising()[0])
```

**Your use:** Discrete Gaussian placement, budget allocation

---

## Learning Path (Practical Steps)

### Week 1: Setup & Fundamentals

**Install:**
```bash
python3 -m venv quantum_env
source quantum_env/bin/activate
pip install qiskit qiskit-ibm-runtime qiskit-machine-learning qiskit-algorithms
pip install numpy scipy matplotlib scikit-learn pandas
```

**IBM Quantum account:**
- Sign up: https://quantum.cloud.ibm.com
- Get API token
- Save: `QiskitRuntimeService.save_account(token='...')`

**Learn:**
- IBM Quantum Learning: "Quantum Computing in Practice"
- Run: `01_quantum_hello_world.py`

**Deliverable:** Verified quantum access works

---

### Week 2: Quantum Kernels (Primitive Classification)

**Learn:**
- Qiskit ML Tutorial: "Quantum Kernels"
- Understanding quantum feature spaces

**Experiment:**
- Run: `02_quantum_kernel_primitive_classification.py`
- Test on synthetic patches (edge vs region)
- Use simulator first (free, unlimited)
- If promising, validate on real quantum (2-5 min of free tier)

**Deliverable:**
- Quantum vs classical kernel accuracy comparison
- Decision: Does quantum find better decision boundaries?

---

### Week 3: Quantum Regression (Formula Learning)

**Learn:**
- Qiskit ML Tutorial: "Neural Network Classifiers and Regressors"
- VQA concepts

**Experiment:**
- Run: `03_quantum_regressor_formula_learning.py`
- Train QNN to learn f_edge
- Compare vs classical NN
- Probe quantum circuit to understand learned function

**Deliverable:**
- Quantum vs classical R² comparison
- If quantum better: extract formula via probing

---

### Week 4: Real Data Experiments

**If quantum shows advantage in Weeks 2-3:**

**Experiment 1: Real Image Patch Classification**
- Extract patches from Kodak images
- Compute real features (gradient, curvature, entropy)
- Quantum kernel classification
- Validate: quantum > classical on real data?

**Experiment 2: Phase 0.5 Data Regression**
- Use actual Phase 0.5 experimental results
- Quantum regressor on real parameter → PSNR data
- Extract formula if quantum learns better

**Deliverable:** Real-world validation of quantum advantage

---

### Month 2-3: Advanced (If Validated)

- Quantum clustering for primitive discovery (10K patches)
- QAOA for Gaussian placement optimization
- Publish findings

---

## Files in This Directory

### Documentation
- `README.md` - This file
- `QUANTUM_LEARNING_PATH.md` - Detailed capabilities and research

### Code Examples
- `01_quantum_hello_world.py` - Verify IBM Quantum access
- `02_quantum_kernel_primitive_classification.py` - Test quantum kernel SVM
- `03_quantum_regressor_formula_learning.py` - Test quantum function learning

### Future
- `04_quantum_clustering_primitives.py` - Discover primitives via quantum clustering
- `05_qaoa_placement.py` - Gaussian placement with QAOA
- `06_hybrid_pipeline.py` - Full quantum discovery → classical deployment

---

## Success Criteria

### Quantum Discovery Succeeds If:

**Primitive Discovery:**
- ✓ Quantum clustering finds groupings with better separability (R² > 0.9)
- ✓ Primitives are interpretable (can explain what they represent)
- ✓ Outperforms classical K-means/hierarchical clustering

**Formula Discovery:**
- ✓ Quantum regressor R² > classical regression R² by >0.05
- ✓ Can extract classical formula from quantum circuit
- ✓ Formula generalizes to held-out test cases

**Placement Discovery:**
- ✓ QAOA placement achieves >1 dB PSNR vs classical heuristics
- ✓ Consistently better across multiple test cases

---

## Current Status

**Environment:** Setting up (installing Qiskit, IBM account)

**Learning:** Researching capabilities, studying tutorials

**Testing:** Running initial experiments on simulator

**Next:** Validate on real quantum hardware (when ready)

---

## Resources

**Official Documentation:**
- IBM Quantum: https://quantum.cloud.ibm.com
- Qiskit Docs: https://docs.quantum.ibm.com
- Qiskit ML: https://qiskit-community.github.io/qiskit-machine-learning

**Tutorials:**
- IBM Quantum Learning: https://quantum.cloud.ibm.com/learning
- Qiskit Textbook: https://qiskit.org/textbook

**Research Papers:**
- "Quantum Annealing for Computer Vision" (2024)
- "Haiqu Quantum Pattern Recognition" (Nov 2025)
- "Variational Quantum Optimization" (2024)

---

**Start with 01_quantum_hello_world.py to verify everything works.**
