"""
QUANTUM KERNEL TEST - Fixed for Qiskit 2025 API
Test if quantum can find better primitive classifications than classical
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Qiskit imports (current API)
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_aer.primitives import SamplerV2 as AerSampler
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as RuntimeSampler

USE_REAL_QUANTUM = False  # Set True to use IBM Quantum hardware

print("=" * 70)
print("QUANTUM KERNEL CLASSIFICATION - Primitive Discovery")
print("=" * 70)

# Generate synthetic data (edge vs region features)
print("\nGenerating synthetic patch features...")
np.random.seed(42)

n_samples = 200
X_edge = np.random.normal([0.8, 0.2, 0.9, 5.0, 0.1], [0.1, 0.05, 0.05, 1.0, 0.02], (n_samples//2, 5))
X_region = np.random.normal([0.1, 0.0, 0.3, 2.0, 0.02], [0.05, 0.01, 0.1, 0.5, 0.01], (n_samples//2, 5))

X = np.vstack([X_edge, X_region])
y = np.array([0]*100 + [1]*100)  # 0=edge, 1=region

# Shuffle
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

print(f"Dataset: {len(X)} patches, {X.shape[1]} features")
print(f"Classes: {np.unique(y)} (0=edge, 1=region)")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Classical baseline
print("\n" + "=" * 70)
print("CLASSICAL RBF KERNEL")
print("=" * 70)

svm_classical = SVC(kernel='rbf', gamma='scale')
svm_classical.fit(X_train_scaled, y_train)

train_acc_classical = svm_classical.score(X_train_scaled, y_train)
test_acc_classical = svm_classical.score(X_test_scaled, y_test)

print(f"Train accuracy: {train_acc_classical:.3f}")
print(f"Test accuracy: {test_acc_classical:.3f}")

# Quantum kernel
print("\n" + "=" * 70)
print("QUANTUM KERNEL")
print("=" * 70)

# Create feature map
feature_dim = X.shape[1]
feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2)

print(f"Quantum circuit: {feature_map.num_qubits} qubits, depth {feature_map.depth()}")

# Setup sampler (simulator or real)
if USE_REAL_QUANTUM:
    print("\n🚀 Using REAL QUANTUM COMPUTER")
    crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/66377f5395bc4ca49acd720d170cdb9f:c9d4885c-bc96-4890-8222-66480cd738ba::"
    service = QiskitRuntimeService(channel="ibm_cloud", instance=crn)
    backend = service.least_busy(operational=True, simulator=False)
    print(f"Backend: {backend.name} ({backend.num_qubits} qubits)")
    sampler = RuntimeSampler(backend)
else:
    print("\n💻 Using simulator (free, unlimited)")
    sampler = AerSampler()

# Create quantum kernel (FIXED API)
fidelity = ComputeUncompute(sampler=sampler)
qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

# Compute kernel matrices
print("\nComputing quantum kernel matrices...")
print("(This evaluates quantum circuits - may take 1-5 minutes)")

K_train = qkernel.evaluate(x_vec=X_train_scaled)
K_test = qkernel.evaluate(x_vec=X_test_scaled, y_vec=X_train_scaled)

print(f"Train kernel: {K_train.shape}")
print(f"Test kernel: {K_test.shape}")

# Train SVM with quantum kernel
svm_quantum = SVC(kernel='precomputed')
svm_quantum.fit(K_train, y_train)

train_acc_quantum = svm_quantum.score(K_train, y_train)
test_pred = svm_quantum.predict(K_test)
test_acc_quantum = accuracy_score(y_test, test_pred)

print(f"\nTrain accuracy: {train_acc_quantum:.3f}")
print(f"Test accuracy: {test_acc_quantum:.3f}")

# Comparison
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Classical RBF:   Test Accuracy = {test_acc_classical:.3f}")
print(f"Quantum Kernel:  Test Accuracy = {test_acc_quantum:.3f}")
print(f"Improvement:     {test_acc_quantum - test_acc_classical:+.3f}")

if test_acc_quantum > test_acc_classical + 0.05:
    print("\n✓ QUANTUM ADVANTAGE DETECTED!")
    print("  Quantum kernel finds better decision boundaries")
    print("  → Quantum Hilbert space reveals structure classical misses")
    print("\nNEXT: Run on real Kodak patches (extract_kodak_features.py)")
elif test_acc_quantum < test_acc_classical - 0.05:
    print("\n✗ Classical is better")
    print("  Quantum overhead not justified for this problem")
else:
    print("\n≈ No significant difference")
    print("  For this simple synthetic data, both work equally well")
    print("\nNEXT: Test on real Kodak data (more complex)")

print("\n" + "=" * 70)
if USE_REAL_QUANTUM:
    print("✓ Quantum experiment complete (used ~2-5 min of quantum time)")
else:
    print("To run on REAL quantum: Set USE_REAL_QUANTUM = True")
    print("This will use ~2-5 minutes of your quantum computing time")
