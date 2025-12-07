"""
Quantum Kernel for Primitive Classification
Test if quantum kernels can classify image patches into primitives better than classical

This is YOUR problem: Given image patch features, classify as edge/region/texture
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Will use simulators first, real quantum later
USE_REAL_QUANTUM = False  # Set True when ready to use free tier minutes


def generate_synthetic_patches(n_samples=200):
    """
    Generate synthetic patch features for testing

    In real version: extract from actual images
    For now: generate with known labels for testing
    """
    np.random.seed(42)

    patches = []

    # Class 1: Edge patches (high gradient, high coherence)
    for i in range(n_samples // 2):
        features = np.array([
            np.random.normal(0.8, 0.1),  # gradient_magnitude (high)
            np.random.normal(0.2, 0.05),  # curvature (low for straight edges)
            np.random.normal(0.9, 0.05),  # coherence (high, directional)
            np.random.normal(5.0, 1.0),  # entropy (medium)
            np.random.normal(0.1, 0.02),  # variance (low, structured)
        ])
        patches.append({'features': features, 'label': 0, 'name': 'edge'})

    # Class 2: Region patches (low gradient, low entropy)
    for i in range(n_samples // 2):
        features = np.array([
            np.random.normal(0.1, 0.05),  # gradient_magnitude (low)
            np.random.normal(0.0, 0.01),  # curvature (near zero)
            np.random.normal(0.3, 0.1),  # coherence (low, isotropic)
            np.random.normal(2.0, 0.5),  # entropy (low)
            np.random.normal(0.02, 0.01),  # variance (very low, smooth)
        ])
        patches.append({'features': features, 'label': 1, 'name': 'region'})

    # Shuffle
    np.random.shuffle(patches)

    X = np.array([p['features'] for p in patches])
    y = np.array([p['label'] for p in patches])

    return X, y, patches


def test_classical_kernel_svm(X_train, y_train, X_test, y_test):
    """Baseline: Classical SVM with RBF kernel"""
    print("\n=== Classical RBF Kernel SVM ===")

    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train
    svm = SVC(kernel='rbf', gamma='scale', C=1.0)
    svm.fit(X_train_scaled, y_train)

    # Evaluate
    train_acc = svm.score(X_train_scaled, y_train)
    test_acc = svm.score(X_test_scaled, y_test)

    print(f"Train accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")

    return {'train_acc': train_acc, 'test_acc': test_acc, 'model': svm}


def test_quantum_kernel_svm(X_train, y_train, X_test, y_test, use_real_quantum=False):
    """Quantum SVM with quantum kernel"""
    print("\n=== Quantum Kernel SVM ===")

    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Quantum feature map
    feature_dimension = X_train.shape[1]  # 5 features
    print(f"Feature dimension: {feature_dimension}")

    feature_map = ZZFeatureMap(feature_dimension=feature_dimension, reps=2)
    print(f"Quantum circuit qubits: {feature_map.num_qubits}")
    print(f"Quantum circuit depth: {feature_map.depth()}")

    # Quantum kernel
    if use_real_quantum:
        # Use real IBM quantum computer (USES FREE TIER MINUTES)
        from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

        print("\n🚀 Using REAL quantum computer (consuming free tier minutes)...")

        service = QiskitRuntimeService()
        backend = service.least_busy(operational=True, simulator=False)
        print(f"Backend: {backend.name}")

        sampler = Sampler(backend)
        qkernel = FidelityQuantumKernel(feature_map=feature_map, sampler=sampler)
    else:
        # Use local simulator (FREE, UNLIMITED)
        print("\n💻 Using local quantum simulator (free, unlimited)")

        from qiskit_aer.primitives import Sampler as AerSampler
        sampler = AerSampler()
        qkernel = FidelityQuantumKernel(feature_map=feature_map, sampler=sampler)

    # Compute quantum kernel matrix (this is the quantum part)
    print("\nComputing quantum kernel matrix...")
    print(f"Train: {len(X_train_scaled)} samples")
    print(f"Test: {len(X_test_scaled)} samples")

    # Train kernel matrix
    K_train = qkernel.evaluate(x_vec=X_train_scaled)
    print(f"Train kernel matrix shape: {K_train.shape}")

    # Test kernel matrix
    K_test = qkernel.evaluate(x_vec=X_test_scaled, y_vec=X_train_scaled)
    print(f"Test kernel matrix shape: {K_test.shape}")

    # Train SVM with precomputed quantum kernel
    svm = SVC(kernel='precomputed')
    svm.fit(K_train, y_train)

    # Evaluate
    train_acc = svm.score(K_train, y_train)
    test_pred = svm.predict(K_test)
    test_acc = np.mean(test_pred == y_test)

    print(f"\nTrain accuracy: {train_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'model': svm,
        'kernel_matrix': K_train
    }


def visualize_kernel_matrices(classical_result, quantum_result):
    """Compare classical vs quantum kernel matrices"""
    print("\n=== Kernel Matrix Visualization ===")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Would need to extract classical kernel matrix for comparison
    # For now, just visualize quantum kernel
    ax = axes[1]
    im = ax.imshow(quantum_result['kernel_matrix'], cmap='viridis')
    ax.set_title('Quantum Kernel Matrix')
    ax.set_xlabel('Sample j')
    ax.set_ylabel('Sample i')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('kernel_comparison.png', dpi=150)
    print("Saved: kernel_comparison.png")


def main():
    print("=" * 70)
    print("QUANTUM KERNEL FOR PRIMITIVE CLASSIFICATION")
    print("=" * 70)

    # Generate test data
    print("\nGenerating synthetic image patch features...")
    X, y, patches = generate_synthetic_patches(n_samples=100)

    print(f"Dataset: {X.shape[0]} patches, {X.shape[1]} features each")
    print(f"Classes: {np.unique(y)} (0=edge, 1=region)")
    print(f"Class balance: {np.sum(y==0)} edges, {np.sum(y==1)} regions")

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Test 1: Classical baseline
    classical_result = test_classical_kernel_svm(X_train, y_train, X_test, y_test)

    # Test 2: Quantum kernel (simulator first)
    quantum_result = test_quantum_kernel_svm(
        X_train, y_train, X_test, y_test,
        use_real_quantum=USE_REAL_QUANTUM
    )

    # Comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"Classical RBF Kernel:  Test Accuracy = {classical_result['test_acc']:.3f}")
    print(f"Quantum ZZ Kernel:     Test Accuracy = {quantum_result['test_acc']:.3f}")

    improvement = quantum_result['test_acc'] - classical_result['test_acc']
    print(f"\nQuantum improvement: {improvement:+.3f}")

    if improvement > 0.05:
        print("✓ Quantum kernel shows advantage!")
    elif improvement < -0.05:
        print("✗ Classical kernel is better")
    else:
        print("≈ No clear advantage either way")

    # Visualize
    # visualize_kernel_matrices(classical_result, quantum_result)

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
If quantum kernel is better:
  → Quantum Hilbert space captures primitive distinctions better
  → Worth exploring for real image patch classification
  → Could discover natural primitive groupings

If classical kernel is better or equal:
  → Problem is simple enough for classical
  → Quantum overhead not justified
  → Stick with classical methods

Next step:
  - If quantum shows advantage on synthetic data → test on real image patches
  - Extract features from Kodak images
  - Classify with quantum kernel
  - See if quantum finds better primitive boundaries
    """)

    print("\nTo run on REAL quantum computer:")
    print("1. Set USE_REAL_QUANTUM = True at top of file")
    print("2. Ensure IBM Quantum account setup (run 01_quantum_hello_world.py first)")
    print("3. This will use ~2-5 minutes of your 10 minute/month free tier")


if __name__ == "__main__":
    main()
