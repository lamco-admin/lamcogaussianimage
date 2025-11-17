"""
Quantum Neural Network Regressor for Formula Learning
Can quantum NN learn f_edge better than classical NN?

This tests: Quantum function learning for (blur, contrast, N) → (sigma_perp)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

USE_REAL_QUANTUM = False  # Set True to use IBM Quantum (uses ~5-10 minutes)


def load_phase05_data():
    """
    Load Phase 0.5 results (when available)

    For now: generate synthetic data that mimics expected relationships
    """
    np.random.seed(42)

    n_samples = 150

    # Generate synthetic data based on Phase 0 findings
    data = []

    for _ in range(n_samples):
        blur = np.random.uniform(0, 4)  # 0-4 pixels
        contrast = np.random.uniform(0.1, 0.8)
        N = np.random.choice([10, 20, 50, 100])

        # True relationship (unknown to learners)
        # Phase 0 suggested: sigma_perp = 1.0 (constant)
        # But maybe there's hidden relationship?
        sigma_perp_true = 1.0 + 0.1 * blur + 0.05 * np.log(N) + np.random.normal(0, 0.1)

        data.append({
            'blur': blur,
            'contrast': contrast,
            'N': N,
            'sigma_perp': sigma_perp_true
        })

    X = np.array([[d['blur'], d['contrast'], d['N']] for d in data])
    y = np.array([d['sigma_perp'] for d in data])

    return X, y


def test_classical_regression(X_train, y_train, X_test, y_test):
    """Baseline: Classical neural network"""
    print("\n=== Classical Neural Network Regression ===")

    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    # Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    # Train
    mlp = MLPRegressor(
        hidden_layer_sizes=(10, 10),
        max_iter=1000,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train_scaled)

    # Predict
    y_pred_scaled = mlp.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Evaluate
    from sklearn.metrics import r2_score, mean_squared_error

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"R² score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return {
        'model': mlp,
        'r2': r2,
        'rmse': rmse,
        'predictions': y_pred
    }


def test_quantum_regression(X_train, y_train, X_test, y_test, use_real_quantum=False):
    """Quantum Neural Network Regressor"""
    print("\n=== Quantum Neural Network Regression ===")

    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.algorithms import NeuralNetworkRegressor
    from sklearn.preprocessing import StandardScaler

    # Scale (quantum works better with normalized inputs)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    feature_dimension = X_train.shape[1]  # 3 features

    # Feature map (encodes inputs)
    feature_map = ZZFeatureMap(feature_dimension=feature_dimension, reps=2)

    # Ansatz (trainable parameters)
    ansatz = RealAmplitudes(num_qubits=feature_dimension, reps=3)

    # Create QNN
    if use_real_quantum:
        print("\n🚀 Using REAL quantum computer...")
        from qiskit_ibm_runtime import QiskitRuntimeService, Estimator

        service = QiskitRuntimeService()
        backend = service.least_busy(operational=True, simulator=False)
        print(f"Backend: {backend.name}")

        estimator = Estimator(backend)
    else:
        print("\n💻 Using local simulator...")
        from qiskit_aer.primitives import Estimator as AerEstimator
        estimator = AerEstimator()

    # EstimatorQNN
    qnn = EstimatorQNN(
        circuit=feature_map.compose(ansatz),
        estimator=estimator,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters
    )

    print(f"QNN input parameters: {len(qnn.input_params)}")
    print(f"QNN weight parameters: {len(qnn.weight_params)}")

    # Regressor
    qregressor = NeuralNetworkRegressor(
        neural_network=qnn,
        optimizer='L_BFGS_B',  # Classical optimizer for hybrid training
        loss='squared_error'
    )

    # Train (this evaluates quantum circuits many times)
    print("\nTraining quantum regressor...")
    print("(This may take several minutes on simulator)")

    qregressor.fit(X_train_scaled, y_train_scaled)

    # Predict
    y_pred_scaled = qregressor.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Evaluate
    from sklearn.metrics import r2_score, mean_squared_error

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"\nR² score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return {
        'model': qregressor,
        'r2': r2,
        'rmse': rmse,
        'predictions': y_pred,
        'qnn': qnn
    }


def probe_quantum_function(qregressor, scaler_X, scaler_y):
    """
    Probe the learned quantum function to understand what it learned

    This extracts classical understanding from quantum circuit
    """
    print("\n=== Probing Quantum-Learned Function ===")

    # Test on grid of inputs to see functional form
    blur_range = np.linspace(0, 4, 20)
    contrast_range = np.linspace(0.1, 0.8, 20)
    N_values = [10, 20, 50, 100]

    results = []

    for N in N_values:
        for blur in blur_range:
            for contrast in contrast_range:
                X_test = np.array([[blur, contrast, N]])
                X_scaled = scaler_X.transform(X_test)
                y_pred_scaled = qregressor.predict(X_scaled)
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))[0, 0]

                results.append({
                    'blur': blur,
                    'contrast': contrast,
                    'N': N,
                    'sigma_perp_predicted': y_pred
                })

    # Analyze patterns
    print(f"Tested {len(results)} combinations")

    # Check if sigma_perp varies with blur
    blur_correlation = []
    for N in N_values:
        subset = [r for r in results if r['N'] == N and abs(r['contrast'] - 0.5) < 0.1]
        blurs = [r['blur'] for r in subset]
        sigmas = [r['sigma_perp_predicted'] for r in subset]
        if len(blurs) > 5:
            corr = np.corrcoef(blurs, sigmas)[0, 1]
            blur_correlation.append(corr)
            print(f"N={N}: blur-sigma_perp correlation = {corr:.3f}")

    # Plot relationship
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Blur vs sigma_perp (N=50, contrast=0.5)
    subset = [r for r in results if r['N'] == 50 and abs(r['contrast'] - 0.5) < 0.1]
    blurs = [r['blur'] for r in subset]
    sigmas = [r['sigma_perp_predicted'] for r in subset]

    axes[0].plot(blurs, sigmas, 'o-')
    axes[0].set_xlabel('Edge Blur (pixels)')
    axes[0].set_ylabel('Predicted sigma_perp')
    axes[0].set_title('Quantum-Learned: sigma_perp vs blur')
    axes[0].grid(True)

    # Contrast vs sigma_perp
    subset = [r for r in results if r['N'] == 50 and abs(r['blur'] - 2.0) < 0.5]
    contrasts = [r['contrast'] for r in subset]
    sigmas = [r['sigma_perp_predicted'] for r in subset]

    axes[1].plot(contrasts, sigmas, 'o-', color='orange')
    axes[1].set_xlabel('Edge Contrast')
    axes[1].set_ylabel('Predicted sigma_perp')
    axes[1].set_title('Quantum-Learned: sigma_perp vs contrast')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('quantum_learned_function.png', dpi=150)
    print("\nSaved: quantum_learned_function.png")

    return results


def main():
    print("=" * 70)
    print("QUANTUM REGRESSOR FOR FORMULA LEARNING")
    print("Test: Can quantum NN learn f_edge better than classical?")
    print("=" * 70)

    # Load data
    print("\nLoading/generating data...")
    X, y = load_phase05_data()

    print(f"Dataset: {len(X)} samples")
    print(f"Features: blur, contrast, N (3 dimensions)")
    print(f"Target: sigma_perp (1 dimension)")
    print(f"Range: sigma_perp ∈ [{y.min():.2f}, {y.max():.2f}]")

    # Split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Test classical
    classical_result = test_classical_regression(X_train, y_train, X_test, y_test)

    # Test quantum
    quantum_result = test_quantum_regression(
        X_train, y_train, X_test, y_test,
        use_real_quantum=USE_REAL_QUANTUM
    )

    # Compare
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"Classical NN:  R² = {classical_result['r2']:.4f}, RMSE = {classical_result['rmse']:.4f}")
    print(f"Quantum QNN:   R² = {quantum_result['r2']:.4f}, RMSE = {quantum_result['rmse']:.4f}")

    r2_improvement = quantum_result['r2'] - classical_result['r2']
    print(f"\nQuantum R² improvement: {r2_improvement:+.4f}")

    if r2_improvement > 0.05:
        print("✓ Quantum learns function better!")
        print("  → QNN captured patterns classical NN missed")
        print("  → Worth probing quantum circuit to understand what it learned")

        # Probe quantum function
        from sklearn.preprocessing import StandardScaler
        scaler_X = StandardScaler().fit(X_train)
        scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))

        probe_results = probe_quantum_function(quantum_result['model'], scaler_X, scaler_y)

    elif r2_improvement < -0.05:
        print("✗ Classical NN is better")
        print("  → Quantum doesn't provide advantage for this function")
        print("  → Stick with classical polynomial/NN regression")
    else:
        print("≈ No significant difference")
        print("  → Quantum overhead not justified")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)

    if quantum_result['r2'] > classical_result['r2']:
        print("""
Quantum shows promise! Next experiments:

1. Probe quantum circuit on dense grid
   - Extract functional form quantum learned
   - Fit simple classical approximation
   - Deploy classically (no quantum at runtime)

2. Try on real Phase 0.5 data (not synthetic)
   - Test if advantage holds on real edge experiments

3. Extend to multi-output
   - Learn all parameters: (blur, contrast, N) → (σ_perp, σ_parallel, spacing, alpha)
   - Multivariate quantum regression
        """)
    else:
        print("""
Classical sufficient for formula learning.

Quantum might still help for:
- Primitive discovery (quantum clustering)
- Placement optimization (QAOA)
- But not formula learning (classical regression works)
        """)


if __name__ == "__main__":
    main()
