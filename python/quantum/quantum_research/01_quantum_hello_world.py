"""
Quantum Hello World: Verify IBM Quantum Access
Test basic quantum computing capabilities before tackling Gaussian problems
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import Aer


def setup_ibm_quantum_account(token=None):
    """
    Setup IBM Quantum account (one-time)

    Get token from: https://quantum.cloud.ibm.com
    Account -> Copy API token
    """
    if token:
        QiskitRuntimeService.save_account(
            channel="ibm_quantum",
            token=token,
            overwrite=True
        )
        print("✓ IBM Quantum account saved")
    else:
        print("Load existing account...")

    try:
        service = QiskitRuntimeService()
        print(f"✓ Connected to IBM Quantum")
        return service
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nTo setup:")
        print("1. Go to https://quantum.cloud.ibm.com")
        print("2. Sign up/login")
        print("3. Get API token from Account page")
        print("4. Run: setup_ibm_quantum_account(token='YOUR_TOKEN')")
        return None


def test_local_simulator():
    """Test quantum circuit on local simulator (free, unlimited)"""
    print("\n=== Test 1: Local Simulator ===")

    # Create simple circuit (Bell state)
    qc = QuantumCircuit(2, 2)
    qc.h(0)  # Hadamard on qubit 0
    qc.cx(0, 1)  # CNOT: qubit 0 controls qubit 1
    qc.measure([0, 1], [0, 1])

    print(f"Circuit:\n{qc}")

    # Run on local simulator
    simulator = Aer.get_backend('qasm_simulator')
    job = simulator.run(qc, shots=1000)
    result = job.result()
    counts = result.get_counts()

    print(f"\nResults: {counts}")
    print("Expected: ~50% |00⟩, ~50% |11⟩ (entangled state)")

    # Check if correct
    if '00' in counts and '11' in counts:
        if abs(counts.get('00', 0) - 500) < 100:  # Within statistical variance
            print("✓ Bell state created successfully")
            return True
    return False


def test_real_quantum_computer(service):
    """
    Test on real IBM quantum computer (uses free tier minutes)

    WARNING: This uses ~0.1-0.5 minutes of your 10 minute/month quota
    """
    if service is None:
        print("✗ No IBM Quantum service available")
        return False

    print("\n=== Test 2: Real Quantum Computer ===")

    # Simple circuit
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    # Get least busy backend
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=2)
    print(f"Using backend: {backend.name}")
    print(f"Qubits: {backend.num_qubits}")

    # Run with Sampler primitive
    from qiskit_ibm_runtime import Sampler

    sampler = Sampler(backend)
    job = sampler.run([qc], shots=1000)

    print(f"Job ID: {job.job_id()}")
    print("Waiting for quantum computer...")

    result = job.result()
    counts = result[0].data.c.get_counts()

    print(f"\nQuantum computer results: {counts}")
    print("(Note: May have noise/errors compared to simulator)")

    if '00' in counts and '11' in counts:
        print("✓ Real quantum computer execution successful")
        return True

    return False


def test_quantum_feature_encoding():
    """Test encoding classical data into quantum state (amplitude encoding)"""
    print("\n=== Test 3: Amplitude Encoding ===")

    # Classical feature vector (e.g., from image patch)
    features = np.array([0.5, 0.3, 0.8, 0.2])  # 4 features
    features_normalized = features / np.linalg.norm(features)  # Normalize

    # Encode in quantum state
    from qiskit.circuit.library import RawFeatureVector

    feature_map = RawFeatureVector(feature_dimension=4)
    qc = feature_map.assign_parameters(features_normalized)

    print(f"Encoding {len(features)} features in {qc.num_qubits} qubits")
    print(f"Circuit depth: {qc.depth()}")

    # Simulate to verify encoding
    from qiskit.quantum_info import Statevector
    statevector = Statevector.from_instruction(qc)
    amplitudes = statevector.data

    print(f"\nOriginal features: {features_normalized}")
    print(f"Quantum amplitudes: {np.abs(amplitudes)}")
    print(f"Match: {np.allclose(features_normalized, np.abs(amplitudes))}")

    if np.allclose(features_normalized, np.abs(amplitudes)):
        print("✓ Amplitude encoding successful")
        return True

    return False


if __name__ == "__main__":
    print("Quantum Computing Hello World - IBM Quantum + Qiskit")
    print("=" * 60)

    # Test 1: Local simulator (free, always works)
    test_local_simulator()

    # Test 2: Amplitude encoding (local, test feature encoding)
    test_quantum_feature_encoding()

    # Test 3: Real quantum computer (uses free tier quota)
    print("\n" + "=" * 60)
    print("OPTIONAL: Test real quantum computer (uses free tier)")
    print("Skip if saving quantum minutes for experiments")
    response = input("Run on real quantum computer? (y/N): ")

    if response.lower() == 'y':
        # Setup account (will prompt for token if not saved)
        service = setup_ibm_quantum_account()

        if service:
            test_real_quantum_computer(service)
        else:
            print("\n✗ IBM Quantum not configured")
            print("Run setup_ibm_quantum_account(token='YOUR_TOKEN') first")
    else:
        print("\nSkipping real quantum test (good - save your minutes!)")

    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Study: IBM Quantum Learning courses")
    print("2. Practice: Qiskit ML tutorials (local simulator)")
    print("3. Run: 02_quantum_kernel_test.py (first real experiment)")
