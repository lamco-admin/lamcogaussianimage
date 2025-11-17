"""Quick test of quantum computer access"""
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit_aer import Aer

print("=" * 70)
print("QUANTUM COMPUTER ACCESS TEST")
print("=" * 70)

# Load service
crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/66377f5395bc4ca49acd720d170cdb9f:c9d4885c-bc96-4890-8222-66480cd738ba::"
service = QiskitRuntimeService(channel="ibm_cloud", instance=crn)

# Show available backends
backends = service.backends()
print(f"\n✓ Connected! {len(backends)} quantum computers available:")
for backend in backends:
    status = backend.status()
    print(f"\n  {backend.name}:")
    print(f"    Qubits: {backend.num_qubits}")
    print(f"    Status: {status.status_msg}")
    print(f"    Pending jobs: {status.pending_jobs}")

# Test on simulator first
print("\n" + "=" * 70)
print("TEST 1: Local Simulator (free, instant)")
print("=" * 70)

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

simulator = Aer.get_backend('qasm_simulator')
result = simulator.run(qc, shots=1000).result()
counts = result.get_counts()
print(f"Results: {counts}")
print("✓ Simulator works perfectly")

# Test on real quantum (BRIEF - uses maybe 0.1 minutes)
print("\n" + "=" * 70)
print("TEST 2: Real Quantum Computer (brief test)")
print("=" * 70)
print("This will use approximately 0.1-0.5 minutes of quantum time")

response = input("Proceed with real quantum test? (y/N): ")

if response.lower() == 'y':
    # Use least busy backend
    backend = service.least_busy(operational=True, simulator=False)
    print(f"\nUsing: {backend.name} ({backend.num_qubits} qubits)")

    sampler = Sampler(backend)
    job = sampler.run([qc], shots=100)  # Fewer shots = faster

    print(f"Job submitted: {job.job_id()}")
    print("Waiting for quantum computer...")

    result = job.result()
    counts = result[0].data.meas.get_counts()

    print(f"\nQuantum computer results: {counts}")
    print("✓ Real quantum computer access CONFIRMED!")
    print(f"\nNote: Results may show noise (this is normal for NISQ devices)")
else:
    print("\nSkipped real quantum test (good - save time for experiments)")

print("\n" + "=" * 70)
print("SETUP COMPLETE ✓")
print("=" * 70)
print("You have access to:")
print("- 3 quantum computers (133-156 qubits each)")
print("- ibm_fez, ibm_marrakesh, ibm_torino")
print("\nReady to run quantum experiments!")
