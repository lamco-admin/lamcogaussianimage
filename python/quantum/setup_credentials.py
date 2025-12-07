"""Setup IBM Quantum credentials"""
from qiskit_ibm_runtime import QiskitRuntimeService
import json

# Load API key from file
with open('/home/greg/Desktop/apikey.json', 'r') as f:
    config = json.load(f)
    token = config['apikey']

# IBM Cloud CRN (Cloud Resource Name)
crn = "crn:v1:bluemix:public:quantum-computing:us-east:a/66377f5395bc4ca49acd720d170cdb9f:c9d4885c-bc96-4890-8222-66480cd738ba::"

print(f"Loaded API key: {config['name']}")
print(f"CRN: {crn[:50]}...")

# Save credentials for IBM Cloud
print("\nSaving IBM Cloud Quantum credentials...")
QiskitRuntimeService.save_account(
    channel="ibm_cloud",
    token=token,
    instance=crn,
    overwrite=True
)

print("✓ IBM Quantum credentials saved")

# Verify access
try:
    service = QiskitRuntimeService(channel="ibm_cloud", instance=crn)
    backends = service.backends()

    print(f"\n✓ Successfully connected to IBM Quantum")
    print(f"Available backends: {len(backends)}")

    # Show some backends
    for backend in backends[:5]:
        print(f"  - {backend.name}: {backend.num_qubits} qubits, status: {backend.status().status_msg}")

    print(f"\n✓ Total available backends: {len(backends)}")

except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Verify IBM Cloud Quantum service is active at: https://cloud.ibm.com/quantum")
    print("2. Check that the service instance exists and is running")
    print("3. Verify API key has correct permissions")
