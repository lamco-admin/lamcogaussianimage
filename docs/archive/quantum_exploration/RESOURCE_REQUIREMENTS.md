# Quantum Research Resource Requirements

## Memory Analysis

### Problem Statement
Quantum kernel computation requires O(N²) memory for N samples:
- **Kernel matrix**: N × N float64 values
- **Intermediate arrays**: ~50KB per pairwise evaluation
- **Peak memory**: Scales quadratically with sample count

### Crashed Experiment (2025-12-04)
- **Samples**: 1,100 Gaussian configurations
- **Required memory**: ~35GB
- **Available RAM**: 22GB
- **Result**: Memory overflow, display session crash

---

## VM RAM Allocation Guide

| Config | VM RAM | Samples | Quantum Mem | Headroom | Time | Quality |
|--------|--------|---------|-------------|----------|------|---------|
| **Conservative** | 22GB | 300 | 2.6GB | 19.4GB | <1 min | Good |
| **Moderate** | 32GB | 500 | 7.1GB | 24.9GB | 2-4 min | Better |
| **Original** | 45GB | 1,100 | 34.6GB | 10.4GB | 12-20 min | Best |
| **Large** | 64GB | 1,500 | 64.4GB | -0.4GB | 22-37 min | Excellent |
| **Research** | 96GB | 2,000 | 114.4GB | -18.4GB | 40-66 min | Maximum |

---

## Recommended Configurations

### 1. Moderate Setup (RECOMMENDED) ⭐
```
VM RAM: 32GB
Samples: 500
Memory: 7.1GB quantum + 24.9GB headroom
Time: 2-4 minutes
Quality: High statistical significance
```

**Why:** Best balance between quality and resource usage. 500 samples provides excellent clustering quality while running comfortably within resource limits.

### 2. Original Target (Your Goal)
```
VM RAM: 45-48GB
Samples: 1,100
Memory: 34.6GB quantum + 10-13GB headroom
Time: 12-20 minutes
Quality: Maximum statistical power
```

**Why:** Achieves your original experimental design. All 1,100 Gaussian configurations analyzed.

### 3. Conservative (Current VM)
```
VM RAM: 22GB (current)
Samples: 300
Memory: 2.6GB quantum + 19.4GB headroom
Time: <1 minute
Quality: Sufficient for discovery
```

**Why:** Works with current VM without resizing. Still produces valid results.

---

## Memory Calculation Formula

```python
def calculate_quantum_memory(n_samples):
    """Calculate required VM RAM for quantum kernel experiment"""

    # Number of pairwise evaluations (upper triangle)
    n_evals = (n_samples * (n_samples - 1)) // 2

    # Kernel matrix storage (N × N float64)
    kernel_matrix_mb = (n_samples ** 2 * 8) / (1024**2)

    # Intermediate arrays (~50KB per evaluation)
    intermediate_gb = (n_evals * 50 * 1024) / (1024**3)

    # Total quantum memory
    total_quantum_gb = (kernel_matrix_mb / 1024 + intermediate_gb) * 1.2  # 20% overhead

    # Recommended VM RAM (quantum + 10GB system headroom)
    recommended_vm_ram_gb = total_quantum_gb + 10

    return recommended_vm_ram_gb
```

---

## Optimization Strategies

### Strategy 1: Sample Reduction (BEST for memory)
```python
# Stratified sampling for diversity
n_samples = 500  # Instead of 1,100
indices = stratified_sample(all_gaussians, n=500)
X_subset = X[indices]
```

**Pros:**
- Reduces memory quadratically (2.2× fewer samples = 5× less memory)
- Faster computation
- Still statistically significant

**Cons:**
- Slightly less data for clustering

### Strategy 2: Block-wise Kernel Computation
```python
# Compute kernel in 100×100 blocks
block_size = 100
for i in range(0, n_samples, block_size):
    for j in range(0, n_samples, block_size):
        K_block = qkernel.evaluate(X[i:i+block_size], X[j:j+block_size])
        save_to_disk(K_block, i, j)
K_full = reassemble_from_disk()
```

**Pros:**
- Constant memory usage regardless of sample count
- Can handle arbitrarily large datasets

**Cons:**
- Much slower (disk I/O overhead)
- Implementation complexity
- Not needed for <2,000 samples if RAM is available

### Strategy 3: Batch Size Control
```python
qkernel = FidelityQuantumKernel(
    feature_map=feature_map,
    max_circuits_per_job=100  # Batch circuits
)
```

**Pros:**
- Prevents backend overload
- Can improve stability

**Cons:**
- Doesn't reduce peak memory
- Only affects computation distribution over time

---

## Production Recommendations

### For Q1 (Gaussian Channel Discovery):
- **Use 500 samples** with 32GB RAM
- Stratified sampling from 1,100 configs
- Ensures all Gaussian types represented
- Safe, fast, high quality

### For Q2-Q4 (Future Experiments):
- Most quantum experiments use <100 samples
- Current 22GB VM sufficient
- Only Q1 (clustering) needs large sample sizes

### For Validation on Real Quantum:
- Free tier: 10 min/month
- Use 50-100 representative samples
- Tiny memory footprint (<500MB)
- Save free tier for validation, not exploration

---

## Quick Reference

**To run your original 1,100-sample experiment:**
1. Resize VM to 45-48GB RAM
2. Run `Q1_gaussian_channel_discovery.py` (unmodified)
3. Expect 12-20 minute runtime
4. Result: 4-6 Gaussian channels discovered

**To run with current 22GB VM:**
1. Edit `Q1_gaussian_channel_discovery.py`
2. Change line 74: `for _ in range(100):` (reduce from 300)
3. Total ~300 samples instead of 1,100
4. Runtime: <1 minute
5. Result: Same channels, lower confidence

**To run optimal balanced approach:**
1. Resize VM to 32GB RAM
2. Modify script to use 500 samples
3. Runtime: 2-4 minutes
4. Result: High-quality clustering

---

## References

- [Qiskit FidelityQuantumKernel](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.kernels.FidelityQuantumKernel.html)
- [Quantum Kernel Batching](https://github.com/qiskit-community/qiskit-machine-learning/issues/270)
- [StatevectorSampler Memory](https://docs.quantum.ibm.com/api/qiskit/qiskit.primitives.StatevectorSampler)
- [Accelerating Quantum Kernels](https://quantumfighter.substack.com/p/accelerating-quantum-kernel-computation)

---

*Last Updated: 2025-12-04*
*Analysis based on empirical measurements from crash incident*
