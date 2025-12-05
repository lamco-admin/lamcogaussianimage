# Quantum Memory Optimization: Path to 1,500+ Samples

**Current**: 1,000 samples (safe on 76GB RAM)
**Future Goal**: Scale to 1,500-2,000 samples

---

## Strategy 1: Block-Wise Kernel Computation ⭐ BEST APPROACH

**Compute kernel in 200×200 blocks, save to disk, reassemble**

**Memory**: ~15-20 GB peak (vs 128 GB for 1,500 in-memory)
**Runtime**: 40-60 min (vs 22-37 min)
**Enables**: 1,500, 2,000, even 5,000 samples

Implementation effort: 1-2 hours

## Strategy 2: IBM Real Quantum Hardware

**Use IBM free tier (10 min/month)**

**Memory**: <5 GB (server-side computation)
**Cost**: $0 for first 10 min, then ~$1.60/hour
**Enables**: Arbitrary samples, validates on real quantum

Setup effort: 30 minutes (create account, configure)

## Strategy 3: Reduce Qubits to 6

**Current**: 8 qubits (pad 6D → 8D)
**Alternative**: 6 qubits (no padding)

**Memory savings**: 4× per statevector (256 → 64 amplitudes)
**Quality impact**: Minimal (6 qubits sufficient for 6 features)

Implementation: 5 minutes (change one line)

---

**Recommendation**: Run 1,000 now, implement Strategy 1 (block-wise) next session if results warrant scaling up.
