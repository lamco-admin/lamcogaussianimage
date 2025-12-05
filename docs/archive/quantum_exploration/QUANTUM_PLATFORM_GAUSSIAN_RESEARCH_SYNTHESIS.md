# Quantum Computing Platforms & Gaussian Image Research Synthesis

**Research Date:** December 4, 2025
**Purpose:** Comprehensive analysis for quantum-accelerated Gaussian image primitive discovery

---

## Part 1: Quantum Computing Platform Comparison

### 1.1 IBM Quantum (Currently Using)

**Hardware:**
- **Condor**: 1,121 qubits (world's largest as of late 2024)
- **Nighthawk** (2025): 120 qubits with 218 tunable couplers, 30% more circuit complexity
- **Roadmap**: 100,000 qubits targeted, fault-tolerant by 2029

**Software:**
- **Qiskit**: Open-source Python SDK, most widely adopted
- **OpenQASM**: Low-level quantum assembly language
- Free tier available with dashboard for education/research

**Pros for Our Use Case:**
- Largest qubit counts for complex optimization
- Mature ecosystem with extensive documentation
- Free tier for prototyping

**Cons:**
- Hardware access can be queue-limited
- Superconducting qubits have shorter coherence times

**Access:** https://quantum.ibm.com/

---

### 1.2 Google Quantum AI

**Hardware:**
- **Sycamore**: Focus on quality over quantity
- Error correction leadership (demonstrated logical qubit improvements)
- 2024: Continued advances in surface code error correction

**Software:**
- **Cirq**: Open-source Python framework for NISQ circuits
- Free simulators (up to ~20 qubits)
- Quantum Virtual Machine for realistic simulation

**Pros:**
- Best-in-class error correction research
- Strong ML integration potential
- Free tools and simulators

**Cons:**
- Hardware access restricted to approved researchers
- Requires Google Cloud project

**Access:** https://quantumai.google/cirq

---

### 1.3 Amazon Braket

**Hardware Access (Multi-Vendor):**
- IonQ (trapped ion): Aria, Forte systems
- Rigetti (superconducting)
- IQM (superconducting)
- QuEra (neutral atom/Rydberg)
- D-Wave (quantum annealing) - removed from Braket as of 2024

**Pricing:**
- Per-shot + per-task model
- IonQ Aria: $0.30/task + $0.03/shot
- Free tier: 1 hour simulator time/month for new users

**Pros:**
- Access multiple hardware types through single interface
- AWS integration for hybrid workflows
- Pay-per-use model

**Access:** https://aws.amazon.com/braket/

---

### 1.4 Azure Quantum

**Hardware Access:**
- IonQ systems
- Quantinuum (H1, H2 trapped ion)
- Rigetti

**Pricing:**
- Per-shot/QCU model
- Monthly subscription: $25,000/month for Aria-Forte plan
- Academic credits available

**Access:** https://azure.microsoft.com/en-us/products/quantum

---

### 1.5 Xanadu (Photonic - Unique Approach)

**Hardware:**
- **Aurora** (2025): 12-qubit universal photonic computer
- World's first scalable, networked, modular quantum computer
- GKP (Gottesman-Kitaev-Preskill) error correction

**Software:**
- **PennyLane**: Most implemented QML framework (47% of quantum programmers)
- **Strawberry Fields**: Photonic quantum circuit library
- 143 university partners across 33 countries

**Roadmap:** 100,000 physical qubits, 1,000 logical qubits by 2029

**Pros for Our Use Case:**
- Continuous-variable quantum computing (natural for Gaussian operations!)
- Strong ML focus with PennyLane
- Room-temperature operation

**Key Insight:** Photonic quantum computing may have natural advantages for Gaussian-related computations due to continuous-variable encoding.

**Access:** https://www.xanadu.ai/ (Borealis available via AWS Braket)

---

### 1.6 D-Wave (Quantum Annealing)

**Hardware:**
- **Advantage 2**: ~4,400 qubits in Zephyr topology
- 2x coherence time vs previous generation
- 40% increase in energy scale

**Software:**
- **Ocean SDK**: Open-source Python tools
- **Quantum AI Toolkit** (August 2025): ML integration
- Demo for image generation using quantum processors

**2025 Achievement:** First quantum computational supremacy on real-world problem

**Pros for Our Use Case:**
- Excellent for optimization/combinatorial problems
- Hybrid solver handles MILP problems
- Image sparse representation research exists

**Cons:**
- Not universal quantum computing
- Limited to QUBO/Ising formulations

**Access:** https://cloud.dwavesys.com/ (free developer access)

---

### 1.7 Quantinuum (Trapped Ion)

**Hardware:**
- **H2-1**: 56 qubits, all-to-all connectivity
- **Helios** (2025): 98 barium ion qubits
- "Racetrack" architecture for direct qubit entanglement

**Research Access:**
- **QCUP** (Quantum Computing User Program): US researchers can apply for credits

**Recent Achievements:**
- Certified random bit generation
- Non-Abelian anyons creation
- High-fidelity magic states (0.01% failure rate)

**Pros:**
- Highest gate fidelities in industry
- All-to-all connectivity reduces circuit depth

**Access:** https://www.quantinuum.com/

---

### 1.8 Rigetti

**Hardware:**
- 100+ qubit chiplet-based system (end 2025)
- 99.5% two-qubit gate fidelity target
- Roadmap: 1,000+ qubits by 2027 (99.8% fidelity)

**Software:**
- **Forest SDK**: Open-source with Quil language
- **QVM**: Open-source quantum virtual machine
- Fine-grained qubit control

**Access:** Via AWS Braket, Azure Quantum, or direct QCS

---

## Part 2: SDK/Language Interoperability

| Platform | Primary SDK | Language | Interop |
|----------|------------|----------|---------|
| IBM | Qiskit | OpenQASM | High |
| Google | Cirq | Python | Medium |
| Xanadu | PennyLane/SF | Python | High (cross-platform) |
| D-Wave | Ocean | Python | QUBO-specific |
| Rigetti | Forest/Quil | Python | Via Braket |
| Amazon | Braket SDK | Python | Multi-vendor |

**Key Finding:** **PennyLane offers best cross-platform compatibility**, supporting IBM, Google, Rigetti, and others through plugins.

---

## Part 3: Gaussian Splatting State-of-the-Art

### 3.1 Core Techniques (120+ papers found)

**Foundational Work:**
- 3DGS (Kerbl et al., 2023): Original 3D Gaussian Splatting

**Compression Advances:**
- **NeuralGS**: 45x model size reduction using MLP encoding
- **Compact 3DGS**: 25x+ storage reduction via learned masks + vector quantization
- **Self-Organizing Gaussian Grids**: 8-26x reduction exploiting perceptual redundancy
- **OMG (Optimized Minimal Gaussians)**: 50% reduction, 600+ FPS rendering

**Video Extensions:**
- **VeGaS**: Folded-Gaussian distributions for video dynamics
- **GaussianVideo**: Neural ODEs for camera motion modeling
- **GauFRe**: Deformable 3D Gaussians for dynamic scenes

**2D/Image Specific:**
- **GaussianImage**: 2D Gaussian splatting achieving 1000-2000 FPS
- 8 parameters per Gaussian (position, covariance, color)
- Comparable to INRs with 3x lower GPU memory, 5x faster fitting

### 3.2 Key Compression Techniques

| Method | Compression | Key Innovation |
|--------|-------------|----------------|
| NeuralGS | 45x | MLP encoding of Gaussian attributes |
| Compact 3DGS | 25x | Learned masks + residual VQ |
| SOG Grids | 8-26x | 2D grid arrangement of parameters |
| OMG | 50% | Sub-vector quantization |

---

## Part 4: Quantum Image Representation

### 4.1 Established Encodings

**FRQI (Flexible Representation of Quantum Images):**
- First quantum image representation
- Encodes pixel intensity in qubit amplitude
- Qubit efficient but limited precision

**NEQR (Novel Enhanced Quantum Representation):**
- Stores grayscale in basis state sequence
- Quadratic speedup in preparation
- 1.5x better compression ratio vs FRQI
- More accurate image retrieval

**QPIXL Framework (2022+):**
- Unifies FRQI, NEQR, MCRQI, NCQI
- **Up to 90% gate reduction** without quality loss
- Uses only Ry and CNOT gates (NISQ-practical)

### 4.2 Recent Advances (2024-2025)

**DCT-EFRQI:**
- Combines classical DCT preprocessing with quantum encoding
- Uses wavelet transforms before quantum representation

**Quantum Image Compression (GLSVLSI 2025):**
- Comparative study: NEQR superior for precision, FRQI for qubit efficiency
- Hybrid processing approaches emerging

---

## Part 5: Quantum-Gaussian Intersection Research

### 5.1 Direct Connections Found

**Quantum Expectation-Maximization for GMM (Kerenidis & Luongo):**
- Quantum algorithm for fitting Gaussian Mixture Models
- Potential speedup for parameter estimation
- **DIRECTLY APPLICABLE to Gaussian primitive discovery**

**Quantum Multi-Model Fitting (Farina et al., 2023):**
- First quantum approach to multi-model fitting
- Uses adiabatic quantum computers
- Tested on 3D geometric data
- **Highly relevant for fitting multiple Gaussians**

**R-QuMF: Robust Quantum Multi-Model Fitting (2025):**
- Handles outliers in model fitting
- Maximum set coverage formulation for AQC
- Real-world 3D dataset results

**Quantum Clustering:**
- Near-optimal quantum coreset construction
- O(√nkd^{3/2}) query complexity
- Quadratic speedup for k-clustering

### 5.2 Related Quantum ML for Images

**Quantum Denoising Diffusion Models:**
- Outperforms classical in FID, SSIM, PSNR
- One-step image generation via unitary sampling
- Fewer parameters than classical equivalents

**Quantum CNNs:**
- 99.21% on MNIST (record for quantum-classical hybrid)
- 8x fewer parameters than classical
- Quanvolutional layers reduce image resolution efficiently

**FRQI on NISQ (2021):**
- Experimental validation on superconducting processors
- Circuit simplification reduces CNOT gates
- Practical limits identified for current hardware

---

## Part 6: Research Opportunities & Open Questions

### 6.1 High-Priority Research Directions

1. **Quantum Optimization for Gaussian Parameter Fitting**
   - VQE/QAOA for position, covariance, color optimization
   - Potential for exploring larger parameter spaces

2. **Continuous-Variable Quantum for Gaussian Primitives**
   - Xanadu's photonic approach naturally suits Gaussians
   - GKP encoding may align with Gaussian distributions

3. **Quantum-Enhanced Compression**
   - Combine QPIXL with Gaussian splatting
   - Hybrid classical-quantum codecs

4. **Quantum Coreset Selection**
   - Use quantum algorithms to select optimal Gaussian subsets
   - Reduce primitive count while maintaining quality

### 6.2 Unanswered Questions

- Can quantum computing find fundamentally new image primitives?
- What qubit count is needed for practical Gaussian optimization?
- How do photonic vs superconducting approaches compare for our use case?
- Can quantum entanglement reveal correlations in image structure?

### 6.3 Recommended Next Steps

1. **Prototype on IBM Quantum** (already in progress)
   - Use Qiskit for Gaussian parameter optimization
   - Compare VQE vs classical gradient descent

2. **Explore Xanadu PennyLane**
   - Test continuous-variable encoding of Gaussians
   - Leverage QML framework for image tasks

3. **Investigate D-Wave for Sparse Representation**
   - QUBO formulation for Gaussian selection
   - Use quantum AI toolkit for image experiments

4. **Apply for Quantinuum QCUP**
   - Access high-fidelity trapped ion hardware
   - Test on circuits requiring deep entanglement

---

## Part 7: Platform Recommendations

### For Your Specific Use Case (Gaussian Primitive Discovery):

| Approach | Recommended Platform | Reason |
|----------|---------------------|--------|
| Parameter optimization | IBM Quantum / Xanadu | VQE/VQC expertise |
| Sparse selection | D-Wave | Native optimization |
| High-fidelity experiments | Quantinuum | Best gate fidelity |
| Cross-platform research | PennyLane | Runs on multiple backends |
| Video/dynamic scenes | IBM + classical hybrid | Circuit depth needs |

### Free/Low-Cost Options:

1. **IBM Quantum** - Free tier with real hardware
2. **Google Cirq** - Free simulators
3. **D-Wave Leap** - Free developer access
4. **PennyLane** - Free, runs locally
5. **Amazon Braket** - 1 hour free simulator/month
6. **Quantinuum QCUP** - Research grant applications

---

## References & Sources

### Quantum Platforms
- [IBM Quantum Newsroom](https://newsroom.ibm.com/2025-11-12-ibm-delivers-new-quantum-processors)
- [Google Cirq Documentation](https://quantumai.google/cirq)
- [Xanadu Push to Public Markets](https://thequantuminsider.com/2025/11/04/a-deeper-look-at-xanadus-push-into-public-markets/)
- [D-Wave Quantum AI Toolkit](https://www.dwavequantum.com/company/newsroom/press-release/d-wave-introduces-new-developer-tools-to-advance-quantum-ai-exploration-and-innovation/)
- [Quantinuum H2 System](https://www.quantinuum.com/products-solutions/quantinuum-systems/system-model-h2)
- [Rigetti Q3 2025 Roadmap](https://www.globenewswire.com/news-release/2025/11/10/3185067/0/en/Rigetti-Computing-Reports-Third-Quarter-2025-Financial-Results-Provides-Technology-Roadmap-Updates-for-2026-and-2027.html)

### Gaussian Splatting
- [VeGaS: Video Gaussian Splatting](https://hf.co/papers/2411.11024)
- [GaussianImage: 2D Gaussian Splatting](https://hf.co/papers/2403.08551)
- [NeuralGS: Neural Field Compression](https://hf.co/papers/2503.23162)
- [Compact 3D Gaussian Splatting](https://hf.co/papers/2408.03822)

### Quantum Image Processing
- [QPIXL Framework](https://www.nature.com/articles/s41598-022-11024-y)
- [FRQI/NEQR Comparison](https://dl.acm.org/doi/10.1145/3716368.3735286)
- [Quantum Image Compression Review](https://www.mdpi.com/2073-431X/13/8/185)

### Quantum-Gaussian Intersection
- [Quantum EM for GMM](https://proceedings.mlr.press/v119/kerenidis20a.html)
- [Quantum Multi-Model Fitting](https://hf.co/papers/2303.15444)
- [Near-Optimal Quantum Coresets](https://hf.co/papers/2306.02826)
