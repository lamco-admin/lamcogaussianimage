# Q2: Algorithm Discovery - Which Optimizer for Which Channel?

**Question**: Do different quantum channels need fundamentally different optimization algorithms (Adam vs L-BFGS vs SGD), not just different hyperparameters?

**Current State**: We've only tested Adam. Other optimizers failed, but maybe they work for specific channel types.

**Goal**: Build empirical mapping of Channel → Best Algorithm

---

## The Core Hypothesis

**Observation**: Quantum found 8 channels with vastly different loss values:
- Channel 4: loss = 0.018 (Adam succeeds)
- Channel 1: loss = 0.101 (Adam mediocre)
- Channel 5: loss = 0.160 (Adam fails)

**Hypothesis**: Adam works for some channels but not others. Different channels might need:
- **Channel 3-4-7** (small, high quality): Maybe L-BFGS (second-order, precise)
- **Channel 1** (large, medium quality): Adam (first-order, robust)
- **Channel 5** (failure mode): Maybe needs different algorithm entirely (SGD with momentum? Trust region?)

**Test**: For each channel, try multiple optimizers and measure which achieves best quality.

---

## Experimental Design

### Phase 1: Optimizer Implementation Audit

**What optimizers exist in the codebase?**

Current status:
- ✅ Adam: `adam_optimizer.rs` (implemented, tested)
- ⚠️ Gradient descent: `optimizer_v2.rs` (exists but maybe not production-ready)
- ❌ L-BFGS: Unknown if implemented
- ❌ Conjugate gradient: Unknown
- ❌ RMSprop: Unknown
- ❌ SGD with momentum: Unknown

**Action**: Audit codebase to find what exists

```bash
find packages/lgi-rs -name "*.rs" | xargs grep -l "lbfgs\|LBFGS\|bfgs" -i
find packages/lgi-rs -name "*.rs" | xargs grep -l "conjugate\|rmsprop\|momentum"
ls packages/lgi-rs/lgi-core/src/*optim*.rs
```

**Decision point**:
- If L-BFGS exists → use it
- If not → implement or use existing library (argmin-rs, optimization-rs)

### Phase 2: Channel Characterization

**For each quantum channel, extract representative Gaussians**:

```python
# File: quantum_research/characterize_channels_for_optimizers.py

import json
import pickle
import numpy as np

# Load quantum results
with open('gaussian_channels_kodak_quantum.json') as f:
    results = json.load(f)

with open('kodak_gaussians_quantum_ready.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
labels = np.array(results['labels_quantum'])

# For each channel, extract characteristics
for ch_id in range(results['n_clusters']):
    mask = labels == ch_id
    ch_gaussians = X[mask]

    print(f"Channel {ch_id}:")
    print(f"  Count: {np.sum(mask)}")
    print(f"  σ_x range: [{ch_gaussians[:,0].min():.4f}, {ch_gaussians[:,0].max():.4f}]")
    print(f"  σ_y range: [{ch_gaussians[:,1].min():.4f}, {ch_gaussians[:,1].max():.4f}]")
    print(f"  Loss range: [{ch_gaussians[:,3].min():.4f}, {ch_gaussians[:,3].max():.4f}]")

    # Analyze characteristics that suggest which optimizer
    mean_scale = np.sqrt(ch_gaussians[:,0].mean() * ch_gaussians[:,1].mean())
    anisotropy = np.abs(ch_gaussians[:,0] - ch_gaussians[:,1]).mean()

    # Heuristics from optimization literature:
    if mean_scale < 0.005 and anisotropy < 0.002:
        suggested_optimizer = "L-BFGS (small scales, smooth problem)"
    elif mean_scale > 0.02:
        suggested_optimizer = "Adam (large scales, robust needed)"
    elif ch_gaussians[:,3].mean() > 0.15:
        suggested_optimizer = "Try alternatives (Adam fails here)"
    else:
        suggested_optimizer = "Adam (already works)"

    print(f"  Suggested: {suggested_optimizer}")
    print()
```

### Phase 3: Systematic Optimizer Testing

**Design**: Test matrix of Channel × Algorithm

**Test harness**:

```rust
// File: packages/lgi-rs/lgi-encoder-v2/examples/test_optimizer_per_channel.rs

/// Test which optimizer works best for each quantum channel
///
/// For each channel:
/// - Extract representative Gaussian configurations
/// - Initialize encoder with those configurations
/// - Try multiple optimizers (Adam, L-BFGS, SGD, etc.)
/// - Measure final quality and convergence speed
///
/// Output: Empirical mapping of channel → best algorithm

use lgi_encoder_v2::*;

struct OptimizerConfig {
    name: String,
    implementation: OptimizerImpl,
}

enum OptimizerImpl {
    Adam(AdamOptimizer),
    LBFGS(LBFGSOptimizer),  // If exists
    SGD(SGDOptimizer),      // If exists
    // etc
}

fn test_channel_with_optimizer(
    channel_gaussians: &[GaussianConfig],
    optimizer_config: &OptimizerConfig,
    test_image: &ImageBuffer<f32>
) -> OptimizerResult {
    // Initialize Gaussians matching channel characteristics
    let mut gaussians = initialize_from_config(channel_gaussians, test_image);

    // Optimize with specific algorithm
    let start = Instant::now();
    let final_loss = run_optimizer(
        &mut gaussians,
        test_image,
        &optimizer_config.implementation
    );
    let time = start.elapsed();

    // Measure quality
    let rendered = render(&gaussians, test_image.width, test_image.height);
    let psnr = compute_psnr(test_image, &rendered);

    OptimizerResult {
        optimizer: optimizer_config.name.clone(),
        final_psnr: psnr,
        final_loss,
        time_seconds: time.as_secs_f32(),
        convergence_quality: final_loss / initial_loss,
    }
}

fn main() {
    // Load quantum channel definitions
    let channels = load_quantum_channels("gaussian_channels_kodak_quantum.json");

    // Define optimizers to test
    let optimizers = vec![
        OptimizerConfig { name: "Adam", ... },
        OptimizerConfig { name: "L-BFGS", ... },
        OptimizerConfig { name: "SGD+Momentum", ... },
        OptimizerConfig { name: "RMSprop", ... },
        OptimizerConfig { name: "ConjugateGradient", ... },
    ];

    // Test images
    let test_images = vec!["kodim03.png", "kodim08.png", "kodim15.png"];

    // Results matrix: Channel × Algorithm × Image
    let mut results = Vec::new();

    for (ch_id, channel) in channels.iter().enumerate() {
        println!("Testing Channel {}: {} Gaussians", ch_id, channel.n_gaussians);

        for optimizer in &optimizers {
            println!("  Optimizer: {}", optimizer.name);

            for image_path in &test_images {
                let image = load_png(image_path)?;

                let result = test_channel_with_optimizer(
                    &channel.representative_gaussians,
                    optimizer,
                    &image
                );

                println!("    {}: PSNR={:.2} dB, Loss={:.4}, Time={:.1}s",
                    image_path, result.final_psnr, result.final_loss, result.time_seconds);

                results.push((ch_id, optimizer.name.clone(), image_path.clone(), result));
            }
        }
    }

    // Analyze: Which optimizer wins for each channel?
    print_summary_table(&results);
}
```

**Output Example**:
```
CHANNEL → ALGORITHM MAPPING
================================================================
Channel | Best Algorithm | Avg PSNR | Improvement vs Adam
--------|----------------|----------|--------------------
0       | Adam           | 12.3 dB  | baseline
1       | Adam           | 15.4 dB  | baseline
2       | Adam           | 14.1 dB  | baseline
3       | L-BFGS         | 18.7 dB  | +5.2 dB ⭐
4       | L-BFGS         | 21.3 dB  | +8.7 dB ⭐⭐
5       | SGD+Momentum   | 14.2 dB  | +12.1 dB (was failing with Adam!)
6       | Adam           | 13.8 dB  | baseline
7       | L-BFGS         | 19.5 dB  | +6.8 dB ⭐
```

### Phase 4: Data Collection for Optimizer Discovery

**Comprehensive experiment** (many hours of automated testing):

```python
# File: quantum_research/collect_optimizer_performance_data.py

"""
Systematically test all optimizer algorithms on Gaussians from each quantum channel.

Generates massive dataset mapping:
  (channel, gaussian_config, optimizer, hyperparams) → (final_loss, psnr, iterations, time)

This data enables quantum discovery of optimal algorithm per channel.
"""

import subprocess
import json
import itertools
from pathlib import Path

# Quantum channels
channels = load_quantum_channels('gaussian_channels_kodak_quantum.json')

# Algorithms to test
algorithms = [
    {'name': 'Adam', 'lr': [0.001, 0.01, 0.1], 'beta1': [0.9], 'beta2': [0.999]},
    {'name': 'L-BFGS', 'max_iter': [10, 20, 50], 'history_size': [10, 20]},
    {'name': 'SGD', 'lr': [0.001, 0.01, 0.1], 'momentum': [0.0, 0.5, 0.9]},
    {'name': 'RMSprop', 'lr': [0.001, 0.01], 'decay': [0.9, 0.99]},
    {'name': 'ConjugateGradient', 'restart': [10, 50, 100]},
]

# Test images (subset for speed)
test_images = ['kodim03.png', 'kodim08.png', 'kodim15.png']

# Results collection
results = []
total_experiments = 0

# Count experiments
for channel in channels:
    for algorithm in algorithms:
        # Enumerate hyperparameter combinations
        param_names = [k for k in algorithm.keys() if k != 'name']
        param_values = [algorithm[k] for k in param_names]
        combos = list(itertools.product(*param_values))
        total_experiments += len(combos) * len(test_images)

print(f"Total experiments: {total_experiments}")
print(f"Estimated time: {total_experiments * 2 / 60:.1f} hours (2 min per experiment)")
print()

# Run experiments
for ch_id, channel in enumerate(channels):
    print(f"="*80)
    print(f"CHANNEL {ch_id}: {channel['n_gaussians']} Gaussians")
    print(f"  Characteristics: σ={channel['sigma_x_mean']:.4f}, loss={channel['loss_mean']:.4f}")
    print(f"="*80)

    for algorithm in algorithms:
        algo_name = algorithm['name']

        # Enumerate hyperparameter combinations
        param_names = [k for k in algorithm.keys() if k != 'name']
        param_values = [algorithm[k] for k in param_names]

        for params in itertools.product(*param_values):
            param_dict = dict(zip(param_names, params))

            for image_path in test_images:
                print(f"  Testing: {algo_name} with {param_dict} on {image_path}")

                # Call Rust encoder with specific algorithm + params
                result = run_encoding_experiment(
                    channel_id=ch_id,
                    channel_config=channel,
                    algorithm=algo_name,
                    hyperparams=param_dict,
                    test_image=image_path
                )

                results.append({
                    'channel_id': ch_id,
                    'algorithm': algo_name,
                    'hyperparams': param_dict,
                    'image': image_path,
                    'final_psnr': result['psnr'],
                    'final_loss': result['loss'],
                    'iterations': result['iterations'],
                    'time_seconds': result['time'],
                    'converged': result['converged'],
                })

                print(f"    → PSNR: {result['psnr']:.2f} dB, Loss: {result['loss']:.4f}")

# Save massive dataset
with open('optimizer_performance_matrix.json', 'w') as f:
    json.dump(results, f, indent=2)

# Analyze: Which algorithm wins for each channel?
analyze_best_algorithms(results)
```

**Runtime**: If 2 min per experiment × 1000+ experiments = **33+ hours**

Can be parallelized:
- 16 CPUs available
- Run 8-10 experiments in parallel
- Reduces to 3-4 hours

---

## What Optimizers to Implement/Test

### Tier 1: Must Have (Proven in Literature)

**1. L-BFGS** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)
- Second-order optimizer (uses Hessian approximation)
- Good for: Smooth, well-conditioned problems with accurate gradients
- **Literature**: 3D Gaussian splatting uses this for fine-tuning
- **Hypothesis**: Channels 3, 4, 7 (small, precise) might need this

**2. SGD with Momentum**
- First-order with momentum term
- Good for: Non-convex landscapes, escaping local minima
- **Hypothesis**: Channel 5 (failure mode) might work with momentum

**3. Conjugate Gradient**
- Finds conjugate directions in parameter space
- Good for: Quadratic problems, coupled parameters
- **Hypothesis**: Anisotropic Gaussians (if they exist) with σ_x ↔ θ coupling

### Tier 2: Worth Testing

**4. RMSprop**
- Adaptive learning rate per parameter
- Good for: Non-stationary objectives
- **Hypothesis**: Gaussians where loss landscape changes during optimization

**5. AdaGrad**
- Accumulates gradient history
- Good for: Sparse updates, rare features
- **Hypothesis**: Rare channel types (0.5-2% of Gaussians)

**6. Trust Region**
- Constrains update step size
- Good for: Ill-conditioned problems
- **Hypothesis**: Highly anisotropic or unstable Gaussians

### Tier 3: Experimental

**7. Genetic Algorithms**
- Population-based, derivative-free
- Good for: Multi-modal, when gradients misleading
- **Hypothesis**: Complex Gaussians where gradient descent fails

**8. Simulated Annealing**
- Probabilistic global search
- Good for: Escaping local minima
- **Hypothesis**: Channels where Adam gets stuck

---

## Immediate Action Plan (Next 4-6 Hours)

### Hour 1: Implementation Audit & Preparation

**Task 1.1**: Check what optimizers exist (15 min)
```bash
cd packages/lgi-rs
grep -r "pub struct.*Optimizer\|pub fn.*optimize" lgi-core/src/*.rs lgi-encoder-v2/src/*.rs | grep -v adam
```

**Task 1.2**: Implement missing critical optimizers (45 min)

If L-BFGS doesn't exist, implement basic version:
```rust
// File: packages/lgi-rs/lgi-encoder-v2/src/lbfgs_optimizer.rs

pub struct LBFGSOptimizer {
    pub max_iterations: usize,
    pub history_size: usize,  // Typically 10-20
    pub line_search_max_iter: usize,
    // ... other params
}

impl LBFGSOptimizer {
    pub fn optimize(
        &mut self,
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>
    ) -> f32 {
        // L-BFGS two-loop recursion implementation
        // OR use external crate like argmin-rs
    }
}
```

**Or** use existing library:
```toml
# Cargo.toml
[dependencies]
argmin = "0.10"
argmin-math = "0.4"
```

### Hour 2-3: Build Test Harness (implement + test)

**Task 2.1**: Create comparison binary (30 min)

```rust
// File: packages/lgi-rs/lgi-encoder-v2/examples/compare_optimizers_per_channel.rs

/// Compare multiple optimizers on Gaussians from each quantum channel
///
/// Tests: Adam, L-BFGS, SGD+Momentum on representative Gaussians
/// Measures: Which optimizer achieves best quality per channel

struct TestConfig {
    channel_id: usize,
    channel_params: ChannelParams,  // Representative σ_x, σ_y from quantum
    image: ImageBuffer<f32>,
}

enum OptimizerType {
    Adam { lr: f32, beta1: f32, beta2: f32 },
    LBFGS { history_size: usize },
    SGDMomentum { lr: f32, momentum: f32 },
}

fn test_single_configuration(
    config: &TestConfig,
    optimizer: OptimizerType
) -> f32 {
    // Initialize Gaussians matching channel characteristics
    let mut gaussians = initialize_from_channel(&config.channel_params, 50);

    // Optimize with specified algorithm
    let loss = match optimizer {
        OptimizerType::Adam { lr, beta1, beta2 } => {
            let mut opt = AdamOptimizer::default();
            opt.learning_rate = lr;
            opt.beta1 = beta1;
            opt.beta2 = beta2;
            opt.optimize(&mut gaussians, &config.image)
        },
        OptimizerType::LBFGS { history_size } => {
            let mut opt = LBFGSOptimizer::new(history_size);
            opt.optimize(&mut gaussians, &config.image)
        },
        OptimizerType::SGDMomentum { lr, momentum } => {
            let mut opt = SGDOptimizer::new(lr, momentum);
            opt.optimize(&mut gaussians, &config.image)
        },
    };

    // Compute final PSNR
    let rendered = RendererV2::render(&gaussians, config.image.width, config.image.height);
    compute_psnr(&config.image, &rendered)
}

fn main() {
    // Load quantum channels
    let channels = load_quantum_channel_json("../../quantum_research/gaussian_channels_kodak_quantum.json");

    // Test image
    let image = load_png("../../test-data/kodak-dataset/kodim03.png").unwrap();

    // Test matrix
    println!("{}", "=".repeat(80));
    println!("OPTIMIZER × CHANNEL PERFORMANCE MATRIX");
    println!("{}", "=".repeat(80));
    println!();

    // Table header
    println!("{:8} | {:12} | {:12} | {:12} | {:12}",
        "Channel", "Adam", "L-BFGS", "SGD+Mom", "Best");
    println!("{}", "-".repeat(80));

    for (ch_id, channel) in channels.iter().enumerate() {
        let config = TestConfig {
            channel_id: ch_id,
            channel_params: channel.clone(),
            image: image.clone(),
        };

        // Test each optimizer
        let psnr_adam = test_single_configuration(&config,
            OptimizerType::Adam { lr: 0.01, beta1: 0.9, beta2: 0.999 });

        let psnr_lbfgs = test_single_configuration(&config,
            OptimizerType::LBFGS { history_size: 10 });

        let psnr_sgd = test_single_configuration(&config,
            OptimizerType::SGDMomentum { lr: 0.01, momentum: 0.9 });

        // Determine best
        let best = psnr_adam.max(psnr_lbfgs).max(psnr_sgd);
        let best_name = if (psnr_lbfgs - best).abs() < 0.01 {
            "L-BFGS"
        } else if (psnr_sgd - best).abs() < 0.01 {
            "SGD"
        } else {
            "Adam"
        };

        println!("{:8} | {:10.2} dB | {:10.2} dB | {:10.2} dB | {}",
            ch_id, psnr_adam, psnr_lbfgs, psnr_sgd, best_name);
    }
}
```

**Task 2.2**: Test on single channel (30 min)

Validate test harness works correctly before running full matrix.

### Hour 4-6: Initial Optimizer Matrix Collection

**Run**: 3 optimizers × 8 channels × 3 images = 72 experiments

**Parallel execution**: 8 concurrent × 2 min each = ~18 minutes per batch = 2 hours total

**Output**: Initial mapping of which algorithms work for which channels

---

## Extended Experiment (Many Hours)

### Full Hyperparameter Grid

Once basic algorithm mapping is known, expand to hyperparameter tuning:

**For each channel's best algorithm**, test hyperparameter grid:

```python
# Channel 3 → L-BFGS is best
l_bfgs_grid = {
    'history_size': [5, 10, 20, 30],
    'max_iter': [10, 20, 50, 100],
    'line_search_max_iter': [10, 20],
    'tolerance': [1e-3, 1e-4, 1e-5],
}

# 4 × 4 × 2 × 3 = 96 combinations
# × 3 test images = 288 experiments for this channel
# × 2 min each = 9.6 hours for complete tuning

# But can be parallelized:
# 16 concurrent = 36 minutes for complete tuning
```

**Total for all 8 channels with full grid**: ~5-8 hours parallelized

### Even Larger: Per-Image Variation

Test if optimal algorithm/hyperparams vary by image content:

```
24 Kodak images × 8 channels × 3 algorithms × avg 5 hyperparams
= 24 × 8 × 3 × 5 = 2,880 experiments
× 2 min each = 96 hours serial
÷ 16 parallel = 6 hours parallel
```

**This would generate comprehensive empirical mapping** of:
- (Image characteristics, Channel, Algorithm, Hyperparams) → Performance

---

## Alternative: Use Existing Research to Guide Implementation

**Practical approach**: Don't blindly test everything. Use literature to guide:

### From 3D Gaussian Splatting Literature

**What they use**:
1. **Adam** for early training (robust, handles noise)
2. **L-BFGS** for fine-tuning (precise, converges to tight minima)
3. **Adaptive densification** (split/clone/prune operations)

**Our mapping**:
- Channel 1 (large, medium quality): Adam (matches literature)
- Channels 3-4-7 (small, high quality): Try L-BFGS (literature says good for fine-tuning)
- Channel 5 (failure): Try something different (SGD? Trust region?)

### From Classical Optimization Literature

**Problem characteristics → Algorithm**:

| Problem Type | Best Algorithm | Why | Our Channels |
|--------------|----------------|-----|--------------|
| Smooth, well-conditioned | L-BFGS | Second-order convergence | Channels 3,4,7? |
| Non-convex, many local minima | SGD + momentum | Escapes traps | Channel 5? |
| Quadratic or near-quadratic | Conjugate Gradient | Conjugate directions | Unknown |
| Sparse gradients | AdaGrad | Adapts per-parameter | Rare channels? |
| Non-stationary | RMSprop | Adaptive scaling | Channel 1? |

**Hypothesis-driven testing**: Match channel characteristics to problem types, test those algorithms first.

---

## Quantum Discovery of Algorithm Mapping (Q2 Proper)

**After collecting empirical data**, use quantum to discover patterns:

```python
# File: quantum_research/Q2_algorithm_discovery.py

"""
Q2: Discover optimal algorithm for each channel using quantum analysis.

Input: Empirical performance data from optimizer × channel matrix
Output: Quantum-discovered algorithm recommendations per channel

Method: Encode performance as quantum features, cluster in algorithm-performance space
"""

# Data structure:
experiments = [
    {
        'channel_id': 3,
        'gaussian_config': {'sigma_x': 0.0011, 'sigma_y': 0.0011, ...},
        'algorithm': 'L-BFGS',
        'hyperparams': {'history_size': 10, ...},
        'final_loss': 0.021,
        'convergence_speed': 0.15,  # Fast
        'iterations_needed': 45,
    },
    # ... thousands more
]

# Feature vector for quantum kernel:
features = [
    'channel_id',  # One-hot encoded
    'sigma_geometric_mean',
    'anisotropy_ratio',
    'algorithm',  # One-hot encoded
    'learning_rate',  # Normalized
    'final_loss',
    'convergence_speed',
    'iterations_needed',
]

# Quantum clustering in algorithm-performance space
# Discovers: Which (channel, algorithm) combinations naturally cluster as "successful"

# Output: Recommendations like
# "Channel 3 + L-BFGS(history=10) → high success cluster"
# "Channel 5 + Adam → failure cluster (avoid)"
```

**This is true Q2** - using quantum to discover algorithm patterns from empirical data.

---

## My Recommendations (Prioritized)

### Phase 1: Quick Wins (Next 2-3 Hours)

**1. Audit existing optimizers** (15 min)
- Check if L-BFGS exists in lgi-core
- Check for SGD, conjugate gradient, etc.

**2. Implement missing L-BFGS** (30-45 min if needed)
- Use argmin-rs library (don't implement from scratch)
- Create `lbfgs_optimizer.rs` wrapper

**3. Single-channel test** (30 min)
- Pick Channel 4 (highest quality with Adam)
- Test: Adam vs L-BFGS on same Gaussian configurations
- Measure: Does L-BFGS achieve even better quality?

**Decision point**: If L-BFGS wins on Channel 4, this validates the hypothesis. Proceed to full matrix.

### Phase 2: Algorithm × Channel Matrix (3-6 Hours Parallel)

**4. Build test harness** (1 hour)
- `compare_optimizers_per_channel.rs` as specified above
- Support parallel execution

**5. Run matrix** (3-5 hours parallelized)
- 3 algorithms × 8 channels × 3 images
- Collect performance data
- Identify which algorithm wins per channel

**6. Analyze results** (30 min)
- Build empirical mapping
- Document findings
- Decide: Is per-channel algorithm selection valuable?

### Phase 3: If Phase 2 Shows Promise (Many More Hours)

**7. Hyperparameter tuning per (channel, algorithm) pair** (6-10 hours)
- Full grid search for optimal hyperparameters
- Parallelized across 16 CPUs

**8. Full image validation** (variable)
- Encode images using per-channel optimal algorithms
- Measure overall PSNR improvement
- Compare to uniform Adam baseline

**9. Quantum pattern discovery** (30 min)
- Use empirical data as input to quantum clustering
- Discover success patterns in algorithm-performance space

---

## What This Would Prove

**If different channels need different algorithms**:
- Validates Q2 hypothesis
- Justifies quantum research direction
- Provides actionable encoder improvements
- Opens door to Q3, Q4

**If Adam works universally best**:
- Hyperparameter tuning might still help (LR per channel)
- Algorithm selection is not the key variable
- Focus shifts to other aspects (initialization, architecture)

**If no pattern emerges**:
- Maybe Gaussians don't cluster by optimization needs
- Compositional framework might not apply to optimization behavior
- Valuable negative result - informs future research direction

---

## Runtime Estimates

| Experiment | Serial Time | Parallel (16 CPU) | When |
|------------|-------------|-------------------|------|
| Optimizer audit | 15 min | 15 min | Hour 1 |
| Implement L-BFGS | 45 min | 45 min | Hour 1 |
| Single-channel test | 30 min | 30 min | Hour 1-2 |
| **Phase 1 Total** | **1.5 hours** | **1.5 hours** | **Hours 1-2** |
| Build test harness | 1 hour | 1 hour | Hour 3 |
| Algorithm matrix (3×8×3) | 12 hours | 1.5 hours | Hours 4-5 |
| Analysis | 30 min | 30 min | Hour 6 |
| **Phase 2 Total** | **13.5 hours** | **3 hours** | **Hours 3-6** |
| Hyperparameter grid | 96 hours | 6 hours | Hours 7-13 |
| Full validation | Variable | Variable | Hours 14+ |
| **Phase 3 Total** | **100+ hours** | **8-10 hours** | **Hours 7-16** |

**Realistically**:
- Phase 1: 1.5-2 hours (sequential, tonight)
- Phase 2: 3-4 hours (parallel, tonight)
- Phase 3: 8-10 hours (parallel, next session or overnight)

**Total: 12-16 hours of compute** that can run overnight/unattended

---

## What to Start Right Now

**Immediate Priority 1**: Optimizer audit
```bash
cd /home/greg/gaussian-image-projects/lgi-project/packages/lgi-rs
find . -name "*.rs" -path "*/lgi-core/src/*" | xargs grep -l "bfgs\|conjugate\|rmsprop" -i
ls lgi-core/src/ | grep -i optim
```

**Immediate Priority 2**: Check if L-BFGS dependency exists
```bash
grep -r "argmin\|optimization\|nlopt" */Cargo.toml
```

**Immediate Priority 3**: Design experiment properly based on what exists

If L-BFGS exists → use it immediately
If not → implement basic version or use library
Then run Phase 1 single-channel test tonight

**This validates your hypothesis**: Different algorithms for different channel types

---

**Should I proceed with optimizer audit and begin Phase 1 implementation?**
