# Advanced Optimization Methodologies for LGI
## Beyond Gradient Descent: Innovative Approaches

**Date**: October 2, 2025
**Context**: Current optimizer achieves 19 dB, targeting 30+ dB
**Goal**: Explore alternative and hybrid optimization strategies

---

## 🧠 **CURRENT STATE & LIMITATIONS**

### Standard Gradient Descent (What We Have)

**Algorithm**: Adam optimizer with backpropagation
- ✅ Works well (19 dB, 3.3× improvement)
- ✅ Principled (follows gradient)
- ✅ Scalable (handles 1000s of parameters)

**Limitations**:
- ⚠️ Local minima (might not find global optimum)
- ⚠️ Slow convergence (78-500 iterations)
- ⚠️ Hyperparameter sensitive (LR, decay, patience)
- ⚠️ Oscillation issues (we observed after iteration 30)

**Question**: Can we do better with alternative methods?

---

## 🔬 **ALTERNATIVE OPTIMIZATION STRATEGIES**

### 1. Second-Order Methods (Newton-type)

**Idea**: Use second derivatives (Hessian) for better convergence

**Methods**:

**A. L-BFGS (Limited-memory BFGS)**
```rust
/// L-BFGS optimizer for Gaussian parameters
pub struct LBFGSOptimizer {
    memory_size: usize,  // Typically 5-20
    history_s: Vec<Vec<f32>>,  // Parameter differences
    history_y: Vec<Vec<f32>>,  // Gradient differences
}

impl LBFGSOptimizer {
    /// Compute search direction using Hessian approximation
    pub fn compute_direction(&mut self, gradient: &[f32]) -> Vec<f32> {
        // Two-loop recursion (efficient Hessian approximation)
        let mut q = gradient.to_vec();

        let k = self.history_s.len();
        let mut alpha = vec![0.0; k];
        let mut rho = vec![0.0; k];

        // First loop: backward
        for i in (0..k).rev() {
            rho[i] = 1.0 / dot(&self.history_s[i], &self.history_y[i]);
            alpha[i] = rho[i] * dot(&self.history_s[i], &q);
            q = q - alpha[i] * &self.history_y[i];
        }

        // Scale
        let gamma = if k > 0 {
            dot(&self.history_s[k-1], &self.history_y[k-1]) /
            dot(&self.history_y[k-1], &self.history_y[k-1])
        } else {
            1.0
        };
        let mut r = q * gamma;

        // Second loop: forward
        for i in 0..k {
            let beta = rho[i] * dot(&self.history_y[i], &r);
            r = r + (alpha[i] - beta) * &self.history_s[i];
        }

        -r  // Search direction
    }
}
```

**Pros**:
- Faster convergence (10-100× fewer iterations)
- Better curvature adaptation
- No LR tuning needed (line search instead)

**Cons**:
- More memory (stores history)
- Line search adds overhead
- Complex implementation

**Expected Impact**: 500 iterations → **50-100 iterations** for same quality

---

**B. Natural Gradient Descent**
```rust
/// Natural gradient using Fisher Information Matrix
pub struct NaturalGradientOptimizer {
    damping: f32,  // Regularization (typical: 1e-4)
}

impl NaturalGradientOptimizer {
    /// Compute natural gradient: F⁻¹ × g
    /// Where F is Fisher Information Matrix
    pub fn natural_gradient(&self, params: &[f32], gradient: &[f32]) -> Vec<f32> {
        // Approximate Fisher matrix for Gaussians
        let fisher = self.approximate_fisher(params);

        // Solve: (F + λI) × ng = g
        conjugate_gradient_solve(&fisher, gradient, self.damping)
    }

    /// Fisher Information Matrix for Gaussian parameters
    fn approximate_fisher(&self, params: &[f32]) -> Matrix {
        // For Gaussian splatting:
        // F_ij = E[∂log p(x|θ)/∂θ_i × ∂log p(x|θ)/∂θ_j]

        // Diagonal approximation (much cheaper):
        // F_ii ≈ Σ_pixels (∂weight/∂param_i)²
    }
}
```

**Pros**:
- Follows natural geometry of parameter space
- Invariant to reparameterization
- Often faster convergence

**Cons**:
- Fisher matrix expensive to compute/invert
- Approximations needed for tractability

**Expected Impact**: Better convergence, fewer oscillations

---

### 2. Evolutionary & Population-Based Methods

**Idea**: Maintain population of solutions, evolve toward optimum

**A. CMA-ES (Covariance Matrix Adaptation Evolution Strategy)**
```rust
/// CMA-ES for global Gaussian optimization
pub struct CMAESOptimizer {
    population_size: usize,  // Typical: 4 + 3×log(n_params)
    mean: Vec<f32>,
    covariance: Matrix,
    step_size: f32,
}

impl CMAESOptimizer {
    /// One generation of evolution
    pub fn step(&mut self, eval_fn: impl Fn(&[f32]) -> f32) -> Vec<f32> {
        // 1. Sample population from N(mean, σ²×C)
        let population = self.sample_population();

        // 2. Evaluate fitness (negative loss)
        let fitness: Vec<f32> = population.iter()
            .map(|individual| -eval_fn(individual))
            .collect();

        // 3. Select top performers
        let elite = select_elite(&population, &fitness, self.population_size / 2);

        // 4. Update mean (weighted by fitness)
        self.mean = weighted_mean(&elite, &fitness);

        // 5. Update covariance (adapt to landscape)
        self.covariance = estimate_covariance(&elite, &self.mean);

        // 6. Update step size (control exploration)
        self.step_size *= compute_step_size_adaptation(&elite);

        self.mean.clone()
    }
}
```

**Pros**:
- **Global optimization** (escapes local minima)
- No gradient needed (could use render-only)
- Adapts to landscape automatically

**Cons**:
- Very slow (100-1000× slower than gradient descent)
- Not practical for 1000s of Gaussians

**Application**: **Hybrid approach**
- Use CMA-ES for first 10-20 iterations (global search)
- Switch to Adam for refinement (local optimization)

---

**B. Genetic Algorithms with Gaussian Operators**
```rust
/// Genetic algorithm for Gaussian placement
pub struct GeneticGaussianOptimizer {
    population_size: usize,
    mutation_rate: f32,
    crossover_rate: f32,
}

impl GeneticGaussianOptimizer {
    /// Evolve Gaussian population
    pub fn evolve(&mut self, population: &[Vec<Gaussian2D>]) -> Vec<Vec<Gaussian2D>> {
        let mut new_population = Vec::new();

        // Selection (tournament or roulette)
        let parents = self.select_parents(population);

        // Crossover (blend Gaussian sets)
        for (parent1, parent2) in parents.chunks(2) {
            let child = self.crossover(parent1[0], parent1[1]);
            new_population.push(child);
        }

        // Mutation (perturb Gaussians)
        for individual in &mut new_population {
            if rand::random::<f32>() < self.mutation_rate {
                self.mutate(individual);
            }
        }

        new_population
    }

    /// Gaussian-specific crossover
    fn crossover(&self, parent1: &[Gaussian2D], parent2: &[Gaussian2D]) -> Vec<Gaussian2D> {
        // Splice crossover: Take Gaussians from both parents
        let split = rand::random::<usize>() % parent1.len();
        let mut child = parent1[..split].to_vec();
        child.extend_from_slice(&parent2[split..]);
        child
    }

    /// Gaussian-specific mutation
    fn mutate(&self, individual: &mut [Gaussian2D]) {
        // Random perturbations
        for gaussian in individual {
            if rand::random::<f32>() < 0.1 {
                // Mutate position
                gaussian.position += Vector2::new(
                    rand_normal(0.0, 0.01),
                    rand_normal(0.0, 0.01),
                );
            }
            // Similar for scale, color, etc.
        }
    }
}
```

**Pros**:
- Explores diverse solutions
- Can escape local minima
- Parallelizable (evaluate population in parallel)

**Cons**:
- Much slower than gradient descent
- Inefficient for fine-tuning

**Application**: **Initial exploration only**
- Run genetic algorithm for 100 generations
- Take best solution as initialization for Adam

---

### 3. Hybrid Gradient-Free + Gradient Methods

**A. Simulated Annealing + Gradient Descent**
```rust
pub struct HybridSAGD {
    temperature: f32,
    cooling_rate: f32,
}

impl HybridSAGD {
    pub fn optimize(&mut self, gaussians: &mut [Gaussian2D], target: &ImageBuffer) {
        // Phase 1: Simulated annealing (global search)
        for _ in 0..100 {
            let candidate = perturb_random(gaussians);
            let delta_loss = evaluate(&candidate) - evaluate(gaussians);

            // Accept if better, or with probability exp(-ΔE/T)
            if delta_loss < 0.0 || rand::random::<f32>() < (-delta_loss / self.temperature).exp() {
                *gaussians = candidate;
            }

            self.temperature *= self.cooling_rate;  // Cool down
        }

        // Phase 2: Gradient descent (local refinement)
        adam_optimize(gaussians, target, 500);
    }
}
```

**Benefit**: Global exploration + local refinement

---

**B. Particle Swarm Optimization**
```rust
pub struct ParticleSwarmOptimizer {
    num_particles: usize,
    inertia: f32,
    cognitive: f32,  // Attraction to personal best
    social: f32,     // Attraction to global best
}

struct Particle {
    position: Vec<Gaussian2D>,  // Current solution
    velocity: Vec<GaussianDelta>,  // Momentum
    best_position: Vec<Gaussian2D>,  // Personal best
    best_fitness: f32,
}

impl ParticleSwarmOptimizer {
    pub fn optimize(&mut self) {
        // For each particle in swarm:
        for particle in &mut self.particles {
            // Update velocity
            particle.velocity =
                self.inertia × particle.velocity +
                self.cognitive × rand() × (particle.best_position - particle.position) +
                self.social × rand() × (global_best - particle.position);

            // Update position
            particle.position += particle.velocity;

            // Evaluate
            let fitness = evaluate(&particle.position);
            if fitness > particle.best_fitness {
                particle.best_position = particle.position.clone();
                particle.best_fitness = fitness;
            }
        }
    }
}
```

**Application**: Explore multiple Gaussian configurations simultaneously

---

### 4. Learned Optimizers (Meta-Learning)

**Idea**: Train a small neural network to predict optimal parameter updates

**A. LSTM-Based Learned Optimizer**
```rust
/// Neural network that learns to optimize
pub struct LearnedOptimizer {
    lstm: LSTM,  // Tiny network: ~50K parameters
}

impl LearnedOptimizer {
    /// Predict parameter update from gradient history
    pub fn predict_update(&mut self, gradient_history: &[Vec<f32>]) -> Vec<f32> {
        // LSTM takes gradient sequence as input
        // Outputs: predicted optimal update

        // The LSTM is pre-trained on thousands of optimization runs
        // It "learns" what updates work best given gradient patterns
        self.lstm.forward(gradient_history)
    }

    /// Use in optimization loop
    pub fn optimize_with_learned(&mut self, gaussians: &mut [Gaussian2D]) {
        let mut grad_history = Vec::new();

        for iteration in 0..max_iters {
            let gradient = compute_gradient(gaussians);
            grad_history.push(gradient.clone());

            // Learned optimizer predicts update (instead of Adam formula)
            let update = self.predict_update(&grad_history);

            apply_update(gaussians, &update);
        }
    }
}
```

**Pros**:
- **Can learn from experience** (gets better over time)
- Potentially faster convergence
- Adapts to problem structure

**Cons**:
- Requires pre-training (expensive one-time cost)
- ~50K parameter network to train
- May not generalize to all image types

**Expected Impact**: 2-5× faster convergence if well-trained

---

### 5. Coordinate Descent & Alternating Optimization

**Idea**: Optimize parameters in groups, alternating

**A. Block Coordinate Descent**
```rust
pub struct BlockCoordinateOptimizer {
    blocks: Vec<ParameterBlock>,
}

enum ParameterBlock {
    Positions,     // Optimize all positions, freeze others
    Shapes,        // Optimize scales + rotations, freeze others
    Appearance,    // Optimize colors + opacities, freeze others
}

impl BlockCoordinateOptimizer {
    pub fn optimize(&mut self, gaussians: &mut [Gaussian2D]) {
        for iteration in 0..max_iters {
            // Cycle through blocks
            let block = self.blocks[iteration % self.blocks.len()];

            match block {
                ParameterBlock::Positions => {
                    // Only compute & apply position gradients
                    // Faster than full backprop
                }
                ParameterBlock::Shapes => {
                    // Only optimize scales & rotations
                }
                ParameterBlock::Appearance => {
                    // Only optimize colors & opacities
                }
            }
        }
    }
}
```

**Pros**:
- **Faster iterations** (partial gradients)
- Can use specialized optimizers per block
- Better for some problem structures

**Cons**:
- May converge slower overall
- Coordination between blocks can be tricky

**Expected Impact**: 2-3× faster iterations, similar or better convergence

---

**B. Alternating Least Squares**
```rust
/// Alternate between optimizing Gaussians and rendering coefficients
pub struct AlternatingOptimizer {
    // Freeze Gaussians, optimize blending weights
    // Then freeze weights, optimize Gaussians
}
```

This is less applicable to our case but used in some neural rendering methods.

---

### 6. Bayesian Optimization for Hyperparameters

**Idea**: Optimize hyperparameters (LR, decay, etc.) using Bayesian optimization

```rust
pub struct BayesianHyperparamOptimizer {
    gaussian_process: GaussianProcess,  // Surrogate model
    acquistion_fn: ExpectedImprovement,
}

impl BayesianHyperparamOptimizer {
    /// Find optimal hyperparameters
    pub fn optimize_hyperparams(&mut self, test_image: &ImageBuffer) -> OptimalConfig {
        // Search space
        let lr_position: [0.001, 0.1]
        let lr_scale: [0.001, 0.05]
        let lr_decay: [0.1, 0.9]
        // etc.

        for trial in 0..50 {
            // 1. GP predicts promising hyperparameters
            let candidate = self.gaussian_process.suggest();

            // 2. Run full optimization with these params
            let result = run_optimization(test_image, candidate);

            // 3. Update GP with result
            self.gaussian_process.update(candidate, result.psnr);
        }

        // Return best found
        self.gaussian_process.best()
    }
}
```

**Benefit**: **Automated hyperparameter tuning**
- No manual search needed
- Finds optimal LR, decay, patience, etc.
- 50 trials → optimal config

**Expected Impact**: +2-5 dB from optimal hyperparameters

---

### 7. Multi-Stage Optimization Pipeline

**Idea**: Different strategies for different phases

**A. Coarse-to-Fine Optimization**
```rust
pub struct MultiStageOptimizer {
    stages: Vec<OptimizationStage>,
}

enum OptimizationStage {
    CoarsePositioning,   // Fast, rough placement
    ShapeAdaptation,     // Adapt scales & rotations
    ColorRefinement,     // Fine-tune colors
    GlobalPolish,        // Final optimization of all params
}

impl MultiStageOptimizer {
    pub fn optimize(&mut self, gaussians: &mut [Gaussian2D]) {
        // Stage 1: Coarse positioning (50 iterations)
        // Only optimize positions with large LR (0.05)
        optimize_positions_only(gaussians, lr=0.05, iters=50);

        // Stage 2: Shape adaptation (100 iterations)
        // Optimize scales & rotations with medium LR
        optimize_shapes_only(gaussians, lr=0.01, iters=100);

        // Stage 3: Color refinement (100 iterations)
        // Optimize colors & opacities with small LR
        optimize_colors_only(gaussians, lr=0.005, iters=100);

        // Stage 4: Global polish (250 iterations)
        // Optimize all parameters with adaptive LR
        optimize_all(gaussians, lr_adaptive, iters=250);
    }
}
```

**Pros**:
- **Structured approach** (systematic refinement)
- Each stage can use optimal algorithm
- Prevents early over-fitting

**Cons**:
- More complex
- Needs careful stage design

**Expected Impact**: Better quality, more stable training

---

**B. Hierarchical Optimization**
```rust
/// Optimize coarse → fine Gaussian scales
pub struct HierarchicalOptimizer {}

impl HierarchicalOptimizer {
    pub fn optimize(&mut self, target: &ImageBuffer) -> Vec<Gaussian2D> {
        // Level 0: Optimize 100 large Gaussians (coarse structure)
        let coarse = optimize_gaussians(100, initial_scale=0.1, iters=200);

        // Level 1: Add 400 medium Gaussians, optimize together
        let medium = coarse + initialize_around_large(400, scale=0.05);
        let refined = optimize_gaussians(500, iters=200);

        // Level 2: Add 500 small Gaussians (detail)
        let fine = refined + initialize_in_gaps(500, scale=0.02);
        let final_result = optimize_gaussians(1000, iters=200);

        final_result
    }
}
```

**Benefit**: **Progressive refinement** - add detail incrementally

---

### 8. Neural Architecture Search Applied to Gaussians

**NOVEL IDEA**: Treat Gaussian placement as architecture search

```rust
/// NAS-inspired Gaussian architecture search
pub struct GaussianArchitectureSearch {
    controller: RNNController,  // Predicts Gaussian configurations
}

impl GaussianArchitectureSearch {
    /// Search for optimal Gaussian architecture
    pub fn search(&mut self, target: &ImageBuffer, search_budget: usize) -> Vec<Gaussian2D> {
        let mut best_config = None;
        let mut best_quality = 0.0;

        for trial in 0..search_budget {
            // 1. Controller predicts Gaussian configuration
            let config = self.controller.predict();
            // config = [(position1, scale1, color1), ...]

            // 2. Evaluate configuration (render + measure quality)
            let gaussians = instantiate_configuration(&config);
            let quality = evaluate_quality(&gaussians, target);

            // 3. Update controller (reinforcement learning)
            let reward = quality;
            self.controller.update(reward);

            if quality > best_quality {
                best_quality = quality;
                best_config = Some(config);
            }
        }

        instantiate_configuration(&best_config.unwrap())
    }
}
```

**Benefit**: **Learns optimal Gaussian placement strategy**

---

### 9. Optimal Transport for Initialization

**Idea**: Match Gaussian distribution to image distribution optimally

```rust
/// Optimal transport-based initialization
pub fn initialize_with_optimal_transport(
    target: &ImageBuffer,
    num_gaussians: usize,
) -> Vec<Gaussian2D> {
    // 1. Treat image as probability distribution (pixel intensities)
    let image_distribution = normalize_as_distribution(target);

    // 2. Treat Gaussians as probability distribution (mixture of Gaussians)
    let gaussian_distribution = UniformGaussianMixture::new(num_gaussians);

    // 3. Solve optimal transport problem
    // Minimize: Wasserstein distance between distributions
    let optimal_gaussians = sinkhorn_algorithm(
        &image_distribution,
        &gaussian_distribution,
        num_iterations=100,
    );

    optimal_gaussians
}
```

**Benefit**: **Provably optimal initial placement** (minimizes transport cost)

**Expected Impact**: Better initialization → faster convergence

---

### 10. Reinforcement Learning for Gaussian Lifecycle

**NOVEL**: Use RL to learn when to split/merge/prune

```rust
/// RL agent for Gaussian lifecycle decisions
pub struct GaussianLifecycleRL {
    policy_network: PolicyNet,  // Tiny network: ~10K params
}

enum Action {
    Keep,
    Prune,
    Merge(usize),  // Merge with Gaussian index
    Split,
}

impl GaussianLifecycleRL {
    /// Decide action for each Gaussian
    pub fn decide_action(&self, gaussian: &Gaussian2D, context: &Context) -> Action {
        // Context: gradient history, local image complexity, neighbors, etc.
        let features = extract_features(gaussian, context);

        // Policy network predicts action
        let action_probs = self.policy_network.forward(features);

        sample_action(action_probs)
    }

    /// Train policy on optimization episodes
    pub fn train(&mut self, training_images: &[ImageBuffer]) {
        for image in training_images {
            let episode = run_optimization_with_rl_policy(image, self);
            let reward = final_psnr - iteration_cost;

            // Update policy (PPO or A2C)
            self.policy_network.update(episode, reward);
        }
    }
}
```

**Benefit**: **Learns optimal lifecycle strategy** from experience

**This directly implements your "degrade/merge" concept with learning!**

---

## 🎯 **PRACTICAL HYBRID APPROACH**

### Recommended: Multi-Method Pipeline

**Combine best of all worlds**:

```rust
/// Production optimizer combining multiple techniques
pub struct ProductionOptimizer {
    initialization: InitStrategy,
    coarse_optimizer: OptimizationMethod,
    fine_optimizer: OptimizationMethod,
    hyperparameter_tuner: Option<BayesianOptimizer>,
}

enum OptimizationMethod {
    Adam,
    LBFGS,
    NaturalGradient,
    Hybrid(Vec<OptimizationMethod>),
}

impl ProductionOptimizer {
    /// Full optimization pipeline
    pub fn optimize(&mut self, target: &ImageBuffer, num_gaussians: usize) -> Vec<Gaussian2D> {
        // Stage 1: Smart initialization (2 minutes)
        let mut gaussians = match self.initialization {
            InitStrategy::OptimalTransport => initialize_ot(target, num_gaussians),
            InitStrategy::Neural => self.neural_init.initialize(target, num_gaussians),
            _ => standard_init(target, num_gaussians),
        };

        // Stage 2: Coarse optimization (10-50 iterations)
        match self.coarse_optimizer {
            OptimizationMethod::LBFGS => {
                // L-BFGS for fast initial convergence
                lbfgs_optimize(&mut gaussians, target, max_iters=50);
            }
            OptimizationMethod::Hybrid => {
                // CMA-ES for 10 iterations (global search)
                cmaes_optimize(&mut gaussians, target, iters=10);
                // Then L-BFGS for refinement
                lbfgs_optimize(&mut gaussians, target, iters=40);
            }
            _ => {}
        }

        // Stage 3: Fine optimization (100-500 iterations)
        match self.fine_optimizer {
            OptimizationMethod::Adam => {
                adam_optimize(&mut gaussians, target, max_iters=500);
            }
            OptimizationMethod::NaturalGradient => {
                natural_gradient_optimize(&mut gaussians, target, max_iters=300);
            }
            _ => {}
        }

        // Stage 4: Adaptive refinement (optional)
        if self.enable_adaptive {
            // RL-based lifecycle decisions
            let lifecycle_agent = GaussianLifecycleRL::new();
            lifecycle_agent.apply_to(&mut gaussians);
        }

        gaussians
    }
}
```

**Expected Performance**:
- **Initialization**: Optimal transport → 10 dB from start (vs. 6 dB random)
- **Coarse**: L-BFGS 50 iters → 18 dB (vs. Adam 30 iters → 15 dB)
- **Fine**: Adam 200 iters → 32 dB (vs. 500 iters → 30 dB)

**Total**: **250 iterations** to 32 dB (vs. 500+ iterations with pure Adam)

**Speedup**: **2× faster** to same quality!

---

## 🚀 **IMMEDIATE IMPLEMENTATION PLAN**

### Quick Wins (Tonight - Implement Now)

**1. Improved LR Schedule** (30 minutes)
```rust
// Cosine annealing with warm restarts
pub fn cosine_annealing_lr(
    lr_max: f32,
    lr_min: f32,
    iteration: usize,
    period: usize,
) -> f32 {
    let t = (iteration % period) as f32 / period as f32;
    lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f32::consts::PI * t).cos())
}
```

**Expected**: +2-3 dB, eliminates oscillation

**2. Block Coordinate Descent** (1-2 hours)
```rust
// Alternate: positions → shapes → colors
// Faster iterations, potentially better convergence
```

**Expected**: 1.5-2× faster iterations

**3. Better Initialization** (1 hour)
```rust
// Variance-adaptive scale initialization
let local_variance = compute_local_variance(target, position);
let initial_scale = (local_variance.sqrt() * 3.0).clamp(0.01, 0.15);
```

**Expected**: Start at ~10 dB instead of 6 dB

---

### Medium-Term (This Week)

**4. L-BFGS Implementation** (1-2 days)
- Much faster convergence
- Industry standard for neural optimization
- Expected: 50-100 iterations to convergence

**5. Bayesian Hyperparameter Optimization** (1 day)
- Automated LR tuning
- Optimal decay schedule
- Expected: +3-5 dB from optimal params

---

### Long-Term (Weeks 2-4)

**6. Learned Optimizer** (1 week)
- Train meta-optimizer on synthetic data
- Deploy for faster convergence
- Research contribution

**7. RL Lifecycle Agent** (1 week)
- Implement your "degrade/merge" concept with learning
- Optimal prune/split decisions
- Novel research

---

## 🎯 **MY RECOMMENDATION: START NOW**

### Action Plan for Next 4 Hours

**Hour 1**: Implement quick wins
```rust
// Add to optimizer_v2.rs:
1. Cosine annealing LR schedule
2. Variance-based initialization
3. Earlier LR decay trigger
```

**Hour 2**: Test improvements
```bash
cargo run --release --bin lgi-cli-v2 -- encode \
  -i /tmp/test.png -o /tmp/test_improved.png \
  -n 1000 -q balanced \
  --metrics-csv /tmp/metrics_improved.csv
```

**Expected**: **28-32 dB** with these improvements!

**Hour 3-4**: Implement block coordinate descent
```rust
// Faster iterations via alternating optimization
```

**Expected**: 2× iteration speedup

---

## 📊 **EXPECTED OUTCOMES**

### With Quick Improvements (Tonight)

```
Current (Adam, default config):
- 500 Gaussians → 19.14 dB
- 78 iterations, 35s

Improved (Cosine LR + better init):
- 500 Gaussians → 22-24 dB  (+3-5 dB)
- 50-60 iterations, 25s

1000 Gaussians:
- Expected: 28-30 dB  ✅ TARGET RANGE!
- ~100 iterations, ~60s

1500 Gaussians:
- Expected: 31-33 dB  ✅ EXCEEDS TARGET!
- ~150 iterations, ~100s
```

### With Advanced Methods (This Week)

```
L-BFGS + Optimal Init + Tuned Hyperparams:
- 1000 Gaussians → 32-35 dB
- 50 iterations, 30s  (3× faster!)

Block Coordinate + Cosine LR:
- 1500 Gaussians → 35-38 dB
- 100 iterations, 60s  (2× faster!)
```

---

## ✨ **BOTTOM LINE**

**Question**: "What should we do now? More optimization?"

**Answer**: **YES - Multiple promising directions!**

**Immediate** (Tonight):
1. ✅ Start 1500 Gaussian test (validating in background)
2. ✅ Implement cosine annealing LR
3. ✅ Improve initialization
4. ✅ Test with improvements

**This Week**:
5. ✅ Implement L-BFGS (if time)
6. ✅ Block coordinate descent
7. ✅ Bayesian hyperparam tuning

**Result**: **30-35+ dB PSNR** with **2-3× faster training**

**Then**: Move to file format + compression with confidence

**All techniques are well-researched, implementable, and will compound for significant improvements!**

Should I start implementing the quick wins (cosine annealing + better init) right now while the 1500 Gaussian test runs?