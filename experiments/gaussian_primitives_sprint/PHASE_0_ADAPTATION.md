# Phase 0: Adaptation of CCW Framework

**Status:** Framework available, needs adaptation for Phase 0
**Date:** 2025-11-17

---

## What CCW Built

CCW created comprehensive infrastructure (3,309 lines) for optimization-based experiments:
- Gaussian rendering, synthetic data generation
- Multiple initialization strategies
- Adam/SGD optimizers with finite difference gradients
- Comprehensive logging

**Validation:** 75% success rate, achieves 20+ dB PSNR improvements

---

## Phase 0 Needs vs CCW Framework

### Phase 0 Requirements:
- ✓ Synthetic edge generator
- ✓ Gaussian renderer (CPU, simple)
- ✓ Manual Gaussian placement
- ✓ Metrics (PSNR, MSE)
- ✓ Logging and visualization
- ✗ **NO optimization**
- ✗ **NO gradient computation**

### What CCW Provides:
- ✓ Gaussian2D dataclass
- ✓ GaussianRenderer with EWA-inspired splatting
- ✓ SyntheticDataGenerator (edges, straight/curved/blurred)
- ✓ Metrics computation
- ✓ ExperimentLogger
- ⚠️ Optimization framework (not needed for Phase 0, caused timeout)
- ⚠️ Complex initializers (Phase 0 only needs uniform placement)

---

## What to Use from CCW Framework

### Core Components to Reuse:

#### 1. Gaussian2D Class
**File:** `infrastructure/gaussian_primitives.py` (lines 14-48)

```python
from experiments.gaussian_primitives_sprint.infrastructure.gaussian_primitives import Gaussian2D
```

**Clean dataclass with:**
- x, y, sigma_parallel, sigma_perp, theta, color, alpha
- to_dict() / from_dict() for serialization

#### 2. GaussianRenderer
**File:** `infrastructure/gaussian_primitives.py` (lines 50-120)

```python
from experiments.gaussian_primitives_sprint.infrastructure.gaussian_primitives import GaussianRenderer

renderer = GaussianRenderer()
image = renderer.render(gaussians, width=100, height=100, channels=1)
```

**Alpha compositing, CPU-based, works for Phase 0.**

#### 3. SyntheticDataGenerator
**File:** `infrastructure/gaussian_primitives.py` (lines 150-350)

```python
from experiments.gaussian_primitives_sprint.infrastructure.gaussian_primitives import SyntheticDataGenerator

generator = SyntheticDataGenerator()

# Straight edge
edge_img, edge_desc = generator.generate_straight_edge(
    size=(100, 100),
    blur_sigma=2.0,
    contrast=0.5,
    orientation='vertical'
)

# Curved edge
curved_img, curved_desc = generator.generate_curved_edge(
    size=(100, 100),
    radius=100,
    blur_sigma=2.0,
    contrast=0.5
)
```

**Exactly what Phase 0 protocol specifies!**

#### 4. Metrics
**File:** `infrastructure/gaussian_primitives.py` (lines 120-150)

```python
from experiments.gaussian_primitives_sprint.infrastructure.gaussian_primitives import compute_metrics

metrics = compute_metrics(target_image, rendered_image)
# Returns: {'psnr': float, 'mse': float, 'mae': float, 'correlation': float}
```

#### 5. ExperimentLogger (adapted)
**File:** `infrastructure/experiment_logger.py`

Can use for saving results, but simplify (no optimization tracking).

---

## What NOT to Use

### ❌ Optimization Framework
**File:** `infrastructure/optimizers.py`

**Why skip:**
- Phase 0 is manual parameter exploration
- No gradient descent needed
- Finite difference gradients caused timeout (70s/experiment)

**Phase 0 approach:** Place Gaussians with explicit parameters, render, done.

### ❌ Complex Initializers
**File:** `infrastructure/initializers.py`

**Why skip:**
- Phase 0 only tests E1 (uniform spacing)
- E2-E4 are content-adaptive (testing that comes LATER)
- Phase 0 is simpler: place N Gaussians uniformly, vary sigma values

**Phase 0 approach:** Write simple placement function (20 lines).

### ❌ Set 4 Runners
**Files:** `run_experiment_set_4_CRITICAL.py`, etc.

**Why skip:**
- These are for layered vs monolithic comparison
- Phase 0 focuses on understanding single edge primitive
- Different research question

---

## Phase 0 Simplified Runner

### Use CCW Components, Skip Optimization

```python
"""
Phase 0: Edge Function Discovery (Manual Sweeps)
Uses CCW's rendering and data generation, no optimization
"""

import numpy as np
from experiments.gaussian_primitives_sprint.infrastructure.gaussian_primitives import (
    Gaussian2D, GaussianRenderer, SyntheticDataGenerator, compute_metrics
)
import matplotlib.pyplot as plt
import pandas as pd

def place_edge_gaussians_uniform(
    edge_descriptor,
    N=10,
    sigma_perp=1.0,
    sigma_parallel=5.0,
    spacing=2.5,
    alpha=0.5
):
    """Simple uniform placement along edge - NO optimization"""
    gaussians = []

    # Get edge position and orientation from descriptor
    edge_pos = edge_descriptor.get('position', 50)
    orientation = edge_descriptor.get('orientation', 'vertical')

    # Determine theta (perpendicular to edge)
    if orientation == 'vertical':
        theta = 0  # horizontal Gaussians
        positions = [(edge_pos, y) for y in np.linspace(10, 90, N)]
    elif orientation == 'horizontal':
        theta = np.pi/2  # vertical Gaussians
        positions = [(x, edge_pos) for x in np.linspace(10, 90, N)]

    # Create Gaussians
    color_value = edge_descriptor.get('mean_color', 0.5)

    for x, y in positions:
        g = Gaussian2D(
            x=x, y=y,
            sigma_parallel=sigma_parallel,
            sigma_perp=sigma_perp,
            theta=theta,
            color=np.array([color_value]),
            alpha=alpha,
            layer_type='E'
        )
        gaussians.append(g)

    return gaussians


def run_phase0_sweep(test_case_name, target_image, edge_descriptor, param_name, param_values):
    """Run one parameter sweep - render with different values, NO optimization"""

    results = []

    for value in param_values:
        # Set parameters
        if param_name == 'sigma_perp':
            gaussians = place_edge_gaussians_uniform(
                edge_descriptor, N=10,
                sigma_perp=value,
                sigma_parallel=5.0,
                spacing=2.5,
                alpha=0.5
            )
        elif param_name == 'sigma_parallel':
            gaussians = place_edge_gaussians_uniform(
                edge_descriptor, N=10,
                sigma_perp=2.0,
                sigma_parallel=value,
                spacing=value/2,
                alpha=0.5
            )
        # ... other parameters

        # Render (NO optimization, just render once)
        renderer = GaussianRenderer()
        rendered = renderer.render(gaussians, width=100, height=100, channels=1)

        # Compute metrics
        metrics = compute_metrics(target_image, rendered)

        # Save result
        results.append({
            'test_case': test_case_name,
            'parameter': param_name,
            'value': value,
            'psnr': metrics['psnr'],
            'mse': metrics['mse'],
            'rendered': rendered
        })

        print(f"{test_case_name} {param_name}={value}: PSNR={metrics['psnr']:.2f} dB")

    return results


# Main Phase 0 execution
if __name__ == "__main__":
    generator = SyntheticDataGenerator()

    # Test Case 6 from Phase 0 protocol: blur=2px, contrast=0.5
    target, descriptor = generator.generate_straight_edge(
        size=(100, 100),
        blur_sigma=2.0,
        contrast=0.5,
        orientation='vertical'
    )

    # Sweep 1: sigma_perp
    print("Running Sweep 1: sigma_perp")
    sigma_perp_values = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    results_sweep1 = run_phase0_sweep(
        "blur2_contrast05",
        target,
        descriptor,
        "sigma_perp",
        sigma_perp_values
    )

    # Plot results
    df = pd.DataFrame(results_sweep1)
    plt.figure(figsize=(8, 5))
    plt.plot(df['value'], df['psnr'], marker='o')
    plt.xlabel('sigma_perp (pixels)')
    plt.ylabel('PSNR (dB)')
    plt.title('Phase 0 Sweep 1: sigma_perp vs PSNR')
    plt.grid(True)
    plt.savefig('phase0_sweep1_sigma_perp.png')
    plt.close()

    print("\nPhase 0 Sweep 1 complete. Check phase0_sweep1_sigma_perp.png")
```

**Key differences from CCW's runners:**
- ✗ No optimizer.optimize() calls
- ✗ No gradient computation
- ✓ Simple render-and-measure loop
- ✓ Uses CCW's components (Gaussian2D, GaussianRenderer, SyntheticDataGenerator)

---

## Recommended Adaptation Strategy

### Option A: Use CCW Framework Selectively (Recommended)

**Pros:**
- Rendering and data generation already work
- Saves time reimplementing those
- Comprehensive logging available

**Cons:**
- Need to write Phase 0-specific runner
- Ignore 70% of CCW's code (optimizers, initializers)

### Option B: Start Fresh for Phase 0

**Pros:**
- Simpler, focused code
- No unused complexity

**Cons:**
- Duplicate rendering and data generation (~200 lines)

---

## Integration with Phase 0 Protocol

### Modify CCW_PHASE_0_INSTRUCTIONS.md

Add section:

```markdown
## Using CCW Framework Components

CCW built useful infrastructure that can be reused for Phase 0:

**From `experiments/gaussian_primitives_sprint/infrastructure/gaussian_primitives.py`:**
- Gaussian2D class (clean dataclass)
- GaussianRenderer (CPU-based, works well)
- SyntheticDataGenerator (generates all test cases)
- compute_metrics (PSNR, MSE, MAE)

**DO NOT USE:**
- Optimizers (optimizers.py) - Phase 0 is manual only
- Initializers E2-E4 (initializers.py) - Phase 0 tests E1 only
- Set 4 runners - Different experiment

**Simple Phase 0 approach:**
1. Use SyntheticDataGenerator to create 12 test cases
2. Write simple `place_gaussians_uniform()` function (20 lines)
3. For each parameter combination:
   - Place Gaussians
   - Render with GaussianRenderer
   - Compute metrics
   - Log results
4. NO optimization loop needed
```

---

## Performance Note

**CCW's timeout issue:** Finite difference gradients (70s per experiment)

**Phase 0 solution:** No gradients! Just render once per parameter combination.

**Expected Phase 0 timing:**
- Render 100×100 image: <0.5s
- 58 renders total: ~30 seconds
- Much faster than CCW's optimization experiments

---

## Recommendation

**MERGE CCW's framework** and use components:
- ✓ Gaussian2D, GaussianRenderer, SyntheticDataGenerator, compute_metrics
- ✓ Adapt for manual sweeps (no optimization)
- ✓ Write simplified Phase 0 runner (100-200 lines)

**This saves ~500 lines of reimplementation while staying true to Phase 0's manual approach.**

---

**Status:** Ready to integrate. CCW's framework provides solid foundation for Phase 0 if used correctly.
