# Multi-Hour Q2 Experimental Campaign
## Algorithm Discovery Per Quantum Channel

**Goal**: Discover which optimizer algorithm works best for each quantum channel  
**Duration**: 6-12 hours of automated experiments  
**Output**: Empirical mapping of Channel → Best Algorithm → Optimal Hyperparameters

---

## Available Optimizers (Already Implemented)

1. **Adam** (`adam_optimizer.rs`) - TESTED, WORKS
   - First-order adaptive
   - Good for: robust optimization, handles noise
   - Current: Used for everything

2. **L-BFGS** (`optimizer_lbfgs.rs`) - EXISTS, UNTESTED
   - Second-order quasi-Newton
   - Good for: smooth problems, precise convergence
   - Literature: Used in 3D splatting for fine-tuning

3. **OptimizerV2** (`optimizer_v2.rs`) - EXISTS, UNKNOWN STATUS
   - Appears to be gradient descent variant
   - May have edge-weighted loss, MS-SSIM support
   - Need to test

4. **OptimizerV3** (`optimizer_v3_perceptual.rs`) - EXISTS, UNKNOWN
   - Perceptual optimization
   - May use different loss function
   - Need to test

**To Add**: SGD with momentum (if not in V2), RMSprop (if beneficial)

---

## Experiment Structure

### Experiment Set 1: Algorithm Comparison (3-4 hours)

**Matrix**: 8 channels × 3 algorithms × 3 images = 72 experiments

```
For each quantum channel:
  Extract representative Gaussian configuration (σ_x, σ_y from channel mean)
  
  For each algorithm (Adam, L-BFGS, OptimizerV2):
    Initialize 50 Gaussians matching channel characteristics
    
    For each test image (kodim03, kodim08, kodim15):
      Optimize with algorithm
      Measure: final PSNR, iterations to convergence, time
      
  Determine: Which algorithm achieves best PSNR for this channel
```

**Runtime**: 72 experiments × 2-3 min each = 144-216 min = **2.5-3.5 hours** serial  
**Parallelized**: With 8 concurrent = **20-30 minutes**

**Output**: `channel_algorithm_matrix.json`
```json
{
  "channel_0": {
    "Adam": {"avg_psnr": 12.3, "avg_time": 120},
    "L-BFGS": {"avg_psnr": 11.8, "avg_time": 180},
    "OptimizerV2": {"avg_psnr": 12.1, "avg_time": 110},
    "winner": "Adam"
  },
  "channel_3": {
    "Adam": {"avg_psnr": 14.2, "avg_time": 125},
    "L-BFGS": {"avg_psnr": 19.7, "avg_time": 95},
    "OptimizerV2": {"avg_psnr": 13.5, "avg_time": 140},
    "winner": "L-BFGS"  ← Discovery!
  },
  ...
}
```

### Experiment Set 2: Hyperparameter Tuning (6-8 hours)

**For each (channel, winning_algorithm) pair**, grid search hyperparameters:

```
Channel 3 + L-BFGS (winner from Set 1):
  history_size: [5, 10, 20]
  max_iter: [20, 50, 100]
  line_search: [aggressive, moderate, conservative]
  = 3 × 3 × 3 = 27 combinations
  × 3 images = 81 experiments
  × 2 min = 162 min = 2.7 hours for this channel

Do this for channels where non-Adam wins (estimated 3-4 channels)
Total: 3-4 channels × 2.7 hours = 8-11 hours
```

**Parallelized**: With 16 concurrent = **30-45 minutes per channel** = **2-3 hours total**

**Output**: `channel_optimal_hyperparams.json`

### Experiment Set 3: Full Validation (2-3 hours)

**Test discovered algorithms on all 24 Kodak images**:

```
For each Kodak image:
  Baseline: Adam for all Gaussians (current)
  
  Quantum-guided: Per-channel optimal algorithm
    - Assign each Gaussian to nearest channel (by σ_x, σ_y)
    - Optimize with that channel's best algorithm
    - Render composit

ionally
  
  Compare: PSNR, iterations, time
```

**Runtime**: 24 images × 2 encodings × 7 min = **5.6 hours** serial  
**Parallelized**: 4 concurrent = **1.4 hours**

---

## Tonight's Execution Plan

### Phase 1: Setup (Now - 30 min)

1. Export L-BFGS module in lib.rs
2. Create wrapper encode method using L-BFGS
3. Create initial test comparing Adam vs L-BFGS on Channel 4
4. Validate: Does L-BFGS work at all?

### Phase 2: Algorithm Matrix (30 min - 3 hours)

5. Build comparison harness (30 min implementation)
6. Run Set 1 experiments (20-30 min parallelized)
7. Analyze which algorithms win which channels (10 min)

**Decision point**: If L-BFGS or V2 wins on ANY channel → proceed to Phase 3

### Phase 3: Hyperparameter Tuning (3 hours - 6 hours)

8. For winning (channel, algorithm) pairs, grid search hyperparams
9. Run Set 2 experiments (2-3 hours parallelized)
10. Document optimal configurations

### Phase 4: Validation (6 hours - 9 hours)

11. Run Set 3 on all Kodak images
12. Measure overall improvement vs baseline
13. Document findings

**Total**: 9-10 hours if run to completion, but can stop after Phase 2 (3 hours) if results aren't promising

---

## Success Criteria

### Phase 2 Success:
- At least 1 channel has non-Adam winner with +2 dB improvement
- Example: Channel 3 + L-BFGS = 19 dB vs Channel 3 + Adam = 14 dB

### Phase 3 Success:
- Hyperparameter tuning improves winner by +0.5-1 dB
- Example: L-BFGS(history=20) beats L-BFGS(history=10)

### Phase 4 Success:
- Per-channel algorithms improve overall encoding by +1 dB
- Or reduce iterations by 30%
- On full 24-image dataset

---

## What to Implement Right Now

**Immediate** (next 30-60 min):
1. Add `pub mod optimizer_lbfgs;` to lib.rs
2. Create `encode_error_driven_lbfgs()` method
3. Test on single image to verify it works

**Then** (next 30 min):
4. Create `examples/test_algorithms_per_channel.rs`
5. Run initial comparison: Adam vs L-BFGS on Channels 3-4

**Then** (let run for hours):
6. Full algorithm × channel matrix
7. Hyperparameter tuning for winners
8. Full Kodak validation

**This runs overnight, results ready tomorrow**
