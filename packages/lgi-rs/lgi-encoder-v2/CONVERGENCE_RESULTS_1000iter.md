# Convergence Test Results (1000 Iterations)

## Test Configuration
- Image size: 64×64
- N Gaussians: 50 (Baselines 1 & 2), 20 initial (Baseline 3)
- Iterations: 1000
- Test image: Gradient + checkerboard pattern

## Results Summary

### Baseline 1: GaussianImage (Random Init, Fixed N)
- **Initial loss**: 0.245190
- **Final loss**: 0.075008
- **Improvement**: 69.41%
- **Speed**: 161.5 iter/s
- **Convergence**: Steady, continuous improvement throughout all 1000 iterations
- **Status**: ✅ WORKS WELL

**Convergence curve**:
- 100 iter: 0.115 (53% improvement)
- 200 iter: 0.101 (59% improvement)
- 500 iter: 0.089 (64% improvement)
- 1000 iter: 0.075 (69% improvement)

**Analysis**:
- Still improving at iteration 1000, hasn't plateaued
- Random initialization gives poor starting point (0.245) but Adam optimizer recovers well
- Fixed N=50 sufficient for this simple test image
- No signs of overfitting or instability

---

### Baseline 2: Video Codec (K-means Init, Cosine LR)
- **Initial loss**: 0.057248 (4.28× better than Baseline 1!)
- **Final loss**: 0.189739
- **Improvement**: -231.43% (DIVERGED!)
- **Speed**: 199.6 iter/s
- **Status**: ❌ CATASTROPHIC FAILURE

**What happened**:
1. Iterations 1-8: Great progress! 0.057 → 0.036 (37% improvement)
2. Iterations 9-32: Rapid divergence! 0.036 → 0.099 (loss tripled!)
3. Iterations 33-1000: Plateaued at ~0.190, no recovery

**Root cause**: Cosine annealing LR schedule INCOMPATIBLE with current setup
- Initial LR too high (0.005 vs 0.001 for Baseline 1)
- Cosine schedule drops LR too quickly early on
- K-means init is good, but can't overcome bad LR schedule

**Fix needed**: Either:
1. Use constant LR like Baseline 1
2. Lower initial LR significantly
3. Use different schedule (exponential decay, step decay)

---

## Key Findings

### 1. Initialization Matters
- K-means gives **4.28× better** starting loss than random
- But good init can't save bad optimization schedule

### 2. Learning Rate Critical
- Baseline 1's constant LR=0.001 works reliably
- Baseline 2's cosine annealing with LR=0.005 causes divergence
- Need to re-test Baseline 2 with fixed LR

### 3. Convergence Still Happening at 1000 Iterations
- Baseline 1 still improving (loss 0.089 → 0.075 from iter 500-1000)
- Should test with 5k-10k iterations to find true convergence point

### 4. Corrected Gradients Working
- No instability in Baseline 1
- Divergence in Baseline 2 is LR issue, not gradient issue

### Baseline 3: 3D-GS (Adaptive Densification)
- **Initial N**: 20
- **Final N**: 1792 (89.6× growth!)
- **Initial loss**: 1.266855
- **Final loss**: 0.539315
- **Improvement**: 57.43%
- **Speed**: 41.0 iter/s (slowest due to large N)
- **Status**: ✅ WORKS, but has issues

**Densification timeline**:
- Iter 100: 20 → 40 (+20)
- Iter 200: 40 → 71 (+31)
- Iter 300: 71 → 124 (+53)
- Iter 400: 124 → 215 (+91)
- Iter 500: 215 → 372 (+157)
- Iter 600: 372 → 635 (+263)
- Iter 700: 635 → 1084 (+449)
- Iter 800: 1084 → 1792 (+708)

**Analysis**:
- Aggressive growth - each densification adds more Gaussians than last
- Loss oscillates after each densification (temporary degradation, then recovery)
- Major improvements at iter 700 (0.962 → 0.686) when N=635
- Still much slower convergence than Baseline 1 (57% vs 69% improvement)
- Final loss (0.539) is **7.2× worse** than Baseline 1 (0.075)!
- Speed penalty from large N (41 vs 161 iter/s)

**Problems identified**:
1. **Overgrowth**: 1792 Gaussians for a 64×64 image is excessive (0.44 Gaussians per pixel!)
2. **Inefficiency**: Despite 89× more Gaussians, achieves worse results
3. **Densification thresholds too aggressive**: Need tuning
4. **No pruning happening**: Should remove low-opacity Gaussians

---

## Comparison of All Three Baselines

### Final Loss Rankings (Lower is Better)
1. **Baseline 1**: 0.075 ⭐ WINNER
2. **Baseline 2**: 0.190 (DIVERGED - invalid)
3. **Baseline 3**: 0.539

### Efficiency Rankings (Loss per Gaussian)
1. **Baseline 1**: 0.075 / 50 = 0.0015 per Gaussian ⭐
2. **Baseline 3**: 0.539 / 1792 = 0.0003 per Gaussian
3. **Baseline 2**: N/A (diverged)

Baseline 1 is **7.2× better** than Baseline 3 while using **35.8× fewer** Gaussians!

### Speed Rankings
1. **Baseline 2**: 199.6 iter/s (but diverged)
2. **Baseline 1**: 161.5 iter/s ⭐
3. **Baseline 3**: 41.0 iter/s (slow due to large N)

---

## Key Findings

### 1. Initialization Matters
- K-means gives **4.28× better** starting loss than random
- But good init can't save bad optimization schedule

### 2. Learning Rate Critical
- Baseline 1's constant LR=0.001 works reliably
- Baseline 2's cosine annealing with LR=0.005 causes divergence
- Need to re-test Baseline 2 with fixed LR

### 3. Adaptive N is a Double-Edged Sword
- Baseline 3 grows N aggressively but achieves **worse** results
- Fixed N=50 (Baseline 1) outperforms adaptive N=1792 (Baseline 3)
- Densification thresholds need careful tuning

### 4. Simpler is Better (for this test)
- Baseline 1 (simplest approach) gives best results
- Random init + fixed N + constant LR + Adam = reliable convergence
- Complex strategies (adaptive N, cosine LR) introduce failure modes

### 5. Convergence Still Happening at 1000 Iterations
- Baseline 1 still improving (loss 0.089 → 0.075 from iter 500-1000)
- Baseline 3 plateauing around iter 900-1000
- Should test with 5k-10k iterations to find true convergence point

### 6. Corrected Gradients Working
- No instability in Baseline 1
- Baseline 3 shows expected temporary loss increase after densification, then recovery
- Divergence in Baseline 2 is LR issue, not gradient issue

---

## Next Steps

1. **Fix Baseline 2 LR and re-test**
   - Use constant LR=0.001 (same as Baseline 1)
   - Keep K-means init (proven good)
   - Re-run 1000 iteration test

2. **Tune Baseline 3 densification**
   - Reduce gradient threshold (less aggressive growth)
   - Enable pruning of low-opacity Gaussians
   - Test with different initial N

3. **Run 5k-10k iteration tests** on all baselines
   - Find true convergence points
   - Measure final quality limits

4. **Test on more complex images**
   - Natural images with more detail
   - Validate findings beyond simple test pattern

5. **Optimize Baseline 3 parameters**
   - Find sweet spot for densification frequency
   - Tune opacity and scale thresholds
   - Balance N growth vs quality

## Current Status Summary

| Baseline | Status | Final Loss | Improvement | N | Speed | Notes |
|----------|--------|------------|-------------|---|-------|-------|
| 1: GaussianImage | ✅ BEST | 0.075 | 69.4% | 50 | 161 iter/s | Simple, reliable, best quality |
| 2: Video Codec | ❌ FAILED | 0.190 | -231% | 50 | 200 iter/s | Good init, bad LR schedule |
| 3: 3D-GS | ⚠️ WORKS | 0.539 | 57.4% | 1792 | 41 iter/s | Over-grows, inefficient |

**Winner**: Baseline 1 (GaussianImage) - Random init + Fixed N + Constant LR
