# Q2 Algorithm Discovery Experiments - IN PROGRESS

**Started**: 2025-12-04 ~23:00-24:00  
**Status**: RUNNING  
**Expected Duration**: 3-4 hours (24 images × 3 algorithms)

---

## What's Running

**Experiment**: Q2 Algorithm Comparison  
**Binary**: `packages/lgi-rs/target/release/examples/q2_algorithm_comparison`  
**Log**: `quantum_research/q2_algorithm_results/experiment_log.txt`  
**Output**: `quantum_research/q2_algorithm_results/full_comparison.json`

**Testing**:
- 24 Kodak images (kodim01-24)
- 3 optimization algorithms per image
- Total: 72 encoding experiments

**Algorithms**:
1. **Adam** (current standard) - adaptive, momentum-based
2. **OptimizerV2** (gradient descent + MS-SSIM)
3. **OptimizerV3** (perceptual: MS-SSIM + edge-weighted)

---

## Expected Results

**Per image output**:
```
[X/24] kodimXX.png
  Adam:        XX.XX dB | Time: XXXs
  OptimizerV2: XX.XX dB | Time: XXXs  
  OptimizerV3: XX.XX dB | Time: XXXs
  Winner: [algorithm]
```

**Final summary**:
```
Average PSNR:
  Adam:        XX.XX dB (baseline)
  OptimizerV2: XX.XX dB (+X.XX vs Adam)
  OptimizerV3: XX.XX dB (+X.XX vs Adam)

Win distribution:
  Adam: X/24
  OptimizerV2: X/24
  OptimizerV3: X/24
```

---

## What This Tests

**Hypothesis**: Different optimization algorithms work better for different types of Gaussians (quantum channels).

**This experiment** doesn't separate by channel yet - it tests which algorithm is universally best across all Gaussians.

**Next step** (if one algorithm wins): Test per-channel algorithm assignment.

**Next step** (if Adam still wins): Maybe algorithm choice doesn't matter as much as hyperparameters.

---

## How to Monitor

```bash
# Check progress
tail -f quantum_research/q2_algorithm_results/experiment_log.txt

# Check how many completed
grep "COMPARISON:" quantum_research/q2_algorithm_results/experiment_log.txt | wc -l

# Expected completion
# Started: ~23:00
# Duration: 3-4 hours
# Complete: ~02:00-03:00
```

---

## What Happens When Complete

Results will show one of:

**Scenario 1: OptimizerV2 or V3 wins significantly** (+2 dB or more)
→ Adopt that optimizer as new standard
→ Run per-channel tests to see if different channels prefer different algorithms

**Scenario 2: Results are mixed** (each wins on some images)
→ Implement per-channel algorithm selection
→ Match image characteristics to algorithm

**Scenario 3: Adam still best overall**
→ Algorithm choice isn't the key variable
→ Focus on hyperparameter tuning per channel instead
→ Or focus on other aspects (initialization, architecture)

---

**Status**: Experiment running unattended. Check back in 3-4 hours for results.
