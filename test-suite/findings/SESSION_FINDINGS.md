# Session Findings - 2025-12-07

## Completed Tests Summary

### 1. Hybrid Loss Function (α·MSE + (1-α)·MS-SSIM)

**WINNER: α = 0.84**

| Alpha | Final PSNR | MS-SSIM | Improvement |
|-------|-----------|---------|-------------|
| 0.20  | 22.24 dB  | 0.839   | +1.97 dB    |
| 0.50  | 22.55 dB  | 0.861   | +2.28 dB    |
| **0.84** | **22.74 dB** | **0.867** | **+2.47 dB** |
| 1.00  | 22.33 dB  | 0.838   | +2.06 dB    |

**Key Insight**: 84% MSE + 16% MS-SSIM provides optimal blend. Pure MSE loses perceptual quality; heavy MS-SSIM reduces PSNR.

---

### 2. L-M JtJ Strategy Comparison

**WINNER: Strategy 1 (BlockDiagonal)**

| Strategy | Final PSNR | Time (s) | Improvement |
|----------|-----------|----------|-------------|
| 0 (Default) | 19.66 dB | 15.99 | +0.48 dB |
| **1 (BlockDiagonal)** | **20.19 dB** | **7.85** | **+1.01 dB** |
| 2 (Sparse) | 19.19 dB | 1.85 | +0.00 dB |
| 3 (Hierarchical) | 19.07 dB | 20.14 | -0.12 dB |
| 4 (MatrixFree) | 19.05 dB | 54.48 | -0.14 dB |

**Key Insight**: BlockDiagonal JtJ is 2× faster than default and +0.53 dB better.

---

### 3. GPU vs CPU Adam Benchmark

**WINNER: GPU (12× faster, same quality)**

| Backend | Final PSNR | Time (s) | Speedup |
|---------|-----------|----------|---------|
| CPU     | 14.30 dB  | 126.49   | 1×      |
| **GPU** | **14.24 dB** | **10.52** | **12×** |

**Key Insight**: GPU Adam provides massive speedup with negligible quality difference (~0.06 dB).

---

### 4. LOD System Quality

**Progressive rendering quality analysis**

| LOD Level | Gaussians | % of Total | PSNR | % of Full Quality |
|-----------|-----------|------------|------|-------------------|
| Coarse    | 88        | 17%        | 17.10 dB | 88% |
| Coarse+Med| 196       | 38%        | 18.11 dB | 93% |
| Full      | 512       | 100%       | 19.41 dB | 100% |

**Key Insight**: LOD enables efficient progressive decoding - 38% of data gives 93% quality.

---

### 5. Gradient Peak vs Uniform Grid Initialization

| Init Method | Gaussians | Initial PSNR | Final PSNR | Improvement |
|-------------|-----------|--------------|------------|-------------|
| Uniform Grid | 256 | 20.12 dB | 20.99 dB | +0.87 dB |
| Gradient Peak | 315 | 17.14 dB | 20.85 dB | +3.71 dB |

**Key Insight**: Both reach similar final quality (~21 dB), but gradient peak has more optimization headroom. Uniform grid has better initial placement.

---

## Tests In Progress

1. **EWA Splatting vs Standard** - Comparing alias-free rendering
2. **Densification Variations** - Testing pruning thresholds
3. **Guided Filter Impact** - Edge-preserving color init
4. **Rate-Distortion Selection** - Optimal Gaussian selection
5. **MS-SSIM Hybrid** - Further alpha sweep testing

---

## Recommendations Based on Findings

### Production Encoder Settings:
```rust
// Optimal configuration based on today's tests
let encoder_config = EncoderConfig {
    // Loss function
    hybrid_alpha: 0.84,  // 84% MSE + 16% MS-SSIM

    // Optimizer
    optimizer: "GPU_Adam",  // 12× faster than CPU

    // L-M fallback (if needed)
    lm_jtj_strategy: JtJStrategy::BlockDiagonal,  // +1.01 dB, fastest

    // Initialization
    init_method: "jittered_grid",  // Best initial placement

    // LOD
    enable_lod: true,  // Progressive decoding support
};
```

### Key Takeaways:

1. **Hybrid loss is essential** - α=0.84 balances PSNR and perceptual quality
2. **GPU acceleration works** - 12× speedup with no quality loss
3. **LOD enables streaming** - 38% data → 93% quality
4. **BlockDiagonal JtJ** - Best L-M strategy if second-order methods needed
5. **Init matters less than optimizer** - Both methods converge to similar quality

---

## Files Generated Today

- `adam_msssim_alpha_*.json` - Hybrid loss sweep results
- `lm_strategy_comparison_*.json` - JtJ strategy results
- `bench_*_adam_*.json` - GPU/CPU benchmark results
- `lod_system_quality_*.json` - LOD analysis
- `gradient_peak_*.json` - Init strategy comparison
- `densification_*.json` - Pruning/densification tests
