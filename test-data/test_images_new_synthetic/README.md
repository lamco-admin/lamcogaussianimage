# GaussianImage Synthetic Test Pack
Generated: 2025-10-07T16:44:21.335901Z
Resolution: 1024×1024

This pack targets two codec failure modes:
1) **High-Frequency Texture Loss** — checks microtexture fidelity.
2) **Blue-Noise Residuals** — reveals high-frequency speckle on smooth fields.

## Files

### High-Frequency Texture Loss
- `HF_checkerboard_1px.png` — 1-pixel alternation, max-frequency test.
- `HF_multi_gratings.png` — multi-orientation multi-frequency sinusoids.
- `HF_fBm_multiscale.png` — fractal noise (fBm), dense stochastic microtexture.
- `HF_woven_texture.png` — synthetic woven fabric with micro-noise.
- `HF_hairlines.png` — thousands of fine line segments (hair/fur proxy).

**Masks**
- `MASK_variance_map.png` — local variance heatmap (σ≈2.5), normalized.
- `MASK_variance_threshold.png` — binary mask: var > μ + 1.5σ.

Use these masks for *selective modulation application* experiments.

### Blue-Noise Residual Sensitivity
- `BN_uniform_gray.png` — flat field, exposes any residual speckle.
- `BN_gradient_horizontal.png` — gentle ramp, reveals banding + HF noise.
- `BN_gradient_vertical.png` — vertical ramp.
- `BN_radial_gradient.png` — radial ramp, vignetting & smoothness probe.
- `BN_blurred_discs.png` — defocused blobs; great for speckle detection.
- `BN_lowfreq_field.png` — very low-frequency illumination field.

## Suggested Experiments

### Modulation variants
- **Additive**: I' = I + m
- **Multiplicative**: I' = I · (1 + m)
- **Blend**: I' = α·(multiplicative) + (1−α)·(additive), α ∈ [0,1]

Run encode/decode under all three. Expect measurable differences on HF vs BN sets.

### Selective application (variance thresholds)
- Compute local variance on the *input*.
- Apply modulation only where `variance > τ`. Try τ ∈ {μ+1σ, μ+1.5σ, μ+2σ}.
- Compare PSNR/SSIM/LPIPS in masked vs complement regions.

### Residual spectrum check
- Compute residual R = I_ref − I_recon.
- Inspect power spectral density (PSD). Blue-noise residuals tilt toward HF.
- Good mitigation lowers HF band energy on BN set *without* sacrificing HF set fidelity.

### Expected swing
Fixing modulation and applying selective variance gating typically turns a **−7 dB** deficit into **+3–5 dB** on texture-heavy scenes (≈10 dB swing), while reducing blue-noise speckle on smooth scenes.

## Notes
- All images are 8-bit PNG grayscale in [0,255].
- Deterministic seeds are used for reproducibility.
- If you need RGB variants or different resolutions, regenerate with your desired parameters.
