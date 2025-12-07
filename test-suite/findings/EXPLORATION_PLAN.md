# Exploration Plan: Focused Technique Investigation

**Date:** 2025-12-07
**Goal:** Test specific techniques for specific improvements - NOT holistic solutions

---

## Guiding Principles

1. **No geometric tiling** - Image must remain whole; no grid-based segmentation
2. **Semantic isolation OK** - Different techniques for different feature types
3. **Layering OK** - Coarse-to-fine, residual encoding, multi-pass
4. **Focused improvements** - Each test targets ONE aspect
5. **Adam limitations acknowledged** - May not be compatible with all techniques

---

## Exploration Tracks

### Track A: Analytical Shape Derivation (Bypass Optimization)

**Hypothesis:** Structure tensor can provide Gaussian shapes directly, eliminating need for shape optimization.

| Test | Focus | Expected Outcome |
|------|-------|------------------|
| `test_structure_tensor_shapes` | Derive (σx, σy, θ) from eigenanalysis | Better edge alignment |
| `test_frangi_elongation` | Use vesselness for tubular features | Better text/wood grain |
| `test_coherence_based_anisotropy` | High coherence → more elongated | Adaptive shape selection |

**Key Question:** Can we fix shapes analytically and only optimize position/color?

---

### Track B: Hard Constraints (Not Soft Losses)

**Hypothesis:** Sigma clamping via geodesic distance prevents color bleeding better than loss functions.

| Test | Focus | Expected Outcome |
|------|-------|------------------|
| `test_geodesic_sigma_clamping` | EDT-based sigma limits | Sharp edges without bleeding |
| `test_edge_barrier_constraints` | Gaussians can't cross detected edges | Clean boundaries |

**Key Question:** Are hard geometric constraints more effective than weighted losses?

---

### Track C: Hierarchical/Layered Encoding

**Hypothesis:** Multiple passes at different scales capture detail better than single-pass.

| Test | Focus | Expected Outcome |
|------|-------|------------------|
| `test_coarse_to_fine` | Large Gaussians first, then smaller | Better convergence |
| `test_residual_encoding` | Each layer encodes residual of previous | Systematic refinement |
| `test_scale_separated_optimization` | Optimize large, freeze, add small | Stable hierarchy |

**Key Question:** Does layering improve final quality or just convergence speed?

---

### Track D: Feature-Specific Strategies

**Hypothesis:** Different image features need fundamentally different approaches.

| Test | Focus | Expected Outcome |
|------|-------|------------------|
| `test_text_stroke_targeting` | Frangi/structure tensor for text | Readable text |
| `test_smooth_region_handling` | Large overlapping Gaussians for sky | Efficient smooth coverage |
| `test_texture_anisotropy` | Aligned Gaussians for directional texture | Better wood grain |

**Key Question:** Can we identify feature type and apply specialized technique?

---

### Track E: Alternative Loss Functions

**Hypothesis:** Perceptual losses guide optimization toward visually better results.

| Test | Focus | Expected Outcome |
|------|-------|------------------|
| `test_edge_weighted_mse` | 3× weight at gradient magnitude | Sharper edges |
| `test_gradient_loss` | Match image gradients, not just pixels | Better structure |
| `test_saliency_weighted` | Higher weight on salient regions | Better perceptual quality |

**Key Question:** Which loss modification yields best visual improvement?

---

### Track F: Closed-Form Solutions

**Hypothesis:** Some parameters have analytical solutions - no iteration needed.

| Test | Focus | Expected Outcome |
|------|-------|------------------|
| `test_closed_form_colors` | Linear least squares for RGB | Optimal colors instantly |
| `test_analytical_positions` | Place at importance maxima | No position optimization |
| `test_hybrid_analytical_adam` | Analytical init → Adam refinement | Faster convergence |

**Key Question:** How much can we solve analytically vs. iteratively?

---

## Immediate Experiments to Run

### Priority 1 (Starting Now)
1. `test_structure_tensor_shapes` - Analytical shape derivation
2. `test_geodesic_sigma_clamping` - Hard edge constraints
3. `test_coarse_to_fine` - Hierarchical encoding
4. `test_edge_weighted_mse` - Perceptual loss

### Priority 2 (Queue After)
5. `test_text_stroke_targeting` - Feature-specific
6. `test_closed_form_colors` - Analytical colors
7. `test_residual_encoding` - Layered refinement

---

## Success Criteria

Each experiment should answer:
1. Does this technique improve PSNR for its targeted aspect?
2. Does the visual output show improvement in the targeted feature?
3. Is this technique compatible with Adam, or does it need different optimization?
4. What are the computational costs?

**NOT asking:** "Is this the optimal holistic solution?"

---

## Framework Compliance

All experiments will:
- Use TestResult with `save_with_image()`
- Record all parameters via `set_param()`
- Show iteration progress with direction indicators
- Abort and save partial results if diverging
- Compare against baseline (grid init + Adam)

---

*This plan focuses on understanding individual techniques, not finding a magic configuration.*
