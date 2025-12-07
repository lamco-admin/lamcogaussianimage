# Image Interpretation Tools Analysis

**Date:** 2025-12-07
**Purpose:** Catalog existing tools, identify gaps, plan for feature-driven adaptive encoding

---

## The Core Problem

Universal techniques applied to whole images fail because:
- Sky needs large smooth Gaussians
- Text needs tiny elongated Gaussians
- Edges need edge-aligned anisotropic Gaussians
- Textures need different strategies than smooth regions

**Solution:** Image interpretation must drive Gaussian placement, shape, and optimization strategy.

---

## Existing Modules Inventory

### Tier 1: Solid Foundation (Well-Implemented)

| Module | Purpose | Outputs | Status |
|--------|---------|---------|--------|
| `structure_tensor.rs` | Edge orientation & coherence | eigenvectors, eigenvalues, coherence per-pixel | ✅ Complete |
| `geodesic_edt.rs` | Edge-respecting distance | distance to edges, sigma clamping | ✅ Complete |
| `position_probability_map.rs` | Combine signals for placement | normalized probability distribution | ✅ Complete |
| `laplacian.rs` | Second derivative, peak detection | edge magnitude, local maxima | ✅ Complete |

### Tier 2: Basic (Works But Limited)

| Module | Purpose | Limitations |
|--------|---------|-------------|
| `content_detection.rs` | Classify whole image | Global only, no per-pixel |
| `entropy.rs` | Tile-based complexity | Crude approximation |
| `frangi_vesselness.rs` | Tubular structures | Needs scale sweeping |
| `edge_weighted_loss.rs` | 3× weight at edges | Binary, no gradation |

### Tier 3: Minimal (Stubs)

| Module | Purpose | What's Missing |
|--------|---------|----------------|
| `saliency_detection.rs` | Importance | Just gradient magnitude |
| `text_stroke_detection.rs` | Text handling | Width estimation poor |
| `specular_detection.rs` | Highlights | Just percentile threshold |
| `perceptual_weighting.rs` | Weight map | No actual perceptual model |

---

## Critical Gaps

### 1. **Per-Pixel Feature Classification**

**Need:** Each pixel labeled with feature type (edge, smooth, texture, text, specular).

**Currently:** Only global content type. No semantic pixel map.

**Impact:** Can't apply different strategies to different regions without knowing what's there.

### 2. **Artifact Detection**

**Need:** Identify common Gaussian artifacts:
- Color bleeding (Gaussians crossing edges)
- Ringing (oscillations near edges)
- Blur (insufficient Gaussian count)
- Banding (quantization in smooth regions)

**Currently:** Nothing. Just PSNR.

**Impact:** Can't diagnose what's wrong to fix it.

### 3. **Error Analysis Per Region**

**Need:** Given current reconstruction, which regions have highest error? What type of error?

**Currently:** Global MSE only.

**Impact:** Can't do targeted refinement.

### 4. **Feature → Strategy Mapping**

**Need:** Systematic mapping:
- Smooth region → large overlapping Gaussians, few needed
- Edge → anisotropic Gaussians aligned to edge, geodesic clamping
- Text → small elongated Gaussians along strokes
- Texture → periodic or anisotropic patterns

**Currently:** Manual, ad-hoc.

**Impact:** No principled way to choose strategies.

### 5. **Iterative Guidance**

**Need:** During optimization, analyze intermediate results and adapt:
- Densify where error is high
- Prune where Gaussians overlap unnecessarily
- Adjust shapes based on residual patterns

**Currently:** Blind iteration to fixed count.

**Impact:** No adaptive refinement.

### 6. **Texture Analysis**

**Need:** Detect:
- Texture directionality (wood grain is horizontal)
- Periodicity (repeating patterns)
- Scale (fine vs coarse texture)

**Currently:** Just coherence from structure tensor.

**Impact:** Can't handle textured regions optimally.

### 7. **Boundary Extraction**

**Need:** Explicit contours/boundaries between regions.

**Currently:** Just edge pixels, not connected contours.

**Impact:** Can't reason about object boundaries.

---

## Proposed Architecture: ImageAnalysis

```rust
/// Comprehensive image analysis result
pub struct ImageAnalysis {
    /// Dimensions
    pub width: u32,
    pub height: u32,

    /// Per-pixel feature classification
    pub feature_map: Vec<FeatureType>,

    /// Structure tensor field (orientation, coherence)
    pub structure: StructureTensorField,

    /// Geodesic distance to edges
    pub edge_distance: Vec<f32>,

    /// Importance map for placement priority
    pub importance: Vec<f32>,

    /// Global content statistics
    pub content_stats: ContentStats,
}

#[derive(Clone, Copy, PartialEq)]
pub enum FeatureType {
    Smooth,          // Large Gaussians OK
    Edge,            // Anisotropic, aligned
    Corner,          // Isotropic, precise
    TextStroke,      // Very small, elongated
    Texture,         // Depends on directionality
    Specular,        // Special handling
    SmoothGradient,  // Smooth but color-varying
}

pub struct ContentStats {
    pub dominant_type: FeatureType,
    pub entropy: f32,
    pub edge_density: f32,
    pub texture_directionality: f32,
    pub color_complexity: f32,
}
```

---

## Proposed: FeatureClassifier

```rust
/// Per-pixel feature classification
pub struct FeatureClassifier {
    /// Thresholds for classification
    pub edge_coherence_threshold: f32,    // > 0.7 = edge
    pub corner_coherence_threshold: f32,  // < 0.3 with high energy = corner
    pub text_width_threshold: f32,        // < 3px stroke width = text
    pub specular_percentile: f32,         // > 95th = specular
    pub texture_variance_threshold: f32,  // High local variance = texture
}

impl FeatureClassifier {
    /// Classify every pixel
    pub fn classify(&self, image: &ImageBuffer<f32>) -> Vec<FeatureType> {
        // 1. Compute structure tensor
        // 2. Compute local variance (texture indicator)
        // 3. Detect specular highlights
        // 4. Classify each pixel based on:
        //    - Coherence + energy → edge/corner/smooth
        //    - Stroke width → text
        //    - Luminance → specular
        //    - Local variance → texture
    }
}
```

---

## Proposed: ArtifactDetector

```rust
/// Detect common Gaussian encoding artifacts
pub struct ArtifactDetector;

impl ArtifactDetector {
    /// Detect color bleeding across edges
    pub fn detect_bleeding(
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
        edges: &[bool],
    ) -> Vec<(u32, u32, f32)> {
        // For each edge pixel:
        // - Compare colors on each side in target vs rendered
        // - High cross-edge correlation in rendered = bleeding
    }

    /// Detect blur (insufficient detail)
    pub fn detect_blur(
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
    ) -> Vec<f32> {
        // Compare high-frequency content (Laplacian)
        // Low HF in rendered vs target = blur
    }

    /// Detect ringing (oscillations)
    pub fn detect_ringing(
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
    ) -> Vec<(u32, u32, f32)> {
        // Near edges, look for sign-alternating errors
    }
}
```

---

## Proposed: AdaptiveStrategy

```rust
/// Map features to Gaussian strategies
pub struct AdaptiveStrategy {
    pub strategies: HashMap<FeatureType, GaussianConfig>,
}

pub struct GaussianConfig {
    pub scale_range: (f32, f32),
    pub anisotropy_range: (f32, f32),
    pub use_structure_tensor_orientation: bool,
    pub use_geodesic_clamping: bool,
    pub density_multiplier: f32,
}

impl AdaptiveStrategy {
    /// Generate Gaussians based on feature map
    pub fn generate_gaussians(
        &self,
        analysis: &ImageAnalysis,
        total_budget: usize,
    ) -> Vec<Gaussian2D<f32>> {
        // 1. Count pixels per feature type
        // 2. Allocate budget proportionally (with density_multiplier)
        // 3. For each feature type:
        //    - Sample positions from importance map for that type
        //    - Initialize shapes based on GaussianConfig
        //    - Apply structure tensor orientation if configured
        //    - Apply geodesic clamping if configured
    }
}
```

---

## Proposed: IterativeRefinement

```rust
/// Analyze intermediate results and guide refinement
pub struct IterativeAnalyzer;

impl IterativeAnalyzer {
    /// Analyze current reconstruction error
    pub fn analyze_error(
        rendered: &ImageBuffer<f32>,
        target: &ImageBuffer<f32>,
        feature_map: &[FeatureType],
    ) -> ErrorAnalysis {
        // Per-feature-type error breakdown
        // High-error region identification
        // Artifact detection
    }

    /// Suggest refinement actions
    pub fn suggest_actions(
        analysis: &ErrorAnalysis,
        gaussians: &[Gaussian2D<f32>],
    ) -> Vec<RefinementAction> {
        // Densify high-error regions
        // Prune low-contribution Gaussians
        // Adjust shapes where artifacts detected
    }
}

pub enum RefinementAction {
    AddGaussians { region: Rect, count: usize, config: GaussianConfig },
    SplitGaussian { index: usize },
    PruneGaussian { index: usize },
    AdjustShape { index: usize, new_shape: GaussianParams },
}
```

---

## Implementation Priority

### Phase 1: Feature Classification (HIGHEST)
1. Implement `FeatureClassifier` with per-pixel output
2. Create visualization tool to see feature map
3. Test on kodim03 (caps, text, wood, sky - diverse features)

### Phase 2: Feature → Strategy Mapping
1. Define `GaussianConfig` for each feature type
2. Implement budget allocation
3. Test feature-specific initialization

### Phase 3: Artifact Detection
1. Implement bleeding detection
2. Implement blur detection
3. Create artifact visualization

### Phase 4: Iterative Refinement
1. Error analysis per region
2. Refinement action suggestion
3. Adaptive densification loop

---

## Immediate Next Steps

1. **Create `feature_classifier.rs`** - Per-pixel feature type classification
2. **Test feature classification on kodim03** - Verify it correctly identifies sky/caps/text/wood
3. **Create `image_analysis.rs`** - Unified analysis structure
4. **Implement feature-specific Gaussian generation** - Different strategies per feature

---

## Success Criteria

The system should be able to:
1. Given kodim03, automatically identify:
   - Sky regions (smooth)
   - Cap regions (mixed)
   - "Bahamas" text (text strokes)
   - Wood grain (directional texture)
   - Cap edges (strong edges)

2. Generate Gaussians appropriately:
   - Large Gaussians in sky
   - Small aligned Gaussians for text
   - Anisotropic Gaussians for edges

3. During optimization, detect:
   - If text is blurry → add more small Gaussians
   - If edges are bleeding → apply geodesic clamping
   - If smooth regions are over-detailed → reduce count

---

*This document establishes the roadmap for building image interpretation capabilities that will enable feature-driven adaptive encoding.*
