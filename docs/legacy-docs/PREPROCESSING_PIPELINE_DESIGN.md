# Image Preprocessing Pipeline Design

**Purpose**: Deep image analysis to guide encoder strategy selection

**Language**: Python (rich CV/ML ecosystem)
**Output**: JSON metadata consumed by Rust encoder
**Future**: Compile to native executable via Nuitka

---

## Architecture

```
Input Image (PNG/JPEG)
    ↓
┌─────────────────────────────────────────────────────────┐
│ PREPROCESSING PIPELINE (Python)                         │
├─────────────────────────────────────────────────────────┤
│ 1. Basic Analysis                                       │
│    - Dimensions, color space                            │
│    - Histogram, dynamic range                           │
│                                                          │
│ 2. Complexity Analysis                                  │
│    - Global entropy                                     │
│    - Local entropy map (tile-based)                     │
│    - Texture vs smooth classification                   │
│    - High-frequency content detection                   │
│                                                          │
│ 3. Structure Analysis                                   │
│    - Edge detection (Canny, Sobel)                      │
│    - Gradient magnitude map                             │
│    - Structure tensor (eigenvalues, coherence)          │
│    - Corner detection (Harris, Shi-Tomasi)              │
│                                                          │
│ 4. Semantic Analysis (Optional - requires ML)           │
│    - Object detection (YOLO, Detectron2)                │
│    - Semantic segmentation (Segment Anything)           │
│    - Saliency detection                                 │
│    - Face detection                                     │
│                                                          │
│ 5. Perceptual Analysis                                  │
│    - Saliency map (important regions)                   │
│    - Visual complexity map                              │
│    - Texture analysis (Haralick, LBP)                   │
│                                                          │
│ 6. Strategy Recommendation                              │
│    - Recommended init: grid | slic | kmeans | ppm       │
│    - Recommended N: based on complexity                 │
│    - Per-region processing hints                        │
│    - Parameter suggestions                              │
└─────────────────────────────────────────────────────────┘
    ↓
metadata.json
    ↓
┌─────────────────────────────────────────────────────────┐
│ LGI ENCODER (Rust)                                      │
├─────────────────────────────────────────────────────────┤
│ 1. Load metadata                                        │
│ 2. Select strategy based on recommendations             │
│ 3. Use analysis maps for technique selection            │
│ 4. Encode with guided parameters                        │
└─────────────────────────────────────────────────────────┘
    ↓
output.lgi (compressed file)
```

---

## Metadata Schema

```json
{
  "image": {
    "path": "kodim02.png",
    "width": 768,
    "height": 512,
    "channels": 3,
    "format": "PNG"
  },

  "complexity": {
    "global_entropy": 0.113,
    "mean_gradient": 0.045,
    "texture_percentage": 35.2,
    "smooth_percentage": 48.1,
    "detail_percentage": 16.7,
    "complexity_score": "medium"
  },

  "structure": {
    "edge_count": 1247,
    "corner_count": 342,
    "dominant_orientation": "horizontal",
    "coherence_avg": 0.42
  },

  "semantic": {
    "objects_detected": [
      {"class": "person", "bbox": [100, 150, 300, 400], "confidence": 0.95},
      {"class": "sky", "bbox": [0, 0, 768, 200], "confidence": 0.88}
    ],
    "scene_type": "outdoor",
    "has_faces": true,
    "has_text": false
  },

  "perceptual": {
    "salient_regions": [
      {"bbox": [150, 200, 250, 350], "importance": 0.9, "reason": "face"}
    ],
    "visual_complexity": "medium-high"
  },

  "recommendations": {
    "init_strategy": "grid",
    "recommended_n": 500,
    "n_range": [300, 1000],

    "use_structure_tensor": true,
    "use_edge_weighted": true,
    "use_ms_ssim": false,
    "use_guided_filter": false,

    "region_hints": [
      {"region": "sky", "strategy": "large_gaussians", "n_budget": 50},
      {"region": "face", "strategy": "dense_detail", "n_budget": 200},
      {"region": "background", "strategy": "sparse", "n_budget": 100}
    ],

    "parameter_suggestions": {
      "coherence_threshold": 0.3,
      "structure_tensor_sigma": [1.2, 1.0],
      "learning_rate_scale": 1.0
    }
  },

  "maps": {
    "entropy_map_path": "metadata/kodim02_entropy.npy",
    "gradient_map_path": "metadata/kodim02_gradient.npy",
    "saliency_map_path": "metadata/kodim02_saliency.npy",
    "segmentation_mask_path": "metadata/kodim02_segments.npy"
  }
}
```

---

## Implementation Plan

### Phase 1: Basic Preprocessing (Week 1)

**File**: `tools/preprocess_image.py`

**Dependencies**:

```bash
pip install opencv-python scikit-image numpy scipy pillow
```

**Features**:

- Entropy analysis (tile-based)
- Gradient computation
- Edge/corner detection
- Texture classification
- Recommended N and strategy

**Output**: Basic metadata.json (no ML)

### Phase 2: Enhanced Analysis (Week 2)

**Add**:

- Saliency detection (opencv saliency module)
- Better texture analysis (Haralick features, LBP)
- Statistical complexity metrics
- Multi-scale analysis

### Phase 3: ML Integration (Week 3-4)

**Add** (optional):

- Segment Anything for semantic segmentation
- YOLO for object detection
- Perceptual metrics (LPIPS)

**Compile**: Nuitka for standalone binary

---

## Rust Integration

### Metadata Loader

**File**: `lgi-encoder-v2/src/metadata_loader.rs`

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct ImageMetadata {
    image: ImageInfo,
    complexity: ComplexityAnalysis,
    structure: StructureAnalysis,
    semantic: Option<SemanticAnalysis>,
    recommendations: EncodingRecommendations,
    maps: Option<AnalysisMaps>,
}

pub fn load_metadata(path: &str) -> Result<ImageMetadata, String> {
    let json = std::fs::read_to_string(path)?;
    serde_json::from_str(&json)
        .map_err(|e| format!("Failed to parse metadata: {}", e))
}
```

### Guided Encoder

**File**: `lgi-encoder-v2/src/guided_encoder.rs`

```rust
pub struct GuidedEncoder {
    encoder: EncoderV2,
    metadata: ImageMetadata,
}

impl GuidedEncoder {
    pub fn encode_with_metadata(&self) -> Vec<Gaussian2D> {
        // Use metadata recommendations
        let n = self.metadata.recommendations.recommended_n;

        let gaussians = match self.metadata.recommendations.init_strategy.as_str() {
            "grid" => self.encoder.initialize_gaussians(grid_size_for_n(n)),
            "slic" => self.encoder.initialize_slic(n),
            "kmeans" => self.encoder.initialize_kmeans(n),
            "ppm" => self.encoder.initialize_ppm(n),
            _ => self.encoder.initialize_gaussians(grid_size_for_n(n)),
        };

        // Apply recommended module settings
        let mut opt = OptimizerV2::default();
        opt.use_edge_weighted = self.metadata.recommendations.use_edge_weighted;
        opt.use_ms_ssim = self.metadata.recommendations.use_ms_ssim;

        // Optimize
        opt.optimize(&mut gaussians, &self.encoder.target);

        gaussians
    }
}
```

---

## Example Workflow

```bash
# Step 1: Preprocess image (Python)
python tools/preprocess_image.py kodim02.png --output metadata/kodim02.json

# Step 2: Encode with metadata (Rust)
cargo run --release -- encode kodim02.png --metadata metadata/kodim02.json --output kodim02.lgi

# Or programmatically:
let metadata = load_metadata("metadata/kodim02.json")?;
let encoder = GuidedEncoder::new(image, metadata)?;
let gaussians = encoder.encode_with_metadata();
```

---

## Benefits

**Separation of concerns**:

- Python: Rich analysis (CV/ML libraries)
- Rust: Fast encoding (performance critical)

**Flexibility**:

- Can update preprocessing without recompiling encoder
- Can run preprocessing once, encode multiple times with different settings
- Can build preprocessing profiles for entire datasets

**Intelligence**:

- Encoder makes informed decisions, not blind guesses
- Per-image optimal strategy
- Better quality and/or efficiency

---

## Next Steps

1. ✅ Run visual_strategy_comparison (see what current methods do)
2. ⏳ Build basic preprocess_image.py (entropy, gradient, recommendations)
3. ⏳ Create metadata_loader.rs and test integration
4. ⏳ Validate: Does metadata-guided encoding improve quality?
5. ⏳ Expand preprocessing with more sophisticated analysis

---

*Created: Session 8 (2025-10-07)*
*Status*: Design complete, ready to implement
