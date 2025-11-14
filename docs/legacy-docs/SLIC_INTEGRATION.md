# SLIC Superpixel Integration Guide

**Strategy E**: SLIC (Simple Linear Iterative Clustering) initialization
**Priority**: HIGH (best-rated in research)
**Status**: Requires external Python/scikit-image

---

## What is SLIC?

**SLIC** segments image into perceptually meaningful regions (superpixels):
- Respects object boundaries
- Uniform size distribution
- Fast (linear time)
- One of best segmentation algorithms

**For LGI**: Initialize one Gaussian per superpixel
- Position: superpixel centroid
- Color: mean color
- Covariance: from pixel distribution

---

## Implementation Options

### Option 1: Python Preprocessing (Easiest)

**Python script** (`tools/slic_preprocess.py`):
```python
from skimage.segmentation import slic
from skimage.io import imread
import json
import numpy as np

def generate_slic_init(image_path, n_segments=100):
    # Load image
    image = imread(image_path)
    height, width = image.shape[:2]

    # Run SLIC
    segments = slic(image, n_segments=n_segments, compactness=10, sigma=1)

    # Extract superpixel properties
    gaussians = []

    for segment_id in range(segments.max() + 1):
        mask = (segments == segment_id)
        pixels = np.argwhere(mask)

        if len(pixels) == 0:
            continue

        # Centroid
        cy, cx = pixels.mean(axis=0)

        # Mean color
        segment_pixels = image[mask]
        r, g, b = segment_pixels.mean(axis=0) / 255.0

        # Covariance (for Gaussian shape)
        cov = np.cov(pixels.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        scale_x = np.sqrt(eigenvalues[0]) / width
        scale_y = np.sqrt(eigenvalues[1]) / height
        rotation = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])

        gaussians.append({
            "position": [cx / width, cy / height],
            "scale": [scale_x, scale_y],
            "rotation": rotation,
            "color": [r, g, b],
        })

    # Save to JSON
    with open("slic_init.json", "w") as f:
        json.dump(gaussians, f)

    print(f"Generated {len(gaussians)} Gaussians from SLIC")

if __name__ == "__main__":
    import sys
    generate_slic_init(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 100)
```

**Usage**:
```bash
python tools/slic_preprocess.py kodim02.png 500
# Creates: slic_init.json
```

**Rust side** (`slic_init.rs`):
```rust
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct SLICGaussianInit {
    position: [f32; 2],
    scale: [f32; 2],
    rotation: f32,
    color: [f32; 3],
}

pub fn load_slic_init(path: &str) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let json = std::fs::read_to_string(path).unwrap();
    let inits: Vec<SLICGaussianInit> = serde_json::from_str(&json).unwrap();

    inits.iter().map(|init| {
        Gaussian2D::new(
            Vector2::new(init.position[0], init.position[1]),
            Euler::new(init.scale[0], init.scale[1], init.rotation),
            Color4::new(init.color[0], init.color[1], init.color[2], 1.0),
            1.0,
        )
    }).collect()
}
```

---

### Option 2: Python Subprocess (Automatic)

```rust
use std::process::Command;

pub fn initialize_slic_subprocess(
    image_path: &str,
    n_segments: usize,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    // Run Python script
    let output = Command::new("python3")
        .arg("tools/slic_preprocess.py")
        .arg(image_path)
        .arg(n_segments.to_string())
        .output()
        .expect("Failed to run SLIC preprocessing");

    if !output.status.success() {
        panic!("SLIC failed: {:?}", String::from_utf8_lossy(&output.stderr));
    }

    // Load result
    load_slic_init("slic_init.json")
}
```

---

### Option 3: Pure Rust (Future)

Port SLIC algorithm to Rust (complex, ~500 lines):
- Requires: spatial indexing, connected components, iterative refinement
- Benefit: No Python dependency
- Effort: 2-3 days
- **Defer until we validate SLIC helps**

---

## Integration Steps

**Immediate** (use Option 1):
1. Create `tools/slic_preprocess.py`
2. Create `src/slic_init.rs` loader
3. Add `serde` dependency to Cargo.toml
4. Test on Kodak images

**Usage in encoder**:
```rust
// Preprocess (once):
// $ python tools/slic_preprocess.py kodim02.png 500

// In Rust:
let gaussians = load_slic_init("slic_init.json");
// Optimize as normal
```

**Benchmark**:
```rust
let slic = encoder.initialize_slic(500);  // Loads from JSON
let grid = encoder.initialize_gaussians(grid_size_for_500);

// Compare quality after optimization
```

---

## Expected Results

**Research suggests**:
- SLIC respects boundaries → better edge quality
- Perceptually meaningful regions → better visual quality
- **Expected: +1-3 dB** vs grid for same N

**If it works**:
- Add as alternative init strategy
- Consider pure Rust port for production

**If it doesn't**:
- Document why (photo content doesn't match superpixel assumption?)
- Keep as research reference

---

## Files to Create

1. `tools/slic_preprocess.py` - Python script
2. `lgi-encoder-v2/src/slic_init.rs` - JSON loader
3. `examples/test_slic_vs_grid.rs` - Comparison test

**Status**: Documented, ready to implement when needed.

---

*Created: Session 8 (2025-10-07)*
