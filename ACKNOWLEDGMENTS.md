# Acknowledgments & Third-Party Code

This document acknowledges the research, code, and libraries that influenced or are included in the LGI (Lamco Gaussian Image) project.

---

## Research Foundation

### GaussianImage (ECCV 2024)
**Authors**: Xinjie Zhang, Xingtong Ge, et al.
**Paper**: "GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting"
**arXiv**: https://arxiv.org/abs/2403.08551
**Repository**: https://github.com/Xinjie-Q/GaussianImage

**Contribution**: This paper introduced the concept of 2D Gaussian splatting for image compression. Our LGI implementation is **independently written from scratch in Rust** but follows the conceptual approach from this paper:
- 2D Gaussian representation for images
- Alpha-composited rasterization
- Gradient-based optimization

**Note**: We did NOT use their code directly. Our implementation is original Rust code with different architecture, cross-platform GPU support (wgpu vs CUDA), and additional techniques.

---

### Image-GS (SIGGRAPH 2025)
**Authors**: Content-Adaptive 2D Gaussians research team
**Paper**: "Image-GS: Content-Adaptive Image Representation via 2D Gaussians"
**Repository**: https://github.com/gs-yao/Image-GS

**Contribution**: Research on content-adaptive Gaussian placement and initialization methods. We researched their approaches for:
- Saliency-based initialization
- Adaptive Gaussian count determination
- Content-aware placement strategies

**Note**: We did NOT include their code. We implemented our own version of these concepts in Rust.

---

### gsplat Library
**Authors**: 1kbirds, nerfstudio-project
**Repository**: https://github.com/nerfstudio-project/gsplat
**License**: Apache 2.0

**Contribution**: General-purpose Gaussian splatting CUDA library. We studied their rasterization approach for understanding 3D Gaussian splatting techniques.

**Status in our project**:
- **NOT included** in our production Rust code (`packages/lgi-rs/`)
- **Included** in `packages/lgi-legacy/image-gs/` (research prototype, kept for reference)
- This is the **original GaussianImage paper's dependency**, NOT ours

---

## Legacy Reference Implementations

### packages/lgi-legacy/image-gs/
**Contents**: Python implementation using the official GaussianImage repository code
**License**: MIT (see `packages/lgi-legacy/image-gs/LICENSE`)
**Purpose**: Research and validation only, **not used in production**
**Status**: Preserved as reference, has own separate .git history

**Third-party code included**:
- gsplat CUDA library (Apache 2.0)
- GLM math library (MIT) - see `packages/lgi-legacy/image-gs/gsplat/gsplat/cuda/csrc/third_party/glm/`

---

### packages/lgi-legacy/image-gs-cpu/
**Contents**: **Original Python CPU implementation by Greg Lamberson**
**License**: MIT/Apache-2.0 (dual, same as main project)
**Purpose**: Pure Python CPU rasterizer for validation
**Status**: 100% original code, no third-party dependencies

---

### packages/lgi-tools/fused-ssim/
**Repository**: https://github.com/Po-Hsun-Su/pytorch-ssim
**License**: MIT
**Authors**: Po-Hsun Su
**Purpose**: MS-SSIM (Multi-Scale Structural Similarity) quality metric

**Status**:
- Included for perceptual quality measurement
- Used for validation and benchmarking
- Not required for core codec functionality

---

## Production Code (`packages/lgi-rs/`)

### 100% Original Implementation
**All code in `packages/lgi-rs/` is original work by Lamco Development (Greg Lamberson)**

**Original Rust modules**:
- `lgi-math/` - Math primitives, Gaussian types
- `lgi-core/` - Rendering engine, initialization
- `lgi-encoder/` - Optimization algorithms
- `lgi-encoder-v2/` - Advanced optimization (+8dB quality)
- `lgi-format/` - File I/O, serialization
- `lgi-gpu/` - wgpu-based GPU acceleration
- `lgi-pyramid/` - Multi-level zoom
- `lgi-cli/` - Command-line tools
- `lgi-viewer/` - GUI (Slint framework)
- `lgi-benchmarks/` - Testing suite
- `lgi-ffi/` - C FFI for ecosystem
- `lgi-wasm/` - WebAssembly build

**Architecture**: Completely independent from GaussianImage paper's code
- Cross-platform GPU via wgpu (not CUDA)
- Rust workspace structure
- Original two-track strategy
- 150 mathematical techniques (many original)

---

## Rust Dependencies (Cargo.toml)

All Rust crates used are standard open-source libraries with permissive licenses:

### Math & Numerics
- `glam` (MIT/Apache-2.0) - Vector/matrix math
- `nalgebra` (Apache-2.0) - Linear algebra
- `wide` (Zlib/Apache-2.0) - SIMD operations

### Graphics
- `wgpu` (MIT/Apache-2.0) - Cross-platform GPU
- `image` (MIT) - Image loading/saving
- `png`, `jpeg-decoder` (MIT/Apache-2.0) - Format support

### UI
- `slint` (GPL-3.0 OR LicenseRef-Slint-Royalty-free-1.1) - GUI framework

### Utilities
- `serde` (MIT/Apache-2.0) - Serialization
- `rayon` (MIT/Apache-2.0) - Parallelism
- `thiserror` (MIT/Apache-2.0) - Error handling

**Full dependency list**: See `packages/lgi-rs/Cargo.toml` and individual crate Cargo.toml files

---

## Conceptual Influence vs Code Usage

### Concepts & Algorithms (Research Papers)
**Used conceptually, implemented independently**:
- 2D Gaussian splatting for images (GaussianImage paper)
- Alpha-composited rendering (GaussianImage, Image-GS)
- Gradient-based optimization (standard technique)
- Content-adaptive initialization (Image-GS paper)
- Quantization-aware training (GaussianImage paper)

### No Code Copied
**We did NOT**:
- Copy code from GaussianImage repository
- Copy code from Image-GS repository
- Copy code from gsplat library (for production)
- Use their CUDA implementations

**We DID**:
- Read and understand their papers
- Implement concepts from scratch in Rust
- Create original architecture and techniques
- Add 150+ techniques (many original contributions)

---

## What's In Each Directory

### Production Code (GitHub repo)
```
packages/lgi-rs/          ← 100% original Rust code by Lamco
packages/lgi-legacy/      ← Reference implementations (properly licensed)
  ├── image-gs/           ← GaussianImage paper code (MIT, for reference)
  └── image-gs-cpu/       ← 100% original Python by Lamco
packages/lgi-tools/       ← Third-party tools (properly licensed)
  └── fused-ssim/         ← pytorch-ssim (MIT)
```

### Not in GitHub
```
local-research/           ← Excluded from repository
  ├── All research notes and session logs (private)
  └── Not distributed
```

---

## Attribution Statement

**LGI (Lamco Gaussian Image) Codec** is an **original implementation** by Lamco Development (Greg Lamberson) inspired by academic research on 2D Gaussian splatting.

**Key Differences from Prior Work**:
1. **Language**: Rust (papers used Python+CUDA)
2. **GPU**: wgpu/Vulkan/DX12/Metal (papers used CUDA only)
3. **Architecture**: Original two-track strategy, 150 techniques
4. **Platform**: Cross-platform including WebAssembly
5. **Scope**: Comprehensive codec + ecosystem integration

**Academic Inspiration**: GaussianImage (Zhang et al., ECCV 2024), Image-GS (SIGGRAPH 2025)

**Code**: 100% original implementation (production Rust code)

**Legacy Code**: Reference implementations properly preserved with original licenses

---

## License Compliance

### Our License
**LGI production code**: Dual MIT OR Apache-2.0

### Third-Party Licenses
- **image-gs (legacy)**: MIT ✅ Compatible
- **gsplat (in legacy)**: Apache 2.0 ✅ Compatible
- **fused-ssim**: MIT ✅ Compatible
- **Rust crates**: Various (MIT, Apache-2.0, Zlib) ✅ All compatible

**All third-party code properly attributed and licensed.**

---

## Contributing

If you find any attribution issues or missing acknowledgments, please file an issue at:
https://github.com/lamco-admin/lamcogaussianimage/issues

---

**Last Updated**: November 14, 2025
**Prepared By**: Greg Lamberson (Lamco Development)
**Purpose**: Transparency and proper attribution of research and code sources
