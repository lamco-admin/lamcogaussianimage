# Library & Tool Research for LGI v2 Development

**Purpose**: Identify existing libraries and tools that can accelerate development, improve quality, or provide benchmarking capabilities.

**Created**: Session 8 (2025-10-07)

---

## Table of Contents

1. [Image Analysis Libraries](#image-analysis-libraries)
2. [Optimization Frameworks](#optimization-frameworks)
3. [Benchmarking & Metrics](#benchmarking--metrics)
4. [Codec Research & Reference](#codec-research--reference)
5. [Computer Vision Tools](#computer-vision-tools)
6. [Recommended Integration Plan](#integration-plan)

---

## Image Analysis Libraries

### Rust Native Libraries

#### 1. **`image`** (Already Using)
- **URL**: https://crates.io/crates/image
- **What**: Image I/O and basic manipulation
- **Status**: ✅ Already integrated
- **Use**: Loading PNG/JPEG test images

#### 2. **`imageproc`**
- **URL**: https://crates.io/crates/imageproc
- **What**: Image processing algorithms in Rust
- **Features**:
  - Edge detection (Canny, Sobel)
  - Morphological operations
  - Feature detection (Harris corners, FAST)
  - Template matching
  - Geometric transformations
  - Filtering (Gaussian, median, bilateral)
- **Relevance**:
  - ✅ Edge detection could improve structure tensor
  - ✅ Feature detection for adaptive Gaussian placement
  - ✅ Template matching for texture detection
- **Integration Effort**: Low (pure Rust)
- **Recommendation**: **ADOPT** - Add to dev-dependencies for analysis

#### 3. **`ndarray`**
- **URL**: https://crates.io/crates/ndarray
- **What**: N-dimensional arrays (like NumPy)
- **Features**:
  - Efficient multi-dimensional arrays
  - Linear algebra operations
  - BLAS integration
  - Parallel iterators
- **Relevance**:
  - ✅ Better than Vec<Vec<>> for image processing
  - ✅ Could speed up convolutions, filtering
  - ✅ SIMD-friendly operations
- **Integration Effort**: Medium (refactor ImageBuffer)
- **Recommendation**: **CONSIDER** - For performance-critical code

#### 4. **`nalgebra`** (Similar to what we use)
- **URL**: https://crates.io/crates/nalgebra
- **What**: Linear algebra library
- **Features**:
  - Vectors, matrices, quaternions
  - Decompositions (SVD, QR, Cholesky)
  - Geometry (transforms, rotations)
  - SIMD support
- **Relevance**:
  - ✅ Better Cholesky decomposition than our impl
  - ✅ Optimized matrix operations
- **Status**: We have `lgi-math` custom impl
- **Recommendation**: **BENCHMARK** - Compare performance vs custom code

---

### External Tools (Call via FFI or subprocess)

#### 5. **OpenCV**
- **URL**: https://opencv.org/
- **Rust Binding**: `opencv-rust` (https://crates.io/crates/opencv)
- **Features**:
  - **Image segmentation**: GrabCut, watershed, superpixels
  - **Feature detection**: SIFT, SURF, ORB, AKAZE
  - **Object detection**: Haar cascades, DNN module
  - **Motion analysis**: Optical flow, background subtraction
  - **Computational photography**: HDR, denoising, inpainting
- **Relevance**:
  - ✅ **Superpixel segmentation** - Group pixels for Gaussian init
  - ✅ **Saliency detection** - Identify important regions
  - ✅ **Denoising** - Preprocess noisy images
  - ✅ **DNN module** - Semantic segmentation (sky, face, etc.)
- **Integration Effort**: High (large dependency, C++ FFI)
- **Recommendation**: **USE FOR ANALYSIS** - Call from Python scripts, not runtime

#### 6. **scikit-image** (Python)
- **URL**: https://scikit-image.org/
- **Features**:
  - Region properties and labeling
  - Morphological operations
  - Texture analysis (Haralick, LBP, GLCM)
  - Segmentation (SLIC, Felzenszwalb, watershed)
  - Perceptual metrics (SSIM, MS-SSIM)
- **Relevance**:
  - ✅ **SLIC superpixels** - Better than grid initialization
  - ✅ **Texture analysis** - Detect where to use textures
  - ✅ **SSIM implementation** - Verify our MS-SSIM
- **Integration Effort**: Medium (Python interop)
- **Recommendation**: **USE FOR PROTOTYPING** - Test ideas before Rust impl

#### 7. **Detectron2 / Segment Anything (Meta)**
- **URL**: https://github.com/facebookresearch/detectron2
- **URL**: https://segment-anything.com/
- **Features**:
  - State-of-art semantic segmentation
  - Instance segmentation
  - Panoptic segmentation
  - Object detection and classification
- **Relevance**:
  - ✅ **Semantic regions** - "This is sky, allocate fewer Gaussians"
  - ✅ **Instance masks** - Per-object encoding strategies
  - ✅ **Saliency proxy** - Important objects get more resources
- **Integration Effort**: Very High (PyTorch, GPU required)
- **Recommendation**: **RESEARCH ONLY** - Use for dataset analysis, not runtime

---

## Optimization Frameworks

### Rust Optimization Libraries

#### 8. **`argmin`**
- **URL**: https://crates.io/crates/argmin
- **What**: Mathematical optimization framework
- **Algorithms**:
  - Gradient descent variants (momentum, Nesterov)
  - Conjugate gradient
  - BFGS, L-BFGS
  - Newton methods
  - Trust region methods
  - Simulated annealing
  - Particle swarm
- **Relevance**:
  - ✅ **L-BFGS** - Better than our Adam? Need to test
  - ✅ **Trust region** - Handles constraints naturally
  - ✅ **Provides line search** - Adaptive learning rates
- **Integration Effort**: Medium (refactor optimizer interface)
- **Recommendation**: **EVALUATE** - Compare L-BFGS vs Adam

#### 9. **`optimization`**
- **URL**: https://crates.io/crates/optimization
- **What**: Numerical optimization algorithms
- **Algorithms**:
  - Nelder-Mead (gradient-free)
  - Levenberg-Marquardt
  - BFGS variants
- **Relevance**:
  - ✅ **Levenberg-Marquardt** - Good for least-squares (MSE loss!)
  - ✅ **Gradient-free** - Could help when gradients are noisy
- **Integration Effort**: Medium
- **Recommendation**: **CONSIDER** - For non-differentiable objectives

#### 10. **`argmin-math`**
- **URL**: Part of argmin ecosystem
- **What**: Math backends for argmin
- **Features**:
  - Integrates with ndarray, nalgebra
  - Provides common ops for optimization
- **Recommendation**: Use if adopting argmin

---

### Automatic Differentiation

#### 11. **`autograd`** (Rust)
- **URL**: https://github.com/raskr/rust-autograd
- **What**: Automatic differentiation library
- **Features**:
  - Define computation graphs
  - Automatic gradient computation
  - GPU support (experimental)
- **Relevance**:
  - ✅ **Auto gradients** - No manual derivative math!
  - ✅ **Easier to add loss functions** - Just write forward pass
  - ⚠️ **Overhead** - Slower than analytical gradients
- **Integration Effort**: High (major refactor)
- **Recommendation**: **RESEARCH** - Compare speed vs manual gradients

#### 12. **PyTorch / JAX** (Python)
- **What**: Deep learning frameworks with autograd
- **Relevance**:
  - ✅ **Prototyping** - Test new loss functions quickly
  - ✅ **GPU acceleration** - Mature CUDA support
  - ✅ **Pre-trained models** - For semantic analysis
- **Integration Effort**: Very High (Python dependency)
- **Recommendation**: **USE FOR RESEARCH** - Prototype, then port to Rust

---

## Benchmarking & Metrics

### Image Quality Metrics

#### 13. **`image-quality-tools`** (External)
- **Options**:
  - `libvmaf` (Netflix's VMAF metric)
  - `butteraugli` (Google's perceptual metric)
  - `ssimulacra2` (Cloudinary's perceptual metric)
- **Features**:
  - Perceptual image quality assessment
  - Correlate better with human judgment than PSNR
  - Widely used in codec evaluation
- **Relevance**:
  - ✅ **Better than PSNR** - VMAF correlates with human perception
  - ✅ **Industry standard** - Compare against JPEG, WebP, AVIF
- **Integration**:
  - Call as subprocess: `vmaf -r reference.png -d distorted.png`
  - Parse output for score
- **Recommendation**: **ADOPT** - Add to benchmark suite

#### 14. **SSIM / MS-SSIM**
- **Our Implementation**: `lgi-core/src/ms_ssim_loss.rs`
- **Reference**: `scikit-image.metrics.structural_similarity`
- **To Do**:
  - ✅ Verify our implementation matches scikit-image
  - ✅ Add unit tests with known inputs/outputs
  - ✅ Compare our MS-SSIM vs VMAF scores

#### 15. **LPIPS** (Learned Perceptual Image Patch Similarity)
- **URL**: https://github.com/richzhang/PerceptualSimilarity
- **What**: Deep learning-based perceptual metric
- **Features**:
  - Uses AlexNet/VGG features
  - Correlates very well with human judgment
  - Better than SSIM for textures
- **Relevance**:
  - ✅ **Best perceptual metric** - State-of-art
  - ⚠️ **Requires neural network** - Not pure Rust
- **Integration Effort**: Very High
- **Recommendation**: **USE FOR ANALYSIS** - Benchmark quality, not runtime

---

### Codec Benchmarking Frameworks

#### 16. **`rd_tool`** (AV1 rate-distortion testing)
- **URL**: https://github.com/xiph/rd_tool
- **What**: Automated codec comparison framework
- **Features**:
  - Test multiple codecs (JPEG, WebP, AVIF, etc.)
  - Generate R-D curves automatically
  - Statistical analysis
  - HTML reports
- **Relevance**:
  - ✅ **Compare LGI vs industry codecs**
  - ✅ **Automated testing** - No manual work
  - ✅ **Standard datasets** - Kodak, CLIC, etc.
- **Integration**:
  - Implement LGI encoder wrapper
  - Register with rd_tool
  - Run comparison suite
- **Recommendation**: **ADOPT** - Essential for codec validation

#### 17. **JPEG XL Benchmark Scripts**
- **URL**: https://github.com/libjxl/libjxl/tree/main/tools/benchmark
- **What**: Comprehensive codec comparison suite
- **Features**:
  - Tests on standard datasets
  - Multiple quality metrics (PSNR, SSIM, Butteraugli)
  - Generates comparison charts
- **Relevance**:
  - ✅ **Learn from mature codec** - JPEG XL has similar goals
  - ✅ **Reuse test infrastructure**
- **Recommendation**: **STUDY** - Adapt methodology

---

## Codec Research & Reference

### Similar/Related Codecs to Study

#### 18. **JPEG XL**
- **URL**: https://jpeg.org/jpegxl/
- **Repo**: https://github.com/libjxl/libjxl
- **Techniques Used**:
  - Adaptive block sizes
  - Adaptive prediction
  - Context modeling
  - Spline-based gradients
  - Modular mode for lossless
- **What to Learn**:
  - ✅ **Adaptive strategies** - How they select block sizes
  - ✅ **Rate-distortion optimization** - Lambda selection
  - ✅ **Perceptual tuning** - Psychovisual modeling
- **Recommendation**: **STUDY CODEBASE** - Lots of lessons

#### 19. **BPG** (Better Portable Graphics)
- **URL**: https://bellard.org/bpg/
- **What**: HEVC intra-frame coding for images
- **Techniques**:
  - Transform coding (DCT/DST)
  - Intra prediction modes
  - Deblocking filter
  - Quantization matrices
- **What to Learn**:
  - ✅ **Prediction modes** - Multiple strategies per block
  - ✅ **Quantization** - How to trade quality/size
- **Recommendation**: **REFERENCE** - Different approach, but proven

#### 20. **Neural Image Compression**
- **Papers**:
  - "Variational Image Compression with a Scale Hyperprior" (Ballé et al.)
  - "Learned Image Compression with Discretized Gaussian Mixture Likelihoods"
- **Implementations**:
  - CompressAI: https://github.com/InterDigitalInc/CompressAI
  - Tensorflow Compression: https://github.com/tensorflow/compression
- **What to Learn**:
  - ✅ **Rate-distortion optimization** - Lagrangian approach
  - ✅ **Entropy modeling** - Better than arithmetic coding?
  - ⚠️ **Requires neural network** - Not our approach
- **Recommendation**: **READ PAPERS** - Learn theory, not implementation

---

## Computer Vision Tools

### Spatial Data Structures

#### 21. **`rstar`**
- **URL**: https://crates.io/crates/rstar
- **What**: R*-tree spatial indexing
- **Features**:
  - Fast nearest-neighbor queries
  - Range queries
  - Spatial indexing in 2D/3D
- **Relevance**:
  - ✅ **Gaussian culling** - Only check nearby Gaussians per pixel
  - ✅ **Expected 10-100× speedup** for large N
  - ✅ **Pure Rust, well-maintained**
- **Integration Effort**: Medium
- **Recommendation**: **HIGH PRIORITY** - Implement ASAP

#### 22. **`kdtree`**
- **URL**: https://crates.io/crates/kdtree
- **What**: k-d tree implementation
- **Features**:
  - Fast spatial queries
  - Simpler than R-tree
  - Good for point data
- **Relevance**:
  - ✅ **Alternative to R-tree** - Simpler, maybe faster for our use case
  - ✅ **Hotspot finding** - Quickly find highest-error regions
- **Integration Effort**: Low
- **Recommendation**: **BENCHMARK** - Compare vs rstar

---

### Superpixel Segmentation

#### 23. **SLIC** (Simple Linear Iterative Clustering)
- **Reference**: http://ivrl.epfl.ch/research/superpixels
- **Available In**:
  - scikit-image: `skimage.segmentation.slic`
  - OpenCV: `cv2.ximgproc.createSuperpixelSLIC`
- **What**: Clusters pixels into perceptually meaningful regions
- **Relevance**:
  - ✅ **Better initialization** - One Gaussian per superpixel
  - ✅ **Adaptive N** - More superpixels in complex regions
  - ✅ **Fast** - Linear time complexity
- **Integration**:
  - Call from Python/C++ for preprocessing
  - Use superpixel centers as Gaussian positions
  - Fit Gaussian to superpixel statistics
- **Recommendation**: **PROTOTYPE** - Test if better than grid init

#### 24. **Felzenszwalb Segmentation**
- **Paper**: "Efficient Graph-Based Image Segmentation"
- **Available In**: scikit-image
- **What**: Graph-based region merging
- **Relevance**:
  - ✅ **Hierarchical segmentation** - Multi-scale regions
  - ✅ **Respects boundaries** - Better than k-means
- **Integration Effort**: Medium
- **Recommendation**: **RESEARCH** - Compare vs SLIC

---

## Recommended Integration Plan

### Phase 1: Low-Hanging Fruit (Week 1)

**Add to Cargo.toml:**
```toml
[dev-dependencies]
imageproc = "0.25"  # Edge detection, features
rstar = "0.12"      # Spatial indexing
```

**Implement:**
1. ✅ **Spatial indexing with rstar**
   - Build R-tree of Gaussian bounding boxes
   - Only check nearby Gaussians per pixel
   - Expected: 10-100× speedup for rendering

2. ✅ **Improved edge detection**
   - Use imageproc::edges::canny
   - Compare vs our structure tensor
   - Use for better initialization

**Integration Effort**: 1-2 days
**Expected Impact**: Massive rendering speedup

---

### Phase 2: Optimization Libraries (Week 2)

**Test Alternatives:**
1. ✅ **argmin L-BFGS**
   - Compare vs Adam optimizer
   - Benchmark on Kodak images
   - Measure quality and time

2. ✅ **Levenberg-Marquardt** (from `optimization` crate)
   - Good fit for least-squares (MSE loss)
   - Test if converges better than Adam

**Integration Effort**: 2-3 days (mostly testing)
**Expected Impact**: Fix loss oscillation issue

---

### Phase 3: Better Metrics (Week 3)

**Add External Metrics:**
1. ✅ **VMAF**
   - Install: `cargo install vmaf-rs` or use subprocess
   - Add to benchmark suite
   - Compare PSNR vs VMAF scores

2. ✅ **Butteraugli**
   - Install from Google's libjxl repo
   - Psychovisual metric
   - See if our MS-SSIM correlates

**Integration Effort**: 2-3 days
**Expected Impact**: Better quality assessment

---

### Phase 4: Advanced Analysis (Week 4+)

**Use External Tools for Research:**
1. ✅ **Superpixel initialization**
   - Python script using scikit-image SLIC
   - Export superpixel centers to JSON
   - Load in Rust for initialization
   - Compare vs grid init

2. ✅ **Semantic segmentation**
   - Run Segment Anything on test images
   - Generate region masks
   - Allocate Gaussians based on region type
   - E.g., sky=few Gaussians, face=many Gaussians

3. ✅ **Texture classification**
   - Use Haralick features or LBP
   - Identify textured vs smooth regions
   - Decide where to use per-primitive textures

**Integration Effort**: 1-2 weeks
**Expected Impact**: Smarter technique selection

---

### Phase 5: Production Polish (Later)

1. Port Python analysis to Rust (if worth it)
2. Integrate codec comparison framework (rd_tool)
3. Add continuous benchmarking
4. Optimize hot paths with SIMD

---

## Immediate Actions

**This Week:**
1. ✅ Add `rstar` and `imageproc` to dev-dependencies
2. ✅ Implement R-tree spatial indexing
3. ✅ Benchmark rendering speedup
4. ✅ Test imageproc Canny edge detection

**Next Week:**
1. Install VMAF for quality metrics
2. Test argmin L-BFGS optimizer
3. Prototype SLIC initialization in Python
4. Document findings

---

## Resources & Links

**Codec Standards:**
- JPEG: https://jpeg.org/
- JPEG XL: https://jpeg.org/jpegxl/
- WebP: https://developers.google.com/speed/webp
- AVIF: https://aomedia.org/av1/

**Image Datasets:**
- Kodak: http://r0k.us/graphics/kodak/
- CLIC: https://www.compression.cc/
- DIV2K: https://data.vision.ee.ethz.ch/cvl/DIV2K/

**Academic Resources:**
- Perception-Based Image Coding: https://www.cns.nyu.edu/~lcv/iqa/
- Image Compression Research: https://github.com/topics/image-compression

**Related Projects:**
- CompressAI: https://github.com/InterDigitalInc/CompressAI
- libjxl: https://github.com/libjxl/libjxl
- Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

---

*End of Library Research - Update as we discover more tools!*
