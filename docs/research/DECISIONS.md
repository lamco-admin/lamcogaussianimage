# LGI Project Decisions Log

**Purpose**: Record of all major architectural, strategic, and technical decisions with full context and rationale

**Format**: Each decision documented with context, options, choice, rationale, trade-offs, and outcome

**Period**: September-October 2024

---

## Table of Contents

- [Technology Decisions](#technology-decisions)
- [Architecture Decisions](#architecture-decisions)
- [Strategy Decisions](#strategy-decisions)
- [Implementation Decisions](#implementation-decisions)
- [Format Decisions](#format-decisions)

---

## Technology Decisions

### D-TECH-001: Rust Implementation

**Date**: October 2, 2024 (Session 1)
**Decision ID**: D-TECH-001
**Status**: ✅ Implemented, Validated

**Context**:
- Python prototype working but slow (26 FPS CPU)
- Need production-ready performance
- Cross-platform requirement (Linux/Windows/macOS/Web)
- Integration with C ecosystem (FFmpeg, ImageMagick)

**Options Considered**:

1. **Python + optimization**
   - Pros: Existing code, fast prototyping
   - Cons: GIL limits parallelism, slow even with Cython

2. **C++**
   - Pros: Maximum performance, mature ecosystem
   - Cons: Memory safety issues, build complexity, no WASM

3. **Rust** ✅
   - Pros: Memory safety, SIMD, rayon, WASM, FFI-friendly
   - Cons: Learning curve, longer initial dev time

4. **Zig**
   - Pros: Simple, comptime, good C interop
   - Cons: Immature ecosystem, fewer libraries

**Decision**: **Rust**

**Rationale**:
- Memory safety critical for codec (buffer handling)
- Excellent SIMD support (`wide` crate)
- WebAssembly first-class citizen
- rayon for effortless parallelism
- Strong type system prevents bugs
- C FFI for ecosystem integration
- Growing multimedia ecosystem

**Trade-offs Accepted**:
- Steeper learning curve
- Longer initial development
- Smaller community than C++/Python
- Build times longer than C

**Validation**:
- `lgi-math`: **59× faster than Python** ✅
- FFI working (FFmpeg, ImageMagick) ✅
- WASM build functional ✅
- Type safety caught bugs early ✅

**Outcome**: **Success** - Right choice, no regrets

**Related Decisions**: D-TECH-002 (wgpu), D-TECH-004 (workspace structure)

---

### D-TECH-002: wgpu over CUDA

**Date**: October 3, 2024 (Session 3)
**Decision ID**: D-TECH-002
**Status**: ✅ Implemented, Validated

**Context**:
- Need GPU acceleration for rendering
- GaussianImage paper used CUDA
- Requirement: Cross-platform, vendor-agnostic
- Target platforms: Windows (DX12), Linux (Vulkan), macOS (Metal), Web (WebGPU)

**Options Considered**:

1. **CUDA** (Official GaussianImage approach)
   - Pros: Proven (1500 FPS in paper), mature, excellent docs
   - Cons: **NVIDIA-only**, no AMD/Intel, no WebGPU, Linux-centric

2. **Vulkan compute**
   - Pros: Cross-platform, low-level control
   - Cons: Verbose, complex, no DX12/Metal abstraction

3. **wgpu** ✅
   - Pros: **Rust-native**, backends for all platforms, WebGPU compatible
   - Cons: Newer API, less mature, some performance overhead vs raw Vulkan

4. **OpenCL**
   - Pros: Cross-vendor support
   - Cons: Deprecated on macOS, less modern

**Decision**: **wgpu v27**

**Rationale**:
- **Vendor-agnostic**: Intel, AMD, NVIDIA all supported
- **Platform coverage**:
  - Windows: DirectX 12
  - Linux: Vulkan
  - macOS: Metal
  - Web: WebGPU
- **Rust integration**: Native, type-safe, async-ready
- **Future-proof**: WebGPU is the web standard
- **Single codebase**: One shader (WGSL) for all platforms

**Trade-offs Accepted**:
- 5-10% performance vs raw CUDA (acceptable)
- Newer API (less Stack Overflow answers)
- Abstraction layer overhead (minimal)

**Validation** (RTX 4060 via Vulkan):
```
Performance: 1,176 FPS @ 1080p ✅
Correctness: Bit-for-bit match with CPU ✅
Backends:    Vulkan, DX12, Metal all tested ✅
WebGPU:      Build successful ✅
```

**Unexpected Benefit**: No CUDA install needed → easier deployment

**Outcome**: **Major Success**
- Proved cross-platform GPU viable
- 61× speedup over CPU ✅
- No vendor lock-in ✅

**Related Decisions**: D-ARCH-001 (global GPU manager)

---

### D-TECH-003: Slint UI Framework

**Date**: October 3, 2024 (Session 2)
**Decision ID**: D-TECH-003
**Status**: ✅ Implemented (viewer functional)

**Context**:
- Need GUI viewer for LGI files
- Interactive zoom, export, encoding
- Cross-platform requirement

**Options Considered**:

1. **egui** (immediate mode)
   - Pros: Simple, Rust-native, good for tools
   - Cons: Immediate mode (repaint every frame), GPU overhead

2. **Slint** ✅
   - Pros: Declarative, efficient, native feel, good docs
   - Cons: Newer, smaller ecosystem

3. **iced**
   - Pros: Elm architecture, pure functional
   - Cons: Less mature, verbose

4. **Qt (qml-rust)**
   - Pros: Mature, comprehensive
   - Cons: Large dependency, C++ interop complexity

**Decision**: **Slint**

**Rationale**:
- Declarative UI (QML-like) easy to reason about
- Efficient retained-mode rendering
- Native feel on each platform
- Excellent documentation
- Active development

**Trade-offs**:
- Thread safety constraints (main thread only)
- Async encoding tricky (had to work around)

**Outcome**: **Success** - Viewer works well (843 LOC)

**Known Issue**: Encoding blocks UI (design limitation, not critical)

---

### D-TECH-004: Workspace Structure (Cargo)

**Date**: October 2, 2024 (Session 1)
**Decision ID**: D-TECH-004
**Status**: ✅ Implemented

**Context**:
- Multiple related crates needed
- Code organization critical
- Shared dependencies

**Decision**: **Cargo workspace with 12 crates**

**Structure**:
```
lgi-rs/ (workspace root)
├── lgi-math/       # Math primitives (foundation)
├── lgi-core/       # Rendering, initialization
├── lgi-encoder/    # Optimization (v1)
├── lgi-encoder-v2/ # Optimization (v2, active)
├── lgi-format/     # File I/O
├── lgi-gpu/        # GPU acceleration
├── lgi-pyramid/    # Multi-level zoom
├── lgi-cli/        # Command-line tools
├── lgi-benchmarks/ # Testing
├── lgi-ffi/        # C FFI
├── lgi-viewer/     # GUI
└── lgi-wasm/       # WebAssembly
```

**Rationale**:
- **Separation of concerns**: Each crate has clear purpose
- **Dependency management**: Shared via workspace
- **Incremental compilation**: Only rebuild changed crates
- **Independent versioning**: Can evolve separately
- **Clear boundaries**: Prevents tight coupling

**Guidelines**:
- **lgi-math**: No dependencies, pure algorithms
- **lgi-core**: Depends on lgi-math only
- **lgi-encoder**: Depends on lgi-core
- **lgi-gpu**: Optional feature, independent
- **lgi-ffi**: C-compatible, minimal deps

**Outcome**: **Excellent organization**
- Easy to navigate
- Clear dependency graph
- Facilitates testing
- Good for documentation

---

## Architecture Decisions

### D-ARCH-001: Global GPU Manager (Singleton)

**Date**: October 3, 2024 (Session 3)
**Decision ID**: D-ARCH-001
**Status**: ✅ Implemented

**Context**:
- Discovered crash: Multiple GPU instance creation
- wgpu doesn't support multiple device instances
- Viewer, encoder, FFmpeg integration all need GPU
- Shared resource management needed

**Problem**:
```
Viewer creates GPU instance → OK
Encoder creates GPU instance → Crash!
"wgpu device can only be created once"
```

**Options**:

1. **Pass GPU instance around**
   - Pros: Explicit ownership
   - Cons: Lifetime complexity, API pollution

2. **Global singleton** ✅
   - Pros: Simple API, one initialization point
   - Cons: Global state, potential contention

3. **Thread-local**
   - Pros: Thread-safe
   - Cons: Can't share across threads

**Decision**: **Global GPU Manager (Singleton)**

**Implementation**: `lgi-gpu/src/manager.rs` (71 lines)

```rust
pub struct GpuManager {
    instance: OnceCell<Arc<GpuContext>>,
}

impl GpuManager {
    pub fn global() -> &'static GpuManager { /* ... */ }

    pub fn initialize(&self) -> Result<()> {
        self.instance.get_or_try_init(|| {
            // Create wgpu instance once
        })
    }

    pub fn render(&self, gaussians: &[Gaussian2D]) -> Result<Image> {
        let ctx = self.instance.get().ok_or(NotInitialized)?;
        ctx.render(gaussians)
    }
}
```

**API Usage**:
```rust
// Application startup
GpuManager::global().initialize()?;

// Anywhere in code
let image = GpuManager::global().render(&gaussians)?;
```

**Rationale**:
- **One initialization**: App startup, explicit
- **Simple API**: No passing references everywhere
- **Thread-safe**: Arc + OnceCell
- **Lazy init**: Only create if GPU needed

**Trade-offs**:
- Global state (not ideal, but pragmatic)
- Can't have multiple GPU contexts (not needed)

**Outcome**: **Solved the crash** ✅

**Related Decisions**: D-TECH-002 (wgpu choice)

---

### D-ARCH-002: Renderer Traits

**Date**: October 2, 2024 (Session 1)
**Decision ID**: D-ARCH-002
**Status**: ✅ Implemented

**Context**:
- Need both CPU and GPU renderers
- May add more backends (CUDA, Metal-specific, etc.)
- API should be backend-agnostic

**Decision**: **Trait-based renderer abstraction**

**Interface**:
```rust
pub trait Renderer {
    fn render(
        &self,
        gaussians: &[Gaussian2D],
        width: u32,
        height: u32,
    ) -> Result<Image>;

    fn render_region(
        &self,
        gaussians: &[Gaussian2D],
        region: Rect,
    ) -> Result<Image>;
}

// Implementations
impl Renderer for CpuRenderer { /* ... */ }
impl Renderer for GpuRenderer { /* ... */ }
```

**Rationale**:
- **Polymorphism**: Swap renderers at runtime
- **Testing**: Mock renderer for unit tests
- **Fallback**: GPU fails → fall back to CPU
- **Benchmarking**: Compare CPU vs GPU easily

**Usage**:
```rust
let renderer: Box<dyn Renderer> = if gpu_available {
    Box::new(GpuRenderer::new()?)
} else {
    Box::new(CpuRenderer::new())
};

let image = renderer.render(&gaussians, 1920, 1080)?;
```

**Outcome**: **Clean abstraction** ✅
- Easy to add new backends
- Testing simplified
- No backend-specific code in high-level API

---

### D-ARCH-003: Error Handling Strategy

**Date**: October 2, 2024 (Session 1)
**Decision ID**: D-ARCH-003
**Status**: ✅ Implemented

**Context**:
- Codec has many error modes (file I/O, GPU, optimization)
- Need clear error reporting
- FFI boundary needs C-compatible errors

**Decision**: **`thiserror` for Rust, error codes for FFI**

**Rust Side**:
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LgiError {
    #[error("Invalid LGI file: {0}")]
    InvalidFormat(String),

    #[error("GPU initialization failed: {0}")]
    GpuError(#[from] wgpu::Error),

    #[error("Optimization failed after {iters} iterations")]
    OptimizationFailed { iters: usize },

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, LgiError>;
```

**FFI Side** (C-compatible):
```c
typedef enum {
    LGI_OK = 0,
    LGI_ERROR_INVALID_FORMAT = 1,
    LGI_ERROR_GPU_FAILED = 2,
    LGI_ERROR_IO = 3,
    // ...
} lgi_error_code_t;

// Last error accessible
const char* lgi_last_error_message();
```

**Rationale**:
- **Rust**: Idiomatic Result<T, E> with rich errors
- **FFI**: Integer codes (C ABI compatible)
- **Debugging**: Full error chain in Rust, message string for C

**Outcome**: **Works well** ✅
- FFmpeg/ImageMagick integration smooth
- Error messages helpful during debugging

---

## Strategy Decisions

### D-STRAT-001: Two-Track Development Strategy

**Date**: October 6, 2024 (Session 6)
**Decision ID**: D-STRAT-001
**Status**: ✅ Approved, Active
**User Approval**: ✅ Explicit approval given

**Context**:
- 150 mathematical techniques identified
- 20 format specification features needed
- Sequential approach would take 75-150 sessions
- User wants ALL 150 techniques (even if tests don't show benefit)

**Problem**:
- Track 1 (algorithms) and Track 2 (format) blocking each other
- Can't validate compression without format features
- Can't test format without quality algorithms
- Monolithic approach too slow

**Decision**: **Parallel Two-Track Development**

**Track 1: Mathematical Techniques** (150 total)
- Algorithmic enhancements
- Quality improvements
- Optimization methods
- Mathematical toolkit
- **Philosophy**: Build comprehensive foundation, all 150 matter

**Track 2: Format Specification Features** (20 total)
- File format capabilities
- Quantization profiles
- Compression pipeline
- Progressive rendering
- **Philosophy**: Production requirements

**Priority System**:
- **P0**: Critical (blocks production)
- **P1**: High value (15 techniques Track 1, 6 features Track 2)
- **P2**: Medium value
- **P3**: Enhancement

**Rationale**:
- **Independent workstreams**: Can progress in parallel
- **Faster to results**: P1s achievable in reasonable time
- **Can merge later**: Techniques + format = complete codec
- **User mandate**: "Implement ALL 150" requires systematic approach

**User's Philosophy** (from session docs):
> "Even if tests don't show benefit, implement ALL 150 techniques. Mathematical foundation matters."

**Validation** (Session 7):
- 4-hour focused session
- All TOP 10 P1 techniques integrated
- **+8 dB quality improvement** ✅
- Proves parallel approach works

**Trade-offs**:
- Need to track two roadmaps
- Integration points must be planned
- Documentation must cover both

**Outcome**: **Major Success** ✅
- Session 7 achieved breakthrough
- Systematic integration > ad-hoc development
- User-approved strategy working

**Related Decisions**: D-STRAT-002 (master integration plan)

---

### D-STRAT-002: Master Integration Plan Approach

**Date**: October 6, 2024 (Session 7)
**Decision ID**: D-STRAT-002
**Status**: ✅ Implemented, Validated

**Context**:
- After Session 6, audited all modules
- Discovery: **22/56 modules production-ready but unused (39%)**
- Bottleneck is **integration**, not implementation

**Problem**:
> "We have all these amazing tools, we're just not using them!"

**Options**:

1. **Continue ad-hoc**: Integrate as needed
   - Cons: Slow, random order, miss opportunities

2. **Big-bang**: Integrate everything at once
   - Cons: Risky, hard to debug, untestable

3. **Master Integration Plan** ✅
   - Pros: Systematic, prioritized, validated incrementally

**Decision**: **Master Integration Plan with TOP 10 priorities**

**TOP 10 Selected**:
1. entropy.rs → Auto N selection
2. better_color_init.rs → Gaussian-weighted sampling
3. error_driven.rs → Adaptive refinement
4. adam_optimizer.rs → Fast convergence
5. rate_distortion.rs → Quality/size targeting
6. geodesic_edt.rs → Anti-bleeding
7. ewa_splatting_v2.rs → Alias-free rendering
8. trigger_actions.rs → Complete TODO handlers
9. renderer_gpu.rs → GPU acceleration
10. ms_ssim_loss.rs → Perceptual quality

**Selection Criteria**:
- Production-ready code exists ✅
- High impact on quality/speed
- Independent (can integrate separately)
- Well-tested modules

**Expected Impact**: +5.1-6.4 dB combined, 200-5000× speedup potential

**Execution** (Session 7, 4 hours):
- All 10 integrated ✅
- 400+ lines added to encoder
- 17 new public methods
- Comprehensive benchmarks

**Results**:
```
Sharp edges:     +9.69 dB ✅
Complex patterns: +6.46 dB ✅
Average:         +8.08 dB ✅ (exceeded expectations!)
```

**Rationale**:
- **Systematic > heroic**: Planning beats long coding sessions
- **Integration is hard**: Deserves explicit methodology
- **Prioritization**: Focus on highest impact first
- **Validation**: Benchmark after each integration

**Lesson**:
> "Implementation ≠ Integration. Integration is the bottleneck."

**Outcome**: **Spectacular Success** ✅
- Proved approach works
- +8 dB in one 4-hour session
- Now template for future integrations

**Related Decisions**: D-STRAT-001 (two-track strategy)

---

### D-STRAT-003: Implement All 150 Techniques

**Date**: October 6, 2024 (Session 6)
**Decision ID**: D-STRAT-003
**Status**: ✅ User mandate, in progress (32/150)

**Context**:
- Research identified 150 mathematical techniques
- Some techniques showing minimal benefit in initial tests
- Temptation to skip "low value" techniques

**User's Position**:
> "Even if tests don't show benefit, implement ALL 150 techniques. Mathematical foundation matters."

**Rationale (User's Philosophy)**:
- **Future-proofing**: May help specific content types not yet tested
- **Comprehensive toolkit**: Better to have and not need
- **Research-backed**: All 150 have academic justification
- **Unknown unknowns**: Can't predict all use cases
- **Foundation building**: Not just for immediate metrics

**Examples of "Low Benefit" Techniques That Still Get Implemented**:
- Technique X: +0.1 dB on test images
  - May be +2 dB on medical images
  - May matter for specific artistic content
  - Completeness > shortcuts

**Counter-Arguments** (engineering perspective):
- Code complexity increases
- Maintenance burden
- Test coverage effort
- User may not need 95% of techniques

**User's Rebuttal**:
- This is a **research platform**, not just an app
- Experimental codec exploring new space
- Mathematical rigor > minimal product
- Want comprehensive foundation

**Decision**: **Implement all 150** ✅

**Approach**:
- Track 1 P1: 15 critical (67% done) ← Focus first
- Track 1 P2: 20 high-value (next)
- Track 1 P3: 65 enhancements (later)
- Already implemented: 22 (not yet integrated)

**Progress Tracking**: `UNIFIED_PRIORITY_ROADMAP.md`

**Outcome**: **In Progress**
- 32/150 integrated (21%)
- User-approved philosophy
- Systematic roadmap exists

**Related Decisions**: D-STRAT-001 (two-track), D-STRAT-002 (integration plan)

---

## Implementation Decisions

### D-IMPL-001: Conditional Anisotropy

**Date**: October 5, 2024 (Session 5 - corrected understanding)
**Decision ID**: D-IMPL-001
**Status**: ✅ Implemented

**Context**:
- Session 4 incorrectly concluded "isotropic better than anisotropic"
- Debugging revealed misunderstanding
- Both needed, conditionally

**Early Misunderstanding**:
> "Isotropic is better than anisotropic" ❌ WRONG

**Corrected Understanding**:
> "BOTH are essential, applied conditionally based on content" ✅

**Decision**: **Conditional application based on coherence**

**Implementation**:
```rust
// Compute local structure tensor
let coherence = compute_coherence(gradient_x, gradient_y);

let (σ_perp, σ_para) = if coherence < 0.2 {
    // Flat regions → ISOTROPIC
    // No dominant direction
    (σ_base, σ_base)
} else {
    // Structured regions → ANISOTROPIC
    // Align with gradient direction
    (σ_base * 0.5, σ_base * 2.0)
};
```

**Coherence Metric** (from structure tensor):
```
coherence = (λ₁ - λ₂) / (λ₁ + λ₂)

where λ₁, λ₂ are eigenvalues of structure tensor

coherence ≈ 0: Isotropic (flat region, noise)
coherence ≈ 1: Anisotropic (strong edge)
```

**Rationale**:
- **Flat regions**: Isotropic works best (no edge to align to)
- **Edges**: Anisotropic essential (align along edge)
- **Content-adaptive**: Let image dictate technique
- **No universal winner**: Context determines choice

**Validation**:
```
Isotropic only:      20.6 dB
Anisotropic only:    27.6 dB (+7.0 dB, but artifacts on flat)
Conditional:         29.5 dB (+8.9 dB) ✅ WINNER
```

**Lesson**:
> "Beware either/or thinking. Build toolkit, use conditionally."

**Outcome**: **Critical insight**
- Content-adaptive > one-size-fits-all
- Both techniques essential
- Now principle applied throughout codec

---

### D-IMPL-002: Euler Angles vs Rotation Matrices

**Date**: October 2, 2024 (Session 1)
**Decision ID**: D-IMPL-002
**Status**: ✅ Implemented

**Context**:
- Need to represent 2D rotation for anisotropic Gaussians
- Storage size matters (file format)
- Gradient optimization considerations

**Options**:

1. **2×2 Rotation Matrix**
   ```rust
   struct Rotation {
       m00: f32, m01: f32,
       m10: f32, m11: f32,
   }
   ```
   - Storage: 4 floats = 16 bytes
   - Rotation: 4 muls + 2 adds (fast)
   - Gradient: 4 parameters to optimize

2. **Euler Angle (single θ)** ✅
   ```rust
   struct Rotation {
       angle: f32,  // radians
   }
   ```
   - Storage: 1 float = 4 bytes ✅ (4× smaller)
   - Rotation: 2 sin/cos + 4 muls + 2 adds
   - Gradient: 1 parameter ✅ (simpler optimization)

3. **Quaternion**
   - Overkill for 2D
   - 4 floats (no benefit over matrix)

**Decision**: **Euler Angle**

**Rationale**:
- **4× storage reduction** (16 → 4 bytes per Gaussian)
- **Simpler optimization** (1 parameter vs 4)
- **Better compressibility** (single value quantizes better)
- **Sufficient precision** (f32 angle has 23-bit mantissa)

**Trade-offs**:
- Slower rotation (sin/cos vs muls)
- But: Rotation is small part of total time

**Benchmark** (1M rotations):
```
Matrix:  12ms
Euler:   18ms (1.5× slower, acceptable)
```

**File Format Impact**:
```
1000 Gaussians with rotation matrix: 16KB
1000 Gaussians with Euler:            4KB (4× smaller) ✅
```

**Outcome**: **Good trade-off** ✅
- Storage reduction significant
- Performance cost acceptable
- Simpler optimization

---

### D-IMPL-003: Grid Size Calculation

**Date**: October 2-7, Sessions 1-8
**Decision ID**: D-IMPL-003
**Status**: ⏳ Ongoing research

**Context**:
- Need to determine optimal number of Gaussians (N)
- Affects quality, file size, encode time
- Currently using arbitrary formulas

**Evolution of Approaches**:

**Version 1** (Session 1): Linear scaling
```rust
N = γ × (width × height)  // γ = 0.01
```
- Problem: Doesn't scale well with resolution

**Version 2** (Session 2): Square root
```rust
N = (sqrt(width × height) / 20).round()
```
- Problem: Arbitrary constant, no content awareness

**Version 3** (Session 7): Entropy-based ✅
```rust
pub fn adaptive_gaussian_count(image: &ColorSource) -> usize {
    let entropy = compute_entropy(image);
    let base = sqrt(W × H) / 10;
    (base × (1.0 + entropy_factor)).round()
}
```
- Better: Adapts to image complexity
- Issue: Uses 80× more Gaussians than arbitrary

**Version 4** (Session 8): Hybrid (under test)
```rust
N = 0.6 × N_entropy + 0.4 × N_gradient
```
- Balance complexity and edges
- Still experimental

**Current Status**: **No perfect solution yet** ⏳

**Research Questions**:
- Does entropy advantage persist after optimization?
- Is there an optimal N for given target PSNR?
- Can we learn N from content analysis?

**Temporary Decision**: Use **entropy-based** for now
- Best quality (+2.93 dB pre-optimization)
- Trade-off: Encode time (O(N²) optimization)
- Needs empirical R-D curve fitting

**Related Research**: Session 8 ongoing validation

---

## Format Decisions

### D-FMT-001: LGIV Spec Before Implementation

**Date**: October 3, 2024 (Session 3)
**Decision ID**: D-FMT-001
**Status**: ✅ Spec complete (1,027 lines)

**Context**:
- LGI image format working
- Video codec is natural extension
- Question: Spec now or later?

**Arguments Against Specifying Now**:
- Don't need video yet
- Spec may change during implementation
- Premature optimization

**Arguments For Specifying Now** ✅:
- **Influences image format design**
- Example: Keyframes must be valid LGI images
- Example: Metadata fields needed for temporal prediction
- **Architecture coherence**: Avoid retrofitting
- **Vision alignment**: Understand full scope

**Decision**: **Specify LGIV now, implement later**

**Specification Created**: `LGIV_VIDEO_FORMAT_SPECIFICATION.md` (1,027 lines)

**Contents**:
- Temporal prediction
- GOP (Group of Pictures) structure
- Keyframe requirements
- Streaming support (HLS/DASH)
- Container integration (MP4/MKV)
- Random access
- HDR/VR support

**Impact on LGI Image Format**:
- Added frame metadata fields
- Reserved chunk types for temporal info
- Designed for temporal coherence
- Multi-level pyramid (also useful for video)

**Rationale**:
- **Future-proofing**: Image format won't need breaking changes
- **Clean design**: Full context when designing
- **Vision document**: Aligns team on goals
- **Low cost**: Spec is just text, no code commitment

**Trade-offs**:
- Spec may evolve during implementation
- Some details premature
- Effort spent on "future work"

**Outcome**: **No regrets** ✅
- Image format cleaner because of it
- No architectural debt
- Clear vision

**Related Decisions**: D-FMT-002 (chunk-based format)

---

### D-FMT-002: Chunk-Based Binary Format

**Date**: October 2, 2024 (Session 1)
**Decision ID**: D-FMT-002
**Status**: ✅ Implemented

**Context**:
- Need file format for LGI codec
- Requirements: Extensible, streamable, backward-compatible

**Options**:

1. **JSON** (text-based)
   - Pros: Human-readable, easy debugging
   - Cons: Large files, slow parsing, no streaming

2. **Protocol Buffers**
   - Pros: Compact, schema validation
   - Cons: Requires .proto, not streamable

3. **Custom binary (flat)**
   - Pros: Simple, fast
   - Cons: Not extensible, version hell

4. **Chunk-based (PNG/MP4 style)** ✅
   - Pros: Extensible, streamable, skip unknown chunks
   - Cons: Slightly more complex

**Decision**: **Chunk-based binary format**

**Structure**:
```
LGI File Format:
[Magic: "LGI\0"] (4 bytes)
[Version: u32]   (4 bytes)
[Chunk]*

Chunk:
[Type: u32]      (4 bytes, e.g., 'GAUS', 'META', 'DATA')
[Length: u32]    (4 bytes)
[Data: bytes]    (length bytes)
[CRC32: u32]     (4 bytes)
```

**Chunk Types**:
- `GAUS`: Gaussian parameters
- `META`: Image metadata (width, height, color space)
- `ATTR`: Custom attributes
- `DATA`: Compressed data (zstd)
- `PYRD`: Pyramid levels
- `TEMP`: Temporal info (for LGIV video)

**Rationale**:
- **Extensibility**: New chunk types without breaking old readers
- **Streaming**: Can parse incrementally
- **Skippable**: Unknown chunks safely ignored
- **Validation**: CRC32 per chunk
- **Standard pattern**: PNG, MP4, MKV all use this

**Backward Compatibility**:
```rust
// Reader ignores unknown chunks
match chunk_type {
    b"GAUS" => parse_gaussians(),
    b"META" => parse_metadata(),
    b"TEMP" => parse_temporal(),  // New in v2
    _ => skip_chunk(),  // Unknown? Skip it ✅
}
```

**Outcome**: **Flexible format** ✅
- Easy to extend
- Future LGIV video compatible
- Tools can add custom chunks

---

### D-FMT-003: Float32 vs Float16 for Storage

**Date**: October 4, 2024 (Session 4)
**Decision ID**: D-FMT-003
**Status**: ✅ Float32 chosen

**Context**:
- Gaussian parameters stored as floats
- File size vs precision trade-off
- Quantization considerations

**Options**:

1. **Float32** ✅
   - Storage: 4 bytes per value
   - Precision: 23-bit mantissa (7 decimal digits)
   - Range: ±3.4×10³⁸

2. **Float16** (half precision)
   - Storage: 2 bytes per value (50% smaller)
   - Precision: 10-bit mantissa (3 decimal digits)
   - Range: ±65,504

**Test**: Encode with f32, quantize to f16, measure quality loss

**Results**:
```
Original (f32):      28.3 dB
Quantized (f16):     23.1 dB (-5.2 dB) ❌ Unacceptable
Quantized (f16) optimized: 26.4 dB (-1.9 dB) ⚠️ Marginal
```

**Analysis**:
- Position precision: f16 sufficient (image space 0-1)
- Color precision: f16 causes banding
- Scale precision: f16 too coarse for small Gaussians

**Decision**: **Float32 with quantization layer**

**Implementation**:
- Store as f32 internally
- Optional quantization profiles (LGIQ-B/S/H/X)
- Vector quantization + zstd for compression
- Not direct f16 conversion

**Rationale**:
- f32 precision needed for quality
- Compression via vector quantization better than f16
- Can add f16 profile later if needed

**Outcome**: **Better approach found** ✅
- VQ + zstd gives better size/quality than f16
- Maintained precision where needed

---

## Appendix: Decision Template

**For Future Decisions**:

```markdown
### D-[CATEGORY]-[NUMBER]: [Decision Title]

**Date**: [Date]
**Decision ID**: D-[CATEGORY]-[NUMBER]
**Status**: [Proposed / Approved / Implemented / Validated / Rejected]

**Context**:
[What situation led to this decision?]

**Options Considered**:
1. Option A
   - Pros: ...
   - Cons: ...
2. Option B ✅
   - Pros: ...
   - Cons: ...

**Decision**: [Chosen option]

**Rationale**:
[Why was this chosen? Key factors?]

**Trade-offs Accepted**:
[What downsides accepted?]

**Validation** (if applicable):
[How was the decision validated?]

**Outcome**: [Success / Failure / Mixed / TBD]

**Related Decisions**: [IDs of related decisions]
```

---

## Conclusion

These decisions represent the **architectural foundation** of the LGI project. Key patterns:

1. **Cross-platform first**: Rust + wgpu for vendor independence
2. **Systematic > heroic**: Planning and prioritization matter
3. **Content-adaptive**: No one-size-fits-all, conditional application
4. **Future-proofing**: LGIV spec now, chunk-based format
5. **User vision**: All 150 techniques, comprehensive foundation

Each decision documented with **full context and rationale** to prevent future confusion and enable informed evolution.

---

**Last Updated**: November 14, 2024
**Total Decisions**: 16 major decisions documented
**Status**: Living document, updated as project evolves
