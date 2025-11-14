# LGI Codec - Complete Documentation Index

**Last Updated**: October 2, 2025
**Status**: Authoritative documentation for production codec
**Organization**: Structured, archived, and comprehensive

---

## 📚 Quick Navigation

**New to LGI?** Start here:
1. [Project Overview](../README.md) - What is LGI?
2. [User Guide](guides/USER_GUIDE.md) - How to use the codec
3. [Quick Start](guides/QUICK_START.md) - Get running in 5 minutes

**Developer?** Go here:
1. [API Reference](api/API_REFERENCE.md) - Complete API documentation
2. [Technical Specification](technical/TECHNICAL_SPECIFICATION.md) - Format details
3. [Implementation Guide](guides/IMPLEMENTATION_GUIDE.md) - How it works

**Integrator?** Check:
1. [Integration Guide](guides/INTEGRATION_GUIDE.md) - Embed LGI in your app
2. [Performance Guide](guides/PERFORMANCE_GUIDE.md) - Optimization tips
3. [Troubleshooting](guides/TROUBLESHOOTING.md) - Common issues

---

## 📖 Documentation Categories

### 1. Technical Specifications (Authoritative)

**Core Specifications**:
- [Technical Specification](technical/TECHNICAL_SPECIFICATION.md) - Complete format spec
- [LGI Image Format](technical/LGI_FORMAT_SPEC.md) - Image codec details
- [LGIV Video Format](technical/LGIV_VIDEO_SPEC.md) - Video codec details
- [Compression Specification](technical/COMPRESSION_SPEC.md) - Quantization, VQ, zstd
- [GPU Architecture](technical/GPU_ARCHITECTURE.md) - wgpu v27 implementation

**Implementation Specs**:
- [Quantization Profiles](technical/QUANTIZATION_PROFILES.md) - LGIQ-B/S/H/X details
- [File Format Layout](technical/FILE_FORMAT_LAYOUT.md) - Binary structure
- [Chunk Specification](technical/CHUNK_SPECIFICATION.md) - HEAD, GAUS, meta, INDE

---

### 2. API Documentation

**Public APIs**:
- [API Reference](api/API_REFERENCE.md) - Complete API documentation
- [Core API](api/CORE_API.md) - lgi-core crate
- [Encoder API](api/ENCODER_API.md) - lgi-encoder crate
- [Format API](api/FORMAT_API.md) - lgi-format crate (file I/O)
- [GPU API](api/GPU_API.md) - lgi-gpu crate (rendering)
- [Pyramid API](api/PYRAMID_API.md) - lgi-pyramid crate (zoom)

**CLI Documentation**:
- [CLI Reference](api/CLI_REFERENCE.md) - Command-line tools
- [lgi-cli-v2 Guide](api/CLI_V2_GUIDE.md) - Production CLI

---

### 3. User Guides

**Getting Started**:
- [Quick Start](guides/QUICK_START.md) - 5-minute tutorial
- [User Guide](guides/USER_GUIDE.md) - Complete user manual
- [Installation](guides/INSTALLATION.md) - Build & install
- [Examples](guides/EXAMPLES.md) - Code examples

**Usage Guides**:
- [Compression Guide](guides/COMPRESSION_GUIDE.md) - Choose compression modes
- [Quality Guide](guides/QUALITY_GUIDE.md) - Optimize for quality
- [Performance Guide](guides/PERFORMANCE_GUIDE.md) - Optimize for speed
- [GPU Guide](guides/GPU_GUIDE.md) - GPU acceleration

**Advanced Topics**:
- [Integration Guide](guides/INTEGRATION_GUIDE.md) - Embed in applications
- [Zoom Applications](guides/ZOOM_GUIDE.md) - Multi-level pyramid usage
- [Video Guide](guides/VIDEO_GUIDE.md) - LGIV video codec (future)

---

### 4. Implementation Documentation

**Architecture**:
- [System Architecture](technical/ARCHITECTURE.md) - Overall design
- [Compression Architecture](technical/COMPRESSION_ARCHITECTURE.md) - Multi-stage pipeline
- [GPU Architecture](technical/GPU_ARCHITECTURE.md) - wgpu v27 rendering
- [Pyramid Architecture](technical/PYRAMID_ARCHITECTURE.md) - Multi-level system

**Implementation Details**:
- [Optimizer Implementation](technical/OPTIMIZER_IMPLEMENTATION.md) - Full backprop
- [VQ Implementation](technical/VQ_IMPLEMENTATION.md) - Vector quantization
- [QA Training](technical/QA_TRAINING.md) - Quantization-aware training
- [Entropy Adaptive Count](technical/ADAPTIVE_COUNT.md) - Auto Gaussian allocation

**Design Decisions**:
- [Quantization Decision](technical/QUANTIZATION_DECISION.md) - Byte-aligned choice
- [Rendering Modes](technical/RENDERING_MODES.md) - Alpha vs accumulated
- [Backend Selection](technical/BACKEND_SELECTION.md) - wgpu vs alternatives

---

### 5. Testing & Benchmarking

**Test Documentation**:
- [Testing Strategy](guides/TESTING_STRATEGY.md) - Test approach
- [Test Results](TESTING_RESULTS.md) - Current test status (65/65 passing)
- [Benchmark Results](BENCHMARK_RESULTS.md) - Performance data

**Benchmark Suites**:
- [Compression Benchmarks](benchmarks/COMPRESSION_BENCHMARKS.md) - All profiles
- [GPU Benchmarks](benchmarks/GPU_BENCHMARKS.md) - CPU vs GPU
- [Quality Benchmarks](benchmarks/QUALITY_BENCHMARKS.md) - PSNR/SSIM data

---

### 6. Project Management

**Current Status**:
- [Implementation Status](IMPLEMENTATION_STATUS.md) - What's done
- [Roadmap](ROADMAP.md) - What's next
- [Changelog](CHANGELOG.md) - Version history

**Historical**:
- [Session Logs](archive/SESSION_LOGS.md) - Implementation sessions
- [Decision Log](archive/DECISION_LOG.md) - Key decisions made
- [Research Notes](archive/RESEARCH_NOTES.md) - Paper analysis

---

## 🗂️ File Organization

```
/home/greg/gaussian-image-projects/
├── README.md                          ← Start here!
├── LGI_FORMAT_SPECIFICATION.md        ← Authoritative spec
├── LGIV_VIDEO_FORMAT_SPECIFICATION.md ← Video spec
│
├── lgi-rs/                            ← Implementation
│   ├── README.md                      ← Implementation overview
│   ├── docs/
│   │   ├── DOCUMENTATION_INDEX.md     ← This file
│   │   ├── technical/                 ← Technical specs
│   │   ├── api/                       ← API docs
│   │   ├── guides/                    ← User guides
│   │   └── archive/                   ← Historical docs
│   │
│   ├── lgi-math/                      ← Math library
│   ├── lgi-core/                      ← Core rendering
│   ├── lgi-encoder/                   ← Optimization
│   ├── lgi-format/                    ← File I/O
│   ├── lgi-gpu/                       ← GPU rendering
│   ├── lgi-pyramid/                   ← Multi-level zoom
│   └── lgi-cli/                       ← CLI tools
│
└── archive/                           ← Old/superseded docs
    ├── session-logs/
    ├── early-prototypes/
    └── deprecated/
```

---

## 📋 Documentation Status

### ✅ Complete & Current
- LGI Format Specification
- LGIV Video Format Specification
- Implementation Status
- API documentation (inline docs)
- Test results

### ⏳ To Be Created
- Consolidated User Guide
- Complete API Reference
- Performance Guide
- Integration Guide
- Troubleshooting Guide

### 📦 To Be Archived
- Session-specific summaries (30+ files)
- Early implementation logs
- Superseded decision documents
- Old roadmaps

---

## 🎯 Navigation by Role

### For Users
1. Start: [README.md](../README.md)
2. Install: [Installation Guide](guides/INSTALLATION.md)
3. Use: [User Guide](guides/USER_GUIDE.md)
4. Examples: [Examples Guide](guides/EXAMPLES.md)

### For Developers
1. Overview: [Technical Specification](technical/TECHNICAL_SPECIFICATION.md)
2. API: [API Reference](api/API_REFERENCE.md)
3. Contribute: [Implementation Guide](guides/IMPLEMENTATION_GUIDE.md)
4. Test: [Testing Strategy](guides/TESTING_STRATEGY.md)

### For Integrators
1. Integration: [Integration Guide](guides/INTEGRATION_GUIDE.md)
2. Performance: [Performance Guide](guides/PERFORMANCE_GUIDE.md)
3. FFI: [C API Guide](api/C_API.md) (future)
4. Examples: [Integration Examples](guides/INTEGRATION_EXAMPLES.md)

---

## 🔄 Documentation Maintenance

**Update Frequency**:
- Technical specs: Stable (update on format changes)
- API docs: Per release (update on API changes)
- Guides: As needed (update on feature changes)
- Benchmarks: Monthly (update on performance changes)

**Versioning**:
- Specs tied to format version (v1.0)
- Implementation docs tied to crate version (0.1.0)
- Archived docs dated and immutable

---

**Last Full Audit**: October 2, 2025
**Next Audit**: When implementing LGIV video codec
**Maintainer**: LGI Project Team
