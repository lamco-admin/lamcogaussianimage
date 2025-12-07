# LGI Project Documentation Index
## Master Guide to All Documentation

**Last Updated:** 2025-12-07
**Purpose:** Central navigation for all project documentation

---

## 🎯 START HERE

### Essential Documents (Read These First)

1. **[RESEARCH_PHILOSOPHY.md](RESEARCH_PHILOSOPHY.md)** - **READ FIRST**
   - Why we're building a toolkit, not finding "optimal configuration"
   - The 960,000+ configuration space problem
   - Research approach: systematic micro-level testing

2. **[CLAUDE.md](../CLAUDE.md)** - Project instructions for Claude Code
   - Build commands, test commands
   - Architecture overview
   - Development standards

3. **[GROUND_TRUTH.md](../GROUND_TRUTH.md)** - Verified facts only
   - What actually works (validated)
   - Performance benchmarks
   - No speculation

4. **[NEXT_STEPS.md](../NEXT_STEPS.md)** - Immediate priorities
   - Current focus areas
   - Open questions
   - Research tasks

---

## 📚 CORE CATALOGS (Technical Reference)

### Architecture Catalogs (Dec 2025 - Current)

These catalogs document **what exists in the code** as of Dec 7, 2025:

1. **[OPTIMIZER_CATALOG.md](OPTIMIZER_CATALOG.md)** - 9 optimization algorithms
   - 6 CPU optimizers (Adam, OptimizerV2, OptimizerV3, L-BFGS, L-M, Hybrid)
   - 3 GPU optimizers (Adam, L-BFGS, L-M with JtJ strategies)
   - 4,671 lines of code
   - Loss functions, gradient methods, learning rate schedules

2. **[ENCODER_CATALOG.md](ENCODER_CATALOG.md)** - 13 encoder entry points
   - Complete encoding pipelines (init + optimizer + refinement)
   - Error-driven, GPU-accelerated, rate-distortion variants
   - 1,778 lines in lib.rs
   - Parameter tables, flow diagrams

3. **[RENDERER_CATALOG.md](RENDERER_CATALOG.md)** - 7 rendering engines
   - 6 CPU renderers, 1 GPU renderer
   - 1,326 lines Rust + 1,995 lines GPU shaders
   - AlphaComposite vs AccumulatedSum modes
   - Performance characteristics

4. **[IMAGE_ANALYSIS_CATALOG.md](IMAGE_ANALYSIS_CATALOG.md)** - 62 analysis modules
   - Toolkit modules (structure tensor, geodesic EDT, saliency, etc.)
   - Edge detection, feature classification, quality metrics
   - 11,841 lines in lgi-core
   - Module index with purposes

5. **[INITIALIZATION_CATALOG.md](INITIALIZATION_CATALOG.md)** - 14 placement strategies
   - Grid, random, gradient-based, k-means, GDGS, etc.
   - 1,700 lines of initialization code
   - Feature dimensions, configuration options

6. **[FORMAT_COMPRESSION_CATALOG.md](FORMAT_COMPRESSION_CATALOG.md)** - File format & compression
   - 4 LGIQ quantization profiles (B/S/H/X)
   - Compression pipeline (quantization → VQ → zstd)
   - PNG-like chunk structure
   - 2,116 lines in lgi-format crate

7. **[THEORETICAL_FOUNDATIONS.md](THEORETICAL_FOUNDATIONS.md)** - Core concepts & math
   - 9-parameter Gaussian definition
   - 4 parameterizations (Euler/Cholesky/LogRadius/InvCov)
   - Layered representation concept (M/E/J/R/B/T)
   - Coverage formulas and scaling laws
   - Fundamental principles (layers never tiles)

8. **[TOOLKIT_MODULES.md](TOOLKIT_MODULES.md)** - Toolkit component guide
   - 6 active toolkit modules
   - Module composition patterns
   - Integration examples

---

## 🔬 TOOLKIT RESEARCH (Dec 2025)

### Active Research Documents

**[TOOLKIT_MODULES.md](../packages/lgi-rs/lgi-encoder-v2/test-results/TOOLKIT_MODULES.md)**
- 6 toolkit components under investigation
- Structure tensor, geodesic EDT, saliency, Frangi, text stroke, perceptual weighting
- Module purposes and research questions

**[SESSION_HANDOVER_2025-12-07_TOOLKIT.md](../packages/lgi-rs/lgi-encoder-v2/test-results/SESSION_HANDOVER_2025-12-07_TOOLKIT.md)**
- Toolkit paradigm shift documentation
- Isolated tests vs optimization-time tests
- Technical details for custom optimization loops

**Test Results:**
- `packages/lgi-rs/lgi-encoder-v2/test-results/` - 678 JSON result files
- Organized by date: 2025-12-05, 06, 07, 19, 20
- Toolkit test results in progress

---

## 📖 HISTORICAL CATALOGS (Nov 2025 - Reference)

**Location:** `docs/archive/`

These Nov 15 catalogs are being superseded by Dec 7 catalogs above:

- RENDERING_ENGINES_CATALOG.md (789 lines) - Now: RENDERER_CATALOG.md
- CODEC_CATALOG.md (1,127 lines) - Being split into focused catalogs
- IMAGE_PROCESSING_CATALOG.md (868 lines) - Now: IMAGE_ANALYSIS_CATALOG.md
- RESEARCH_TOOLS_MASTER_CATALOG.md (742 lines) - Now: This index + focused catalogs
- PYTHON_TOOLS_CATALOG.md (999 lines) - Python infrastructure docs

**Status:** Reference only - content being assimilated into new catalogs

---

## 🏗️ ARCHITECTURE & DEVELOPMENT

### Architecture Docs

**[DEVELOPMENT_STANDARDS.md](DEVELOPMENT_STANDARDS.md)**
- Code quality requirements
- Testing standards
- Non-negotiable rules

**[GPU_CPU_ARCHITECTURE.md](../packages/lgi-rs/lgi-gpu/GPU_CPU_ARCHITECTURE.md)**
- GPU vs CPU implementation patterns
- When to use which optimizer

### Research Findings

**[COMPREHENSIVE_FINDINGS_2025-12-06.md](../packages/lgi-rs/lgi-encoder-v2/test-results/COMPREHENSIVE_FINDINGS_2025-12-06.md)**
- Dec 6 experimental results
- Optimizer comparisons
- Placement strategy results

**[OPTIMIZER_INVENTORY.md](../packages/lgi-rs/lgi-encoder-v2/test-results/OPTIMIZER_INVENTORY.md)**
- Detailed optimizer status (Dec 7)
- Implementation completeness
- Research notes

**[DENSIFICATION_RESEARCH_SUMMARY.md](../packages/lgi-rs/lgi-encoder-v2/test-results/DENSIFICATION_RESEARCH_SUMMARY.md)**
- Adaptive densification experiments
- Split/clone/prune research

---

## 🗂️ DOCUMENTATION ORGANIZATION

### Current Structure

```
lgi-project/
├── docs/
│   ├── README.md (this file) - Master index
│   ├── RESEARCH_PHILOSOPHY.md - Core principles
│   ├── OPTIMIZER_CATALOG.md - 9 optimizers
│   ├── ENCODER_CATALOG.md - 13 encoders
│   ├── RENDERER_CATALOG.md - 7 renderers
│   ├── IMAGE_ANALYSIS_CATALOG.md - 62 modules
│   ├── INITIALIZATION_CATALOG.md - 14 strategies
│   ├── DEVELOPMENT_STANDARDS.md - Code quality
│   ├── research/ - Active research docs
│   │   ├── PROJECT_HISTORY.md
│   │   ├── EXPERIMENTS.md
│   │   └── ...
│   └── archive/ - Historical reference (Nov 2025)
│       ├── CODEC_CATALOG.md
│       ├── RENDERING_ENGINES_CATALOG.md
│       ├── IMAGE_PROCESSING_CATALOG.md
│       └── ...
├── GROUND_TRUTH.md - Verified facts
├── NEXT_STEPS.md - Priorities
├── CLAUDE.md - Project instructions
└── packages/lgi-rs/lgi-encoder-v2/test-results/
    ├── TOOLKIT_MODULES.md - Toolkit component guide
    ├── SESSION_HANDOVER_* - Recent session notes
    └── 2025-12-*/ - Test result data
```

---

## 🔍 HOW TO FIND INFORMATION

### "I need to understand the architecture"
→ Read catalogs in order:
1. ENCODER_CATALOG (entry points)
2. OPTIMIZER_CATALOG (parameter updates)
3. RENDERER_CATALOG (forward passes)
4. INITIALIZATION_CATALOG (placement)
5. IMAGE_ANALYSIS_CATALOG (feature detection)

### "I want to implement a new technique"
→ Check IMAGE_ANALYSIS_CATALOG for existing modules
→ Follow DEVELOPMENT_STANDARDS.md
→ Add to appropriate layer (encoder/optimizer/renderer/init)

### "I want to understand what's been tested"
→ packages/lgi-rs/lgi-encoder-v2/test-results/
→ 678 JSON files with experimental data
→ COMPREHENSIVE_FINDINGS_2025-12-06.md

### "I want to understand the toolkit approach"
→ RESEARCH_PHILOSOPHY.md (why toolkit not monolith)
→ TOOLKIT_MODULES.md (6 toolkit components)
→ SESSION_HANDOVER_2025-12-07_TOOLKIT.md (technical details)

### "I need historical context"
→ docs/archive/ - Nov 2025 catalogs
→ docs/research/PROJECT_HISTORY.md
→ SESSION_HANDOVER files

---

## 📊 STATISTICS SUMMARY

### Code Base (Dec 2025)
- **Total Rust Files:** 403
- **Optimizer Code:** 4,671 lines
- **Encoder Code:** 1,778 lines (lib.rs)
- **Renderer Code:** 1,326 lines + 1,995 shader lines
- **Image Analysis:** 11,841 lines (lgi-core)
- **Initialization:** 1,700 lines
- **Test Examples:** 212 files
- **Test Results:** 678 JSON files

### Documentation (Dec 2025)
- **Core Catalogs:** 6 documents (this index + 5 technical)
- **Research Docs:** 8+ active documents
- **Archived Docs:** 30+ historical documents
- **Session Handovers:** 20 files

---

## 🎓 LEARNING PATH

### For New Contributors

1. Read RESEARCH_PHILOSOPHY.md - Understand the approach
2. Read GROUND_TRUTH.md - Know what's validated
3. Skim CLAUDE.md - Build/test commands
4. Pick ONE catalog relevant to your interest
5. Look at code files referenced in catalog
6. Find related test examples in lgi-encoder-v2/examples/

### For Understanding a Specific Area

**Optimizers:** OPTIMIZER_CATALOG.md → adam_optimizer.rs → test examples
**Encoders:** ENCODER_CATALOG.md → lib.rs specific function
**Image Analysis:** IMAGE_ANALYSIS_CATALOG.md → lgi-core/src/module.rs

---

## 📝 DOCUMENT STATUS

| Document | Status | Date | Lines | Purpose |
|----------|--------|------|-------|---------|
| RESEARCH_PHILOSOPHY.md | ✅ Current | Dec 7 | 219 | Core principles |
| OPTIMIZER_CATALOG.md | ✅ Current | Dec 7 | ~480 | 9 optimizers |
| ENCODER_CATALOG.md | ✅ Current | Dec 7 | ~560 | 13 encoders |
| RENDERER_CATALOG.md | ✅ Current | Dec 7 | ~380 | 7 renderers |
| IMAGE_ANALYSIS_CATALOG.md | ✅ Current | Dec 7 | ~450 | 62 modules |
| INITIALIZATION_CATALOG.md | ✅ Current | Dec 7 | ~280 | 14 strategies |
| FORMAT_COMPRESSION_CATALOG.md | ✅ Current | Dec 7 | ~480 | File format & compression |
| THEORETICAL_FOUNDATIONS.md | ✅ Current | Dec 7 | ~310 | Math & core concepts |
| TOOLKIT_MODULES.md | ✅ Current | Dec 7 | 211 | Toolkit components |
| GROUND_TRUTH.md | ✅ Current | Dec 5 | 321 | Verified facts |
| NEXT_STEPS.md | ✅ Current | Dec 5 | 244 | Priorities |

---

*This index will be updated as new catalogs are created and documentation evolves.*
