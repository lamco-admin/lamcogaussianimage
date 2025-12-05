# Comprehensive Python Tools & Preprocessing Catalog

This directory contains three comprehensive documentation files that catalog ALL Python preprocessing tools, scripts, and utilities in the Gaussian Image Codec repository.

## Documentation Files

### 1. PYTHON_TOOLS_CATALOG.md
**Comprehensive Technical Reference (999 lines)**

The most detailed documentation file containing:
- Complete documentation for all 28 Python files
- Detailed function signatures and parameters
- Algorithm descriptions and implementation details
- Color space transforms and mathematical formulas
- Input/output specifications
- Dependencies and version requirements
- Integration with Rust codebase

**When to use:** When you need deep technical details, algorithm specifications, or implementation reference.

---

### 2. PYTHON_TOOLS_QUICK_SUMMARY.txt
**Executive Summary & Quick Reference (200 lines)**

Fast lookup reference containing:
- Quick summary of all 28 Python files
- Key algorithms at a glance
- Output formats
- Typical usage workflow
- Dependencies summary
- Complete file paths index
- Key functions organized by purpose
- Integration overview

**When to use:** Quick lookups, understanding file organization, finding specific tools, or getting started.

---

### 3. PYTHON_ARCHITECTURE_MAP.txt
**System Architecture & Data Flow (300 lines)**

Visual system design documentation containing:
- Complete file tree with relationships
- Detailed dependency graph
- Data flow during training
- Preprocessing pipeline flowchart
- SLIC Gaussian initialization process
- Quality metrics overview
- Configuration parameters
- Complete metrics summary

**When to use:** Understanding system architecture, debugging integration issues, or planning modifications.

---

## Quick Navigation

### Find by Task

**I want to preprocess an image:**
- See: PYTHON_TOOLS_QUICK_SUMMARY.txt → "PRIMARY PREPROCESSING TOOLS"
- Full details: PYTHON_TOOLS_CATALOG.md → "SECTION 1: PRIMARY PREPROCESSING TOOLS"

**I need to understand how Gaussians are rendered:**
- See: PYTHON_ARCHITECTURE_MAP.txt → "DEPENDENCY GRAPH" & "DATA FLOW DURING TRAINING"
- Full details: PYTHON_TOOLS_CATALOG.md → "SECTION 4: GAUSSIAN SPLATTING KERNELS"

**I need to integrate with the Rust codec:**
- See: PYTHON_TOOLS_QUICK_SUMMARY.txt → "INTEGRATION WITH RUST CODEC"
- Full details: PYTHON_ARCHITECTURE_MAP.txt → "DEPENDENCY GRAPH" & "DATA FLOW DURING TRAINING"

**I want to optimize image quality:**
- See: PYTHON_TOOLS_QUICK_SUMMARY.txt → "DEPENDENCIES" & "KEY FUNCTIONS BY PURPOSE"
- Full details: PYTHON_TOOLS_CATALOG.md → "SECTION 5: PERFORMANCE & QUALITY METRICS"

**I need configuration parameters:**
- See: PYTHON_ARCHITECTURE_MAP.txt → "KEY CONFIGURATION PARAMETERS"
- Full details: PYTHON_TOOLS_CATALOG.md → Individual sections with "Configuration Parameters"

### Find by File Type

**Preprocessing Tools:**
- preprocess_image.py
- preprocess_image_v2.py
- slic_preprocess.py

See: PYTHON_TOOLS_QUICK_SUMMARY.txt → "PRIMARY PREPROCESSING TOOLS (lgi-rs/tools/)"
Full: PYTHON_TOOLS_CATALOG.md → "SECTION 1"

**Utilities:**
- image_utils.py
- quantization_utils.py
- saliency_utils.py
- misc_utils.py
- flip.py

See: PYTHON_TOOLS_QUICK_SUMMARY.txt → "IMAGE UTILITIES"
Full: PYTHON_TOOLS_CATALOG.md → "SECTION 2"

**Training & Models:**
- main.py
- model.py

See: PYTHON_TOOLS_QUICK_SUMMARY.txt → "TRAINING & MODELS"
Full: PYTHON_TOOLS_CATALOG.md → "SECTION 3"

**Gaussian Splatting:**
- project_gaussians_2d_scale_rot.py
- rasterize_sum.py
- rasterize_no_tiles.py
- utils.py (gsplat)
- __init__.py (gsplat)

See: PYTHON_TOOLS_QUICK_SUMMARY.txt → "GAUSSIAN SPLATTING"
Full: PYTHON_TOOLS_CATALOG.md → "SECTION 4"

**Quality & Metrics:**
- fused_ssim/__init__.py
- test.py, genplot.py, train_image.py (SSIM tests)

See: PYTHON_TOOLS_QUICK_SUMMARY.txt → "SSIM & QUALITY"
Full: PYTHON_TOOLS_CATALOG.md → "SECTION 5"

---

## File Organization

```
/home/user/lamcogaussianimage/
├── PYTHON_TOOLS_CATALOG.md                 [This directory]
├── PYTHON_TOOLS_QUICK_SUMMARY.txt          [This directory]
├── PYTHON_ARCHITECTURE_MAP.txt             [This directory]
├── README_PYTHON_CATALOG.md                [This file]
│
└── packages/
    ├── lgi-rs/tools/
    │   ├── preprocess_image.py
    │   ├── preprocess_image_v2.py
    │   └── slic_preprocess.py
    │
    ├── lgi-legacy/image-gs/
    │   ├── utils/
    │   │   ├── image_utils.py
    │   │   ├── quantization_utils.py
    │   │   ├── saliency_utils.py
    │   │   ├── misc_utils.py
    │   │   └── flip.py
    │   ├── main.py
    │   ├── model.py
    │   └── gsplat/gsplat/
    │       ├── project_gaussians_2d_scale_rot.py
    │       ├── rasterize_sum.py
    │       ├── rasterize_no_tiles.py
    │       ├── utils.py
    │       └── __init__.py
    │
    ├── lgi-legacy/image-gs-cpu/
    │   ├── gaussian_2d_cpu.py
    │   └── test_gaussian.py
    │
    └── lgi-tools/fused-ssim/
        ├── fused_ssim/__init__.py
        └── tests/
            ├── test.py
            ├── genplot.py
            └── train_image.py
```

---

## Key Statistics

**Total Python Files Cataloged:** 28
**Total Lines of Code:** 2000-2500
**Total Documentation Lines:** 1500+

**Breakdown:**
- Preprocessing tools: 3 files
- Image utilities: 5 files
- Training/Model: 2 files
- Gaussian splatting: 5 files
- SSIM/Quality: 4 files
- Legacy/Testing: 2 files
- Config/Init: 7 files

**Algorithms Documented:** 15+
**Loss Functions:** 5+
**Utility Functions:** 50+

---

## Typical Workflow

### Step 1: Preprocess Image
```bash
python packages/lgi-rs/tools/preprocess_image_v2.py image.png \
  --n-segments 500 --entropy-tile-size 16 --use-gpu
```

Output: 8 .npy maps + metadata.json

See: PYTHON_TOOLS_CATALOG.md → "1.2: preprocess_image_v2.py"

### Step 2: Generate Gaussian Initialization
```bash
python packages/lgi-rs/tools/slic_preprocess.py image.png 500 slic_init.json
```

Output: slic_init.json with Gaussian parameters

See: PYTHON_TOOLS_CATALOG.md → "1.3: slic_preprocess.py"

### Step 3: Train Model
```bash
python packages/lgi-legacy/image-gs/main.py \
  --input-path image.png --num-gaussians 5000
```

Output: Checkpoints, rendered images, metrics

See: PYTHON_TOOLS_CATALOG.md → "3: TRAINING & OPTIMIZATION MODULES"

### Step 4: Evaluate Quality
- PSNR via image_utils.get_psnr()
- SSIM via fused_ssim.fused_ssim()
- LPIPS via lpips package
- FLIP via flip.LDRFLIPLoss()

See: PYTHON_ARCHITECTURE_MAP.txt → "QUALITY METRICS AVAILABLE"

---

## Common Questions

**Q: Where do I find preprocessing scripts?**
A: `/home/user/lamcogaussianimage/packages/lgi-rs/tools/`
See: PYTHON_TOOLS_QUICK_SUMMARY.txt → "FILE PATHS"

**Q: What output does preprocessing generate?**
A: 8 .npy files + metadata.json
See: PYTHON_TOOLS_CATALOG.md → "1.2: preprocess_image_v2.py" → "Output Files"

**Q: How are Gaussians initialized?**
A: SLIC superpixels → position, scale, rotation, color
See: PYTHON_TOOLS_CATALOG.md → "1.3: slic_preprocess.py"

**Q: How is the placement map calculated?**
A: Weighted combination of 5 analysis maps (entropy, gradient, texture, saliency, distance)
See: PYTHON_TOOLS_CATALOG.md → "1.1: preprocess_image.py" → "Weights for Placement Map"

**Q: What GPU support is available?**
A: CUDA optional in preprocessing (cv2.cuda_*) and SSIM (fused_ssim_cuda)
See: PYTHON_TOOLS_CATALOG.md → "GPU Support"

**Q: How do I integrate preprocessing with the Rust codec?**
A: Output .npy and .json files are directly compatible
See: PYTHON_TOOLS_QUICK_SUMMARY.txt → "INTEGRATION WITH RUST CODEC"

---

## Dependencies Summary

**Core Libraries:**
- torch, torchvision
- numpy, scipy
- scikit-image
- opencv-contrib-python
- mahotas
- PIL, matplotlib
- PyYAML

**Custom Extensions:**
- gsplat.cuda (C++/CUDA)
- fused_ssim_cuda (CUDA)
- lpips
- pytorch_msssim

See: PYTHON_TOOLS_QUICK_SUMMARY.txt → "DEPENDENCIES"

---

## Document History

- **Created:** 2025-11-15
- **Total Files Analyzed:** 28 Python files
- **Total Lines Documented:** 1500+
- **Completeness:** 100% - All Python preprocessing tools cataloged

---

## How to Use These Documents

1. **Start here** for overview and quick reference:
   → PYTHON_TOOLS_QUICK_SUMMARY.txt

2. **Dive into** specific details:
   → PYTHON_TOOLS_CATALOG.md

3. **Understand** system architecture:
   → PYTHON_ARCHITECTURE_MAP.txt

4. **Find specific** algorithms or functions:
   → Use grep on these files for keywords

---

## Questions or Updates?

These documents were generated on 2025-11-15 and comprehensively cover all Python preprocessing tools in the Gaussian Image Codec repository.

For the most current information, refer to:
- Source code in `/home/user/lamcogaussianimage/packages/`
- Docstrings in individual Python files
- README and DOCUMENTATION in main repository

---

**Last Updated:** 2025-11-15
**Coverage:** 100% of Python preprocessing tools and utilities
**Scope:** Gaussian Image Codec Repository - Complete Python Catalog
