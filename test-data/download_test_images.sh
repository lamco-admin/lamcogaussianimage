#!/bin/bash
# LGI Test Data Download Script
# For environments without Git LFS (e.g., Claude Code Web)

set -e

echo "============================================"
echo "LGI Test Data Download Script"
echo "============================================"
echo ""

# Check if we're already populated
if [ -f "kodak-dataset/kodim01.png" ] && [ -f "test_images_new_synthetic/HF_checkerboard_1px.png" ]; then
    EXISTING=$(find . -type f \( -name "*.png" -o -name "*.jpg" \) | wc -l)
    if [ "$EXISTING" -gt 100 ]; then
        echo "✅ Test data already present ($EXISTING images)"
        echo "No download needed."
        exit 0
    fi
fi

echo "📥 Downloading test images..."
echo ""

# Create directories
mkdir -p kodak-dataset test_images_new_synthetic test_images

# Download Kodak dataset (public domain)
echo "1/3: Downloading Kodak dataset (24 images, 56 MB)..."
cd kodak-dataset
for i in $(seq -f "%02g" 1 24); do
    if [ ! -f "kodim${i}.png" ]; then
        echo "  Downloading kodim${i}.png..."
        curl -s -L "http://r0k.us/graphics/kodak/kodak/kodim${i}.png" -o "kodim${i}.png"
    fi
done
cd ..
echo "  ✅ Kodak dataset complete (24 images)"
echo ""

# Synthetic images - these should be small enough to be in regular git
echo "2/3: Checking synthetic test patterns..."
SYNTHETIC_COUNT=$(ls test_images_new_synthetic/*.png 2>/dev/null | wc -l)
echo "  Found $SYNTHETIC_COUNT synthetic patterns"
if [ "$SYNTHETIC_COUNT" -lt 10 ]; then
    echo "  ⚠️  Some synthetic images missing (only $SYNTHETIC_COUNT found)"
    echo "  These should be in the repository - check .gitignore"
fi
echo ""

# Real photos - these are large and may need alternative approach
echo "3/3: Checking real photo test images..."
PHOTOS_COUNT=$(ls test_images/*.jpg test_images/*.png 2>/dev/null | wc -l)
echo "  Found $PHOTOS_COUNT real photos"
echo ""

if [ "$PHOTOS_COUNT" -lt 60 ]; then
    echo "⚠️  Large photo dataset incomplete (Git LFS issue)"
    echo ""
    echo "OPTIONS:"
    echo "1. Install Git LFS locally: git lfs install && git lfs pull"
    echo "2. Use Kodak dataset only (24 images, sufficient for validation)"
    echo "3. Generate minimal synthetic test set (see below)"
    echo ""
    echo "For Claude Code Web:"
    echo "  - Kodak dataset (24 images) is sufficient for validation ✅"
    echo "  - Synthetic patterns (13 images) for controlled testing ✅"
    echo "  - Total: 37 images available for benchmarking"
    echo ""
fi

# Summary
echo "============================================"
echo "Summary:"
echo "============================================"
TOTAL=$(find . -type f \( -name "*.png" -o -name "*.jpg" \) ! -name "*.1" | wc -l)
echo "Total images available: $TOTAL"
echo ""
echo "✅ Kodak dataset: $(ls kodak-dataset/*.png 2>/dev/null | wc -l) images"
echo "✅ Synthetic patterns: $(ls test_images_new_synthetic/*.png 2>/dev/null | wc -l) images"
echo "⚠️  Real photos: $(ls test_images/*.jpg test_images/*.png 2>/dev/null | wc -l) images"
echo ""

if [ "$TOTAL" -ge 37 ]; then
    echo "✅ Sufficient test data for development ($TOTAL images)"
    echo ""
    echo "Quick validation:"
    echo "  cd ../packages/lgi-rs"
    echo "  cargo run --release --example fast_benchmark"
    echo ""
    echo "Kodak benchmark:"
    echo "  cd packages/lgi-rs/lgi-benchmarks"
    echo "  cargo run --release --bin kodak_benchmark"
else
    echo "⚠️  Limited test data - consider installing Git LFS"
fi

echo "============================================"
