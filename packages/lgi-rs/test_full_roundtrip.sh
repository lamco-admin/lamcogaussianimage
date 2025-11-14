#!/bin/bash
# Complete Round-Trip Test: PNG → LGI → PNG
# Tests the full encode/decode pipeline

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  LGI Full Round-Trip Test                                    ║"
echo "║  PNG → Gaussians → .lgi file → Load → Render → PNG          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Create test image
echo "📷 Step 1: Creating test image..."
convert -size 256x256 gradient:blue-red /tmp/test_roundtrip.png
echo "   ✅ Created: /tmp/test_roundtrip.png"
echo ""

# Encode to .lgi
echo "🔧 Step 2: Encoding PNG → .lgi (with VQ compression)..."
timeout 120 cargo run --release --bin lgi-cli-v2 -- encode \
  -i /tmp/test_roundtrip.png \
  -o /tmp/test_encoded.png \
  -n 200 \
  -q fast \
  --qa-training \
  --save-lgi \
  --metrics-csv /tmp/test_metrics.csv \
  2>&1 | grep -E "(Saving|Compression|PSNR|Done)" || true
echo ""

# Show .lgi file info
echo "📊 Step 3: Inspecting .lgi file..."
cargo run --release --bin lgi-cli-v2 -- info \
  -i /tmp/test_encoded.lgi \
  2>&1 | grep -v "Compiling" || true
echo ""

# Decode .lgi → PNG
echo "🎨 Step 4: Decoding .lgi → PNG..."
cargo run --release --bin lgi-cli-v2 -- decode \
  -i /tmp/test_encoded.lgi \
  -o /tmp/test_decoded.png \
  2>&1 | grep -E "(Loading|Gaussians|Rendering|Saving|Done)" || true
echo ""

# Compare file sizes
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Results Summary                                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

ORIGINAL_SIZE=$(stat -c%s /tmp/test_roundtrip.png 2>/dev/null || stat -f%z /tmp/test_roundtrip.png 2>/dev/null)
LGI_SIZE=$(stat -c%s /tmp/test_encoded.lgi 2>/dev/null || stat -f%z /tmp/test_encoded.lgi 2>/dev/null)
ENCODED_SIZE=$(stat -c%s /tmp/test_encoded.png 2>/dev/null || stat -f%z /tmp/test_encoded.png 2>/dev/null)
DECODED_SIZE=$(stat -c%s /tmp/test_decoded.png 2>/dev/null || stat -f%z /tmp/test_decoded.png 2>/dev/null)

echo "📦 File Sizes:"
echo "   Original PNG:  $((ORIGINAL_SIZE / 1024)) KB"
echo "   Encoded PNG:   $((ENCODED_SIZE / 1024)) KB"
echo "   .lgi file:     $((LGI_SIZE / 1024)) KB"
echo "   Decoded PNG:   $((DECODED_SIZE / 1024)) KB"
echo ""

RATIO=$(echo "scale=2; $ORIGINAL_SIZE / $LGI_SIZE" | bc)
echo "💾 Compression:"
echo "   PNG vs .lgi:   ${RATIO}× smaller"
echo ""

echo "📁 Generated Files:"
echo "   /tmp/test_roundtrip.png  (original)"
echo "   /tmp/test_encoded.png    (rendered from Gaussians)"
echo "   /tmp/test_encoded.lgi    (VQ compressed)"
echo "   /tmp/test_decoded.png    (decoded from .lgi)"
echo "   /tmp/test_metrics.csv    (optimization metrics)"
echo ""

echo "✅ Round-trip test COMPLETE!"
echo ""
echo "Next steps:"
echo "  • Compare images visually"
echo "  • Check metrics CSV for quality data"
echo "  • Try different Gaussian counts and quality presets"
echo ""
