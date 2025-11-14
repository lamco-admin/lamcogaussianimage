#!/bin/bash
# Test QA Training and VQ Compression
# Compares quality with and without QA training

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  QA Training & VQ Compression Test                           ║"
echo "║  Tests impact of Quantization-Aware training on quality      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Create test image (simple gradient)
echo "📷 Creating test image (256×256 gradient)..."
convert -size 256x256 gradient:blue-red /tmp/test_gradient.png
echo "   ✅ Created: /tmp/test_gradient.png"
echo ""

# Test 1: Without QA training (baseline)
echo "Test 1: WITHOUT QA training (baseline)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
timeout 120 cargo run --release --bin lgi-cli-v2 -- encode \
  -i /tmp/test_gradient.png \
  -o /tmp/test_no_qa.png \
  -n 200 \
  -q fast \
  --metrics-csv /tmp/metrics_no_qa.csv \
  2>&1 | grep -E "(PSNR|iteration|QA training)" || true
echo ""

# Test 2: WITH QA training
echo "Test 2: WITH QA training"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
timeout 120 cargo run --release --bin lgi-cli-v2 -- encode \
  -i /tmp/test_gradient.png \
  -o /tmp/test_with_qa.png \
  -n 200 \
  -q fast \
  --qa-training \
  --metrics-csv /tmp/metrics_with_qa.csv \
  2>&1 | grep -E "(PSNR|iteration|QA training|VQ|codebook)" || true
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Results Summary                                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Extract final PSNR from CSV files
echo "Extracting metrics..."
if [ -f /tmp/metrics_no_qa.csv ]; then
    PSNR_NO_QA=$(tail -1 /tmp/metrics_no_qa.csv | cut -d',' -f3)
    echo "   Without QA: PSNR = ${PSNR_NO_QA} dB"
fi

if [ -f /tmp/metrics_with_qa.csv ]; then
    PSNR_WITH_QA=$(tail -1 /tmp/metrics_with_qa.csv | cut -d',' -f3)
    echo "   With QA:    PSNR = ${PSNR_WITH_QA} dB"
fi

echo ""
echo "✅ Test complete!"
echo ""
echo "Expected behavior:"
echo "  • QA training activates at 70% of iterations"
echo "  • VQ codebook trained with 256 entries"
echo "  • Quality difference should be <1 dB (QA maintains quality)"
echo "  • Actual compression would be 5-10× (not measured here)"
echo ""
echo "Files generated:"
echo "  • /tmp/test_gradient.png (input)"
echo "  • /tmp/test_no_qa.png (baseline)"
echo "  • /tmp/test_with_qa.png (with QA)"
echo "  • /tmp/metrics_no_qa.csv"
echo "  • /tmp/metrics_with_qa.csv"
