#!/bin/bash
# LGI Codec Diagnostic Test Suite
# Run this to check if basic encoding works AT ALL

set -e
cd /home/greg/gaussian-image-projects/lgi-rs

echo "======================================"
echo "LGI CODEC DIAGNOSTIC TEST SUITE"
echo "======================================"
echo ""

# Test 1: Single solid color
echo "TEST 1: Encoding solid red image with 5 Gaussians"
echo "---------------------------------------------------"
convert -size 128x128 xc:red /tmp/diag_red.png
./target/release/lgi-cli encode -i /tmp/diag_red.png -o /tmp/diag_red_out.png -n 5 -q fast 2>&1 | tee /tmp/diag1.log | tail -20

echo ""
echo "Input image:"
identify /tmp/diag_red.png
echo "Output image:"
identify /tmp/diag_red_out.png

echo ""
echo "First pixel of INPUT (should be red ~255,0,0):"
convert /tmp/diag_red.png -crop 1x1+0+0 txt:- | grep -v "^#"

echo "First pixel of OUTPUT (should be reddish):"
convert /tmp/diag_red_out.png -crop 1x1+0+0 txt:- | grep -v "^#"

echo ""
echo "Output image statistics:"
convert /tmp/diag_red_out.png -format "Mean: %[mean]\nMin: %[min]\nMax: %[max]" info:

echo ""
echo "---------------------------------------------------"
echo ""

# Test 2: Simple gradient
echo "TEST 2: Encoding gradient image with 20 Gaussians"
echo "---------------------------------------------------"
convert -size 128x128 gradient:blue-red /tmp/diag_grad.png
./target/release/lgi-cli encode -i /tmp/diag_grad.png -o /tmp/diag_grad_out.png -n 20 -q fast 2>&1 | tee /tmp/diag2.log | tail -20

echo ""
echo "Input gradient:"
identify /tmp/diag_grad.png
echo "Output gradient:"
identify /tmp/diag_grad_out.png

echo ""
echo "Output gradient statistics:"
convert /tmp/diag_grad_out.png -format "Mean: %[mean]\nMin: %[min]\nMax: %[max]" info:

echo ""
echo "---------------------------------------------------"
echo ""

# Check for errors
echo "ERROR CHECK"
echo "---------------------------------------------------"
echo "Checking for NaN in logs:"
rg "NaN" /tmp/diag*.log || echo "  No NaN found (good)"

echo ""
echo "Checking for panics/errors:"
rg -i "panic|error" /tmp/diag*.log || echo "  No panics/errors found (good)"

echo ""
echo "---------------------------------------------------"
echo ""

# Visual comparison
echo "VISUAL COMPARISON RESULTS"
echo "---------------------------------------------------"
echo "Created comparison images:"
echo "  Input red:     /tmp/diag_red.png"
echo "  Output red:    /tmp/diag_red_out.png"
echo "  Input grad:    /tmp/diag_grad.png"
echo "  Output grad:   /tmp/diag_grad_out.png"
echo ""
echo "Open these with: display /tmp/diag_red_out.png"
echo ""

# Check if outputs are valid
if [ ! -s /tmp/diag_red_out.png ]; then
    echo "ERROR: Red output is empty!"
fi

if [ ! -s /tmp/diag_grad_out.png ]; then
    echo "ERROR: Gradient output is empty!"
fi

# Check pixel values
red_mean=$(convert /tmp/diag_red_out.png -format "%[mean]" info: | cut -d. -f1)
grad_mean=$(convert /tmp/diag_grad_out.png -format "%[mean]" info: | cut -d. -f1)

echo "QUICK DIAGNOSIS:"
echo "---------------------------------------------------"
if [ "$red_mean" -lt "1000" ]; then
    echo "❌ FAIL: Red output is too dark (mean=$red_mean) - likely all black"
    echo "   → Rendering is BROKEN"
elif [ "$red_mean" -gt "60000" ]; then
    echo "❌ FAIL: Red output is too bright (mean=$red_mean) - likely all white"
    echo "   → Rendering is BROKEN"
else
    echo "✓ PASS: Red output has reasonable brightness (mean=$red_mean)"
fi

echo ""
echo "Full logs saved to:"
echo "  /tmp/diag1.log (red test)"
echo "  /tmp/diag2.log (gradient test)"
echo ""
echo "======================================"
echo "DIAGNOSTIC COMPLETE"
echo "======================================"
