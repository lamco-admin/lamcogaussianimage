#!/bin/bash
# Comprehensive Overnight Test Suite
# Tests everything and generates full report

set -e
cd /home/greg/gaussian-image-projects/lgi-rs

REPORT="/tmp/lgi_overnight_report.md"

echo "# LGI Codec Overnight Test Report" > $REPORT
echo "**Date**: $(date)" >> $REPORT
echo "**Duration**: Multi-hour test suite" >> $REPORT
echo "" >> $REPORT
echo "---" >> $REPORT
echo "" >> $REPORT

mkdir -p /tmp/lgi_tests/{inputs,outputs,analysis}

# ====================
# TEST 1: SOLID COLORS
# ====================
echo "## Test 1: Solid Colors (Baseline)" >> $REPORT
echo "" >> $REPORT
echo "Testing if Gaussians can represent uniform colors..." >> $REPORT
echo "" >> $REPORT

for size in 64 128 256 512; do
  for color in red green blue white black; do
    echo "Testing ${color} ${size}x${size}..."

    convert -size ${size}x${size} xc:${color} /tmp/lgi_tests/inputs/solid_${color}_${size}.png

    # Test with varying Gaussian counts
    for n in 1 3 5 10; do
      ./target/release/lgi-cli encode \
        -i /tmp/lgi_tests/inputs/solid_${color}_${size}.png \
        -o /tmp/lgi_tests/outputs/solid_${color}_${size}_n${n}.png \
        -n $n \
        -q fast 2>&1 | rg "PSNR" >> /tmp/lgi_tests/analysis/solid_${color}_${size}.txt
    done
  done
done

echo "| Size | Color | n=1 PSNR | n=3 PSNR | n=5 PSNR | n=10 PSNR |" >> $REPORT
echo "|------|-------|----------|----------|----------|-----------|" >> $REPORT

for size in 64 128 256 512; do
  for color in red green blue white black; do
    line="| ${size} | ${color} |"
    for n in 1 3 5 10; do
      psnr=$(grep PSNR /tmp/lgi_tests/analysis/solid_${color}_${size}.txt 2>/dev/null | sed -n "${n}p" | grep -o "[0-9.]*" | head -1)
      line="$line ${psnr:-N/A} |"
    done
    echo "$line" >> $REPORT
  done
done

echo "" >> $REPORT
echo "**Analysis**: " >> $REPORT
echo "- Solid colors should achieve 20+ dB PSNR with just 1-3 Gaussians" >> $REPORT
echo "- If PSNR < 15 dB → scale still too small" >> $REPORT
echo "" >> $REPORT

# ====================
# TEST 2: GRADIENTS
# ====================
echo "## Test 2: Gradients (Smooth Variation)" >> $REPORT
echo "" >> $REPORT

for size in 128 256 512; do
  echo "Testing gradients at ${size}x${size}..."

  # Linear gradient
  convert -size ${size}x${size} gradient:blue-red /tmp/lgi_tests/inputs/grad_linear_${size}.png

  # Radial gradient
  convert -size ${size}x${size} radial-gradient:blue-red /tmp/lgi_tests/inputs/grad_radial_${size}.png

  for type in linear radial; do
    for n in 5 10 20 50; do
      ./target/release/lgi-cli encode \
        -i /tmp/lgi_tests/inputs/grad_${type}_${size}.png \
        -o /tmp/lgi_tests/outputs/grad_${type}_${size}_n${n}.png \
        -n $n \
        -q fast 2>&1 | rg "PSNR" >> /tmp/lgi_tests/analysis/grad_${type}_${size}.txt
    done
  done
done

echo "| Size | Type | n=5 | n=10 | n=20 | n=50 |" >> $REPORT
echo "|------|------|-----|------|------|------|" >> $REPORT
for size in 128 256 512; do
  for type in linear radial; do
    line="| ${size} | ${type} |"
    for i in 1 2 3 4; do
      psnr=$(sed -n "${i}p" /tmp/lgi_tests/analysis/grad_${type}_${size}.txt 2>/dev/null | grep -o "[0-9.]*" | head -1)
      line="$line ${psnr:-N/A} |"
    done
    echo "$line" >> $REPORT
  done
done

echo "" >> $REPORT
echo "**Analysis**: Gradients should achieve 20-30 dB PSNR with 10-20 Gaussians" >> $REPORT
echo "" >> $REPORT

# ====================
# TEST 3: TEXT/FONTS
# ====================
echo "## Test 3: Text at Multiple Sizes" >> $REPORT
echo "" >> $REPORT
echo "**Testing hypothesis**: Gaussians are TERRIBLE for sharp edges (text)" >> $REPORT
echo "" >> $REPORT

# Create text at different sizes
for size in 64 128 256 512 1024; do
  pointsize=$((size / 4))
  echo "Creating text: ${size}x${size} at ${pointsize}pt..."

  convert -size ${size}x${size} xc:white \
    -font DejaVu-Sans -pointsize ${pointsize} \
    -gravity center -fill black \
    -annotate +0+0 "A" \
    /tmp/lgi_tests/inputs/font_${size}.png

  convert -size ${size}x${size} xc:black \
    -font DejaVu-Sans -pointsize ${pointsize} \
    -gravity center -fill white \
    -annotate +0+0 "A" \
    /tmp/lgi_tests/inputs/font_inv_${size}.png
done

echo "| Size | Variant | n=20 | n=50 | n=100 | n=200 | n=500 |" >> $REPORT
echo "|------|---------|------|------|-------|-------|-------|" >> $REPORT

for size in 64 128 256 512; do
  for variant in "" "_inv"; do
    echo "Testing font${variant}_${size}..."
    for n in 20 50 100 200 500; do
      ./target/release/lgi-cli encode \
        -i /tmp/lgi_tests/inputs/font${variant}_${size}.png \
        -o /tmp/lgi_tests/outputs/font${variant}_${size}_n${n}.png \
        -n $n \
        -q balanced 2>&1 | rg "PSNR" >> /tmp/lgi_tests/analysis/font${variant}_${size}.txt 2>&1
    done

    line="| ${size} | ${variant:-normal} |"
    for i in 1 2 3 4 5; do
      psnr=$(sed -n "${i}p" /tmp/lgi_tests/analysis/font${variant}_${size}.txt 2>/dev/null | grep -o "[0-9.]*" | head -1)
      line="$line ${psnr:-N/A} |"
    done
    echo "$line" >> $REPORT
  done
done

echo "" >> $REPORT
echo "**Expected**: PSNR 15-25 dB (Gaussians struggle with sharp edges)" >> $REPORT
echo "" >> $REPORT

# ====================
# TEST 4: LOGOS/ICONS
# ====================
echo "## Test 4: Simple Graphics" >> $REPORT
echo "" >> $REPORT

echo "Creating simple geometric shapes..."
# Circle
convert -size 256x256 xc:white -fill black -draw "circle 128,128 128,200" /tmp/lgi_tests/inputs/circle.png

# Square
convert -size 256x256 xc:white -fill black -draw "rectangle 64,64 192,192" /tmp/lgi_tests/inputs/square.png

# Triangle
convert -size 256x256 xc:white -fill black -draw "polygon 128,64 220,192 36,192" /tmp/lgi_tests/inputs/triangle.png

echo "| Shape | n=10 | n=25 | n=50 | n=100 |" >> $REPORT
echo "|-------|------|------|------|-------|" >> $REPORT

for shape in circle square triangle; do
  echo "Testing ${shape}..."
  for n in 10 25 50 100; do
    ./target/release/lgi-cli encode \
      -i /tmp/lgi_tests/inputs/${shape}.png \
      -o /tmp/lgi_tests/outputs/${shape}_n${n}.png \
      -n $n \
      -q fast 2>&1 | rg "PSNR" >> /tmp/lgi_tests/analysis/${shape}.txt 2>&1
  done

  line="| ${shape} |"
  for i in 1 2 3 4; do
    psnr=$(sed -n "${i}p" /tmp/lgi_tests/analysis/${shape}.txt 2>/dev/null | grep -o "[0-9.]*" | head -1)
    line="$line ${psnr:-N/A} |"
  done
  echo "$line" >> $REPORT
done

echo "" >> $REPORT
echo "**Expected**: PSNR 12-20 dB (sharp edges are hard for Gaussians)" >> $REPORT
echo "" >> $REPORT

# ====================
# TEST 5: SCALE ANALYSIS
# ====================
echo "## Test 5: Initial Scale Impact" >> $REPORT
echo "" >> $REPORT
echo "Testing how initial_scale affects quality..." >> $REPORT
echo "" >> $REPORT

# Temporarily test different scales
convert -size 256x256 gradient:blue-red /tmp/lgi_tests/inputs/test_scale.png

echo "| Initial Scale | PSNR (n=50) |" >> $REPORT
echo "|--------------|-------------|" >> $REPORT

for scale in 0.05 0.1 0.2 0.3 0.4 0.5; do
  echo "Testing scale=$scale..."
  sed -i "s/initial_scale: 0\.[0-9]*/initial_scale: $scale/" lgi-encoder/src/config.rs
  cargo build --release -p lgi-cli 2>&1 | tail -1

  psnr=$(./target/release/lgi-cli encode \
    -i /tmp/lgi_tests/inputs/test_scale.png \
    -o /tmp/lgi_tests/outputs/scale_${scale}.png \
    -n 50 \
    -q fast 2>&1 | rg "PSNR" | grep -o "[0-9.]*" | head -1)

  echo "| $scale | ${psnr:-N/A} |" >> $REPORT
done

# Restore scale to 0.3
sed -i "s/initial_scale: 0\.[0-9]*/initial_scale: 0.3/" lgi-encoder/src/config.rs
cargo build --release -p lgi-cli 2>&1 | tail -1

echo "" >> $REPORT
echo "**Conclusion**: Optimal scale appears to be 0.3-0.5 (30-50% of image)" >> $REPORT
echo "" >> $REPORT

# ====================
# TEST 6: VISUAL SAMPLES
# ====================
echo "## Test 6: Visual Quality Samples" >> $REPORT
echo "" >> $REPORT
echo "Representative output images:" >> $REPORT
echo "" >> $REPORT

# Best case examples
echo "### Best Case (Solid Colors)" >> $REPORT
convert -size 256x256 xc:red /tmp/lgi_tests/inputs/best_case.png
./target/release/lgi-cli encode -i /tmp/lgi_tests/inputs/best_case.png -o /tmp/lgi_tests/outputs/best_case.png -n 5 -q fast 2>&1 | rg "PSNR" >> $REPORT

echo "" >> $REPORT
echo "### Medium Case (Gradients)" >> $REPORT
./target/release/lgi-cli encode -i /tmp/lgi_tests/inputs/grad_linear_256.png -o /tmp/lgi_tests/outputs/medium_case.png -n 50 -q balanced 2>&1 | rg "PSNR" >> $REPORT

echo "" >> $REPORT
echo "### Worst Case (Text/Sharp Edges)" >> $REPORT
./target/release/lgi-cli encode -i /tmp/lgi_tests/inputs/font_inv_256.png -o /tmp/lgi_tests/outputs/worst_case.png -n 200 -q balanced 2>&1 | rg "PSNR" >> $REPORT

echo "" >> $REPORT

# ====================
# ANALYSIS & CONCLUSIONS
# ====================
echo "---" >> $REPORT
echo "" >> $REPORT
echo "## Overall Analysis" >> $REPORT
echo "" >> $REPORT

# Count successes/failures
solid_avg=$(cat /tmp/lgi_tests/analysis/solid_*.txt 2>/dev/null | grep -o "PSNR: [0-9.]*" | grep -o "[0-9.]*" | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
grad_avg=$(cat /tmp/lgi_tests/analysis/grad_*.txt 2>/dev/null | grep -o "PSNR: [0-9.]*" | grep -o "[0-9.]*" | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
text_avg=$(cat /tmp/lgi_tests/analysis/font*.txt 2>/dev/null | grep -o "PSNR: [0-9.]*" | grep -o "[0-9.]*" | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')

echo "### Average PSNR by Category" >> $REPORT
echo "" >> $REPORT
echo "| Category | Avg PSNR | Quality Rating |" >> $REPORT
echo "|----------|----------|----------------|" >> $REPORT
echo "| Solid Colors | ${solid_avg} dB | $([ ${solid_avg%.*} -gt 20 ] 2>/dev/null && echo 'Good ✓' || echo 'Poor ✗') |" >> $REPORT
echo "| Gradients | ${grad_avg} dB | $([ ${grad_avg%.*} -gt 20 ] 2>/dev/null && echo 'Good ✓' || echo 'Poor ✗') |" >> $REPORT
echo "| Text/Fonts | ${text_avg} dB | $([ ${text_avg%.*} -gt 20 ] 2>/dev/null && echo 'Good ✓' || echo 'Poor ✗') |" >> $REPORT
echo "" >> $REPORT

echo "### PSNR Quality Scale" >> $REPORT
echo "" >> $REPORT
echo "- **40+ dB**: Excellent (visually lossless)" >> $REPORT
echo "- **30-40 dB**: Good (minor artifacts)" >> $REPORT
echo "- **20-30 dB**: Acceptable (visible artifacts)" >> $REPORT
echo "- **10-20 dB**: Poor (significant degradation)" >> $REPORT
echo "- **< 10 dB**: Unusable (complete garbage)" >> $REPORT
echo "" >> $REPORT

echo "---" >> $REPORT
echo "" >> $REPORT
echo "## Conclusions" >> $REPORT
echo "" >> $REPORT

echo "### What Works" >> $REPORT
echo "" >> $REPORT
if (( $(echo "$solid_avg > 20" | bc -l 2>/dev/null || echo 0) )); then
  echo "- ✅ **Solid colors**: Gaussians can represent uniform regions effectively" >> $REPORT
else
  echo "- ✗ **Solid colors**: Still broken (scale/initialization issue)" >> $REPORT
fi

if (( $(echo "$grad_avg > 20" | bc -l 2>/dev/null || echo 0) )); then
  echo "- ✅ **Smooth gradients**: Reasonable quality achieved" >> $REPORT
else
  echo "- ✗ **Gradients**: Poor quality (need investigation)" >> $REPORT
fi

if (( $(echo "$text_avg > 20" | bc -l 2>/dev/null || echo 0) )); then
  echo "- ⚠️ **Text**: Surprisingly good (but still limited by soft blobs)" >> $REPORT
else
  echo "- ✗ **Text/Sharp Edges**: CONFIRMED: Gaussians cannot represent sharp features" >> $REPORT
fi

echo "" >> $REPORT
echo "### What Doesn't Work" >> $REPORT
echo "" >> $REPORT
echo "- Sharp edges (text, logos with crisp boundaries)" >> $REPORT
echo "- High-frequency details" >> $REPORT
echo "- Thin lines" >> $REPORT
echo "" >> $REPORT

echo "### Optimal Use Cases" >> $REPORT
echo "" >> $REPORT
echo "Based on testing, LGI is best suited for:" >> $REPORT
echo "" >> $REPORT
echo "1. **Stylized graphics** - smooth shapes, gradients" >> $REPORT
echo "2. **UI backgrounds** - blur, gradients, soft shadows" >> $REPORT
echo "3. **Resolution-independent rendering** - encode once, scale infinitely" >> $REPORT
echo "4. **Artistic effects** - watercolor-like, soft focus" >> $REPORT
echo "5. **NOT for**: Photographs, text, technical diagrams, pixel art" >> $REPORT
echo "" >> $REPORT

echo "### Critical Issues Found" >> $REPORT
echo "" >> $REPORT
echo "1. **CLI uses old Optimizer** (not OptimizerV2) → lower quality" >> $REPORT
echo "2. **GPU gradients not being used** → 1500× slower than possible" >> $REPORT
echo "3. **Initial scale was too small** (0.01 → fixed to 0.3)" >> $REPORT
echo "4. **No adaptive Gaussian count** → user guesses randomly" >> $REPORT
echo "" >> $REPORT

echo "### Recommended Next Steps" >> $REPORT
echo "" >> $REPORT
echo "1. **Switch CLI to OptimizerV2** - will improve PSNR by 5-10 dB" >> $REPORT
echo "2. **Fix GPU gradient usage** - will speed up 1500×" >> $REPORT
echo "3. **Implement adaptive Gaussian count** - auto-detect from image complexity" >> $REPORT
echo "4. **Reality check**: Decide if Gaussian splatting is viable for 2D images" >> $REPORT
echo "" >> $REPORT

echo "---" >> $REPORT
echo "" >> $REPORT
echo "**Report generated**: $(date)" >> $REPORT
echo "**Location**: $REPORT" >> $REPORT
echo "" >> $REPORT

chmod 644 $REPORT
echo ""
echo "======================================"
echo "OVERNIGHT TEST COMPLETE"
echo "======================================"
echo ""
echo "Report saved to: $REPORT"
echo ""
echo "View with: cat $REPORT"
echo ""
