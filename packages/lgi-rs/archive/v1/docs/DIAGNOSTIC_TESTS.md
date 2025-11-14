# LGI Diagnostic Test Suite

**Purpose**: Verify the codec actually works before worrying about GPU optimization.

---

## CRITICAL ISSUE IDENTIFIED

**Observation**: Encoding produces garbage output (blank or corrupted images)
**Iterations**: Taking 5.5s each (CPU gradients, NOT using GPU!)
**Result Quality**: Unacceptable

This suggests **fundamental implementation problems**, not just performance issues.

---

## Test 1: Simple Synthetic Image (MOST IMPORTANT)

**Goal**: Verify basic Gaussian rendering works at all

### Create Test Image:
```bash
cd /home/greg/gaussian-image-projects/lgi-rs

# Create a simple 3-colored square image (256x256)
convert -size 256x256 xc:red xc:green xc:blue +append /tmp/test_simple.png
```

### Encode with MINIMAL Gaussians:
```bash
# Use lgi-cli with VERY FEW Gaussians
./target/release/lgi-cli encode \
  -i /tmp/test_simple.png \
  -o /tmp/test_simple_output.png \
  -n 10 \
  -q fast

# Check the output
display /tmp/test_simple_output.png  # Or open it
```

**Expected**: Should see SOMETHING resembling 3 colored regions
**If Broken**: Complete garbage = fundamental rendering bug

---

## Test 2: Single Gaussian Test

**Goal**: Can we even render ONE Gaussian?

### Create Test Script:
```bash
cat > /tmp/test_single_gaussian.sh << 'EOF'
#!/bin/bash
cd /home/greg/gaussian-image-projects/lgi-rs

# Create solid red 128x128 image
convert -size 128x128 xc:red /tmp/single_test.png

# Encode with JUST 1 Gaussian
./target/release/lgi-cli encode \
  -i /tmp/single_test.png \
  -o /tmp/single_output.png \
  -n 1 \
  -q fast

echo "Input:"
identify /tmp/single_test.png
echo ""
echo "Output:"
identify /tmp/single_output.png

# Show pixel values
echo ""
echo "Output first pixel (should be reddish):"
convert /tmp/single_output.png -crop 1x1+0+0 txt:-
EOF

chmod +x /tmp/test_single_gaussian.sh
/tmp/test_single_gaussian.sh
```

**Expected**: Output should be a blob of red (not perfect, but RED)
**If Broken**: Black/white/garbage = Gaussian math is wrong

---

## Test 3: Check CPU vs GPU Usage

**Goal**: Confirm if GPU is actually being used

### Check Iteration 0 Logs:
```bash
rg "Iteration 0.*render:" /tmp/viewer.log -A 1
```

**Should Show**:
```
📊 Iteration 0 render: X.XXms (using GPU)
📊 Iteration 0 gradients: X.XXms (using GPU)
```

**If Shows**:
```
📊 Iteration 0 render: X.XXms (using CPU)
📊 Iteration 0 gradients: 5000ms (using CPU)
```
= GPU is NOT being used!

---

## Test 4: Validate Gradient Computation

**Goal**: Are gradients even calculating correctly?

### Check for NaN in gradients:
```bash
rg "grad_scale=NaN" /tmp/viewer.log | wc -l
```

**If > 0**: Gradient computation is producing NaN (BROKEN!)

### Check gradient magnitudes:
```bash
rg "grad_mean|grad_max" /tmp/viewer.log | head -5
```

**Expected**: Non-zero, non-NaN numbers
**If Zero/NaN**: Gradients aren't working

---

## Test 5: Render-Only Test (No Optimization)

**Goal**: Test if rendering works without optimization

### Create Direct Render Test:
```bash
cat > test_render_only.sh << 'EOF'
#!/bin/bash
cd /home/greg/gaussian-image-projects/lgi-rs

# Create a .lgi file manually with known Gaussians
cat > /tmp/test_manual.lgi << 'LGIEOF'
LGI\0
[Header bytes - width, height, num_gaussians, etc]
LGIEOF

# Or use the viewer to load and immediately re-render
# without encoding
EOF
```

---

## Test 6: Check Image Format Compatibility

**Goal**: Verify input image loading works

```bash
# Test different formats
convert -size 64x64 gradient:red-blue /tmp/gradient.png
./target/release/lgi-cli encode -i /tmp/gradient.png -o /tmp/gradient_out.png -n 20 -q fast

# Check if it processed
ls -lh /tmp/gradient_out.png
file /tmp/gradient_out.png
```

---

## Test 7: Initialization Strategy Test

**Goal**: Check if Gaussian initialization is sensible

### Check Initial Gaussians:
```bash
rg "Initialized.*Gaussians" /tmp/viewer.log
rg "Starting optimization" /tmp/viewer.log -A 5
```

**Should show**:
- Reasonable number of Gaussians (not 0, not 1000000)
- Initial loss value (should be high but not infinite)

---

## CRITICAL DIAGNOSTICS TO RUN NOW

### Quick Test Commands:

```bash
# 1. Create simple test image
convert -size 128x128 xc:red /tmp/red.png

# 2. Encode with minimal settings
./target/release/lgi-cli encode -i /tmp/red.png -o /tmp/red_out.png -n 5 -q fast 2>&1 | tee /tmp/encode_test.log

# 3. Check output
echo "Input:"
identify /tmp/red.png
echo "Output:"
identify /tmp/red_out.png
echo "First pixel of output:"
convert /tmp/red_out.png -crop 1x1+0+0 txt:-

# 4. Check for errors
rg "ERROR|panic|NaN" /tmp/encode_test.log
```

---

## LIKELY BUGS TO INVESTIGATE

Based on "garbage output", check these:

### 1. Opacity Clamping Issue
```bash
rg "opacity.*clamp" lgi-encoder/src/optimizer_v2.rs
```
**Possible Bug**: Opacity clamped to 0, making everything invisible

### 2. Color Space Issue
```bash
rg "ColorSpace|srgb|linear" lgi-core/src/lib.rs
```
**Possible Bug**: sRGB vs linear RGB mismatch

### 3. Normalization Issue
```bash
rg "normalize|/ 255" lgi-encoder/src/
```
**Possible Bug**: Pixel values not normalized to [0,1]

### 4. Rendering Accumulation
```bash
rg "AccumulatedSum|AlphaComposite" lgi-gpu/src/shaders/
```
**Possible Bug**: Wrong compositing mode

### 5. Scale Initialization
```bash
rg "initial_scale|scale_x.*0\." lgi-encoder/src/
```
**Possible Bug**: Gaussians too small/large to be visible

---

## Expected Results vs Reality

### What SHOULD Happen:
- Simple images (solid colors, gradients) → Recognizable output
- 10-50 Gaussians → Blurry but visible approximation
- 500+ Gaussians → Reasonable quality

### What You're Seeing:
- Garbage/blank output = **FUNDAMENTAL BUG**
- Not a GPU/performance issue!

---

## Next Steps

1. **Run Test 1** (simple synthetic image) - 5 minutes
2. **Run Test 2** (single Gaussian) - 2 minutes
3. **Check logs** for NaN/errors - 1 minute
4. **Report findings** so we can fix the actual bug

**DO NOT worry about GPU optimization until basic rendering works!**

---

## Debugging Commands for You

```bash
# Quick visual test
convert -size 256x256 gradient:blue-red /tmp/grad.png
./target/release/lgi-cli encode -i /tmp/grad.png -o /tmp/grad_out.png -n 50 -q fast
compare /tmp/grad.png /tmp/grad_out.png /tmp/diff.png
display /tmp/diff.png

# Check if output is all black
convert /tmp/grad_out.png -format "%[mean]" info:
# If output is "0" = all black = BROKEN

# Check if output is all white
convert /tmp/grad_out.png -format "%[max]" info:
# If output is "65535" and mean is "65535" = all white = BROKEN
```

Run these and tell me what you see!
