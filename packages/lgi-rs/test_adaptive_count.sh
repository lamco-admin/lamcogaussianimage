#!/bin/bash
# Test adaptive Gaussian count estimation

cd /home/greg/gaussian-image-projects/lgi-rs

echo "======================================"
echo "ADAPTIVE GAUSSIAN COUNT TESTS"
echo "======================================"
echo ""

# Test 1: Solid color (should need 1 Gaussian!)
echo "TEST 1: Solid Red - Should estimate ~1 Gaussian"
echo "----------------------------------------------"
cat > /tmp/test_count.rs << 'EOF'
use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use lgi_encoder::{estimate_gaussian_count, QualityTarget};

fn main() {
    // Solid red image
    let mut image = ImageBuffer::new(256, 256);
    for pixel in &mut image.data {
        *pixel = Color4::new(1.0, 0.0, 0.0, 1.0);
    }

    let count = estimate_gaussian_count(&image, QualityTarget::Balanced);
    println!("Estimated Gaussians for solid red 256x256: {}", count);

    if count == 1 {
        println!("✓ CORRECT: Solid color needs 1 Gaussian");
    } else {
        println!("✗ WRONG: Should be 1, got {}", count);
    }
}
EOF

rustc --edition 2021 -L target/release/deps -L target/release \
    --extern lgi_core=target/release/liblgi_core.rlib \
    --extern lgi_math=target/release/liblgi_math.rlib \
    --extern lgi_encoder=target/release/liblgi_encoder.rlib \
    /tmp/test_count.rs -o /tmp/test_count 2>&1 | head -5

if [ -f /tmp/test_count ]; then
    /tmp/test_count
else
    echo "Compilation failed, using CLI instead..."
fi

echo ""
echo "======================================"
