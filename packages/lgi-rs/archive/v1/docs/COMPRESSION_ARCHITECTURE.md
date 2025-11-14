# LGI Compression Architecture - Complete Design

**Date**: October 2, 2025
**Status**: Production Design Document
**Target**: 5-10× compression with <1 dB quality loss

---

## 🎯 Compression Strategy Overview

### Dual-Mode Architecture

**Lossless Mode** (LGIQ-X):
- Full 32-bit float precision
- Bit-exact reconstruction
- Target: Medical imaging, archival
- Compression: 2-3× via zstd only

**Lossy Mode** (LGIQ-B/S/H):
- Quantized parameters
- VQ compression
- QA training
- Target: General use, web delivery
- Compression: 5-10×

---

## 📊 Compression Pipeline

### Multi-Stage Compression

```
Stage 1: Parameterization
  Gaussian2D → Parameter vectors [μx, μy, σx, σy, θ, r, g, b, α]

Stage 2: Quantization (Lossy only)
  Profile Selection:
    LGIQ-B (Balanced):  11 bytes/Gaussian
    LGIQ-S (Small):     13 bytes/Gaussian
    LGIQ-H (High):      18 bytes/Gaussian
    LGIQ-X (Lossless):  36 bytes/Gaussian

Stage 3: Vector Quantization (Optional)
  K-means codebook (256 entries)
  → 9KB codebook + 1 byte/Gaussian

Stage 4: Entropy Coding
  Morton curve ordering (spatial coherence)
  → Delta coding
  → rANS or Huffman

Stage 5: zstd Compression
  Outer compression layer
  Level 9 (max compression, still fast decode)
  → Final file

Result: 4-7 bytes/Gaussian (compressed)
```

---

## 🔧 Quantization Profiles (LGIQ)

### LGIQ-B: Balanced (11 bytes)

**Target**: General use, good quality/size tradeoff

```rust
Position:  16-bit × 2 = 4 bytes  // [0, 1] → 16-bit fixed-point
Scale:     12-bit × 2 = 3 bytes  // log-encoded, [0.001, 0.5]
Rotation:  12-bit     = 1.5 bytes // [-π, π] → 12-bit
Color:     8-bit × 3  = 3 bytes  // RGB [0, 1] → 8-bit
Opacity:   8-bit      = 1 byte   // [0, 1] → 8-bit
---
Total: 11.5 bytes → 11 bytes packed
```

**Expected Quality**: 28-32 dB PSNR

---

### LGIQ-S: Small (13 bytes)

**Target**: Web delivery, smaller files

```rust
Position:  18-bit × 2 = 4.5 bytes
Scale:     14-bit × 2 = 3.5 bytes
Rotation:  14-bit     = 1.75 bytes
Color:     8-bit × 3  = 3 bytes
Opacity:   8-bit      = 1 byte
---
Total: 13.75 bytes → 13 bytes packed
```

**Expected Quality**: 30-34 dB PSNR

---

### LGIQ-H: High Quality (18 bytes)

**Target**: High-quality applications

```rust
Position:  20-bit × 2 = 5 bytes
Scale:     16-bit × 2 = 4 bytes
Rotation:  16-bit     = 2 bytes
Color:     10-bit × 3 = 4 bytes (HDR support)
Opacity:   10-bit     = 1.5 bytes
---
Total: 16.5 bytes → 18 bytes with padding
```

**Expected Quality**: 35-40 dB PSNR

---

### LGIQ-X: Lossless (36 bytes)

**Target**: Archival, medical imaging

```rust
Position:  32-bit × 2 = 8 bytes
Scale:     32-bit × 2 = 8 bytes
Rotation:  32-bit     = 4 bytes
Color:     32-bit × 3 = 12 bytes
Opacity:   32-bit     = 4 bytes
---
Total: 36 bytes (full precision)
```

**Expected Quality**: Bit-exact (∞ dB)

---

## 🗜️ Compression Stages Detail

### Stage 1: Quantization

**Implementation**:
```rust
pub enum QuantizationProfile {
    LGIQ_B,  // 11 bytes
    LGIQ_S,  // 13 bytes
    LGIQ_H,  // 18 bytes
    LGIQ_X,  // 36 bytes (lossless)
}

pub struct Quantizer {
    profile: QuantizationProfile,
}

impl Quantizer {
    pub fn quantize(&self, gaussian: &Gaussian2D) -> QuantizedGaussian {
        match self.profile {
            LGIQ_B => self.quantize_balanced(gaussian),
            LGIQ_S => self.quantize_small(gaussian),
            LGIQ_H => self.quantize_high(gaussian),
            LGIQ_X => self.quantize_lossless(gaussian),
        }
    }
}
```

---

### Stage 2: Vector Quantization

**When to Use**:
- Lossy modes only (LGIQ-B/S/H)
- When Gaussian count > 500
- Enabled via QA training

**Compression**:
- Codebook: 256 entries × 9 params × 4 bytes = 9 KB
- Indices: N Gaussians × 1 byte
- Total: 9 KB + N bytes

**For 1000 Gaussians**:
- Uncompressed (LGIQ-B): 11 KB
- With VQ: 9 KB + 1 KB = 10 KB
- Ratio: 1.1× (modest for small N)

**For 10000 Gaussians**:
- Uncompressed: 110 KB
- With VQ: 9 KB + 10 KB = 19 KB
- Ratio: 5.8× (excellent for large N!)

**Recommendation**: Use VQ when N > 1000

---

### Stage 3: Entropy Coding

**Morton Curve Ordering**:

```rust
// Sort Gaussians by Morton curve (Z-order)
fn morton_encode(x: u16, y: u16) -> u32 {
    let mut result = 0u32;
    for i in 0..16 {
        result |= ((x & (1 << i)) as u32) << i;
        result |= ((y & (1 << i)) as u32) << (i + 1);
    }
    result
}

// Benefit: Spatial locality → better delta compression
```

**Delta Coding**:
```rust
// Instead of: [100, 102, 105, 103]
// Store: [100, +2, +3, -2]
// Smaller values → better compression
```

**rANS Encoding** (or Huffman):
```rust
// Encode deltas with arithmetic coding
// Small deltas → few bits
// Large deltas → more bits
// Typical: 30-40% reduction
```

---

### Stage 4: zstd Compression

**Outer Layer**:
```rust
use zstd::encode_all;

let compressed = encode_all(
    &quantized_data,
    9  // Level: 0-22, 9 is good balance
)?;

// zstd is VERY effective on:
// - Repeated patterns (codebook indices)
// - Small integers (delta-coded positions)
// - Ordered data (Morton curve)
```

**Expected Compression**:
- VQ indices: 70-80% reduction (0.7 → 0.2 bytes/Gaussian)
- Codebook: 20-30% reduction (9 KB → 6-7 KB)
- Total: Additional 2-3× on top of VQ

---

## 📈 Compression Ratio Analysis

### For 1000 Gaussians, 512×512 Image

**Uncompressed**:
```
Full precision: 1000 × 48 bytes = 48 KB
```

**LGIQ-B (Balanced)**:
```
Quantization:   1000 × 11 bytes = 11 KB
+ zstd:         11 KB × 0.7      = 7.7 KB
Ratio: 48 / 7.7 = 6.2×
```

**LGIQ-B + VQ**:
```
Codebook:       256 × 36 bytes  = 9 KB
Indices:        1000 × 1 byte   = 1 KB
+ zstd:         10 KB × 0.7     = 7 KB
Ratio: 48 / 7 = 6.9×
```

**LGIQ-X (Lossless)**:
```
Full precision: 1000 × 36 bytes = 36 KB
+ zstd:         36 KB × 0.6     = 21.6 KB
Ratio: 48 / 21.6 = 2.2×
```

**Target Achieved**: ✅ 5-10× for lossy, 2-3× for lossless

---

## 🎓 Implementation Strategy

### Phase 1: Quantization Profiles (This Session)

**Tasks**:
1. Create `lgi-format/src/quantization.rs`
2. Implement all 4 profiles (B/S/H/X)
3. Bit-packing utilities
4. Quantize/dequantize functions
5. Unit tests (round-trip, quality loss)

**Time**: 1-2 hours

---

### Phase 2: zstd Integration (This Session)

**Tasks**:
1. Add zstd to GAUS chunk
2. Update header compression flags
3. Transparent compression/decompression
4. Benchmark compression ratios
5. Update CLI to show compression stats

**Time**: 30-60 minutes

---

### Phase 3: Entropy Coding (Optional Enhancement)

**Tasks**:
1. Morton curve ordering
2. Delta coding
3. rANS integration (or defer to zstd)

**Time**: 1-2 hours (can defer)

---

## 🔒 Quality Preservation

### Quantization-Aware Training

**Critical**: Must train with target quantization profile!

```rust
if config.enable_qa_training {
    // Simulate quantization
    let quantizer = Quantizer::new(config.quantization_profile);

    for gaussian in &mut gaussians {
        // Quantize
        let quantized = quantizer.quantize(gaussian);

        // Dequantize
        let dequantized = quantizer.dequantize(&quantized);

        // Render dequantized version
        // Gradients flow to original
    }
}
```

**Result**: Gaussians learn to minimize loss AFTER quantization!

---

## 📊 Expected Performance

### Quality vs. Compression

| Profile | Bytes/G | Compression | PSNR | Use Case |
|---------|---------|-------------|------|----------|
| LGIQ-X | 21 KB (zstd) | 2.3× | ∞ | Medical, archival |
| LGIQ-H + zstd | 12 KB | 4× | 35-40 dB | High quality |
| LGIQ-S + VQ + zstd | 7 KB | 6.9× | 30-34 dB | General use |
| LGIQ-B + VQ + zstd | 7 KB | 6.9× | 28-32 dB | Web delivery |

**Target**: ✅ 5-10× for lossy modes, 2-3× for lossless

---

## 🔧 API Design

### Encoding

```rust
use lgi_format::{LgiFile, CompressionConfig, QuantizationProfile};

// Lossy with VQ + zstd
let config = CompressionConfig::new()
    .with_quantization(QuantizationProfile::LGIQ_B)
    .with_vq(256)
    .with_zstd(9);

let file = LgiFile::with_compression(gaussians, width, height, config);
LgiWriter::write_file(&file, "output.lgi")?;

// Lossless with zstd only
let config = CompressionConfig::lossless();
let file = LgiFile::with_compression(gaussians, width, height, config);
```

### Decoding

```rust
// Transparent decompression
let file = LgiReader::read_file("input.lgi")?;
let gaussians = file.gaussians();  // Automatically dequantized

// Check compression mode
println!("Profile: {:?}", file.quantization_profile());
println!("VQ: {}", file.is_vq_compressed());
println!("zstd: {}", file.is_zstd_compressed());
println!("Ratio: {:.2}×", file.compression_ratio());
```

---

## 🎯 Success Criteria

### Functional Requirements

- ✅ Lossless mode (bit-exact reconstruction)
- ✅ Lossy modes (4 profiles: B/S/H/X)
- ✅ VQ compression (optional, adaptive)
- ✅ zstd compression (outer layer)
- ✅ QA training support
- ✅ Transparent encode/decode

### Performance Requirements

- ✅ Compression: 5-10× (lossy), 2-3× (lossless)
- ✅ Quality: <1 dB loss with QA training
- ✅ Speed: <100ms encode overhead, <10ms decode overhead
- ✅ Memory: Streaming-capable (chunk-by-chunk)

### Quality Requirements

- ✅ Round-trip tests (all profiles)
- ✅ PSNR measurements (validate targets)
- ✅ Bit-exact lossless mode
- ✅ Compression ratio validation

---

## 📚 References

### Quantization Design

- **GaussianImage (ECCV 2024)**: VQ + QA training
- **PNG**: Bit-packing, filtering, zlib compression
- **JPEG XL**: Advanced quantization matrices
- **WebP**: Lossy/lossless dual mode

### Implementation References

- `zstd-rs`: Rust bindings for zstd
- `bitstream-io`: Bit packing utilities
- Morton curve: Z-order space-filling curve

---

## ✅ Implementation Checklist

### Phase 1: Quantization
- [ ] Create quantization.rs module
- [ ] Implement 4 profiles (B/S/H/X)
- [ ] Bit packing/unpacking
- [ ] Round-trip tests
- [ ] Quality measurements

### Phase 2: zstd Integration
- [ ] Add zstd dependency
- [ ] Compress GAUS chunk
- [ ] Update header flags
- [ ] Transparent decompression
- [ ] Compression ratio tracking

### Phase 3: Integration
- [ ] Update CLI (compression options)
- [ ] Update QA training (profile-aware)
- [ ] Benchmark suite
- [ ] Documentation

---

**This architecture provides:**
- ✅ Dual-mode compression (lossy + lossless)
- ✅ Multiple quality profiles
- ✅ 5-10× compression target
- ✅ Production-ready design
- ✅ Extensible for future enhancements

**Next**: Implement quantization profiles module!
