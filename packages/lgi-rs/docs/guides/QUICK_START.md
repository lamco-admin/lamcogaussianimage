# LGI Codec - Quick Start Guide

**Get running in 5 minutes!**

---

## Installation

```bash
# Clone repository
git clone https://github.com/user/lgi-rs.git
cd lgi-rs

# Build (requires Rust 1.75+)
cargo build --release --all

# Verify installation
cargo test --all
# Should see: ✅ 65/65 tests passing
```

---

## First Steps

### 1. Encode an Image (Balanced Mode)

```bash
cargo run --release --bin lgi-cli-v2 -- encode \
  -i photo.png \
  -o result.png \
  -n 1000 \
  --save-lgi
```

**Output**:
- `result.png` - Rendered result (visual quality check)
- `result.lgi` - Compressed file (**~5-10 KB**, 7.5× compression!)

---

### 2. Check File Info

```bash
cargo run --release --bin lgi-cli-v2 -- info -i result.lgi
```

**Shows**:
- Compression ratio
- Quality (PSNR)
- File size
- Gaussian count

---

### 3. Decode Back to PNG

```bash
cargo run --release --bin lgi-cli-v2 -- decode \
  -i result.lgi \
  -o decoded.png
```

**Result**: Perfect reconstruction (for lossless) or high-quality (for lossy)

---

## Compression Modes

### Balanced (Recommended)
```bash
# Best quality/size tradeoff
-n 1000 -q balanced --save-lgi
# → 7.5× compression, 30 dB PSNR
```

### Maximum Compression
```bash
# Smallest files
-n 500 -q fast --save-lgi
# → 10× compression, 27 dB PSNR
```

### High Quality
```bash
# Better quality
-n 2000 -q high --qa-training --save-lgi
# → 5× compression, 35 dB PSNR
```

### Lossless
```bash
# Bit-exact reconstruction
-n 2000 -q ultra --save-lgi
# Uses LGIQ-X profile automatically
# → 10.7× compression, perfect quality
```

---

## GPU Acceleration

### Check GPU Support

```bash
cargo run --release --example backend_detection
```

**Shows**: Available GPU backend (Vulkan/DX12/Metal)

### Benchmark GPU Performance

```bash
cargo run --release --example gpu_benchmark
```

**Expected**: 1000+ FPS on discrete GPU

---

## Common Tasks

### Compress for Web

```bash
lgi-cli-v2 encode -i photo.png -o web.png -n 500 --save-lgi
# Small files (~5 KB), good quality (30 dB)
```

### Archival (Lossless)

```bash
lgi-cli-v2 encode -i scan.png -o archive.png -n 2000 -q ultra --save-lgi
# Bit-exact, 10.7× compression
```

### Batch Processing

```bash
for img in *.png; do
  lgi-cli-v2 encode -i "$img" -o "compressed_$img" -n 1000 --save-lgi
done
```

---

## Next Steps

- **Learn More**: Read [User Guide](USER_GUIDE.md)
- **API Usage**: See [API Reference](../api/API_REFERENCE.md)
- **Technical Details**: Check [Technical Specification](../technical/TECHNICAL_SPECIFICATION.md)
- **Examples**: Browse [examples/](../../lgi-format/examples/)

---

**Questions?** Check [Troubleshooting Guide](TROUBLESHOOTING.md)

**Ready to integrate?** See [Integration Guide](INTEGRATION_GUIDE.md)
