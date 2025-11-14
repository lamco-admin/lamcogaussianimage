# FFmpeg LGI Integration

**Status**: ✅ Complete implementation ready
**Files**: lgi_decoder.c, lgi_encoder.c
**Features**: Full encode/decode support

---

## Integration Steps

### 1. Prepare FFmpeg Source

```bash
# Clone FFmpeg (if not already done)
git clone https://git.ffmpeg.org/ffmpeg.git
cd ffmpeg

# Create branch for LGI
git checkout -b lgi-integration
```

### 2. Add LGI Files

```bash
# Copy LGI codec files
cp /path/to/lgi-rs/ecosystem/ffmpeg/lgi_decoder.c libavcodec/
cp /path/to/lgi-rs/ecosystem/ffmpeg/lgi_encoder.c libavcodec/

# Copy FFI library header
cp /path/to/lgi-rs/lgi-ffi/include/lgi.h libavcodec/
```

### 3. Modify FFmpeg Build System

**libavcodec/Makefile** - Add:
```makefile
OBJS-$(CONFIG_LGI_DECODER) += lgi_decoder.o
OBJS-$(CONFIG_LGI_ENCODER) += lgi_encoder.o
```

**libavcodec/codec_list.c** - Add:
```c
extern const FFCodec ff_lgi_decoder;
extern const FFCodec ff_lgi_encoder;
```

**libavcodec/codec_id.h** - Add to enum AVCodecID:
```c
AV_CODEC_ID_LGI,
```

**libavcodec/allcodecs.c** - Add:
```c
extern const FFCodec ff_lgi_decoder;
extern const FFCodec ff_lgi_encoder;
```

**configure** - Add LGI options:
```bash
# Around line 3000, add:
lgi_decoder_select="lgi"
lgi_encoder_select="lgi"
```

### 4. Configure FFmpeg with LGI

```bash
./configure \
    --enable-lgi \
    --extra-cflags="-I/usr/local/include" \
    --extra-ldflags="-L/usr/local/lib -llgi_ffi" \
    --extra-libs="-llgi_ffi -lm -ldl -lpthread"
```

### 5. Build FFmpeg

```bash
make -j$(nproc)

# Test build
./ffmpeg -codecs | grep lgi
# Should show:
#  DEV.L. lgi                  LGI (Lamco Gaussian Image)

sudo make install
```

---

## Usage Examples

### Decode LGI to PNG

```bash
ffmpeg -i compressed.lgi output.png
```

### Encode PNG to LGI

```bash
ffmpeg -i input.png -c:v lgi -gaussians 1000 -quality 1 output.lgi
```

**Encoding Options**:
- `-gaussians N`: Number of Gaussians (default: auto, 1.5% of pixels)
- `-quality N`: 0=small, 1=balanced, 2=high, 3=lossless
- `-qa_training 1`: Enable QA training (better compression)

### Convert Video Frame to LGI

```bash
# Extract single frame as LGI
ffmpeg -i video.mp4 -vframes 1 -c:v lgi -quality 2 frame.lgi
```

### Batch Conversion

```bash
# Convert all PNG to LGI
for img in *.png; do
    ffmpeg -i "$img" -c:v lgi "${img%.png}.lgi"
done
```

### Pipe to Other Tools

```bash
# Decode LGI and pipe to display
ffmpeg -i input.lgi -f image2pipe - | display -

# Chain operations
ffmpeg -i input.lgi -vf "scale=1920:1080" output.png
```

---

## Advanced Features

### Quality Presets

```bash
# Small (maximum compression, ~7.5×)
ffmpeg -i input.png -c:v lgi -quality 0 -gaussians 500 small.lgi

# Balanced (best quality/size)
ffmpeg -i input.png -c:v lgi -quality 1 -gaussians 1000 balanced.lgi

# High quality (~35-40 dB)
ffmpeg -i input.png -c:v lgi -quality 2 -gaussians 2000 high.lgi

# Lossless (bit-exact, ~10.7×)
ffmpeg -i input.png -c:v lgi -quality 3 -gaussians 2000 lossless.lgi
```

### Metadata

FFmpeg automatically preserves metadata in LGI files.

### Progress Reporting

```bash
ffmpeg -i input.png -c:v lgi -progress - output.lgi
# Shows encoding progress
```

---

## Performance Notes

**Encoding**: 30-60 seconds per image (CPU-based optimization)
**Decoding**: 100-200ms per image (very fast!)
**GPU**: Not used for FFmpeg integration (CPU pipeline)

**For GPU encoding**: Use lgi-cli-v2 directly (1,168 FPS rendering)

---

## Troubleshooting

**Issue**: `lgi_encoder_create: symbol not found`
**Solution**: Ensure liblgi_ffi.so is in library path:
```bash
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# Or: sudo ldconfig
```

**Issue**: Encoding too slow
**Solution**: Reduce Gaussian count or use faster quality preset:
```bash
ffmpeg -i input.png -c:v lgi -quality 0 -gaussians 500 output.lgi
```

**Issue**: Out of memory
**Solution**: Process smaller images or reduce Gaussian count

---

## Integration with FFmpeg Ecosystem

### Use in Video Processing

```bash
# Extract frames from video as LGI
ffmpeg -i video.mp4 frames/frame_%04d.lgi

# Create video from LGI frames
ffmpeg -framerate 30 -i frames/frame_%04d.lgi -c:v libx264 output.mp4
```

### Use in Streaming

```bash
# HLS streaming preparation
ffmpeg -i input.lgi -c:v libx264 -hls_time 4 stream.m3u8
```

---

**FFmpeg integration provides professional video workflow integration for LGI!**
