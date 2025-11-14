# Gaussian Image Codec: Technical Implementation Framework

## Architecture Overview and Algorithm Specifications

This document provides detailed technical specifications for building a production-grade Gaussian image codec, including algorithm pseudocode, architectural decisions, and performance characteristics.

---

## Section 1: Core Representation and Rendering

### 1.1 Data Structure Specification

```python
# Gaussian Representation (8 parameters per Gaussian)
class Gaussian2D:
    # Spatial parameters
    position_x: float32       # μx
    position_y: float32       # μy
    
    # Covariance via Cholesky decomposition
    cholesky_1: float32       # l₁ (diagonal element)
    cholesky_2: float32       # l₂ (diagonal element)
    cholesky_3: float32       # l₃ (off-diagonal element)
    
    # Color parameters
    red: float32              # R value [0, 1]
    green: float32            # G value [0, 1]
    blue: float32             # B value [0, 1]
    
    # Optional: opacity (can be folded into color via pre-multiplication)
    opacity: float32          # α ∈ [0, 1]
    
    # Methods
    def covariance_matrix() -> Matrix2x2:
        """Reconstruct 2×2 covariance matrix from Cholesky factors"""
        L = [[cholesky_1, 0],
             [cholesky_3, cholesky_2]]
        return L @ L.T
    
    def inverse_covariance() -> Matrix2x2:
        """For efficiency, cache or compute on-demand"""
        Σ = self.covariance_matrix()
        return inverse(Σ)
    
    def evaluate_at(x, y) -> float:
        """Evaluate Gaussian value at pixel (x, y)"""
        dx = x - position_x
        dy = y - position_y
        
        # Compute Mahalanobis distance
        Σ_inv = self.inverse_covariance()
        dist_sq = dx**2 * Σ_inv[0,0] + dy**2 * Σ_inv[1,1] + \
                  2 * dx * dy * Σ_inv[0,1]
        
        return opacity * exp(-0.5 * dist_sq)
```

### 1.2 Forward Rendering Algorithm (Tile-Based)

```python
def render_image_tiled(gaussians: List[Gaussian2D], 
                       image_width: int, 
                       image_height: int,
                       tile_size: int = 16) -> NDArray:
    """
    Render image from Gaussians using tile-based rasterization
    Optimal for GPU parallelization
    """
    
    # Allocate output
    output = zeros((image_height, image_width, 3))
    
    # Sort Gaussians by depth (z-order) or use painter's algorithm
    # For 2D, typically use raster-scan order or front-to-back
    sorted_gaussians = sort_by_depth_or_size(gaussians)
    
    # Process in tiles
    for tile_y in range(0, image_height, tile_size):
        for tile_x in range(0, image_width, tile_size):
            
            # Determine tile boundaries
            y_end = min(tile_y + tile_size, image_height)
            x_end = min(tile_x + tile_size, image_width)
            
            # Initialize tile with white background
            tile = ones((y_end - tile_y, x_end - tile_x, 3))
            tile_alpha = zeros((y_end - tile_y, x_end - tile_x))
            
            # Accumulate Gaussians for this tile
            for gaussian in sorted_gaussians:
                
                # Early termination if tile fully opaque
                if all(tile_alpha >= 0.99):
                    break
                
                # Check if Gaussian contributes to tile (bounding box check)
                gauss_bounds = get_gaussian_support(gaussian, 
                                                    threshold=0.01)
                if not intersects_tile(gauss_bounds, (tile_x, tile_y, 
                                                       x_end, y_end)):
                    continue
                
                # Evaluate Gaussian over tile pixels
                for local_y in range(y_end - tile_y):
                    for local_x in range(x_end - tile_x):
                        
                        # World coordinates
                        x = tile_x + local_x
                        y = tile_y + local_y
                        
                        # Evaluate Gaussian
                        g_value = gaussian.evaluate_at(x, y)
                        
                        # Alpha blend
                        alpha_contrib = g_value
                        color_contrib = alpha_contrib * gaussian_color(gaussian)
                        
                        # Update pixel
                        tile[local_y, local_x] = \
                            tile[local_y, local_x] * (1 - alpha_contrib) + \
                            color_contrib
                        
                        tile_alpha[local_y, local_x] += alpha_contrib
            
            # Copy tile to output
            output[tile_y:y_end, tile_x:x_end] = tile
    
    return output
```

### 1.3 CUDA Kernel Pseudocode (Highly Optimized)

```cuda
// CUDA kernel for tile-based rasterization
__global__ void rasterize_gaussians_tiles(
    const Gaussian2D* gaussians,      // Device memory
    int num_gaussians,
    int image_width,
    int image_height,
    int tile_size,
    float3* output                    // Output image (width × height)
) {
    // Block dimensions: 16×16 threads per tile
    int tile_x = blockIdx.x * blockDim.x;
    int tile_y = blockIdx.y * blockDim.y;
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;
    
    int x = tile_x + local_x;
    int y = tile_y + local_y;
    
    if (x >= image_width || y >= image_height) return;
    
    // Shared memory for tile
    __shared__ float3 tile_color[16][16];
    __shared__ float tile_alpha[16][16];
    
    // Initialize with white
    tile_color[local_y][local_x] = make_float3(1.0f, 1.0f, 1.0f);
    tile_alpha[local_y][local_x] = 0.0f;
    __syncthreads();
    
    // Process each Gaussian
    for (int g = 0; g < num_gaussians; ++g) {
        Gaussian2D gauss = gaussians[g];
        
        // Early termination
        if (tile_alpha[local_y][local_x] > 0.99f) break;
        
        // Evaluate Gaussian at this pixel
        float dx = x - gauss.position_x;
        float dy = y - gauss.position_y;
        
        // Compute inverse covariance × distance vector
        float2x2 sigma_inv = inverse_cholesky(gauss.cholesky_1,
                                               gauss.cholesky_2,
                                               gauss.cholesky_3);
        
        float mahal_sq = dx * dx * sigma_inv.m00 +
                         dy * dy * sigma_inv.m11 +
                         2 * dx * dy * sigma_inv.m01;
        
        float gauss_value = exp(-0.5f * mahal_sq);
        float alpha_contrib = gauss.opacity * gauss_value;
        
        // Alpha blend
        float3 color = make_float3(gauss.red, gauss.green, gauss.blue);
        
        tile_color[local_y][local_x] = 
            tile_color[local_y][local_x] * (1 - alpha_contrib) +
            color * alpha_contrib;
        
        tile_alpha[local_y][local_x] += alpha_contrib;
    }
    
    __syncthreads();
    
    // Write output
    int pixel_idx = y * image_width + x;
    output[pixel_idx] = tile_color[local_y][local_x];
}
```

### 1.4 Backward Pass: Gradient Computation

```python
def compute_gradients(loss, gaussians, rendering_cache):
    """
    Compute gradients of loss w.r.t. Gaussian parameters
    Using automatic differentiation through rendering pipeline
    """
    
    # These are computed via PyTorch autograd
    # But structure matters for numerical stability:
    
    # Gradient w.r.t. position
    ∂L/∂μₓ ∝ (pixel_error) × (∂G/∂μₓ)  where G is Gaussian value
    ∂L/∂μᵧ ∝ (pixel_error) × (∂G/∂μᵧ)
    
    # Gradient w.r.t. Cholesky factors (via chain rule)
    ∂L/∂l₁ = ∂L/∂Σ × ∂Σ/∂l₁  (computed via Cholesky derivatives)
    ∂L/∂l₂ = ∂L/∂Σ × ∂Σ/∂l₂
    ∂L/∂l₃ = ∂L/∂Σ × ∂Σ/∂l₃
    
    # Gradient w.r.t. color
    ∂L/∂c_r ∝ alpha_contrib × (pixel_error_r)
    ∂L/∂c_g ∝ alpha_contrib × (pixel_error_g)
    ∂L/∂c_b ∝ alpha_contrib × (pixel_error_b)
    
    # Gradient w.r.t. opacity
    ∂L/∂α ∝ gaussian_value × (pixel_error · color)
    
    return {
        'grad_position': grad_pos,
        'grad_cholesky': grad_chol,
        'grad_color': grad_color,
        'grad_opacity': grad_opacity
    }
```

---

## Section 2: Optimization Pipeline

### 2.1 Training Algorithm for Image Fitting

```python
def fit_image_to_gaussians(image: NDArray, 
                           num_gaussians: int,
                           learning_rate: float = 0.01,
                           max_iterations: int = 10000,
                           use_content_aware_init: bool = True):
    """
    Main optimization loop to fit Gaussians to image
    """
    
    # Step 1: Initialization
    if use_content_aware_init:
        gaussians = initialize_from_superpixels(image, num_gaussians)
    else:
        gaussians = initialize_random(image.shape, num_gaussians)
    
    # Step 2: Optimizer setup
    optimizer = Adam(gaussians.parameters(), lr=learning_rate)
    
    # Step 3: Loss configuration
    loss_weights = {
        'L1': 0.2,
        'L2': 0.8,
        'SSIM': 0.1
    }
    
    # Step 4: Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    for iteration in range(max_iterations):
        
        # Forward pass
        rendered = render_image(gaussians, image.shape)
        
        # Compute loss (multi-component)
        loss_l1 = L1Loss(rendered, image)
        loss_l2 = L2Loss(rendered, image)
        loss_ssim = 1 - SSIM(rendered, image)
        
        loss = (loss_weights['L1'] * loss_l1 +
                loss_weights['L2'] * loss_l2 +
                loss_weights['SSIM'] * loss_ssim)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Optional: Adaptive density control
        if iteration % 100 == 0:
            perform_adaptive_densification(gaussians, rendered, image)
        
        # Logging and convergence checking
        if iteration % 100 == 0:
            psnr = compute_psnr(rendered, image)
            print(f"Iter {iteration}: Loss={loss:.4f}, PSNR={psnr:.2f}")
            
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > 500:
                    print("Early stopping")
                    break
    
    return gaussians
```

### 2.2 Adaptive Densification

```python
def perform_adaptive_densification(gaussians, rendered, target_image,
                                    gradient_threshold=0.0002):
    """
    Add/remove Gaussians based on reconstruction error
    """
    
    # Compute error map
    error = abs(rendered - target_image).mean(axis=2)
    
    # Compute gradient of error (indicates where more detail needed)
    error_gradient = compute_gradient_magnitude(error)
    
    # Identify Gaussians with high gradients → split them
    for gaussian in gaussians:
        if error_gradient[gaussian.position] > gradient_threshold:
            # Clone and slightly perturb this Gaussian
            gaussian_clone = clone_gaussian(gaussian)
            gaussian_clone.position += small_random_offset()
            gaussians.append(gaussian_clone)
    
    # Prune low-opacity Gaussians
    opacities = [g.opacity for g in gaussians]
    opacity_threshold = 0.01
    gaussians = [g for g in gaussians if g.opacity > opacity_threshold]
    
    return gaussians
```

### 2.3 Quantization-Aware Training

```python
def quantization_aware_training(gaussians, image,
                                 num_bits={'position': 16,
                                          'cholesky': 6,
                                          'color': 8},
                                 fine_tune_iterations=1000):
    """
    Fine-tune Gaussian parameters with simulated quantization
    """
    
    # Define quantizers
    quantizers = {
        'position': Quantizer(bits=num_bits['position'], signed=True),
        'cholesky': Quantizer(bits=num_bits['cholesky'], signed=False),
        'color': Quantizer(bits=num_bits['color'], signed=False)
    }
    
    optimizer = Adam(gaussians.parameters(), lr=0.001)
    
    for iteration in range(fine_tune_iterations):
        
        # Forward pass with fake quantization
        gaussians_quant = quantize_gaussians(gaussians, quantizers)
        
        # Render with quantized parameters
        rendered = render_image(gaussians_quant, image.shape)
        
        # Compute loss (now aware of quantization)
        loss = reconstruction_loss(rendered, image)
        
        # Backward pass (straight-through estimator)
        optimizer.zero_grad()
        loss.backward()
        
        # Update original (high-precision) parameters
        optimizer.step()
        
        if iteration % 100 == 0:
            psnr = compute_psnr(rendered, image)
            print(f"QAT Iter {iteration}: PSNR={psnr:.2f}")
    
    return gaussians
```

---

## Section 3: Compression Pipeline

### 3.1 Attribute Quantization and Encoding

```python
class CompressionPipeline:
    """
    Convert optimized Gaussians to compressed bitstream
    """
    
    def __init__(self, quantization_config):
        self.pos_quantizer = UniformQuantizer(bits=16)      # FP16
        self.cov_quantizer = UniformQuantizer(bits=6)       # 6-bit
        self.color_quantizer = ResidualVectorQuantizer(
            stages=3, bits_per_stage=8)
        self.entropy_coder = ANSCoder()
    
    def encode_image(self, gaussians, image_metadata):
        """
        Encode Gaussians to binary bitstream
        Returns: byte array
        """
        
        bitstream = ByteBuffer()
        
        # Step 1: Write header
        bitstream.write_u32(len(gaussians))  # Number of Gaussians
        bitstream.write_u32(image_metadata['width'])
        bitstream.write_u32(image_metadata['height'])
        
        # Step 2: Quantize all Gaussian attributes
        quantized_data = []
        for gaussian in gaussians:
            
            # Position (FP16)
            pos_quant = self.pos_quantizer.quantize(
                [gaussian.position_x, gaussian.position_y])
            
            # Covariance (6-bit per element)
            chol_quant = self.cov_quantizer.quantize(
                [gaussian.cholesky_1, gaussian.cholesky_2, 
                 gaussian.cholesky_3])
            
            # Color (RVQ)
            color_quant = self.color_quantizer.quantize(
                [gaussian.red, gaussian.green, gaussian.blue])
            
            quantized_data.append({
                'position': pos_quant,
                'cholesky': chol_quant,
                'color': color_quant
            })
        
        # Step 3: Entropy encode quantized values
        for attr_type in ['position', 'cholesky', 'color']:
            attribute_values = [d[attr_type] for d in quantized_data]
            
            # Learn probability distribution
            pmf = estimate_pmf(attribute_values)
            
            # Encode with ANS
            encoded = self.entropy_coder.encode(attribute_values, pmf)
            bitstream.write_bytes(encoded)
        
        return bitstream.to_bytes()
    
    def decode_image(self, bitstream):
        """
        Decode bitstream back to Gaussians
        """
        
        reader = ByteReader(bitstream)
        
        # Read header
        num_gaussians = reader.read_u32()
        width = reader.read_u32()
        height = reader.read_u32()
        
        gaussians = []
        
        # Decode each Gaussian
        for i in range(num_gaussians):
            
            # Decode position
            pos_quant = self.entropy_coder.decode(reader)
            position = self.pos_quantizer.dequantize(pos_quant)
            
            # Decode covariance
            chol_quant = self.entropy_coder.decode(reader)
            cholesky = self.cov_quantizer.dequantize(chol_quant)
            
            # Decode color
            color_quant = self.entropy_coder.decode(reader)
            color = self.color_quantizer.dequantize(color_quant)
            
            # Reconstruct Gaussian
            gaussian = Gaussian2D(
                position_x=position[0],
                position_y=position[1],
                cholesky_1=cholesky[0],
                cholesky_2=cholesky[1],
                cholesky_3=cholesky[2],
                red=color[0],
                green=color[1],
                blue=color[2]
            )
            
            gaussians.append(gaussian)
        
        return gaussians, (width, height)
```

### 3.2 Bits-Back Coding Integration (Optional)

```python
def encode_with_bits_back(gaussians, image):
    """
    Optional: Use bits-back coding for further compression
    Trade-off: More complex decoding
    """
    
    # Initialize ANS state with random bits
    ans = ANSCoder()
    initial_bits = 1024  # Or computed based on data
    
    # Encode Gaussian attributes as sequence
    sequence = flatten_gaussians(gaussians)
    
    # Apply bits-back iteratively
    for stage in range(num_stages):
        
        # Encode using conditional distribution
        pmf = get_conditional_pmf(sequence, stage)
        encoded = ans.encode(sequence[stage], pmf)
    
    return ans.finalize()
```

---

## Section 4: Performance Characteristics and Optimization

### 4.1 Memory Requirements Analysis

```python
# For 1000×1000 image with N Gaussians

def estimate_memory(N, image_width, image_height):
    """
    Memory requirements breakdown
    """
    
    # Gaussian parameters (full precision)
    gaussian_params = N * 8 * 4  # 8 float32s per Gaussian
    
    # Tile buffers (for 16×16 tiles)
    num_tiles = (image_width // 16) * (image_height // 16)
    tile_buffers = num_tiles * (16 * 16 * 4) * 4  # float32 RGBA
    
    # Intermediate gradients during training
    gradient_storage = N * 8 * 4  # Same as params
    
    # Rendering intermediate buffers
    render_buffer = image_width * image_height * 12  # RGB float32
    
    total = (gaussian_params + tile_buffers + 
             gradient_storage + render_buffer)
    
    return {
        'gaussian_params_mb': gaussian_params / 1e6,
        'tile_buffers_mb': tile_buffers / 1e6,
        'gradient_storage_mb': gradient_storage / 1e6,
        'render_buffer_mb': render_buffer / 1e6,
        'total_mb': total / 1e6
    }

# Example: 10,000 Gaussians
# gaussian_params: ~0.3 MB
# tile_buffers: ~64 MB  
# gradient_storage: ~0.3 MB
# render_buffer: ~12 MB
# TOTAL: ~76 MB (much smaller than NeRF/INR methods)
```

### 4.2 Rendering Performance Metrics

```python
# Benchmark configurations

# Config 1: Fast rendering (1000 FPS)
# - 1000×1000 image
# - 5000 Gaussians
# - 16×16 tile size
# - No antialiasing
# GPU: NVIDIA RTX 4090
# Expected: 0.5-1.0 ms per frame

# Config 2: High quality (500 FPS)
# - 1024×1024 image
# - 10000 Gaussians
# - 8×8 tile size
# - 2×2 supersampling
# GPU: NVIDIA RTX 4090
# Expected: 2-4 ms per frame

# Config 3: Mobile (60 FPS)
# - 512×512 image
# - 2000 Gaussians
# - 32×32 tile size
# GPU: NVIDIA Jetson Orin
# Expected: 16-20 ms per frame
```

### 4.3 Training Speed Improvements

```python
# Initialization strategy comparison

# Random initialization
# Time to PSNR 30: ~13-20 seconds

# Content-aware (superpixel) initialization
# Time to PSNR 30: ~1.5-2 seconds
# Speedup: 8-10×

# Multi-scale progressive addition
# Time to PSNR 30: ~2-3 seconds
# Speedup: 6-8×

# Recommendation: Use content-aware for 88% faster encoding
```

---

## Section 5: File Format Specification

### 5.1 GaussianCodec File Format (`.ggc`)

```
Header (64 bytes):
├─ Magic number (4 bytes): "GGC\x00"
├─ Version (2 bytes): 0x0100
├─ Flags (2 bytes): [compression_type, has_metadata, ...]
├─ Image width (4 bytes): uint32
├─ Image height (4 bytes): uint32
├─ Number of Gaussians (4 bytes): uint32
├─ Quantization config (4 bytes):
│  ├─ Position bits (8 bits): 16
│  ├─ Cholesky bits (8 bits): 6
│  ├─ Color bits (8 bits): 8
│  └─ Reserved (8 bits)
├─ Metadata size (4 bytes): uint32
└─ CRC32 checksum (4 bytes)

Metadata (optional):
├─ Creation timestamp (8 bytes)
├─ Original image hash (4 bytes)
├─ Quality metrics (PSNR, SSIM): variable
└─ Custom tags: variable

Gaussian data (compressed):
├─ Position stream: entropy-coded
├─ Cholesky stream: entropy-coded
├─ Color stream: RVQ-encoded and entropy-coded
└─ Optional bits-back data

Footer:
└─ CRC32 checksum of data
```

### 5.2 Serialization Functions

```python
def save_gaussian_codec(gaussians, image_metadata, filename):
    """Save to .ggc file"""
    with open(filename, 'wb') as f:
        # Write header
        f.write(b'GGC\x00')
        f.write(pack('>H', 0x0100))  # Version
        f.write(pack('>H', 0x0000))  # Flags
        f.write(pack('>I', image_metadata['width']))
        f.write(pack('>I', image_metadata['height']))
        f.write(pack('>I', len(gaussians)))
        # ... write rest of header
        
        # Compress and write Gaussian data
        compressed = CompressionPipeline().encode_image(gaussians, 
                                                         image_metadata)
        f.write(compressed)

def load_gaussian_codec(filename):
    """Load from .ggc file"""
    with open(filename, 'rb') as f:
        # Parse header
        magic = f.read(4)
        assert magic == b'GGC\x00'
        # ... parse rest of header
        
        # Decompress Gaussian data
        compressed_data = f.read()
        gaussians, metadata = CompressionPipeline().decode_image(
            compressed_data)
        
        return gaussians, metadata
```

---

## Section 6: Integration Points and API Design

### 6.1 Python API

```python
import gaussian_codec as gc

# Encoding workflow
image = cv2.imread('photo.jpg')
encoder = gc.GaussianEncoder(num_gaussians=10000)
gaussians = encoder.fit(image, max_time_seconds=5)
encoder.save('photo.ggc')

# Decoding workflow  
decoder = gc.GaussianDecoder()
gaussians, metadata = decoder.load('photo.ggc')
reconstructed = gaussians.render(metadata['width'], 
                                  metadata['height'])

# Quality metrics
psnr = gc.evaluate.psnr(image, reconstructed)
ssim = gc.evaluate.ssim(image, reconstructed)
lpips = gc.evaluate.lpips(image, reconstructed)
```

### 6.2 Command-Line Interface

```bash
# Encode image
$ gaussian-codec encode photo.jpg --output photo.ggc --quality 30

# Decode
$ gaussian-codec decode photo.ggc --output reconstructed.png

# Compare
$ gaussian-codec compare original.jpg reconstructed.png

# Batch processing
$ gaussian-codec batch-encode *.jpg --quality 30

# Extract metadata
$ gaussian-codec info photo.ggc
```

---

## Section 7: Testing and Validation

### 7.1 Unit Test Structure

```python
class TestGaussian2D(unittest.TestCase):
    def test_covariance_reconstruction(self):
        """Verify Cholesky reconstruction"""
        g = Gaussian2D(...)
        Σ = g.covariance_matrix()
        assert Σ.is_positive_definite()
    
    def test_gaussian_evaluation(self):
        """Verify Gaussian value computation"""
        g = Gaussian2D(position=(0, 0), ...)
        assert g.evaluate_at(0, 0) > g.evaluate_at(5, 5)
    
    def test_alpha_blending(self):
        """Verify alpha compositing correctness"""
        # Test against reference implementation

class TestRenderingPipeline(unittest.TestCase):
    def test_tile_consistency(self):
        """Verify tile rendering matches full rendering"""
    
    def test_gradient_correctness(self):
        """Verify gradients via finite differences"""
    
    def test_cuda_kernel_accuracy(self):
        """Compare CUDA output with CPU reference"""

class TestCompressionPipeline(unittest.TestCase):
    def test_quantization_invertible(self):
        """Verify quantize/dequantize round-trip"""
    
    def test_entropy_coding(self):
        """Verify ANS encoding/decoding"""
    
    def test_file_format(self):
        """Verify file format parsing"""
```

### 7.2 Integration Tests on Standard Datasets

```python
def test_kodak_dataset():
    """Test on 24-image Kodak benchmark"""
    results = {}
    for image_path in kodak_images:
        encoder = GaussianEncoder()
        gaussians = encoder.fit(image)
        
        reconstructed = gaussians.render()
        
        psnr = compute_psnr(image, reconstructed)
        ssim = compute_ssim(image, reconstructed)
        bitrate = gaussians.compressed_size() * 8 / (width * height)
        
        results[image_path] = {'psnr': psnr, 'ssim': ssim, 'bpp': bitrate}
    
    # Compare to JPEG, JPEG2000, COIN++
    print_comparison_table(results)
```

---

## Conclusion

This technical framework provides concrete implementation guidance for building a production-grade Gaussian image codec. The modular architecture supports:

- **Incremental development**: Core → Optimization → Compression → Advanced features
- **Testing at each stage**: Unit tests validate mathematical correctness
- **Performance profiling**: Identify bottlenecks early
- **Multiple optimization paths**: Choose between speed and quality trade-offs

Key implementation priorities:
1. **GPU tile-based rasterization** is critical for 1000+ FPS rendering
2. **Content-aware initialization** is essential for fast encoding
3. **Quantization-aware training** determines compression performance
4. **Entropy coding quality** separates good from great codecs
