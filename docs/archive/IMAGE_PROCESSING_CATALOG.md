# COMPREHENSIVE CATALOG OF IMAGE PROCESSING, FEATURE DETECTION, AND COMPUTER VISION UTILITIES
## Gaussian Image Codec Repository

**Generated:** 2025-11-15  
**Repository:** /home/user/lamcogaussianimage  
**Scope:** All image processing, feature detection, computer vision algorithms, preprocessing, and analysis utilities

---

## TABLE OF CONTENTS
1. [Image Processing Algorithms](#image-processing-algorithms)
2. [Feature Detection Methods](#feature-detection-methods)
3. [Image Analysis Utilities](#image-analysis-utilities)
4. [Color Space Conversions](#color-space-conversions)
5. [Spatial Frequency & Filtering](#spatial-frequency--filtering)
6. [Structure Detection Algorithms](#structure-detection-algorithms)
7. [Preprocessing Pipelines](#preprocessing-pipelines)
8. [Quality Metrics & Loss Functions](#quality-metrics--loss-functions)
9. [Saliency Detection](#saliency-detection)
10. [Library Stack Summary](#library-stack-summary)

---

## IMAGE PROCESSING ALGORITHMS

### 1. Gaussian Blur / Gaussian Filtering
**File:** `/home/user/lamcogaussianimage/packages/lgi-rs/tools/preprocess_image.py`  
**Line:** 46 (imports from skimage.filters)  
**Function:** `skimage.filters.gaussian()` via scikit-image  
**Location in Code:**
- Used in saliency_utils.py line 29: `smap = filters.gaussian(smap, filter_size)` with configurable filter size
- Post-processing of saliency maps with 15-pixel filter (default)

**Purpose:** Smooth saliency maps to reduce noise; Gaussian blur for image processing  
**Parameters:** Filter size (sigma), typically 15 pixels for saliency  
**Implementation Details:**
- Smooths per-pixel saliency values
- Normalizes output: `smap -= smap.min(); smap /= smap.max()`

---

### 2. Image Resizing / Downsampling
**File:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/image_utils.py`  
**Lines:** 61-96 (load_images function)  
**Libraries:** PIL (Image.Resampling.BILINEAR, Image.Resampling.BOX)  

**Implementation Details:**
```python
# Bilinear downsampling for loading
image = image.resize((round(image.width/downsample_ratio), 
                     round(image.height/downsample_ratio)), 
                     resample=Image.Resampling.BILINEAR)

# BOX filter for zoom (line 122)
image = image.resize((round(width*zoom), round(height*zoom)), 
                     resample=Image.Resampling.BOX)
```

**Purpose:** Image downsampling for multi-resolution processing; scaling for visualization

---

### 3. Sobel Edge Detection
**File:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/image_utils.py`  
**Lines:** 51-58 (compute_image_gradients function)  
**Library:** scipy.ndimage.sobel  

**Implementation:**
```python
def compute_image_gradients(image):
    gy, gx = [], []
    for image_channel in image:
        gy.append(sobel(image_channel, 0))  # Vertical edges
        gx.append(sobel(image_channel, 1))  # Horizontal edges
    gy = norm(np.stack(gy, axis=0), ord=2, axis=0).astype(np.float32)
    gx = norm(np.stack(gx, axis=0), ord=2, axis=0).astype(np.float32)
    return gy, gx
```

**Output:** Per-pixel gradient magnitude in Y and X directions  
**Purpose:** Compute image gradients for edge strength analysis

---

### 4. Sobel Operator (OpenCV GPU-optimized)
**File:** `/home/user/lamcogaussianimage/packages/lgi-rs/tools/preprocess_image.py`  
**Lines:** 256-291 (_compute_gradient_map method)  
**Library:** cv2.Sobel (CPU), cv2.cuda (GPU optional)  

**Features:**
- **Kernel size:** 3×3
- **Bit depth:** CV_16S (signed 16-bit)
- **GPU acceleration:** Optional CUDA support via cv2.cuda.createSobelFilter
- **Normalization:** L2 magnitude normalization to [0,1]

**GPU Path:**
```python
gpu_grad_x = cv2.cuda.createSobelFilter(cv2.CV_8U, cv2.CV_16S, 1, 0, ksize=3)
gpu_grad_y = cv2.cuda.createSobelFilter(cv2.CV_8U, cv2.CV_16S, 0, 1, ksize=3)
grad_x = gpu_grad_x.apply(gpu_img).download()
grad_y = gpu_grad_y.apply(gpu_img).download()
```

---

### 5. Image Format Conversion
**File:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/image_utils.py`  
**Lines:** 99-114 (to_output_format function)  

**Conversions Supported:**
- Torch to NumPy tensors
- 3D to 2D (squeeze single channels)
- Range clipping to [0, 1]
- Gamma correction (inverse)
- Uint8 conversion (255 normalization)

---

## FEATURE DETECTION METHODS

### 1. Edge Detection (FLIP Feature Detector)
**File:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/flip.py`  
**Lines:** 529-567 (feature_detection function)  
**Type:** Gaussian-based edge detector using partial derivatives  

**Implementation:**
```python
def feature_detection(img_y, pixels_per_degree, feature_type):
    w = 0.082  # Peak-to-trough (2σ) of edge detection filter
    sd = 0.5 * w * pixels_per_degree
    radius = int(np.ceil(3 * sd))
    
    # Compute 2D Gaussian
    [x, y] = np.meshgrid(range(-radius, radius+1), range(-radius, radius+1))
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sd * sd))
    
    if feature_type == 'edge':
        Gx = np.multiply(-x, g)  # First derivative
    else:  # point detector
        Gx = np.multiply(x ** 2 / (sd * sd) - 1, g)  # Second derivative
```

**Feature Types:**
- **Edge:** First derivative Gaussian (detects luminance discontinuities)
- **Point:** Second derivative Gaussian (detects local extrema)

**Normalization:** Separate normalization for positive and negative weights  
**Output:** 2-channel feature maps (X and Y components)

---

### 2. Corner/Point Detection (Harris-like)
**File:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/flip.py`  
**Lines:** 550-567 (feature_detection with 'point' type)  
**Method:** Laplacian of Gaussian (LoG) approximation

---

### 3. Saliency Detection (Spectral Residual + EML-NET)

#### A. Spectral Residual Method (Fast)
**File:** `/home/user/lamcogaussianimage/packages/lgi-rs/tools/preprocess_image.py`  
**Lines:** 355-376 (_compute_saliency_map method)  
**Library:** cv2.saliency.StaticSaliencySpectralResidual_create()  

**Implementation:**
```python
saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
success, saliency = saliency_detector.computeSaliency(image_uint8)
# Normalize to [0, 1]
saliency = saliency / saliency.max()
```

**Computational Speed:** ~100ms for 768×512 images  
**Reference:** OpenCV built-in implementation of spectral residual

#### B. EML-NET Saliency (Deep Learning)
**File:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/saliency_utils.py`  
**Lines:** 1-35 (get_smap function)  

**Architecture:**
- **ImageNet backbone:** ResNet50 pretrained on ImageNet
- **Places backbone:** ResNet50 pretrained on Places dataset
- **Decoder:** Custom fusion decoder combining both feature streams
- **Models:** 3 weight files (res_imagenet.pth, res_places.pth, res_decoder.pth)

**Implementation Details:**
```python
def get_smap(image, path, filter_size=15):
    imagenet_model = resnet.resnet50(f"{path}/emlnet/res_imagenet.pth").cuda().eval()
    places_model = resnet.resnet50(f"{path}/emlnet/res_places.pth").cuda().eval()
    decoder_model = decoder.build_decoder(f"{path}/emlnet/res_decoder.pth", sod_res, 5, 5).cuda().eval()
    
    # Resize to standard SOD resolution
    image_sod = resize(image, (480, 640)).unsqueeze(0)
    
    with torch.no_grad():
        imagenet_feat = imagenet_model(image_sod, decode=True)
        places_feat = places_model(image_sod, decode=True)
        smap = decoder_model([imagenet_feat, places_feat])
    
    # Resize back to original
    smap = resize(smap.squeeze(0).detach().cpu(), image.shape[1:]).squeeze(0)
    
    # Post-process with Gaussian filter
    smap = filters.gaussian(smap, filter_size)
    smap -= smap.min()
    smap /= smap.max()
```

**Reference:** EML-NET (https://arxiv.org/abs/1805.01047)

---

## IMAGE ANALYSIS UTILITIES

### 1. Local Entropy Computation (Tile-Based)
**File:** `/home/user/lamcogaussianimage/packages/lgi-rs/tools/preprocess_image.py`  
**Lines:** 221-254 (_compute_entropy_map method)  
**Algorithm:** Histogram-based Shannon entropy

**Implementation:**
```python
def _compute_entropy_map(self, image: np.ndarray) -> np.ndarray:
    # Tile-based entropy computation
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = gray[y:y_end, x:x_end]
            # Histogram-based entropy
            hist, _ = np.histogram(tile, bins=256, range=(0, 1))
            hist = hist.astype(float) / hist.sum()
            hist = hist[hist > 0]  # Remove zeros
            entropy = -np.sum(hist * np.log2(hist))  # Shannon entropy formula
            entropy_map[y:y_end, x:x_end] = entropy
```

**Parameters:**
- **Tile size:** Default 16×16 pixels (configurable via --entropy-tile-size)
- **Bins:** 256 levels
- **Formula:** Shannon entropy: H = -Σ p(i) * log₂(p(i))

**Output:** Normalized entropy map [0, 1], higher values = more complex regions

---

### 2. Gradient Magnitude Analysis
**File:** `/home/user/lamcogaussianimage/packages/lgi-rs/tools/preprocess_image.py`  
**Lines:** 256-291  
**Formula:** ||∇I|| = √(Gx² + Gy²)

**Features:**
- **GPU acceleration:** Optional CUDA path
- **Normalization:** L2 norm to [0, 1]
- **Output:** Per-pixel edge strength

---

### 3. Texture Analysis (Haralick Features)
**File:** `/home/user/lamcogaussianimage/packages/lgi-rs/tools/preprocess_image.py`  
**Lines:** 293-334 (_compute_texture_map method)  
**Library:** mahotas.features.haralick  

**Implementation:**
```python
def _compute_texture_map(self, image: np.ndarray) -> np.ndarray:
    # Tile-based texture analysis
    tile_size = 32  # Larger tiles for texture analysis
    
    for y in range(0, height - tile_size, tile_size // 2):
        for x in range(0, width - tile_size, tile_size // 2):
            tile = gray_uint8[y:y+tile_size, x:x+tile_size]
            
            # Haralick texture features (13 features × 4 directions)
            haralick = mh.features.haralick(tile)
            
            # Use contrast and energy
            contrast = haralick[:, 1].mean()    # Feature 1: Contrast
            energy = haralick[:, 8].mean()       # Feature 8: Energy/Uniformity
            
            # Texture score: High contrast + low energy = texture
            texture_score = contrast / (energy + 0.01)
```

**Haralick Features Used:**
- **Index 1:** Contrast (texture roughness)
- **Index 8:** Energy/Uniformity (smoothness)

**Texture Score Formula:** contrast / (energy + ε)
- High score = textured regions
- Low score = smooth regions

**Reference:** Haralick et al., Textural Features for Image Classification (IEEE 1973)

---

### 4. Histogram Analysis
**File:** `/home/user/lamcogaussianimage/packages/lgi-rs/tools/preprocess_image_v2.py`  
**Lines:** 166-181  
**Method:** NumPy histogram with 256 bins

**Usage:** Foundation for entropy computation

---

## COLOR SPACE CONVERSIONS

### Complete Color Space Transformation Pipeline
**File:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/flip.py`  
**Lines:** 570-715 (color_space_transform function)  
**Comprehensive Dictionary of Supported Conversions:**

| Conversion | Forward Transform | Reference |
|-----------|------------------|-----------|
| sRGB ↔ Linear RGB | Gamma correction (γ=2.4) | ITU-R BT.709 |
| Linear RGB ↔ CIE XYZ | 3×3 matrix (D65) | ISO 11664-1 |
| CIE XYZ ↔ CIE L*a*b* | Cubic root transform | CIE 1976 |
| CIE XYZ ↔ YCxCz | Channel transform | FLIP paper |
| sRGB → YCxCz | Composite (sRGB→linRGB→XYZ→YCxCz) | Multi-step |
| Linear RGB → L*a*b* | Composite | Multi-step |
| YCxCz ↔ Linear RGB | Inverse transforms | Reversible |

**Key Implementation Details:**

#### sRGB ↔ Linear RGB
```python
if fromSpace2toSpace == "srgb2linrgb":
    limit = 0.04045
    transformed_color = torch.where(
        input_color > limit,
        torch.pow((torch.clamp(input_color, min=limit) + 0.055) / 1.055, 2.4),
        input_color / 12.92
    )
```

#### Linear RGB ↔ CIE XYZ (D65 Illuminant)
```python
# Forward: Linear RGB → XYZ
a11, a12, a13 = 10135552 / 24577794, 8788810 / 24577794, 4435075 / 24577794
# (3 rows × 3 columns matrix multiplication)

# Inverse: XYZ → Linear RGB (precomputed inverse)
a11, a12, a13 = 3.241003275, -1.537398934, -0.498615861
```

#### CIE XYZ ↔ L*a*b* (with Hunt Adjustment)
```python
# Non-linear transform with cubic root and piecewise linear
delta = 6/29, delta_sq = (6/29)², delta_cube = (6/29)³
factor = 1 / (3 * delta_sq)

# Apply Hunt adjustment (multiplicative, line 475-491)
img_h[:, 1:2, :, :] = (0.01 * L) * img[:, 1:2, :, :]  # a*
img_h[:, 2:3, :, :] = (0.01 * L) * img[:, 2:3, :, :]  # b*
```

**D65 Reference Illuminant:**
```python
reference_illuminant = torch.tensor([[[0.950428545]], [[1.000000000]], [[1.088900371]]])
inv_reference_illuminant = torch.tensor([[[1.052156925]], [[1.000000000]], [[0.918357670]]])
```

---

## SPATIAL FREQUENCY & FILTERING

### 1. Contrast Sensitivity Function (CSF) Filters
**File:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/flip.py`  
**Lines:** 391-441 (generate_spatial_filter function)  
**Purpose:** Model human visual perception for image quality assessment

**Three Channels Implemented:**

#### A. Achromatic CSF (Channel 'A')
- **Parameters:** a1=1.0, b1=0.0047, a2=0, b2=1e-5
- **Formula:** g = a1*√(π/b1)*exp(-π²z/b1) + a2*√(π/b2)*exp(-π²z/b2)

#### B. Red-Green CSF (Channel 'RG')
- **Parameters:** a1=1.0, b1=0.0053, a2=0, b2=1e-5
- **Color opponent channel**

#### C. Blue-Yellow CSF (Channel 'BY')
- **Parameters:** a1=34.1, b1=0.04, a2=13.5, b2=0.025
- **Higher sensitivity** (more terms)

**Filter Generation:**
```python
def generate_spatial_filter(pixels_per_degree, channel):
    # Determine radius based on CSF parameters
    r = int(np.ceil(3 * np.sqrt(max_scale_param / (2 * π²)) * pixels_per_degree))
    
    # Create 2D spatial domain
    x, y = np.meshgrid(range(-r, r+1), range(-r, r+1))
    z = (x * deltaX)² + (y * deltaX)²  # squared distance
    
    # Generate Gaussian mixture weights
    g = a1 * np.sqrt(π / b1) * np.exp(-π² * z / b1) + 
        a2 * np.sqrt(π / b2) * np.exp(-π² * z / b2)
    
    g = g / np.sum(g)  # Normalize
```

**Application:**
```python
# Replicate-padded convolution (lines 444-472)
img_pad = torch.zeros((dim[0], dim[1], dim[2] + 2*radius, dim[3] + 2*radius))
img_pad[:, 0:1, :, :] = nn.functional.pad(img[:, 0:1, :, :], 
                                         (radius, radius, radius, radius), 
                                         mode='replicate')
img_tilde[:, 0:1, :, :] = nn.functional.conv2d(img_pad[:, 0:1, :, :], s_a)
```

---

### 2. Spatial Filtering with CSF
**File:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/flip.py`  
**Lines:** 444-472 (spatial_filter function)  

**Process:**
1. Replicate padding to handle boundaries
2. Channel-specific convolution (Achromatic, Red-Green, Blue-Yellow)
3. Transform to linear RGB
4. Clamp to [0, 1] box

---

## STRUCTURE DETECTION ALGORITHMS

### 1. Distance Transform
**File:** `/home/user/lamcogaussianimage/packages/lgi-rs/tools/preprocess_image.py`  
**Lines:** 378-414 (_compute_distance_transform method)  
**Library:** cv2.distanceTransform (CPU), cv2.cuda.createDistanceTransform (GPU)  

**Algorithm:** Euclidean distance transform (DIST_L2)  
**Kernel:** 3×3 connectivity (cv2.DIST_L2, 3)

**Implementation:**
```python
# Per-segment distance transform
for seg_id in range(n_segments):
    mask = (segments == seg_id).astype(np.uint8)
    
    if self.use_gpu and self.has_cuda:
        gpu_dist = cv2.cuda.createDistanceTransform(cv2.DIST_L2, 3)
        dist = gpu_dist.apply(gpu_mask).download()
    else:
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    
    distance_map = np.maximum(distance_map, dist)
```

**Output:** Per-pixel distance to nearest segment boundary  
**Use:** Identify region centers for Gaussian placement

---

### 2. Skeleton Extraction / Medial Axis Transform
**File:** `/home/user/lamcogaussianimage/packages/lgi-rs/tools/preprocess_image.py`  
**Lines:** 416-443 (_compute_skeleton method)  
**Library:** skimage.morphology.skeletonize  

**Algorithm:** Binary morphological skeletonization

**Implementation:**
```python
def _compute_skeleton(self, image: np.ndarray, segments: np.ndarray):
    skeleton_map = np.zeros(image.shape[:2], dtype=np.float32)
    
    for seg_id in range(n_segments):
        mask = (segments == seg_id)
        
        if mask.sum() < 100:  # Skip very small segments
            continue
        
        # Skeletonize (binary morphology)
        skel = morphology.skeletonize(mask)
        skeleton_map[skel] = 1.0
    
    return skeleton_map
```

**Output:** Binary skeleton map (1.0 on medial axis, 0 elsewhere)  
**Use:** Place Gaussians along thin structural elements

---

## SEGMENTATION ALGORITHMS

### 1. SLIC Superpixel Segmentation
**File:** `/home/user/lamcogaussianimage/packages/lgi-rs/tools/preprocess_image.py`  
**Lines:** 336-353 (_compute_slic_segments method)  
**Library:** skimage.segmentation.slic  

**Implementation:**
```python
def _compute_slic_segments(self, image: np.ndarray) -> np.ndarray:
    segments = segmentation.slic(
        image,
        n_segments=self.n_segments,  # Default: 500
        compactness=10,               # Balance color vs spatial
        sigma=1,                       # Gaussian smoothing
        start_label=0,
        channel_axis=-1
    )
    return segments
```

**Parameters:**
- **n_segments:** Requested superpixel count (default 500, configurable)
- **compactness:** 10 (balance between color similarity and spatial compactness)
- **sigma:** 1 (Gaussian smoothing pre-filter)

**Output:** Integer label map where each label is a superpixel  
**Actual segments:** Usually 90-100% of requested count

**Reference:** SLIC paper (Achanta et al., IEEE TPAMI 2012)

---

## PREPROCESSING PIPELINES

### Complete Image Preprocessing Pipeline (v2)
**File:** `/home/user/lamcogaussianimage/packages/lgi-rs/tools/preprocess_image_v2.py`  
**Lines:** 51-164 (preprocess method)  

**8-Stage Pipeline:**

1. **Entropy Map** (Line 79) → Tile-based Shannon entropy
2. **Gradient Map** (Line 83) → Sobel magnitude
3. **Texture Map** (Line 87) → Haralick features
4. **SLIC Segments** (Line 91) → Superpixels
5. **Saliency Map** (Line 98) → Spectral residual
6. **Distance Transform** (Line 102) → Region centers
7. **Skeleton** (Line 106) → Medial axis
8. **Placement Map** (Line 110) → Weighted combination

**Output Files:**
```
kodim02.json                 # Metadata + statistics
kodim02_entropy.npy          # [H, W] float32
kodim02_gradient.npy         # [H, W] float32
kodim02_texture.npy          # [H, W] float32
kodim02_saliency.npy         # [H, W] float32
kodim02_distance.npy         # [H, W] float32
kodim02_skeleton.npy         # [H, W] float32
kodim02_segments.npy         # [H, W] int32
kodim02_placement.npy        # [H, W] float32 (normalized sum=1.0)
```

---

### SLIC-Based Gaussian Initialization
**File:** `/home/user/lamcogaussianimage/packages/lgi-rs/tools/slic_preprocess.py`  
**Lines:** 21-120 (generate_slic_gaussians function)  

**Process:**
1. Run SLIC segmentation
2. For each superpixel:
   - Compute centroid (mean position)
   - Extract mean color
   - Compute covariance matrix
   - Extract eigendecomposition
   - Convert to Gaussian parameters

**Output Format (JSON):**
```json
{
  "gaussians": [
    {
      "position": [0.45, 0.23],           # Normalized [0, 1]
      "scale": [0.05, 0.04],              # Normalized width, height
      "rotation": 0.123,                  # Radians
      "color": [0.8, 0.2, 0.1],          # RGB [0, 1]
      "segment_id": 42,
      "pixel_count": 1234
    },
    ...
  ]
}
```

---

## QUALITY METRICS & LOSS FUNCTIONS

### 1. Fused SSIM (Structural Similarity)
**File:** `/home/user/lamcogaussianimage/packages/lgi-tools/fused-ssim/fused_ssim/__init__.py`  
**Lines:** 8-42  

**Architecture:**
- **CUDA backend:** Custom CUDA kernel (fusedssim_cuda)
- **Forward:** Computes per-pixel SSIM with intermediate derivatives
- **Backward:** Gradient computation for optimization

**Constants:**
- **C1:** (0.01)² = 0.0001
- **C2:** (0.03)² = 0.0009

**Implementation:**
```python
class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, padding="same", train=True):
        # CUDA kernel call
        ssim_map, dm_dmu1, dm_dsigma1_sq, dm_dsigma12 = fusedssim(C1, C2, img1, img2, train)
        
        if padding == "valid":
            ssim_map = ssim_map[:, :, 5:-5, 5:-5]  # Crop 5px border
        
        return ssim_map
    
    @staticmethod
    def backward(ctx, opt_grad):
        # Gradient propagation
        grad = fusedssim_backward(...)
        return None, None, grad, None, None, None

def fused_ssim(img1, img2, padding="same", train=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    map = FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
    return map.mean()  # Return scalar loss
```

**Padding Modes:**
- **"same":** Output same size as input
- **"valid":** Crop 5-pixel border (11×11 window area)

---

### 2. FLIP (LDR/HDR) Loss Functions
**File:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/flip.py`  
**Lines:** 58-227  

#### A. LDR-FLIP (Low Dynamic Range)
**Class:** LDRFLIPLoss (lines 132-167)  
**Input:** sRGB images [0, 1]

**Two-Channel Pipeline:**

**Color Pipeline:**
1. Spatial filtering with CSF (Achromatic, Red-Green, Blue-Yellow)
2. Transform to L*a*b* perceptual space
3. Hunt adjustment (luminance-dependent color scaling)
4. HyAB distance computation
5. Error redistribution to [0, 1] range

**Feature Pipeline:**
1. Extract Y (luminance) channel
2. Edge detection (Gaussian partial derivatives)
3. Point detection (Gaussian Laplacian)
4. Norm computation (feature prominence)

**Final Error:** Combined color and feature using power law (line 227)
```python
return torch.pow(deltaE_c, 1 - deltaE_f)
```

#### B. HDR-FLIP (High Dynamic Range)
**Class:** HDRFLIPLoss (lines 58-129)  
**Input:** Linear RGB, non-negative (HDR)

**Process:**
1. Compute start/stop exposures (lines 92-97)
2. Loop over exposure range
3. Tone map at each exposure (lines 116-117)
4. Compute LDR-FLIP at each exposure
5. Take per-pixel maximum (lines 127-128)

**Tone Mappers Supported:**
- **Reinhard:** Simple/smooth tone mapping
- **Hable:** Filmic tone mapping (more aggressive)
- **ACES:** Academy Color Encoding System (default, cinematic)

---

### 3. PSNR (Peak Signal-to-Noise Ratio)
**File:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/image_utils.py`  
**Lines:** 35-40 (get_psnr function)  

**Formula:**
```python
def get_psnr(image1, image2, max_value=1.0):
    mse = torch.mean((image1-image2)**2)
    if mse.item() <= 1e-7:
        return float('inf')
    psnr = 20*torch.log10(max_value/torch.sqrt(mse))
    return psnr
```

**Formula:** PSNR = 20*log₁₀(MAX / √MSE)  
**Default max_value:** 1.0 (normalized images)  
**Output:** Scalar value in dB

---

## SALIENCY DETECTION

### Architecture: Dual-Stream ResNet50 + Decoder
**Files:**
- Saliency models: `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/saliency/`
- Decoder: `decoder.py` (lines 1-65)
- ResNet backbone: `resnet.py` (lines 1-165+)

### ResNet50 Architecture
**File:** `resnet.py` (lines 72-165+)

**Layers:**
- **conv1:** 3→64 channels, kernel 7×7, stride 2
- **layer1:** BasicBlock (64 channels)
- **layer2:** Bottleneck (128 channels, stride 2)
- **layer3:** Bottleneck (256 channels, stride 2)
- **layer4:** Bottleneck (512 channels, stride 2)
- **outputs:** 5 readout layers (one per layer + initial)

**Output Heads:**
```python
self.output0 = self._make_output(64, readout=1)
self.output1 = self._make_output(256, readout=1)
self.output2 = self._make_output(512, readout=1)
self.output3 = self._make_output(1024, readout=1)
self.output4 = self._make_output(2048, readout=1)
self.combined = self._make_output(5, sigmoid=True)
```

### Decoder Architecture
**File:** `decoder.py` (lines 8-58)

**Components:**
- **ImageNet stream:** Sequential processing of ImageNet features
- **Places stream:** Sequential processing of Places features
- **Fusion:** Concatenate both streams → output conv

**Processing:**
```python
# For each feature in both streams
for a, b in zip(img_feat, self.img_model):
    f = F.interpolate(b(a), self.shape)  # Conv2d + BN + ReLU + resize
    feat.append(f)

# Combine streams
feat = torch.cat(feat, dim=1)
feat = self.combined(feat)  # Final conv → sigmoid
```

---

## LOSS FUNCTIONS IMPLEMENTED

### 1. PSNR
- Per-image metric
- Location: image_utils.py:35-40
- Output: Scalar dB value

### 2. SSIM (via Fused SSIM)
- Per-pixel and aggregated
- Location: fused-ssim package
- Output: Scalar [0, 1]

### 3. FLIP (LDR and HDR)
- Per-pixel error maps + aggregated
- Location: flip.py:58-227
- Designed for perceptual error assessment

### 4. Quantization Loss
**File:** `/home/user/lamcogaussianimage/packages/lgi-legacy/image-gs/utils/quantization_utils.py`  
**Lines:** 4-17 (ste_quantize function)  

**Straight-Through Estimator (STE):**
```python
def ste_quantize(x: torch.Tensor, num_bits: int = 16) -> torch.Tensor:
    qmin, qmax = 0, 2**num_bits - 1
    min_val, max_val = x.min().item(), x.max().item()
    scale = max((max_val - min_val) / (qmax - qmin), 1e-8)
    
    # Forward: quantize
    q_x = torch.round((x - min_val) / scale).clamp(qmin, qmax)
    dq_x = q_x * scale + min_val
    
    # Backward: identity (straight-through)
    dq_x = x + (dq_x - x).detach()
    return dq_x
```

**Purpose:** Quantize Gaussian parameters while maintaining differentiability  
**Bit precision:** Default 16-bit (configurable)

---

## LIBRARY STACK SUMMARY

| Library | Version | Key Functions | Category |
|---------|---------|----------------|----------|
| **OpenCV** | 4.12.0 | Sobel, distance transform, saliency, morphology | Image processing, features |
| **scikit-image** | 0.25.2 | SLIC, skeletonize, filters, morphology | Segmentation, structure |
| **Mahotas** | 1.4.18 | Haralick texture features | Texture analysis |
| **PyTorch** | (latest) | FLIP loss, SSIM, tensor ops | Deep learning, optimization |
| **NumPy** | (latest) | Numerics, histograms, linear algebra | Foundation |
| **PIL/Pillow** | (latest) | Image I/O, resizing, format conversion | File handling |
| **SciPy** | (latest) | Sobel, filters, morphology | Image processing |

### GPU Support
- **OpenCV:** CUDA-accelerated Sobel, distance transform (optional, requires rebuild)
- **PyTorch:** CUDA-native tensor operations
- **Mahotas:** CPU only
- **scikit-image:** CPU only

---

## ALGORITHM COMPLEXITY & PERFORMANCE

### Computational Costs (768×512 image)

| Operation | Time | Library | GPU | Notes |
|-----------|------|---------|-----|-------|
| Entropy (16×16 tiles) | ~100ms | NumPy | CPU | Tile-based |
| Gradient (Sobel) | ~50ms | OpenCV | Fast | Can use CUDA |
| Texture (Haralick) | ~1.8s | Mahotas | CPU | Per-tile computation |
| SLIC (500 segments) | ~300ms | scikit-image | CPU | Compact superpixels |
| Saliency (spectral) | ~100ms | OpenCV | CPU | Fast method |
| Distance transform | ~200ms | OpenCV | Fast | Per-segment |
| Skeleton | ~2.5s | scikit-image | CPU | Morphological |
| **Total** | **~5-6s** | Mixed | ~2s GPU | Production time |

---

## KEY IMPLEMENTATION DECISIONS

1. **Triple Library Stack:** Best-of-breed approach
   - Mahotas (only Haralick)
   - scikit-image (skeleton, SLIC)
   - OpenCV (speed, optional GPU)

2. **Tile-Based Analysis:** Enables local/global metrics
   - Entropy: 16×16 tiles (configurable)
   - Texture: 32×32 tiles with 50% overlap

3. **Perceptual Color Spaces:** Human visual system modeling
   - YCxCz for FLIP
   - L*a*b* with Hunt adjustment

4. **CSF Filters:** Incorporates human visual perception
   - Three separate channels (A, RG, BY)
   - Frequency-dependent sensitivity

5. **Weighted Placement Combination:**
   - Entropy: 0.30 (complexity)
   - Gradient: 0.25 (edges)
   - Texture: 0.20 (textured regions)
   - Saliency: 0.15 (visual importance)
   - Distance: 0.10 (region centers)

---

## REFERENCES & PAPERS

1. **EML-NET Saliency:** https://arxiv.org/abs/1805.01047
2. **FLIP (LDR):** https://research.nvidia.com/publication/2020-07_FLIP
3. **HDR-FLIP:** https://research.nvidia.com/publication/2021-05_HDR-FLIP
4. **SLIC Superpixels:** Achanta et al., IEEE TPAMI 2012
5. **Haralick Texture:** Haralick et al., IEEE 1973
6. **STE Quantization:** https://arxiv.org/abs/1308.3432
7. **CSF (Human Vision):** Perception of contrast and image sharpness

---

**Document Version:** 1.0  
**Generated:** 2025-11-15  
**Status:** Complete Comprehensive Catalog
