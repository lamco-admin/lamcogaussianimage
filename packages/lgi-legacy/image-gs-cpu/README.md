# Image-GS CPU Implementation

A simplified CPU-compatible implementation of **Image-GS: Content-Adaptive Image Representation via 2D Gaussians** (SIGGRAPH 2025).

## Paper Information

**Title:** Image-GS: Content-Adaptive Image Representation via 2D Gaussians  
**Authors:** Yunxiang Zhang, Bingxuan Li, Alexandr Kuznetsov, Akshay Jindal, Stavros Diolatzis, Kenneth Chen, Anton Sochenov, Anton Kaplanyan, Qi Sun  
**Conference:** SIGGRAPH 2025  
**Paper:** [arXiv:2407.01866](https://arxiv.org/abs/2407.01866)  
**Original Implementation:** [NYU-ICL/image-gs](https://github.com/NYU-ICL/image-gs)

## Overview

This implementation provides a simplified, educational version of Image-GS that:
- Runs on CPU (no CUDA required)
- Uses minimal dependencies (only PyTorch, NumPy, PIL, matplotlib)
- Demonstrates the core concepts of 2D Gaussian image representation
- Includes progressive optimization capabilities

## Key Concepts

Image-GS reconstructs images by:
1. **Adaptive Gaussian Allocation**: Using a set of 2D Gaussians with learnable parameters
2. **Progressive Optimization**: Gradually refining Gaussian parameters to match the target image
3. **Content-Adaptive Representation**: Gaussians naturally concentrate in areas with more detail

Each Gaussian has the following parameters:
- **Position** (x, y): Location in the image
- **Scale** (sx, sy): Width and height
- **Rotation**: Orientation angle
- **Color** (r, g, b): RGB color values
- **Opacity**: Transparency/contribution weight

## Installation

```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision numpy pillow matplotlib tqdm
```

## Usage

### Basic Usage

```python
from gaussian_2d_cpu import ImageGS, ImageGSTrainer, load_image, save_image

# Load target image
target = load_image("path/to/image.jpg", size=(256, 256))

# Create model with 1000 Gaussians
model = ImageGS(num_gaussians=1000, image_size=(256, 256))

# Train
trainer = ImageGSTrainer(model, target, learning_rate=0.01)
losses = trainer.train(num_iterations=2000)

# Get result
rendered = model.render()
save_image(rendered, "output.png")
```

### Command Line Interface

```bash
# Reconstruct an image
python gaussian_2d_cpu.py --image path/to/image.jpg --num_gaussians 1000 --iterations 2000

# Run demo mode (creates synthetic target)
python gaussian_2d_cpu.py --num_gaussians 500 --iterations 1000
```

### Run Tests

```bash
python test_gaussian.py
```

## Implementation Details

### Simplified Rendering

Instead of the optimized tile-based rendering in the original paper, this implementation uses a straightforward approach:
1. Create a coordinate grid for the output image
2. For each Gaussian, compute its contribution at every pixel
3. Accumulate weighted color contributions
4. Apply opacity blending

### Loss Function

The training uses a combination of:
- **L2 Loss**: Pixel-wise mean squared error
- **Simplified SSIM**: Structural similarity to preserve perceptual quality

### Optimization

- Uses Adam optimizer with learnable parameters for all Gaussian attributes
- Applies sigmoid activation to colors and opacities to keep them in valid ranges
- Uses absolute value for scales to ensure positive dimensions

## Differences from Original

This simplified implementation differs from the original in several ways:
1. **No CUDA acceleration**: Runs on CPU for accessibility
2. **Simplified rendering**: No tile-based optimization
3. **Basic loss function**: No LPIPS or advanced perceptual losses
4. **No quantization**: Full precision parameters
5. **No saliency-guided initialization**: Random initialization only

## Results

The implementation can achieve reasonable reconstructions with:
- 100-500 Gaussians: Basic structure and colors
- 1000-2000 Gaussians: Good detail preservation
- 2000+ Gaussians: High-quality reconstruction

## Future Improvements

Potential enhancements:
1. GPU acceleration for faster training
2. Implement tile-based rendering for efficiency
3. Add quantization for compression
4. Implement saliency-guided initialization
5. Add support for texture stacks
6. Implement level-of-detail hierarchy

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@inproceedings{zhang2025image,
  title={Image-gs: Content-adaptive image representation via 2d gaussians},
  author={Zhang, Yunxiang and Li, Bingxuan and Kuznetsov, Alexandr and Jindal, Akshay and Diolatzis, Stavros and Chen, Kenneth and Sochenov, Anton and Kaplanyan, Anton and Sun, Qi},
  booktitle={Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
  pages={1--11},
  year={2025}
}
```

## License

This simplified implementation is provided for educational purposes. Please refer to the original repository for licensing information.