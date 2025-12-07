"""
Core Gaussian Primitives Infrastructure
Provides: Gaussian atom, rendering, synthetic data generation
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class Gaussian2D:
    """2D Gaussian atom - canonical representation"""
    x: float  # center x
    y: float  # center y
    sigma_parallel: float  # std dev along major axis
    sigma_perp: float  # std dev along minor axis
    theta: float  # orientation (radians)
    color: np.ndarray  # RGB [0,1] or grayscale
    alpha: float = 1.0  # opacity [0,1]
    layer_type: str = "unknown"  # M/E/J/R/B/T

    def to_dict(self) -> dict:
        return {
            "x": float(self.x),
            "y": float(self.y),
            "sigma_parallel": float(self.sigma_parallel),
            "sigma_perp": float(self.sigma_perp),
            "theta": float(self.theta),
            "color": self.color.tolist() if isinstance(self.color, np.ndarray) else self.color,
            "alpha": float(self.alpha),
            "layer_type": self.layer_type
        }

    @staticmethod
    def from_dict(d: dict) -> 'Gaussian2D':
        return Gaussian2D(
            x=d["x"], y=d["y"],
            sigma_parallel=d["sigma_parallel"],
            sigma_perp=d["sigma_perp"],
            theta=d["theta"],
            color=np.array(d["color"]),
            alpha=d.get("alpha", 1.0),
            layer_type=d.get("layer_type", "unknown")
        )


class GaussianRenderer:
    """Simple CPU renderer for 2D Gaussians - EWA splatting inspired"""

    @staticmethod
    def render(gaussians: List[Gaussian2D],
               width: int, height: int,
               channels: int = 1) -> np.ndarray:
        """Render Gaussians to image using alpha compositing"""

        # Create meshgrid for all pixels
        y_coords, x_coords = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing='ij'
        )

        # Initialize output image
        if channels == 1:
            image = np.zeros((height, width), dtype=np.float32)
        else:
            image = np.zeros((height, width, channels), dtype=np.float32)

        alpha_accumulated = np.zeros((height, width), dtype=np.float32)

        # Render each Gaussian (back to front)
        for g in gaussians:
            # Compute rotation matrix
            cos_t = np.cos(g.theta)
            sin_t = np.sin(g.theta)

            # Transform coordinates to Gaussian space
            dx = x_coords - g.x
            dy = y_coords - g.y

            # Rotate
            dx_rot = cos_t * dx + sin_t * dy
            dy_rot = -sin_t * dx + cos_t * dy

            # Compute Gaussian weight
            exponent = -0.5 * (
                (dx_rot / g.sigma_parallel) ** 2 +
                (dy_rot / g.sigma_perp) ** 2
            )

            # Clip exponent to avoid numerical issues
            exponent = np.clip(exponent, -50, 0)
            weight = g.alpha * np.exp(exponent)

            # Alpha compositing (over operator)
            alpha_blend = weight * (1.0 - alpha_accumulated)

            if channels == 1:
                # Grayscale
                color_val = g.color[0] if isinstance(g.color, np.ndarray) else g.color
                image += alpha_blend * color_val
            else:
                # RGB
                for c in range(channels):
                    image[:, :, c] += alpha_blend * g.color[c]

            alpha_accumulated += alpha_blend

            # Early termination if fully opaque
            if np.all(alpha_accumulated >= 0.999):
                break

        return np.clip(image, 0, 1)

    @staticmethod
    def render_accumulate(gaussians: List[Gaussian2D],
                         width: int, height: int,
                         channels: int = 1) -> np.ndarray:
        """Render using simple accumulation (GaussianImage ECCV 2024 style)"""

        y_coords, x_coords = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing='ij'
        )

        if channels == 1:
            image = np.zeros((height, width), dtype=np.float32)
        else:
            image = np.zeros((height, width, channels), dtype=np.float32)

        for g in gaussians:
            cos_t = np.cos(g.theta)
            sin_t = np.sin(g.theta)

            dx = x_coords - g.x
            dy = y_coords - g.y

            dx_rot = cos_t * dx + sin_t * dy
            dy_rot = -sin_t * dx + cos_t * dy

            exponent = -0.5 * (
                (dx_rot / g.sigma_parallel) ** 2 +
                (dy_rot / g.sigma_perp) ** 2
            )
            exponent = np.clip(exponent, -50, 0)
            weight = g.alpha * np.exp(exponent)

            if channels == 1:
                color_val = g.color[0] if isinstance(g.color, np.ndarray) else g.color
                image += weight * color_val
            else:
                for c in range(channels):
                    image[:, :, c] += weight * g.color[c]

        return np.clip(image, 0, 1)


class SyntheticDataGenerator:
    """Generate synthetic test images with known features"""

    @staticmethod
    def generate_edge(
        image_size: Tuple[int, int] = (100, 100),
        edge_type: str = 'straight',
        radius: Optional[float] = None,
        blur_sigma: float = 0.0,
        contrast: float = 0.5,
        orientation: float = 0.0
    ) -> Tuple[np.ndarray, Dict]:
        """
        Generate synthetic edge image

        Args:
            image_size: (height, width)
            edge_type: 'straight' or 'curved'
            radius: radius for curved edges (pixels)
            blur_sigma: edge blur (0 = sharp)
            contrast: intensity difference across edge
            orientation: angle for straight edges (radians)

        Returns:
            (image, descriptor) where descriptor contains edge parameters
        """
        h, w = image_size
        image = np.zeros((h, w), dtype=np.float32)

        y_coords, x_coords = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing='ij'
        )

        if edge_type == 'straight':
            # Centered straight edge with given orientation
            cx, cy = w / 2, h / 2

            # Rotate coordinates
            cos_o = np.cos(orientation)
            sin_o = np.sin(orientation)
            x_rot = cos_o * (x_coords - cx) + sin_o * (y_coords - cy)

            # Distance from edge (perpendicular distance)
            if blur_sigma > 0:
                # Smooth edge (error function)
                from scipy.special import erf
                edge_profile = 0.5 + 0.5 * erf(x_rot / (blur_sigma * np.sqrt(2)))
            else:
                # Sharp edge (step function)
                edge_profile = (x_rot >= 0).astype(np.float32)

            # Apply contrast
            image = edge_profile * contrast

            descriptor = {
                'type': 'straight_edge',
                'orientation': float(orientation),
                'blur_sigma': float(blur_sigma),
                'contrast': float(contrast),
                'curvature': 0.0
            }

        elif edge_type == 'curved':
            if radius is None:
                radius = min(h, w) / 3

            # Circular arc edge
            cx, cy = w / 2, h / 2
            distance = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)

            if blur_sigma > 0:
                from scipy.special import erf
                edge_profile = 0.5 + 0.5 * erf((distance - radius) / (blur_sigma * np.sqrt(2)))
            else:
                edge_profile = (distance >= radius).astype(np.float32)

            image = edge_profile * contrast

            descriptor = {
                'type': 'curved_edge',
                'radius': float(radius),
                'blur_sigma': float(blur_sigma),
                'contrast': float(contrast),
                'curvature': 1.0 / radius if radius > 0 else 0.0
            }
        else:
            raise ValueError(f"Unknown edge_type: {edge_type}")

        return image, descriptor

    @staticmethod
    def generate_region(
        image_size: Tuple[int, int] = (100, 100),
        shape: str = 'rectangle',
        fill_type: str = 'constant',
        vertices: Optional[List[Tuple[float, float]]] = None,
        variance: float = 0.0
    ) -> Tuple[np.ndarray, Dict]:
        """Generate region test image"""
        h, w = image_size
        image = np.zeros((h, w), dtype=np.float32)

        # Create mask
        if shape == 'rectangle':
            margin = 0.2
            x1, y1 = int(w * margin), int(h * margin)
            x2, y2 = int(w * (1 - margin)), int(h * (1 - margin))
            mask = np.zeros((h, w), dtype=bool)
            mask[y1:y2, x1:x2] = True

        elif shape == 'ellipse':
            y_coords, x_coords = np.meshgrid(
                np.arange(h, dtype=np.float32),
                np.arange(w, dtype=np.float32),
                indexing='ij'
            )
            cx, cy = w / 2, h / 2
            rx, ry = w / 3, h / 3
            dist = ((x_coords - cx) / rx)**2 + ((y_coords - cy) / ry)**2
            mask = dist <= 1.0

        else:
            # Default to rectangle
            mask = np.ones((h, w), dtype=bool)

        # Fill region
        if fill_type == 'constant':
            image[mask] = 0.5
        elif fill_type == 'linear_gradient':
            y_coords = np.arange(h, dtype=np.float32) / h
            for i in range(h):
                image[i, mask[i, :]] = y_coords[i]
        elif fill_type == 'radial_gradient':
            y_coords, x_coords = np.meshgrid(
                np.arange(h, dtype=np.float32),
                np.arange(w, dtype=np.float32),
                indexing='ij'
            )
            cx, cy = w / 2, h / 2
            dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            dist_norm = dist / dist.max()
            image[mask] = dist_norm[mask]

        # Add noise if requested
        if variance > 0:
            noise = np.random.normal(0, np.sqrt(variance), (h, w))
            image[mask] += noise[mask]
            image = np.clip(image, 0, 1)

        # Compute descriptor
        area = np.sum(mask)
        # Approximate perimeter (boundary pixels)
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask)
        boundary = mask & ~eroded
        perimeter = np.sum(boundary)

        descriptor = {
            'type': f'{shape}_region',
            'shape': shape,
            'fill_type': fill_type,
            'area': int(area),
            'perimeter': int(perimeter),
            'boundary_complexity': float(perimeter**2 / area) if area > 0 else 0,
            'interior_variance': float(variance)
        }

        return image, descriptor

    @staticmethod
    def generate_junction(
        image_size: Tuple[int, int] = (100, 100),
        junction_type: str = 'L',
        angles: Optional[List[float]] = None,
        arm_blur: float = 0.0,
        arm_contrast: float = 0.5
    ) -> Tuple[np.ndarray, Dict]:
        """Generate junction test image"""
        h, w = image_size
        image = np.zeros((h, w), dtype=np.float32)

        # Junction center
        cx, cy = w / 2, h / 2

        # Define arm angles based on junction type
        if junction_type == 'L':
            arm_angles = angles or [0, np.pi/2]  # 90 degrees
        elif junction_type == 'T':
            arm_angles = angles or [0, np.pi/2, np.pi]  # T-shape
        elif junction_type == 'X':
            arm_angles = angles or [0, np.pi/2, np.pi, 3*np.pi/2]  # Cross
        else:
            arm_angles = angles or [0, np.pi/2]

        # Create each arm as a straight edge
        for angle in arm_angles:
            arm_image, _ = SyntheticDataGenerator.generate_edge(
                image_size=image_size,
                edge_type='straight',
                blur_sigma=arm_blur,
                contrast=arm_contrast,
                orientation=angle
            )
            image = np.maximum(image, arm_image)

        descriptor = {
            'type': f'{junction_type}_junction',
            'junction_type': junction_type,
            'num_arms': len(arm_angles),
            'arm_angles': [float(a) for a in arm_angles],
            'arm_blur': float(arm_blur),
            'arm_contrast': float(arm_contrast)
        }

        return image, descriptor

    @staticmethod
    def generate_blob(
        image_size: Tuple[int, int] = (100, 100),
        blob_type: str = 'gaussian',
        size: float = 5.0,
        eccentricity: float = 0.0
    ) -> Tuple[np.ndarray, Dict]:
        """Generate blob test image"""
        h, w = image_size
        image = np.zeros((h, w), dtype=np.float32)

        y_coords, x_coords = np.meshgrid(
            np.arange(h, dtype=np.float32),
            np.arange(w, dtype=np.float32),
            indexing='ij'
        )

        cx, cy = w / 2, h / 2

        if blob_type == 'gaussian':
            # Isotropic or elliptical Gaussian
            if eccentricity == 0:
                # Isotropic
                dist_sq = (x_coords - cx)**2 + (y_coords - cy)**2
                image = np.exp(-dist_sq / (2 * size**2))
            else:
                # Elliptical
                sigma_major = size
                sigma_minor = size * (1 - eccentricity)
                theta = np.pi / 4  # 45 degrees

                cos_t = np.cos(theta)
                sin_t = np.sin(theta)
                dx = x_coords - cx
                dy = y_coords - cy
                dx_rot = cos_t * dx + sin_t * dy
                dy_rot = -sin_t * dx + cos_t * dy

                image = np.exp(-0.5 * ((dx_rot/sigma_major)**2 + (dy_rot/sigma_minor)**2))

        elif blob_type == 'square':
            # Square blob (for testing non-Gaussian)
            half_size = size
            mask = (np.abs(x_coords - cx) <= half_size) & (np.abs(y_coords - cy) <= half_size)
            image[mask] = 1.0

        elif blob_type == 'star':
            # Star shape (5-pointed)
            dx = x_coords - cx
            dy = y_coords - cy
            angle = np.arctan2(dy, dx)
            radius = np.sqrt(dx**2 + dy**2)

            # 5-pointed star profile
            star_radius = size * (1 + 0.5 * np.cos(5 * angle))
            image = np.exp(-(radius - star_radius)**2 / (2 * (size/3)**2))
            image = np.clip(image, 0, 1)

        descriptor = {
            'type': f'{blob_type}_blob',
            'blob_type': blob_type,
            'size': float(size),
            'eccentricity': float(eccentricity)
        }

        return image, descriptor


def compute_metrics(image1: np.ndarray, image2: np.ndarray) -> Dict[str, float]:
    """Compute quality metrics between two images"""

    # Ensure same shape
    assert image1.shape == image2.shape, "Images must have same shape"

    # MSE
    mse = np.mean((image1 - image2) ** 2)

    # PSNR
    if mse < 1e-10:
        psnr = 100.0  # Perfect match
    else:
        psnr = 10 * np.log10(1.0 / mse)

    # MAE
    mae = np.mean(np.abs(image1 - image2))

    # Simple SSIM (approximate, full SSIM would be more complex)
    # For now, use correlation coefficient as proxy
    correlation = np.corrcoef(image1.flatten(), image2.flatten())[0, 1]

    return {
        'mse': float(mse),
        'psnr': float(psnr),
        'mae': float(mae),
        'correlation': float(correlation)
    }


if __name__ == "__main__":
    # Quick test
    print("Testing Gaussian Primitives Infrastructure...")

    # Test 1: Render a simple Gaussian
    g = Gaussian2D(
        x=50, y=50,
        sigma_parallel=10, sigma_perp=5,
        theta=np.pi/4,
        color=np.array([1.0]),
        alpha=1.0
    )

    renderer = GaussianRenderer()
    img = renderer.render([g], 100, 100, channels=1)
    print(f"✓ Rendered Gaussian: shape={img.shape}, range=[{img.min():.3f}, {img.max():.3f}]")

    # Test 2: Generate synthetic edge
    edge_img, edge_desc = SyntheticDataGenerator.generate_edge(
        edge_type='straight',
        blur_sigma=2.0,
        contrast=0.8
    )
    print(f"✓ Generated edge: {edge_desc['type']}, PSNR vs zeros: {compute_metrics(edge_img, np.zeros_like(edge_img))['psnr']:.1f} dB")

    # Test 3: Generate synthetic region
    region_img, region_desc = SyntheticDataGenerator.generate_region(
        shape='ellipse',
        fill_type='radial_gradient'
    )
    print(f"✓ Generated region: {region_desc['type']}, area={region_desc['area']}")

    print("\n Infrastructure tests passed!")
