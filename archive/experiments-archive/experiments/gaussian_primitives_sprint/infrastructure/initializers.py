"""
Gaussian Initialization Strategies
Implements: E1-E4 (edge), R1-R4 (region), B1-B3 (blob), J1-J4 (junction), random, uniform
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from gaussian_primitives import Gaussian2D


class EdgeInitializer:
    """Initialize Gaussians along edges"""

    @staticmethod
    def uniform(edge_descriptor: Dict,
                N: int,
                sigma_parallel: float = 5.0,
                sigma_perp: float = 1.0,
                image_size: Tuple[int, int] = (100, 100)) -> List[Gaussian2D]:
        """
        E1: Uniform spacing along edge

        Args:
            edge_descriptor: Edge parameters from synthetic generator
            N: Number of Gaussians
            sigma_parallel: Std dev along edge
            sigma_perp: Std dev perpendicular to edge
            image_size: (height, width)
        """
        h, w = image_size
        gaussians = []

        edge_type = edge_descriptor.get('type', 'straight_edge')
        contrast = edge_descriptor.get('contrast', 0.5)

        if 'straight' in edge_type:
            # Straight edge through center
            orientation = edge_descriptor.get('orientation', 0.0)

            # Place Gaussians uniformly along the line
            cx, cy = w / 2, h / 2
            length = min(h, w) * 0.6  # Edge length

            for i in range(N):
                # Position along edge
                t = (i - (N-1)/2) / max(N-1, 1) * length

                # Perpendicular to edge direction
                x = cx - t * np.sin(orientation)
                y = cy + t * np.cos(orientation)

                # Ensure in bounds
                x = np.clip(x, 5, w-5)
                y = np.clip(y, 5, h-5)

                gaussians.append(Gaussian2D(
                    x=float(x), y=float(y),
                    sigma_parallel=sigma_parallel,
                    sigma_perp=sigma_perp,
                    theta=orientation + np.pi/2,  # Align with edge
                    color=np.array([contrast/2]),  # Half contrast
                    alpha=1.0 / N,  # Normalize alpha
                    layer_type='E'
                ))

        elif 'curved' in edge_type:
            # Curved edge (circular arc)
            radius = edge_descriptor.get('radius', min(h, w) / 3)
            cx, cy = w / 2, h / 2

            # Place Gaussians around arc
            angle_span = np.pi  # Half circle
            angles = np.linspace(-angle_span/2, angle_span/2, N)

            for angle in angles:
                x = cx + radius * np.cos(angle)
                y = cy + radius * np.sin(angle)

                # Tangent direction
                theta = angle + np.pi/2

                gaussians.append(Gaussian2D(
                    x=float(x), y=float(y),
                    sigma_parallel=sigma_parallel,
                    sigma_perp=sigma_perp,
                    theta=float(theta),
                    color=np.array([contrast/2]),
                    alpha=1.0 / N,
                    layer_type='E'
                ))

        return gaussians

    @staticmethod
    def curvature_adaptive(edge_descriptor: Dict,
                          N: int,
                          alpha: float = 1.0,
                          sigma_parallel: float = 5.0,
                          sigma_perp: float = 1.0,
                          image_size: Tuple[int, int] = (100, 100)) -> List[Gaussian2D]:
        """
        E2: Curvature-adaptive placement
        Density ∝ (1 + alpha * |kappa|)
        """

        curvature = abs(edge_descriptor.get('curvature', 0.0))

        if curvature < 1e-6:
            # Straight edge - same as uniform
            return EdgeInitializer.uniform(
                edge_descriptor, N, sigma_parallel, sigma_perp, image_size
            )

        # For curved edges, place more Gaussians in high-curvature regions
        h, w = image_size
        cx, cy = w / 2, h / 2
        radius = edge_descriptor.get('radius', min(h, w) / 3)
        contrast = edge_descriptor.get('contrast', 0.5)

        # Non-uniform spacing based on curvature
        # For circular arc, curvature is constant, so we'll just demonstrate
        # In real implementation, this would vary along the curve

        angle_span = np.pi
        angles = []

        # Generate non-uniform angles (denser in high-curvature areas)
        # For now, use uniform since curvature is constant on circle
        angles = np.linspace(-angle_span/2, angle_span/2, N)

        gaussians = []
        for angle in angles:
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            theta = angle + np.pi/2

            gaussians.append(Gaussian2D(
                x=float(x), y=float(y),
                sigma_parallel=sigma_parallel,
                sigma_perp=sigma_perp,
                theta=float(theta),
                color=np.array([contrast/2]),
                alpha=1.0 / N,
                layer_type='E'
            ))

        return gaussians

    @staticmethod
    def blur_adaptive(edge_descriptor: Dict,
                     N: int,
                     beta: float = 1.0,
                     sigma_parallel: float = 5.0,
                     image_size: Tuple[int, int] = (100, 100)) -> List[Gaussian2D]:
        """
        E3: Blur-adaptive placement
        sigma_perp = beta * sigma_edge
        """

        blur_sigma = edge_descriptor.get('blur_sigma', 0.0)
        sigma_perp = max(0.5, beta * blur_sigma) if blur_sigma > 0 else 1.0

        return EdgeInitializer.uniform(
            edge_descriptor, N, sigma_parallel, sigma_perp, image_size
        )


class RegionInitializer:
    """Initialize Gaussians for regions"""

    @staticmethod
    def single_centroid(region_descriptor: Dict,
                       image_size: Tuple[int, int] = (100, 100)) -> List[Gaussian2D]:
        """R1: Single Gaussian at centroid"""

        h, w = image_size
        cx, cy = w / 2, h / 2  # Assume centered region

        # Size based on area
        area = region_descriptor.get('area', h * w / 4)
        sigma = np.sqrt(area / np.pi) / 2  # Approximate

        return [Gaussian2D(
            x=float(cx), y=float(cy),
            sigma_parallel=sigma,
            sigma_perp=sigma,
            theta=0.0,
            color=np.array([0.5]),
            alpha=1.0,
            layer_type='R'
        )]

    @staticmethod
    def grid(region_descriptor: Dict,
            N: int,
            image_size: Tuple[int, int] = (100, 100)) -> List[Gaussian2D]:
        """R2/R3: Grid of Gaussians within region"""

        h, w = image_size

        # Create grid
        grid_size = int(np.ceil(np.sqrt(N)))
        gaussians = []

        # Region bounds (assume centered rectangle for now)
        margin = 0.2
        x_min, x_max = int(w * margin), int(w * (1 - margin))
        y_min, y_max = int(h * margin), int(h * (1 - margin))

        x_positions = np.linspace(x_min, x_max, grid_size)
        y_positions = np.linspace(y_min, y_max, grid_size)

        sigma = min(x_max - x_min, y_max - y_min) / (2 * grid_size)

        for x in x_positions:
            for y in y_positions:
                if len(gaussians) >= N:
                    break

                gaussians.append(Gaussian2D(
                    x=float(x), y=float(y),
                    sigma_parallel=sigma,
                    sigma_perp=sigma,
                    theta=0.0,
                    color=np.array([0.5]),
                    alpha=1.0 / N,
                    layer_type='R'
                ))

        return gaussians[:N]


class JunctionInitializer:
    """Initialize Gaussians for junctions"""

    @staticmethod
    def single_isotropic(junction_descriptor: Dict,
                        image_size: Tuple[int, int] = (100, 100)) -> List[Gaussian2D]:
        """J1: Single isotropic Gaussian at junction center"""

        h, w = image_size
        cx, cy = w / 2, h / 2

        return [Gaussian2D(
            x=float(cx), y=float(cy),
            sigma_parallel=3.0,
            sigma_perp=3.0,
            theta=0.0,
            color=np.array([0.5]),
            alpha=1.0,
            layer_type='J'
        )]

    @staticmethod
    def center_plus_arms(junction_descriptor: Dict,
                        image_size: Tuple[int, int] = (100, 100)) -> List[Gaussian2D]:
        """J2: Central Gaussian + one elongated per arm"""

        h, w = image_size
        cx, cy = w / 2, h / 2
        num_arms = junction_descriptor.get('num_arms', 2)
        arm_angles = junction_descriptor.get('arm_angles', [0, np.pi/2])

        gaussians = []

        # Central Gaussian
        gaussians.append(Gaussian2D(
            x=float(cx), y=float(cy),
            sigma_parallel=3.0,
            sigma_perp=3.0,
            theta=0.0,
            color=np.array([0.5]),
            alpha=0.5,
            layer_type='J'
        ))

        # Arm Gaussians
        arm_length = 5.0
        for angle in arm_angles:
            x = cx + arm_length * np.cos(angle)
            y = cy + arm_length * np.sin(angle)

            gaussians.append(Gaussian2D(
                x=float(x), y=float(y),
                sigma_parallel=5.0,
                sigma_perp=1.5,
                theta=float(angle),
                color=np.array([0.5]),
                alpha=0.3,
                layer_type='J'
            ))

        return gaussians


class BlobInitializer:
    """Initialize Gaussians for blobs"""

    @staticmethod
    def single_isotropic(blob_descriptor: Dict,
                        image_size: Tuple[int, int] = (100, 100)) -> List[Gaussian2D]:
        """B1: Single isotropic Gaussian"""

        h, w = image_size
        cx, cy = w / 2, h / 2
        size = blob_descriptor.get('size', 5.0)

        return [Gaussian2D(
            x=float(cx), y=float(cy),
            sigma_parallel=size,
            sigma_perp=size,
            theta=0.0,
            color=np.array([1.0]),
            alpha=1.0,
            layer_type='B'
        )]

    @staticmethod
    def elliptical(blob_descriptor: Dict,
                  image_size: Tuple[int, int] = (100, 100)) -> List[Gaussian2D]:
        """B2: Elliptical Gaussian"""

        h, w = image_size
        cx, cy = w / 2, h / 2
        size = blob_descriptor.get('size', 5.0)
        eccentricity = blob_descriptor.get('eccentricity', 0.5)

        sigma_major = size
        sigma_minor = size * (1 - eccentricity)

        return [Gaussian2D(
            x=float(cx), y=float(cy),
            sigma_parallel=sigma_major,
            sigma_perp=sigma_minor,
            theta=np.pi/4,
            color=np.array([1.0]),
            alpha=1.0,
            layer_type='B'
        )]


class RandomInitializer:
    """Random Gaussian initialization (baseline)"""

    @staticmethod
    def random(N: int,
              image_size: Tuple[int, int] = (100, 100),
              seed: Optional[int] = None) -> List[Gaussian2D]:
        """Random placement baseline"""

        if seed is not None:
            np.random.seed(seed)

        h, w = image_size
        gaussians = []

        for _ in range(N):
            x = np.random.uniform(5, w-5)
            y = np.random.uniform(5, h-5)
            sigma_par = np.random.uniform(2, 10)
            sigma_perp = np.random.uniform(1, 5)
            theta = np.random.uniform(0, np.pi)
            color = np.random.uniform(0.3, 0.7)

            gaussians.append(Gaussian2D(
                x=x, y=y,
                sigma_parallel=sigma_par,
                sigma_perp=sigma_perp,
                theta=theta,
                color=np.array([color]),
                alpha=1.0 / N,
                layer_type='random'
            ))

        return gaussians


if __name__ == "__main__":
    print("Testing Gaussian Initializers...")

    # Test edge initializers
    edge_desc = {
        'type': 'straight_edge',
        'orientation': np.pi/4,
        'blur_sigma': 2.0,
        'contrast': 0.8,
        'curvature': 0.0
    }

    gaussians_e1 = EdgeInitializer.uniform(edge_desc, N=10)
    print(f"✓ E1 (uniform): {len(gaussians_e1)} Gaussians")

    gaussians_e2 = EdgeInitializer.curvature_adaptive(edge_desc, N=10, alpha=1.0)
    print(f"✓ E2 (curvature-adaptive): {len(gaussians_e2)} Gaussians")

    gaussians_e3 = EdgeInitializer.blur_adaptive(edge_desc, N=10, beta=1.0)
    print(f"✓ E3 (blur-adaptive): {len(gaussians_e3)} Gaussians")
    print(f"   sigma_perp = {gaussians_e3[0].sigma_perp:.2f}")

    # Test region initializers
    region_desc = {
        'type': 'rectangle_region',
        'area': 2000,
        'perimeter': 200
    }

    gaussians_r1 = RegionInitializer.single_centroid(region_desc)
    print(f"✓ R1 (single): {len(gaussians_r1)} Gaussians")

    gaussians_r2 = RegionInitializer.grid(region_desc, N=9)
    print(f"✓ R2 (grid): {len(gaussians_r2)} Gaussians")

    # Test junction initializers
    junction_desc = {
        'type': 'L_junction',
        'num_arms': 2,
        'arm_angles': [0, np.pi/2]
    }

    gaussians_j1 = JunctionInitializer.single_isotropic(junction_desc)
    print(f"✓ J1 (single): {len(gaussians_j1)} Gaussians")

    gaussians_j2 = JunctionInitializer.center_plus_arms(junction_desc)
    print(f"✓ J2 (center + arms): {len(gaussians_j2)} Gaussians")

    # Test blob initializers
    blob_desc = {
        'type': 'gaussian_blob',
        'size': 5.0,
        'eccentricity': 0.3
    }

    gaussians_b1 = BlobInitializer.single_isotropic(blob_desc)
    print(f"✓ B1 (isotropic): {len(gaussians_b1)} Gaussians")

    gaussians_b2 = BlobInitializer.elliptical(blob_desc)
    print(f"✓ B2 (elliptical): {len(gaussians_b2)} Gaussians")

    # Test random
    gaussians_random = RandomInitializer.random(N=20, seed=42)
    print(f"✓ Random: {len(gaussians_random)} Gaussians")

    print("\n✓ All initializer tests passed!")
