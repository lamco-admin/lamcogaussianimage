"""
Build Comprehensive Dataset for Quantum Primitive Discovery
- All 24 Kodak images (natural photos)
- Phase 0/0.5 synthetic edges (controlled properties)
- Real edge cases (extracted edge regions from photos)
- Diverse content (textures, smooth regions, junctions)

Goal: 5,000-10,000 patches covering full spectrum of image features
"""

import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt

KODAK_DIR = Path('/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset')
PHASE0_DIR = Path('/home/greg/gaussian-image-projects/lgi-project/phase_0_results/test_images')


def extract_features(patch):
    """Extract 10 core features from 16x16 patch"""
    gy, gx = np.gradient(patch)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Structure tensor
    Ixx = ndimage.gaussian_filter(gx * gx, sigma=1.0)
    Ixy = ndimage.gaussian_filter(gx * gy, sigma=1.0)
    Iyy = ndimage.gaussian_filter(gy * gy, sigma=1.0)

    center = patch.shape[0] // 2
    tensor = np.array([[Ixx[center, center], Ixy[center, center]],
                       [Ixy[center, center], Iyy[center, center]]])
    eigenvalues = np.linalg.eigvalsh(tensor) + 1e-10

    return np.array([
        np.mean(grad_mag),                      # 0: gradient magnitude
        np.std(grad_mag),                       # 1: gradient variation
        eigenvalues[1],                         # 2: max eigenvalue
        eigenvalues[0],                         # 3: min eigenvalue
        eigenvalues[1]/eigenvalues[0],          # 4: anisotropy
        np.mean(patch),                         # 5: mean intensity
        np.std(patch),                          # 6: std dev
        np.var(patch),                          # 7: variance
        np.max(patch) - np.min(patch),          # 8: intensity range
        np.abs(ndimage.laplace(patch)).mean()   # 9: edge strength
    ])


def extract_from_kodak(n_per_image=200):
    """Extract patches from all Kodak images"""
    print("Extracting from Kodak dataset...")
    patches = []

    for img_num in range(1, 25):
        img_path = KODAK_DIR / f'kodim{img_num:02d}.png'
        if not img_path.exists():
            continue

        img = Image.open(img_path)
        img_gray = np.array(img.convert('L')).astype(float) / 255.0
        h, w = img_gray.shape

        # Random sampling
        for _ in range(n_per_image):
            y = np.random.randint(10, h-26)
            x = np.random.randint(10, w-26)
            patch = img_gray[y:y+16, x:x+16]

            features = extract_features(patch)
            patches.append({
                'features': features,
                'source': 'kodak',
                'image_id': img_num,
                'patch': patch
            })

        if img_num % 5 == 0:
            print(f"  Processed kodim{img_num:02d} ({len(patches)} patches total)")

    return patches


def extract_from_phase0_edges():
    """Extract patches from Phase 0 synthetic edges (known edge properties)"""
    print("\nExtracting from Phase 0 synthetic edges...")
    patches = []

    if not PHASE0_DIR.exists():
        print("  Phase 0 data not found, skipping")
        return patches

    for img_path in PHASE0_DIR.glob('*.png'):
        img = np.array(Image.open(img_path)).astype(float) / 255.0
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)

        h, w = img.shape
        if h < 16 or w < 16:
            continue

        # Extract patches along edge (we know edge is at center)
        for offset in range(-8, 9, 2):
            if w > h:  # Horizontal edge
                y = h//2 + offset
                x = w//2
            else:  # Vertical edge
                y = h//2
                x = w//2 + offset

            if y >= 8 and y < h-8 and x >= 8 and x < w-8:
                patch = img[y-8:y+8, x-8:x+8]
                features = extract_features(patch)
                patches.append({
                    'features': features,
                    'source': 'phase0_edge',
                    'image_id': img_path.stem,
                    'patch': patch
                })

    print(f"  Extracted {len(patches)} edge patches")
    return patches


def extract_specific_features():
    """
    Extract specific feature types for diversity
    Scan Kodak for specific characteristics
    """
    print("\nExtracting specific feature types...")
    patches = []

    for img_num in [1, 2, 5, 8, 15, 19, 23]:  # Diverse Kodak images
        img_path = KODAK_DIR / f'kodim{img_num:02d}.png'
        if not img_path.exists():
            continue

        img = Image.open(img_path)
        img_gray = np.array(img.convert('L')).astype(float) / 255.0
        h, w = img_gray.shape

        # Compute gradient map for targeted sampling
        gy, gx = np.gradient(img_gray)
        grad_mag = np.sqrt(gx**2 + gy**2)

        # Extract HIGH gradient patches (edges)
        high_grad_coords = np.argwhere(grad_mag > 0.2)
        np.random.shuffle(high_grad_coords)
        for y, x in high_grad_coords[:50]:
            if y >= 8 and y < h-8 and x >= 8 and x < w-8:
                patch = img_gray[y-8:y+8, x-8:x+8]
                features = extract_features(patch)
                patches.append({
                    'features': features,
                    'source': 'targeted_edge',
                    'image_id': img_num,
                    'patch': patch
                })

        # Extract LOW gradient patches (smooth regions)
        low_grad_coords = np.argwhere(grad_mag < 0.02)
        np.random.shuffle(low_grad_coords)
        for y, x in low_grad_coords[:50]:
            if y >= 8 and y < h-8 and x >= 8 and x < w-8:
                patch = img_gray[y-8:y+8, x-8:x+8]
                features = extract_features(patch)
                patches.append({
                    'features': features,
                    'source': 'targeted_smooth',
                    'image_id': img_num,
                    'patch': patch
                })

    print(f"  Extracted {len(patches)} targeted patches")
    return patches


def main():
    print("=" * 70)
    print("COMPREHENSIVE DATASET FOR QUANTUM PRIMITIVE DISCOVERY")
    print("=" * 70)

    np.random.seed(42)

    all_patches = []

    # Source 1: Kodak (natural diversity) - 200/image × 24 = 4,800 patches
    kodak_patches = extract_from_kodak(n_per_image=200)
    all_patches.extend(kodak_patches)

    # Source 2: Phase 0 edges (controlled) - ~120 patches
    phase0_patches = extract_from_phase0_edges()
    all_patches.extend(phase0_patches)

    # Source 3: Targeted sampling (specific features) - ~700 patches
    targeted_patches = extract_specific_features()
    all_patches.extend(targeted_patches)

    print("\n" + "=" * 70)
    print(f"TOTAL DATASET: {len(all_patches)} patches")
    print("=" * 70)

    # Convert to arrays
    X = np.array([p['features'] for p in all_patches])

    # Source breakdown
    print("\nDataset composition:")
    for source in ['kodak', 'phase0_edge', 'targeted_edge', 'targeted_smooth']:
        count = sum(1 for p in all_patches if p['source'] == source)
        if count > 0:
            print(f"  {source}: {count} patches ({100*count/len(all_patches):.1f}%)")

    # Feature statistics
    print(f"\nFeature ranges:")
    feature_names = ['grad_mag', 'grad_std', 'eig_max', 'eig_min', 'anisotropy',
                     'mean', 'std', 'var', 'range', 'edge_strength']
    for i, name in enumerate(feature_names):
        print(f"  {name}: [{X[:, i].min():.3f}, {X[:, i].max():.3f}]")

    # Save dataset
    import pickle
    with open('comprehensive_dataset.pkl', 'wb') as f:
        pickle.dump({
            'X': X,
            'patches': all_patches,
            'feature_names': feature_names
        }, f)

    print(f"\n✓ Saved to comprehensive_dataset.pkl")
    print(f"\nThis dataset is ready for quantum clustering")
    print(f"Expected compute time:")
    print(f"  Simulator: ~20-40 minutes (free)")
    print(f"  Real quantum: ~1-2 hours ($6,000-12,000)")
    print(f"\nRecommendation: Use simulator for discovery, it's exact and free!")

    return X, all_patches


if __name__ == "__main__":
    X, patches = main()
