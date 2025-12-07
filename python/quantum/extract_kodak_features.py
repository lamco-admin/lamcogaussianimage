"""
Extract features from Kodak images for quantum experiments
Creates labeled dataset: patches with features → primitive type
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from skimage import feature, filters
import json

KODAK_DIR = Path('../test-data/kodak-dataset')


def load_kodak_images(max_images=24):
    """Load Kodak dataset images"""
    images = []
    for i in range(1, max_images + 1):
        img_path = KODAK_DIR / f'kodim{i:02d}.png'
        if img_path.exists():
            img = np.array(Image.open(img_path))
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
            # Normalize to [0, 1]
            img = img.astype(float) / 255.0
            images.append({'id': i, 'image': img, 'path': str(img_path)})
    return images


def extract_patch(image, center, size=16):
    """Extract square patch centered at position"""
    h, w = image.shape
    half = size // 2
    y, x = center

    y1, y2 = max(0, y - half), min(h, y + half)
    x1, x2 = max(0, x - half), min(w, x + half)

    patch = image[y1:y2, x1:x2]

    # Pad if at edge
    if patch.shape != (size, size):
        padded = np.zeros((size, size))
        padded[:patch.shape[0], :patch.shape[1]] = patch
        return padded

    return patch


def compute_patch_features(patch):
    """
    Compute comprehensive feature vector for patch

    Returns ~20-30 dimensional feature vector
    """
    features = {}

    # 1. Gradient features
    gy, gx = np.gradient(patch)
    grad_mag = np.sqrt(gx**2 + gy**2)

    features['gradient_magnitude_mean'] = np.mean(grad_mag)
    features['gradient_magnitude_std'] = np.std(grad_mag)
    features['gradient_magnitude_max'] = np.max(grad_mag)

    # 2. Structure tensor (edge orientation and coherence)
    Ixx = ndimage.gaussian_filter(gx * gx, sigma=1.0)
    Ixy = ndimage.gaussian_filter(gx * gy, sigma=1.0)
    Iyy = ndimage.gaussian_filter(gy * gy, sigma=1.0)

    # Eigenvalues at center
    center = patch.shape[0] // 2
    tensor = np.array([[Ixx[center, center], Ixy[center, center]],
                       [Ixy[center, center], Iyy[center, center]]])
    eigenvalues = np.linalg.eigvalsh(tensor)

    features['eigenvalue_1'] = eigenvalues[1]  # Larger
    features['eigenvalue_2'] = eigenvalues[0]  # Smaller
    features['eigenvalue_ratio'] = eigenvalues[1] / (eigenvalues[0] + 1e-6)
    features['coherence'] = (eigenvalues[1] - eigenvalues[0]) / (eigenvalues[1] + eigenvalues[0] + 1e-6)

    # 3. Statistics
    features['mean_intensity'] = np.mean(patch)
    features['std_intensity'] = np.std(patch)
    features['variance'] = np.var(patch)
    features['min_intensity'] = np.min(patch)
    features['max_intensity'] = np.max(patch)
    features['intensity_range'] = features['max_intensity'] - features['min_intensity']

    # 4. Entropy
    hist, _ = np.histogram(patch.flatten(), bins=16, range=(0, 1))
    hist = hist + 1e-10  # Avoid log(0)
    hist_normalized = hist / hist.sum()
    features['entropy'] = -np.sum(hist_normalized * np.log2(hist_normalized))

    # 5. Frequency content (simple)
    fft = np.fft.fft2(patch)
    fft_mag = np.abs(fft)

    # High frequency energy (outer regions of FFT)
    h, w = fft_mag.shape
    mask_hf = np.ones_like(fft_mag)
    mask_hf[h//4:3*h//4, w//4:3*w//4] = 0  # Exclude center (low freq)

    features['high_freq_energy'] = np.sum(fft_mag * mask_hf) / np.sum(fft_mag)

    # 6. Laplacian (edge strength)
    laplacian = ndimage.laplace(patch)
    features['laplacian_mean'] = np.abs(laplacian).mean()
    features['laplacian_std'] = laplacian.std()

    # Convert to array (consistent ordering)
    feature_names = sorted(features.keys())
    feature_vector = np.array([features[k] for k in feature_names])

    return feature_vector, feature_names


def label_patch(patch, features_dict):
    """
    Automatically label patch based on features

    Returns: 'edge', 'region', 'texture', 'smooth'
    """
    grad_mag = features_dict['gradient_magnitude_mean']
    coherence = features_dict['coherence']
    variance = features_dict['variance']
    eigenvalue_ratio = features_dict['eigenvalue_ratio']
    high_freq = features_dict['high_freq_energy']

    # Edge: High gradient, high coherence (directional)
    if grad_mag > 0.15 and coherence > 0.5 and eigenvalue_ratio > 3:
        return 'edge'

    # Texture: High frequency, high variance
    elif high_freq > 0.3 and variance > 0.02:
        return 'texture'

    # Smooth/flat: Low variance, low gradient
    elif variance < 0.005 and grad_mag < 0.05:
        return 'smooth'

    # Region: Everything else (medium complexity)
    else:
        return 'region'


def extract_dataset(n_patches_per_image=50, n_images=5, patch_size=16):
    """
    Extract dataset from Kodak images

    Returns: X (features), y (labels), metadata
    """
    print(f"Extracting {n_patches_per_image} patches from {n_images} Kodak images...")

    images = load_kodak_images(max_images=n_images)

    if not images:
        print(f"✗ No Kodak images found in {KODAK_DIR}")
        return None, None, None

    print(f"✓ Loaded {len(images)} images")

    all_patches = []
    feature_names = None

    for img_data in images:
        img = img_data['image']
        h, w = img.shape

        # Sample random patches
        for _ in range(n_patches_per_image):
            # Random center (avoid borders)
            y = np.random.randint(patch_size, h - patch_size)
            x = np.random.randint(patch_size, w - patch_size)

            # Extract patch
            patch = extract_patch(img, (y, x), size=patch_size)

            # Compute features
            feature_vector, feature_names = compute_patch_features(patch)

            # Create feature dict for labeling
            features_dict = {name: val for name, val in zip(feature_names, feature_vector)}

            # Auto-label
            label = label_patch(patch, features_dict)

            all_patches.append({
                'features': feature_vector,
                'label': label,
                'image_id': img_data['id'],
                'position': (y, x),
                'patch': patch
            })

    # Convert to arrays
    X = np.array([p['features'] for p in all_patches])
    y_labels = np.array([p['label'] for p in all_patches])

    # Convert labels to integers
    label_map = {'edge': 0, 'region': 1, 'texture': 2, 'smooth': 3}
    y = np.array([label_map[label] for label in y_labels])

    # Statistics
    print(f"\n✓ Extracted {len(all_patches)} patches")
    print(f"Feature dimension: {X.shape[1]}")
    print(f"\nLabel distribution:")
    for label, idx in label_map.items():
        count = np.sum(y == idx)
        print(f"  {label}: {count} ({100*count/len(y):.1f}%)")

    metadata = {
        'feature_names': feature_names,
        'label_map': label_map,
        'n_patches': len(all_patches),
        'patch_size': patch_size
    }

    return X, y, metadata, all_patches


def save_dataset(X, y, metadata, filename='kodak_features.npz'):
    """Save extracted features for reuse"""
    np.savez(filename,
             X=X,
             y=y,
             feature_names=metadata['feature_names'],
             label_map=json.dumps(metadata['label_map']))
    print(f"\n✓ Dataset saved to {filename}")


if __name__ == "__main__":
    print("=" * 70)
    print("KODAK FEATURE EXTRACTION")
    print("=" * 70)

    # Extract features
    X, y, metadata, patches = extract_dataset(
        n_patches_per_image=50,
        n_images=5,
        patch_size=16
    )

    if X is not None:
        # Save for quantum experiments
        save_dataset(X, y, metadata, 'kodak_features.npz')

        # Show sample patches per class
        fig, axes = plt.subplots(4, 5, figsize=(12, 10))

        label_names = ['edge', 'region', 'texture', 'smooth']
        for i, label_name in enumerate(label_names):
            label_idx = metadata['label_map'][label_name]
            samples = [p for p in patches if p['label'] == label_name][:5]

            for j, sample in enumerate(samples):
                axes[i, j].imshow(sample['patch'], cmap='gray', vmin=0, vmax=1)
                axes[i, j].axis('off')
                if j == 0:
                    axes[i, j].set_title(f'{label_name}', fontsize=10)

        plt.tight_layout()
        plt.savefig('sample_patches_by_label.png', dpi=150)
        print(f"✓ Sample patches saved to sample_patches_by_label.png")
