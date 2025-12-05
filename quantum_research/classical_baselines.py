#!/usr/bin/env python3
"""
Classical Clustering Baselines for Gaussian Channel Discovery

Tests multiple classical clustering methods to establish performance ceiling
before running quantum experiments.

Compositional Framework: All methods cluster in parameter space (NO spatial info)

Usage:
  python3 classical_baselines.py                    # Use default dataset
  python3 classical_baselines.py --enhanced         # Use enhanced dataset with opt features
"""

import numpy as np
import pickle
import json
import sys
from sklearn.cluster import (KMeans, DBSCAN, AgglomerativeClustering,
                               SpectralClustering)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import time

def load_dataset(enhanced=False):
    """Load quantum-ready dataset"""
    if enhanced:
        filename = 'kodak_gaussians_quantum_ready_enhanced.pkl'
    else:
        filename = 'kodak_gaussians_quantum_ready.pkl'

    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        print(f"ERROR: Dataset not found: {filename}")
        print()
        if enhanced:
            print("Run extract_optimization_features.py first to create enhanced dataset")
        else:
            print("Run prepare_quantum_dataset.py first to create dataset")
        sys.exit(1)

def run_clustering_method(method_name, X, k, **kwargs):
    """
    Run a single clustering method.

    Returns: (labels, runtime, additional_metrics)
    """
    start = time.time()

    if method_name == 'kmeans':
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        extra = {'inertia': float(model.inertia_)}

    elif method_name == 'gmm':
        model = GaussianMixture(n_components=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        extra = {
            'bic': float(model.bic(X)),
            'aic': float(model.aic(X))
        }

    elif method_name == 'spectral_rbf':
        model = SpectralClustering(
            n_clusters=k,
            affinity='rbf',
            random_state=42,
            n_init=10
        )
        labels = model.fit_predict(X)
        extra = {}

    elif method_name == 'hierarchical_ward':
        model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = model.fit_predict(X)
        extra = {}

    elif method_name == 'hierarchical_average':
        model = AgglomerativeClustering(n_clusters=k, linkage='average')
        labels = model.fit_predict(X)
        extra = {}

    elif method_name == 'dbscan':
        # DBSCAN doesn't take k as parameter
        eps = kwargs.get('eps', 0.5)
        model = DBSCAN(eps=eps, min_samples=5)
        labels = model.fit_predict(X)
        extra = {
            'eps': eps,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'n_noise': np.sum(labels == -1)
        }

    else:
        raise ValueError(f"Unknown method: {method_name}")

    runtime = time.time() - start

    return labels, runtime, extra

def main():
    print()
    print("="*80)
    print("CLASSICAL CLUSTERING BASELINES")
    print("Compositional Framework: Clustering in Parameter + Behavior Space")
    print("="*80)
    print()

    # Check for --enhanced flag
    enhanced = '--enhanced' in sys.argv

    # Load dataset
    print("Loading dataset...")
    data = load_dataset(enhanced=enhanced)

    X = data['X_scaled']
    n_samples = data['n_samples']
    n_features = data['n_features']
    features = data['features']
    framework = data.get('framework', 'unknown')

    print(f"✓ Loaded: {data['source']}")
    print(f"  Framework: {framework}")
    print(f"  Samples: {n_samples:,}")
    print(f"  Features: {n_features}")
    print()

    print("Features:")
    for i, feat in enumerate(features, 1):
        print(f"  {i}. {feat}")
    print()

    if enhanced:
        print("✓ Using ENHANCED dataset with optimization behavior features")
        print("  → Should discover OPTIMIZATION CLASSES")
    else:
        print("Using original dataset with geometric + context features")
        print("  → Will discover geometric classes")

    print()

    # Test clustering methods
    methods = [
        'kmeans',
        'gmm',
        'spectral_rbf',
        'hierarchical_ward',
        'hierarchical_average',
    ]

    print("="*80)
    print("RUNNING CLUSTERING METHODS")
    print("="*80)
    print()

    all_results = {}
    best_overall = {'method': None, 'k': None, 'silhouette': -1}

    # Test k from 3 to 8
    for k in range(3, 9):
        print(f"Testing {k} clusters...")
        print("-" * 40)

        all_results[k] = {}

        for method in methods:
            labels, runtime, extra = run_clustering_method(method, X, k)

            # Compute metrics
            silhouette = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)

            result = {
                'silhouette': float(silhouette),
                'calinski_harabasz': float(calinski),
                'runtime_seconds': float(runtime),
                'labels': labels.tolist(),
                **extra
            }

            all_results[k][method] = result

            print(f"  {method:20s}: silhouette={silhouette:.3f}, "
                  f"CH={calinski:.1f}, time={runtime:.3f}s")

            # Track best overall
            if silhouette > best_overall['silhouette']:
                best_overall = {
                    'method': method,
                    'k': k,
                    'silhouette': silhouette
                }

        print()

    # Test DBSCAN (doesn't require k)
    print("Testing DBSCAN (density-based, automatic k)...")
    print("-" * 40)

    eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
    dbscan_results = []

    for eps in eps_values:
        labels, runtime, extra = run_clustering_method('dbscan', X, None, eps=eps)

        n_clusters = extra['n_clusters']
        n_noise = extra['n_noise']

        if n_clusters > 1 and n_noise < len(X) * 0.5:  # Valid clustering
            silhouette = silhouette_score(X[labels != -1], labels[labels != -1])

            result = {
                'eps': eps,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': float(silhouette),
                'runtime_seconds': float(runtime),
                'labels': labels.tolist()
            }

            dbscan_results.append(result)

            print(f"  eps={eps:.1f}: k={n_clusters}, silhouette={silhouette:.3f}, "
                  f"noise={n_noise} ({100*n_noise/len(X):.1f}%)")

    all_results['dbscan'] = dbscan_results

    print()

    # Summary
    print("="*80)
    print("BEST CLASSICAL RESULT")
    print("="*80)
    print()

    print(f"Method: {best_overall['method'].upper()}")
    print(f"Clusters: {best_overall['k']}")
    print(f"Silhouette: {best_overall['silhouette']:.3f}")
    print()

    # Analyze best clustering
    best_labels = all_results[best_overall['k']][best_overall['method']]['labels']
    best_labels = np.array(best_labels)

    print(f"Cluster sizes:")
    for cluster_id in range(best_overall['k']):
        count = np.sum(best_labels == cluster_id)
        print(f"  Cluster {cluster_id}: {count:,} samples ({100*count/n_samples:.1f}%)")

    print()

    # Save results
    output_filename = 'classical_clustering_results_enhanced.json' if enhanced else 'classical_clustering_results.json'

    results_output = {
        'dataset': {
            'source': data['source'],
            'framework': framework,
            'n_samples': n_samples,
            'n_features': n_features,
            'features': features,
            'enhanced': enhanced
        },
        'results': all_results,
        'best': best_overall,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(output_filename, 'w') as f:
        json.dump(results_output, f, indent=2)

    print("="*80)
    print("RESULTS SAVED")
    print("="*80)
    print()
    print(f"✓ Output: {output_filename}")
    print()

    print("="*80)
    print("INTERPRETATION")
    print("="*80)
    print()

    if enhanced:
        print("With optimization behavior features, clusters represent:")
        print("  • Optimization classes (fast/slow convergers, stable/unstable)")
        print("  • Defined by HOW Gaussians optimize, not just WHAT they look like")
        print("  • Compositional layers that need different iteration strategies")
    else:
        print("With geometric features, clusters represent:")
        print("  • Geometric types (large/small, isotropic/anisotropic)")
        print("  • Defined by parameter magnitudes")
        print("  • May not reflect optimization behavior")

    print()

    print("="*80)
    print("NEXT STEP")
    print("="*80)
    print()

    if not enhanced:
        print("Run with optimization features:")
        print("  python3 extract_optimization_features.py")
        print("  python3 classical_baselines.py --enhanced")
        print()
        print("Compare: Does adding optimization behavior improve clustering?")
    else:
        print("Run quantum clustering on enhanced features:")
        print("  (Update Q1_production_real_data.py to load enhanced dataset)")
        print("  python3 Q1_production_real_data.py")
        print()
        print("Compare: Does quantum find different optimization classes than classical?")

    print()
    print("="*80)

if __name__ == "__main__":
    main()
