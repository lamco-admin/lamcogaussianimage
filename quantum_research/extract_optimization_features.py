#!/usr/bin/env python3
"""
Extract Optimization Behavior Features from Gaussian Trajectories

Processes 682K Gaussian optimization trajectories to extract features about
HOW each Gaussian optimizes, not just its final parameters.

This transforms geometric clustering into OPTIMIZATION CLASS clustering,
enabling discovery of compositional layers defined by optimization behavior.

Input: kodak_gaussian_data/*.csv (iteration-by-iteration trajectories)
Output: kodak_gaussians_quantum_ready_enhanced.pkl (with optimization features)

Framework: COMPOSITIONAL - no spatial information, pure optimization classes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import pickle
import time
from scipy.stats import linregress

def extract_trajectory_features(trajectory_df):
    """
    Extract optimization behavior features from a Gaussian's trajectory.

    Parameters:
    - trajectory_df: DataFrame with iteration-by-iteration data for one Gaussian

    Returns: dict of optimization features
    """
    iterations = trajectory_df['iteration'].values
    losses = trajectory_df['loss'].values

    if len(losses) < 2:
        # Not enough data for trajectory analysis
        return {
            'convergence_speed': 1.0,
            'loss_slope': 0.0,
            'loss_linearity': 0.0,
            'loss_curvature': 0.0,
            'sigma_x_stability': 0.0,
            'sigma_y_stability': 0.0,
            'parameter_coupling': 0.0,
        }

    # === Convergence Characteristics ===

    final_loss = losses[-1]
    initial_loss = losses[0]
    total_improvement = initial_loss - final_loss

    # Convergence speed: iteration where we reach 90% of final improvement
    if total_improvement > 1e-6:
        target_loss = initial_loss - 0.9 * total_improvement
        convergence_iter = None

        for i, loss in enumerate(losses):
            if loss <= target_loss:
                convergence_iter = iterations[i]
                break

        if convergence_iter is not None:
            # Normalize to [0,1]: 0=fast (early), 1=slow (late)
            max_iter = iterations[-1]
            convergence_speed = convergence_iter / max_iter
        else:
            convergence_speed = 1.0  # Never reached 90%
    else:
        convergence_speed = 1.0  # No improvement

    # === Loss Curve Shape ===

    # Linear fit to loss curve
    slope, intercept, r_value, _, _ = linregress(iterations, losses)
    loss_slope = slope  # Negative for decreasing
    loss_linearity = r_value ** 2  # R² ∈ [0,1]

    # Loss curvature (second derivative - measures non-linearity)
    if len(losses) > 2:
        first_deriv = np.diff(losses)
        second_deriv = np.diff(first_deriv)
        loss_curvature = np.std(second_deriv)  # Higher = more curved/noisy
    else:
        loss_curvature = 0.0

    # === Parameter Stability ===

    if 'sigma_x' in trajectory_df.columns and 'sigma_y' in trajectory_df.columns:
        sigma_x_values = trajectory_df['sigma_x'].values
        sigma_y_values = trajectory_df['sigma_y'].values

        # Coefficient of variation (CV = std/mean)
        # High CV = parameter changes a lot during optimization
        sigma_x_mean = np.mean(sigma_x_values)
        sigma_y_mean = np.mean(sigma_y_values)

        if sigma_x_mean > 1e-6:
            sigma_x_stability = np.std(sigma_x_values) / sigma_x_mean
        else:
            sigma_x_stability = 0.0

        if sigma_y_mean > 1e-6:
            sigma_y_stability = np.std(sigma_y_values) / sigma_y_mean
        else:
            sigma_y_stability = 0.0

        # Parameter coupling: do sigma_x and sigma_y change together?
        if len(sigma_x_values) > 2:
            sigma_x_changes = np.diff(sigma_x_values)
            sigma_y_changes = np.diff(sigma_y_values)

            # Correlation between changes
            if np.std(sigma_x_changes) > 1e-6 and np.std(sigma_y_changes) > 1e-6:
                parameter_coupling = np.corrcoef(sigma_x_changes, sigma_y_changes)[0, 1]
            else:
                parameter_coupling = 0.0
        else:
            parameter_coupling = 0.0

        # Rotation stability (if available)
        if 'rotation' in trajectory_df.columns:
            rotation_values = trajectory_df['rotation'].values
            rotation_drift = np.sum(np.abs(np.diff(rotation_values)))
        else:
            rotation_drift = 0.0
    else:
        sigma_x_stability = 0.0
        sigma_y_stability = 0.0
        parameter_coupling = 0.0
        rotation_drift = 0.0

    return {
        'convergence_speed': float(convergence_speed),
        'loss_slope': float(loss_slope),
        'loss_linearity': float(loss_linearity),
        'loss_curvature': float(loss_curvature),
        'sigma_x_stability': float(sigma_x_stability),
        'sigma_y_stability': float(sigma_y_stability),
        'parameter_coupling': float(parameter_coupling),
    }

def process_all_trajectories():
    """
    Process all 24 Kodak CSV files to extract optimization features.
    """
    print("="*80)
    print("EXTRACTING OPTIMIZATION BEHAVIOR FEATURES")
    print("Compositional Framework: Parameters + Optimization Dynamics")
    print("="*80)
    print()

    data_dir = Path("./kodak_gaussian_data")
    csv_files = sorted(data_dir.glob("kodim*.csv"))

    if len(csv_files) == 0:
        print("ERROR: No CSV files found in kodak_gaussian_data/")
        print()
        print("Run collect_all_kodak_data.py first to generate trajectory data")
        return

    print(f"Found {len(csv_files)} CSV files")
    print()

    all_features = []
    total_trajectories = 0

    start_time = time.time()

    for file_idx, csv_file in enumerate(csv_files, 1):
        print(f"[{file_idx:2d}/24] Processing {csv_file.name}...")

        df = pd.read_csv(csv_file)

        # Group by refinement pass and Gaussian ID (each group is one trajectory)
        grouped = df.groupby(['refinement_pass', 'gaussian_id'])

        trajectories_in_file = 0

        for (pass_id, gauss_id), trajectory in grouped:
            # Extract static features (from final iteration)
            final_row = trajectory.iloc[-1]

            static_features = {
                'image_id': final_row['image_id'],
                'refinement_pass': int(pass_id),
                'gaussian_id': int(gauss_id),
                'sigma_x': float(final_row['sigma_x']),
                'sigma_y': float(final_row['sigma_y']),
                'alpha': float(final_row['alpha']),
                'final_loss': float(final_row['loss']),
            }

            # Extract optimization behavior features (from trajectory)
            opt_features = extract_trajectory_features(trajectory)

            # Combine
            combined = {**static_features, **opt_features}
            all_features.append(combined)

            trajectories_in_file += 1

        print(f"        Extracted {trajectories_in_file:,} Gaussian trajectories")
        total_trajectories += trajectories_in_file

        # Progress estimate
        if file_idx % 6 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / file_idx
            remaining = (24 - file_idx) * avg_time

            print()
            print(f"        Progress: {file_idx}/24 ({100*file_idx/24:.0f}%)")
            print(f"        Elapsed: {elapsed/60:.1f} min")
            print(f"        Estimated remaining: {remaining/60:.0f} min")
            print()

    total_time = time.time() - start_time

    print()
    print(f"✓ Total: {len(all_features):,} Gaussian trajectories processed")
    print(f"  Time: {total_time/60:.1f} minutes")
    print()

    # Convert to DataFrame
    df_features = pd.DataFrame(all_features)

    # Remove rows with NaN in critical features
    critical_features = ['sigma_x', 'sigma_y', 'final_loss', 'convergence_speed']
    nan_mask = df_features[critical_features].isna().any(axis=1)
    nan_count = nan_mask.sum()

    if nan_count > 0:
        print(f"⚠️  Removing {nan_count:,} trajectories with NaN ({100*nan_count/len(df_features):.2f}%)")
        df_features = df_features[~nan_mask]

    print()

    # Summary statistics
    print("="*80)
    print("FEATURE STATISTICS")
    print("="*80)
    print()

    numeric_cols = df_features.select_dtypes(include=[np.number]).columns

    print("Geometric features:")
    for col in ['sigma_x', 'sigma_y', 'alpha']:
        if col in numeric_cols:
            print(f"  {col:25s}: mean={df_features[col].mean():8.4f}, std={df_features[col].std():8.4f}")

    print()
    print("Quality features:")
    if 'final_loss' in numeric_cols:
        print(f"  {'final_loss':25s}: mean={df_features['final_loss'].mean():8.4f}, std={df_features['final_loss'].std():8.4f}")

    print()
    print("Optimization behavior features:")
    opt_features = ['convergence_speed', 'loss_slope', 'loss_linearity', 'loss_curvature',
                    'sigma_x_stability', 'sigma_y_stability', 'parameter_coupling']

    for col in opt_features:
        if col in numeric_cols:
            print(f"  {col:25s}: mean={df_features[col].mean():8.4f}, std={df_features[col].std():8.4f}")

    print()

    # Filter to representative samples
    print("="*80)
    print("FILTERING TO REPRESENTATIVE SAMPLES")
    print("Compositional: Diverse across parameter + optimization behavior space")
    print("="*80)
    print()

    target_samples = 1500
    print(f"Target: {target_samples:,} samples (optimized for 76GB RAM quantum kernel)")
    print()

    # Select features for diversity sampling
    # Include BOTH geometric AND optimization features
    diversity_features = [
        'sigma_x',
        'sigma_y',
        'final_loss',
        'convergence_speed',
        'loss_slope',
        'loss_curvature',
        'sigma_x_stability',
        'sigma_y_stability',
        'parameter_coupling',
    ]

    # Check which features are available
    available_features = [f for f in diversity_features if f in df_features.columns]
    print(f"Using {len(available_features)} features for diversity sampling:")
    for f in available_features:
        print(f"  • {f}")
    print()

    X_diversity = df_features[available_features].values

    # Normalize
    scaler_diversity = StandardScaler()
    X_diversity_scaled = scaler_diversity.fit_transform(X_diversity)

    # K-means clustering to find diverse representatives
    print(f"Running k-means to identify {target_samples:,} diverse representatives...")
    print("  This ensures coverage across BOTH parameter and optimization behavior space")
    print()

    kmeans = MiniBatchKMeans(
        n_clusters=target_samples,
        random_state=42,
        batch_size=1000,
        max_iter=100
    )
    cluster_labels = kmeans.fit_predict(X_diversity_scaled)

    # Select samples closest to centroids
    print("Selecting representatives closest to cluster centroids...")
    selected_indices = []

    for i in range(target_samples):
        cluster_points = np.where(cluster_labels == i)[0]

        if len(cluster_points) > 0:
            centroid = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(X_diversity_scaled[cluster_points] - centroid, axis=1)
            closest_idx = cluster_points[np.argmin(distances)]
            selected_indices.append(closest_idx)

        if (i+1) % 300 == 0:
            print(f"  {i+1}/{target_samples} clusters processed")

    filtered = df_features.iloc[selected_indices].copy()

    print()
    print(f"✓ Selected {len(filtered):,} diverse samples")
    print()

    # Prepare final feature matrix for quantum analysis
    print("="*80)
    print("PREPARING QUANTUM-READY FEATURE MATRIX")
    print("="*80)
    print()

    # Final feature set: parameters + optimization behavior (NO spatial context)
    feature_cols_final = [
        # Geometric (intrinsic properties)
        'sigma_x',
        'sigma_y',
        'alpha',

        # Quality
        'final_loss',

        # Optimization behavior (NEW - defines optimization classes)
        'convergence_speed',
        'loss_slope',
        'loss_curvature',
        'sigma_x_stability',
        'sigma_y_stability',
        'parameter_coupling',
    ]

    # Verify all features exist
    missing_features = [f for f in feature_cols_final if f not in filtered.columns]
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
        print(f"    Using only available features")
        feature_cols_final = [f for f in feature_cols_final if f in filtered.columns]

    X_final = filtered[feature_cols_final].values

    # Normalize for quantum kernel
    scaler_final = StandardScaler()
    X_final_scaled = scaler_final.fit_transform(X_final)

    print(f"Feature matrix: {X_final.shape}")
    print(f"Features ({len(feature_cols_final)}):")
    for i, feat in enumerate(feature_cols_final):
        print(f"  {i+1}. {feat}")
    print()

    print("Feature statistics (normalized):")
    for i, feat in enumerate(feature_cols_final):
        print(f"  {feat:25s}: mean={X_final_scaled[:,i].mean():7.4f}, std={X_final_scaled[:,i].std():7.4f}")

    print()

    # Save enhanced dataset
    print("="*80)
    print("SAVING ENHANCED QUANTUM-READY DATASET")
    print("="*80)
    print()

    dataset = {
        'X': X_final,
        'X_scaled': X_final_scaled,
        'scaler': scaler_final,
        'metadata': filtered[['image_id', 'refinement_pass', 'gaussian_id']].values,
        'n_samples': len(X_final),
        'n_features': len(feature_cols_final),
        'features': feature_cols_final,
        'source': 'Kodak PhotoCD (24 images) with optimization behavior',
        'framework': 'compositional_layers',
        'description': 'Gaussians characterized by parameters + optimization dynamics (NO spatial info)',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'statistics': {
            'total_raw_trajectories': len(df_features),
            'filtered_samples': len(filtered),
            'filter_rate': len(filtered) / len(df_features),
        }
    }

    output_file = 'kodak_gaussians_quantum_ready_enhanced.pkl'

    with open(output_file, 'wb') as f:
        pickle.dump(dataset, f, protocol=4)

    file_size = Path(output_file).stat().st_size

    print(f"✓ Dataset saved: {output_file}")
    print(f"  Samples: {len(X_final):,}")
    print(f"  Features: {len(feature_cols_final)}")
    print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
    print()

    # Comparison with original dataset
    try:
        with open('kodak_gaussians_quantum_ready.pkl', 'rb') as f:
            original_data = pickle.load(f)

        print("="*80)
        print("COMPARISON: ORIGINAL vs ENHANCED")
        print("="*80)
        print()

        print("Original dataset (6D):")
        print(f"  Features: {', '.join(original_data['features'])}")
        print(f"  Framework: Mixed (parameters + image context)")
        print()

        print("Enhanced dataset (10D):")
        print(f"  Features: {', '.join(feature_cols_final)}")
        print(f"  Framework: Pure compositional (parameters + optimization behavior)")
        print()

        print("Key differences:")
        print("  ✓ Added: Optimization behavior features (convergence, stability, coupling)")
        print("  ✓ Removed: Spatial context (edge_coherence, local_gradient)")
        print("  → Enables discovery of OPTIMIZATION CLASSES, not geometric types")
        print()

    except FileNotFoundError:
        pass

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"Processed {total_trajectories:,} Gaussian optimization trajectories")
    print(f"Selected {len(filtered):,} diverse representatives")
    print(f"Filter ratio: {100*len(filtered)/total_trajectories:.1f}%")
    print()
    print("Features capture:")
    print("  ✓ Geometric properties (scale, anisotropy)")
    print("  ✓ Optimization behavior (convergence speed, stability)")
    print("  ✓ Parameter coupling (how dimensions interact during optimization)")
    print("  ✓ Loss landscape topology (curvature, linearity)")
    print("  ✓ NO spatial information (pure compositional framework)")
    print()
    print("Ready for quantum clustering to discover OPTIMIZATION CLASSES!")
    print()

    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print()
    print("1. Run classical baselines on enhanced features:")
    print("     python3 classical_baselines.py")
    print()
    print("2. Run IBM quantum clustering on enhanced features:")
    print("     (Modify Q1_production_real_data.py to load enhanced dataset)")
    print("     python3 Q1_production_real_data.py")
    print()
    print("3. Run Xanadu CV clustering:")
    print("     python3 xanadu_compositional_clustering.py")
    print()
    print("Expected outcome:")
    print("  • Discover 4-6 channels defined by optimization behavior")
    print("  • Each channel: unique convergence profile + optimal strategy")
    print("  • Compositional: all channels contribute everywhere in images")
    print()
    print("="*80)

if __name__ == "__main__":
    try:
        process_all_trajectories()
    except Exception as e:
        print()
        print("="*80)
        print("ERROR")
        print("="*80)
        print()
        print(f"Exception: {e}")
        print()
        import traceback
        traceback.print_exc()
        print()
