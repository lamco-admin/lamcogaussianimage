#!/usr/bin/env python3
"""
Extract Within-Channel Correlation Patterns

Different channels showed opposite correlations:
- Channel 3: coherence vs loss = +0.894 (higher edge = worse)
- Channel 6: coherence vs loss = -0.992 (higher edge = better)

This script systematically extracts all pairwise feature correlations
within each channel to discover channel-specific optimization patterns.
"""

import json
import pickle
import numpy as np
from scipy.stats import pearsonr
import pandas as pd

print("="*80)
print("WITHIN-CHANNEL CORRELATION PATTERNS")
print("="*80)
print()

# Load data
with open('gaussian_channels_kodak_quantum.json') as f:
    results = json.load(f)

with open('kodak_gaussians_quantum_ready.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
features = data['features']
labels = np.array(results['labels_quantum'])

print(f"Features analyzed: {features}")
print(f"Channels: {results['n_clusters']}")
print()

# For each channel, compute all pairwise correlations
all_patterns = {}

for ch_id in range(results['n_clusters']):
    mask = labels == ch_id
    n = np.sum(mask)

    if n < 5:  # Need at least 5 samples for meaningful correlation
        continue

    ch_data = X[mask]

    print("="*80)
    print(f"CHANNEL {ch_id}: {n} Gaussians")
    print("="*80)
    print()

    channel_patterns = {}

    # Compute pairwise correlations
    for i, feat_i in enumerate(features):
        for j, feat_j in enumerate(features):
            if i >= j:  # Skip diagonal and duplicates
                continue

            # Skip alpha (zero variance)
            if feat_i == 'alpha' or feat_j == 'alpha':
                continue

            try:
                corr, p_val = pearsonr(ch_data[:,i], ch_data[:,j])

                # Store if significant
                if abs(corr) > 0.5 and p_val < 0.05:
                    pair_key = f"{feat_i} vs {feat_j}"
                    channel_patterns[pair_key] = {
                        'correlation': float(corr),
                        'p_value': float(p_val),
                        'strength': 'strong' if abs(corr) > 0.7 else 'moderate'
                    }

                    # Print significant patterns
                    strength = "STRONG" if abs(corr) > 0.7 else "moderate"
                    direction = "positive" if corr > 0 else "negative"

                    print(f"  {feat_i:15s} vs {feat_j:15s}: r={corr:6.3f} (p={p_val:.4f}) [{strength}, {direction}]")

            except:
                pass

    if channel_patterns:
        all_patterns[ch_id] = channel_patterns
        print()
    else:
        print("  No strong correlations found")
        print()

# Summary of patterns
print("="*80)
print("PATTERN SUMMARY")
print("="*80)
print()

# Group by correlation type
scale_loss_patterns = {}
coherence_loss_patterns = {}
scale_coherence_patterns = {}

for ch_id, patterns in all_patterns.items():
    for pair_key, stats in patterns.items():
        if 'sigma' in pair_key and 'loss' in pair_key:
            scale_loss_patterns[ch_id] = stats['correlation']
        elif 'coherence' in pair_key and 'loss' in pair_key:
            coherence_loss_patterns[ch_id] = stats['correlation']
        elif 'sigma' in pair_key and 'coherence' in pair_key:
            scale_coherence_patterns[ch_id] = stats['correlation']

print("Scale-Loss Correlations by Channel:")
if scale_loss_patterns:
    for ch_id, corr in scale_loss_patterns.items():
        direction = "smaller scales = better" if corr < 0 else "larger scales = better"
        print(f"  Channel {ch_id}: r={corr:.3f} ({direction})")
else:
    print("  No significant patterns")
print()

print("Coherence-Loss Correlations by Channel:")
if coherence_loss_patterns:
    for ch_id, corr in coherence_loss_patterns.items():
        direction = "sharper edges = better" if corr < 0 else "sharper edges = worse"
        print(f"  Channel {ch_id}: r={corr:.3f} ({direction})")
else:
    print("  No significant patterns")
print()

# Key insight
print("="*80)
print("KEY INSIGHTS")
print("="*80)
print()

if coherence_loss_patterns:
    # Check if pattern is consistent or varies
    corr_values = list(coherence_loss_patterns.values())

    if max(corr_values) * min(corr_values) < 0:  # Opposite signs
        print("CRITICAL FINDING: Coherence-loss correlation REVERSES between channels!")
        print()
        print("Some channels: sharper edges → better quality")
        print("Other channels: sharper edges → worse quality")
        print()
        print("This proves channels have fundamentally different optimization behavior.")
        print("NOT just size groupings - they represent different Gaussian 'types'")
        print("that respond differently to image context.")
    else:
        print("Coherence-loss pattern is consistent across channels")

print()

# Save all patterns
with open('within_channel_patterns.json', 'w') as f:
    json.dump({
        'all_patterns': all_patterns,
        'scale_loss_by_channel': {str(k): float(v) for k, v in scale_loss_patterns.items()},
        'coherence_loss_by_channel': {str(k): float(v) for k, v in coherence_loss_patterns.items()},
        'interpretation': 'Channels have different internal correlation structures'
    }, f, indent=2)

print("Detailed patterns saved: within_channel_patterns.json")
print()
