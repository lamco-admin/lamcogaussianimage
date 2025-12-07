#!/usr/bin/env python3
"""
Q2 Multi-Hour Experiment: Algorithm Discovery Per Quantum Channel

Tests which optimizer works best for each quantum channel by running
systematic experiments across channels × algorithms × images.

Runtime: 3-4 hours with parallelization
Output: Empirical mapping of optimal algorithm per channel
"""

import subprocess
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

# Configuration
RUST_BINARY_PATH = Path("../packages/lgi-rs/target/release/examples")
KODAK_DIR = Path("../test-data/kodak-dataset")
OUTPUT_DIR = Path("./q2_algorithm_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load quantum channels
with open('gaussian_channels_kodak_quantum.json') as f:
    quantum_results = json.load(f)
    channels = quantum_results['quantum_channels']

# Algorithms to test
ALGORITHMS = ['adam', 'optimizerv2', 'optimizerv3']

# Test images
TEST_IMAGES_SMALL = ['kodim03', 'kodim08', 'kodim15']
TEST_IMAGES_FULL = [f'kodim{i:02d}' for i in range(1, 25)]

def run_single_experiment(channel_id, algorithm, image_id, full_image_set=False):
    """
    Run single encoding experiment with specific channel config + algorithm.

    Returns: dict with results or None if failed
    """
    channel = channels[channel_id]
    
    # Build command (will create this Rust binary)
    # For now, return mock data structure
    result = {
        'channel_id': channel_id,
        'algorithm': algorithm,
        'image': image_id,
        'sigma_x_init': channel['sigma_x_mean'],
        'sigma_y_init': channel['sigma_y_mean'],
        'final_psnr': 0.0,  # To be filled by Rust
        'final_loss': 0.0,
        'iterations': 0,
        'time_seconds': 0.0,
        'converged': False,
    }
    
    print(f"  Ch{channel_id} + {algorithm:12s} + {image_id}: ", end='', flush=True)
    
    # TODO: Call Rust binary when implemented
    # For now, print plan
    print(f"[Would run: encode with {algorithm}]")
    
    return result

def main():
    use_full_set = '--full' in sys.argv
    max_workers = int(sys.argv[sys.argv.index('--parallel') + 1]) if '--parallel' in sys.argv else 8
    
    test_images = TEST_IMAGES_FULL if use_full_set else TEST_IMAGES_SMALL
    
    print("="*80)
    print("Q2: ALGORITHM DISCOVERY PER QUANTUM CHANNEL")
    print("="*80)
    print()
    print(f"Channels: {len(channels)}")
    print(f"Algorithms: {len(ALGORITHMS)} ({', '.join(ALGORITHMS)})")
    print(f"Images: {len(test_images)}")
    print(f"Parallel workers: {max_workers}")
    print()
    
    total_exp = len(channels) * len(ALGORITHMS) * len(test_images)
    est_time_serial = total_exp * 2.5 / 60
    est_time_parallel = est_time_serial / max_workers
    
    print(f"Total experiments: {total_exp}")
    print(f"Estimated time (serial): {est_time_serial:.1f} hours")
    print(f"Estimated time (parallel): {est_time_parallel:.1f} hours")
    print()
    print("Starting in 3 seconds...")
    time.sleep(3)
    print()
    
    # Collect all experiment configurations
    experiments = []
    for ch_id in range(len(channels)):
        for algo in ALGORITHMS:
            for image_id in test_images:
                experiments.append((ch_id, algo, image_id, use_full_set))
    
    # Run experiments in parallel
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_experiment, ch_id, algo, img, full): (ch_id, algo, img)
            for ch_id, algo, img, full in experiments
        }
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
            
            completed += 1
            if completed % 10 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total_exp - completed) / rate if rate > 0 else 0
                print(f"Progress: {completed}/{total_exp} ({100*completed/total_exp:.0f}%) | ETA: {remaining/60:.0f} min")
    
    elapsed_total = time.time() - start_time
    
    # Save results
    output_file = OUTPUT_DIR / 'algorithm_comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'experiments': results,
            'n_experiments': len(results),
            'runtime_seconds': elapsed_total,
            'runtime_hours': elapsed_total / 3600,
        }, f, indent=2)
    
    print()
    print("="*80)
    print("EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"Runtime: {elapsed_total/3600:.2f} hours")
    print(f"Results: {output_file}")
    print()
    
    # Analyze: Which algorithm wins per channel?
    analyze_results(results)

def analyze_results(results):
    """Determine best algorithm for each channel"""
    
    channel_summary = {}
    
    for ch_id in range(len(channels)):
        channel_summary[ch_id] = {}
        
        for algo in ALGORITHMS:
            ch_algo_results = [r for r in results 
                               if r['channel_id'] == ch_id and r['algorithm'] == algo]
            
            if ch_algo_results:
                avg_psnr = sum(r['final_psnr'] for r in ch_algo_results) / len(ch_algo_results)
                channel_summary[ch_id][algo] = avg_psnr
        
        # Find winner
        if channel_summary[ch_id]:
            winner = max(channel_summary[ch_id].items(), key=lambda x: x[1])
            channel_summary[ch_id]['winner'] = winner[0]
            channel_summary[ch_id]['best_psnr'] = winner[1]
    
    # Print summary
    print("="*80)
    print("ALGORITHM WINNERS PER CHANNEL")
    print("="*80)
    print()
    print(f"{'Channel':<10} | {'Adam':<10} | {'OptimizerV2':<12} | {'OptimizerV3':<12} | {'Winner':<12}")
    print("-"*80)
    
    for ch_id in sorted(channel_summary.keys()):
        ch_data = channel_summary[ch_id]
        adam_psnr = ch_data.get('adam', 0.0)
        v2_psnr = ch_data.get('optimizerv2', 0.0)
        v3_psnr = ch_data.get('optimizerv3', 0.0)
        winner = ch_data.get('winner', 'unknown')
        
        print(f"{ch_id:<10} | {adam_psnr:>8.2f} dB | {v2_psnr:>10.2f} dB | {v3_psnr:>10.2f} dB | {winner:<12}")
    
    print()

if __name__ == "__main__":
    main()
