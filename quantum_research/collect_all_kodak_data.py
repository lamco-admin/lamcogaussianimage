#!/usr/bin/env python3
"""
Collect Gaussian configuration data from all 24 Kodak images.

Orchestrates the Rust encoder to process each image and collect
real Gaussian configurations for quantum research.

Expected runtime: 2.5-3.5 hours (7-8 min per image × 24 images)
Expected output: ~250,000-400,000 Gaussian configuration snapshots
"""

import subprocess
import time
from pathlib import Path
import sys

# Paths
RUST_BINARY = Path("../packages/lgi-rs/target/release/examples/collect_gaussian_data")
KODAK_DIR = Path("../test-data/kodak-dataset")
OUTPUT_DIR = Path("./kodak_gaussian_data")

def check_setup():
    """Verify everything is in place before starting."""
    print("="*80)
    print("PRE-FLIGHT CHECKS")
    print("="*80)
    print()

    # Check Rust binary
    if not RUST_BINARY.exists():
        print(f"✗ Rust binary not found: {RUST_BINARY}")
        print()
        print("Build it with:")
        print("  cd ../packages/lgi-rs")
        print("  cargo build --release --example collect_gaussian_data")
        return False

    print(f"✓ Rust binary: {RUST_BINARY}")

    # Check Kodak dataset
    kodak_images = sorted(KODAK_DIR.glob("kodim*.png"))
    if len(kodak_images) < 24:
        print(f"✗ Expected 24 Kodak images, found {len(kodak_images)}")
        return False

    print(f"✓ Kodak dataset: {len(kodak_images)} images")

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"✓ Output directory: {OUTPUT_DIR}")

    print()
    return True

def encode_image(image_path, image_id, index, total):
    """
    Encode a single Kodak image with data logging.

    Returns: (success: bool, runtime_seconds: float, snapshots_collected: int)
    """
    output_file = OUTPUT_DIR / f"{image_id}.csv"

    print()
    print("="*80)
    print(f"[{index}/{total}] {image_id.upper()}")
    print("="*80)
    print(f"Input:  {image_path}")
    print(f"Output: {output_file}")
    print()

    start_time = time.time()

    try:
        # Run Rust encoder
        result = subprocess.run(
            [str(RUST_BINARY), str(image_path), str(output_file)],
            capture_output=True,
            text=True,
            timeout=900,  # 15 minute timeout per image
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"✗ Encoding FAILED (exit code {result.returncode})")
            print()
            print("STDERR:")
            print(result.stderr)
            return (False, elapsed, 0)

        # Count snapshots
        with open(output_file, 'r') as f:
            line_count = sum(1 for _ in f) - 1  # Subtract header

        # Extract final PSNR from output
        final_psnr = None
        for line in result.stdout.split('\n'):
            if 'Final PSNR' in line:
                try:
                    final_psnr = float(line.split(':')[-1].strip().replace(' dB', ''))
                except:
                    pass

        print(f"✓ Encoding complete")
        print(f"  Runtime: {elapsed/60:.2f} minutes")
        print(f"  Snapshots: {line_count:,}")
        if final_psnr:
            print(f"  Final PSNR: {final_psnr:.2f} dB")
        print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

        # Check for NaN contamination
        nan_check = subprocess.run(
            ['grep', '-c', 'NaN', str(output_file)],
            capture_output=True,
            text=True
        )

        nan_count = 0 if nan_check.returncode != 0 else int(nan_check.stdout.strip())

        if nan_count > 0:
            print(f"  ⚠️  NaN values detected: {nan_count} lines")
        else:
            print(f"  ✓ Data quality: Clean (no NaN)")

        return (True, elapsed, line_count)

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"✗ TIMEOUT after {elapsed/60:.1f} minutes")
        return (False, elapsed, 0)

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ ERROR: {e}")
        return (False, elapsed, 0)

def main():
    print()
    print("="*80)
    print("KODAK GAUSSIAN DATA COLLECTION - FULL DATASET")
    print("Quantum Research: Real Gaussian Configurations from Image Encoding")
    print("="*80)
    print()

    # Pre-flight checks
    if not check_setup():
        print("Pre-flight checks failed. Aborting.")
        sys.exit(1)

    # Find all Kodak images
    kodak_images = sorted(KODAK_DIR.glob("kodim*.png"))[:24]

    print("="*80)
    print("COLLECTION PLAN")
    print("="*80)
    print(f"  Images: {len(kodak_images)}")
    print(f"  Estimated time per image: 7-8 minutes")
    print(f"  Estimated total time: {len(kodak_images) * 7.5 / 60:.1f} hours")
    print(f"  Expected snapshots: ~250,000-400,000 total")
    print(f"  Expected disk usage: ~30-40 MB compressed")
    print()
    print("Starting collection...")
    print()

    # Collection statistics
    total_start = time.time()
    results = {
        'successful': 0,
        'failed': 0,
        'total_snapshots': 0,
        'total_time': 0.0,
    }

    failed_images = []

    # Process each image
    for idx, image_path in enumerate(kodak_images, 1):
        image_id = image_path.stem  # kodim01, kodim02, etc.

        success, runtime, snapshots = encode_image(image_path, image_id, idx, len(kodak_images))

        if success:
            results['successful'] += 1
            results['total_snapshots'] += snapshots
        else:
            results['failed'] += 1
            failed_images.append(image_id)

        results['total_time'] += runtime

        # Progress update every 5 images
        if idx % 5 == 0:
            elapsed_total = time.time() - total_start
            avg_time = elapsed_total / idx
            remaining = (len(kodak_images) - idx) * avg_time

            print()
            print("-"*80)
            print(f"PROGRESS: {idx}/{len(kodak_images)} ({100*idx/len(kodak_images):.0f}%)")
            print(f"  Average time/image: {avg_time/60:.1f} minutes")
            print(f"  Elapsed: {elapsed_total/60:.1f} minutes ({elapsed_total/3600:.2f} hours)")
            print(f"  Estimated remaining: {remaining/60:.0f} minutes ({remaining/3600:.1f} hours)")
            print(f"  Success rate: {results['successful']}/{idx} ({100*results['successful']/idx:.0f}%)")
            print(f"  Total snapshots so far: {results['total_snapshots']:,}")
            print("-"*80)
            print()

    total_elapsed = time.time() - total_start

    # Final summary
    print()
    print("="*80)
    print("COLLECTION COMPLETE")
    print("="*80)
    print()
    print(f"Total runtime: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    print(f"Successful: {results['successful']}/{len(kodak_images)}")
    print(f"Failed: {results['failed']}")
    print(f"Total snapshots: {results['total_snapshots']:,}")
    print()

    if failed_images:
        print("Failed images:")
        for img in failed_images:
            print(f"  - {img}")
        print()

    # Check output files
    csv_files = list(OUTPUT_DIR.glob("*.csv"))
    total_size = sum(f.stat().st_size for f in csv_files)

    print(f"Output files: {len(csv_files)}")
    print(f"Total disk usage: {total_size / 1024 / 1024:.2f} MB")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")
    print()

    print("="*80)
    print("NEXT STEP")
    print("="*80)
    print()
    print("Run dataset preparation:")
    print("  cd quantum_research")
    print("  python prepare_quantum_dataset.py")
    print()
    print("This will:")
    print("  - Load all CSV files")
    print("  - Filter to ~10,000 representative samples")
    print("  - Normalize features for quantum kernel")
    print("  - Save quantum-ready dataset")
    print()
    print("Expected time: 2-5 minutes")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print()
        print("="*80)
        print("INTERRUPTED BY USER")
        print("="*80)
        print()
        print("Partial results may be available in:")
        print(f"  {OUTPUT_DIR.absolute()}")
        print()
        sys.exit(1)
