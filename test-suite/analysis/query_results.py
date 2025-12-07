#!/usr/bin/env python3
"""Query test results by metadata.

Examples:
    # Find all Adam results
    python query_results.py --optimizer Adam

    # Find toolkit tests above 30 dB
    python query_results.py --module toolkit --min-psnr 30.0

    # Find all tests on kodim03
    python query_results.py --image kodim03
"""

import json
import glob
import argparse
from pathlib import Path
from typing import List, Dict, Any

def load_results(data_dir: str = "../data") -> List[Dict[str, Any]]:
    """Load all JSON result files."""
    json_files = glob.glob(f"{data_dir}/*.json")
    results = []
    for f in json_files:
        try:
            with open(f) as fp:
                results.append(json.load(fp))
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
    return results

def filter_results(results: List[Dict], **kwargs) -> List[Dict]:
    """Filter results by criteria."""
    filtered = results

    if kwargs.get('optimizer'):
        filtered = [r for r in filtered if r.get('optimizer') == kwargs['optimizer']]

    if kwargs.get('module'):
        filtered = [r for r in filtered if kwargs['module'] in r.get('module_under_test', '')]

    if kwargs.get('image'):
        filtered = [r for r in filtered if kwargs['image'] in r.get('image_path', '')]

    if kwargs.get('min_psnr'):
        filtered = [r for r in filtered if r.get('final_psnr', 0) >= kwargs['min_psnr']]

    if kwargs.get('max_psnr'):
        filtered = [r for r in filtered if r.get('final_psnr', 999) <= kwargs['max_psnr']]

    return filtered

def main():
    parser = argparse.ArgumentParser(description="Query test results")
    parser.add_argument('--optimizer', help="Filter by optimizer")
    parser.add_argument('--module', help="Filter by module_under_test")
    parser.add_argument('--image', help="Filter by image name")
    parser.add_argument('--min-psnr', type=float, help="Minimum PSNR")
    parser.add_argument('--max-psnr', type=float, help="Maximum PSNR")
    parser.add_argument('--data-dir', default='../data', help="Data directory")
    parser.add_argument('--output', help="Output file (default: stdout)")

    args = parser.parse_args()

    # Load results
    results = load_results(args.data_dir)
    print(f"Loaded {len(results)} results")

    # Filter
    filtered = filter_results(
        results,
        optimizer=args.optimizer,
        module=args.module,
        image=args.image,
        min_psnr=args.min_psnr,
        max_psnr=args.max_psnr
    )

    print(f"Matched {len(filtered)} results\n")

    # Display
    for r in filtered:
        print(f"{r.get('test_name'):60} | {r.get('optimizer'):15} | {r.get('final_psnr', 0):6.2f} dB | {r.get('image_path', 'unknown'):30}")

if __name__ == '__main__':
    main()
