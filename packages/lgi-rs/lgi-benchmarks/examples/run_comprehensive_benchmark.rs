//! Comprehensive benchmark runner
//!
//! This creates a full benchmark suite testing:
//! - Different image patterns
//! - Various Gaussian counts
//! - Multiple quality presets
//! - Different resolutions

use lgi_benchmarks::test_images::{TestImageGenerator, TestPattern};
use lgi_benchmarks::{BenchmarkRunner, BenchmarkConfig};
use std::path::Path;

fn main() {
    println!("╔════════════════════════════════════════════════╗");
    println!("║   LGI Comprehensive Benchmark Suite          ║");
    println!("╚════════════════════════════════════════════════╝\n");

    // Create output directory
    let output_dir = Path::new("benchmark_results");
    std::fs::create_dir_all(output_dir).unwrap();

    // Test patterns to benchmark
    let test_patterns = vec![
        ("solid_color", TestPattern::SolidColor),
        ("linear_gradient", TestPattern::LinearGradient),
        ("radial_gradient", TestPattern::RadialGradient),
        ("checkerboard", TestPattern::Checkerboard),
        ("concentric_circles", TestPattern::ConcentricCircles),
        ("frequency_sweep", TestPattern::FrequencySweep),
        ("random_noise", TestPattern::RandomNoise),
        ("natural_scene", TestPattern::NaturalScene),
        ("geometric", TestPattern::Geometric),
        ("text_pattern", TestPattern::TextPattern),
    ];

    // Benchmark configuration
    let config = BenchmarkConfig {
        sizes: vec![128, 256],
        gaussian_counts: vec![100, 200, 500],
        quality_presets: vec!["fast".to_string()],
        num_runs: 2,
    };

    // Run benchmarks for each pattern
    for (name, pattern) in test_patterns {
        println!("\n==================================================");
        println!("Testing pattern: {}", name);
        println!("==================================================\n");

        let gen = TestImageGenerator::new(256).with_seed(42);
        let test_image = gen.generate(pattern);

        // Save test image
        let test_img_path = output_dir.join(format!("{}_test.png", name));
        test_image.save(&test_img_path).unwrap();

        let mut runner = BenchmarkRunner::new(config.clone());
        runner.run_suite(&test_image);

        // Export results
        let csv_path = output_dir.join(format!("{}_results.csv", name));
        let json_path = output_dir.join(format!("{}_results.json", name));

        runner.export_csv(&csv_path).unwrap();
        runner.export_json(&json_path).unwrap();

        runner.print_summary();

        println!("\nResults saved:");
        println!("  CSV:  {}", csv_path.display());
        println!("  JSON: {}", json_path.display());
    }

    println!("\n╔════════════════════════════════════════════════╗");
    println!("║   All Benchmarks Complete!                    ║");
    println!("╚════════════════════════════════════════════════╝");
    println!("\nResults in: {}", output_dir.display());
}
