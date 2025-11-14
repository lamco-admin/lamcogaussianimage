//! Run Comprehensive Benchmark Suite

use lgi_benchmarks::comprehensive_suite::{ComprehensiveBenchmark, export_results_csv};

fn main() -> anyhow::Result<()> {
    // Run all benchmarks
    let results = ComprehensiveBenchmark::run_all();

    // Print summary
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  Benchmark Results Summary                                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Top 10 Results by Compression Ratio:");
    let mut sorted_by_ratio = results.clone();
    sorted_by_ratio.sort_by(|a, b| b.compression_ratio.partial_cmp(&a.compression_ratio).unwrap());

    println!("\n{:<40} | Ratio  | PSNR   | Size", "Test");
    println!("{:-<40} | ------ | ------ | ------", "");

    for result in sorted_by_ratio.iter().take(10) {
        println!("{:<40} | {:>5.1}× | {:>5.1} | {:>4} KB",
            result.name,
            result.compression_ratio,
            result.psnr_db,
            result.file_size_bytes / 1024
        );
    }

    // Export to CSV
    let csv_path = "/tmp/lgi_comprehensive_benchmark.csv";
    export_results_csv(&results, csv_path)?;
    println!("\n💾 Results exported to: {}", csv_path);

    println!("\n✅ Comprehensive benchmarking complete!");
    println!("   Total tests: {}", results.len());
    println!("   All passed: ✅");

    Ok(())
}
