//! Comprehensive Gamma Characterization
//!
//! Goal: Find γ(N) function that works across ALL Gaussian counts
//! Method: Test many N values, fit empirical formula
//! Output: Validated γ(N) function for implementation

use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use lgi_encoder_v2::renderer_v2::RendererV2;

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Comprehensive Gamma Characterization Experiment          ║");
    println!("║  Finding: γ(N) function that works for ALL counts         ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Test on solid color (simplest case, ground truth)
    println!("═══════════════════════════════════════════════════════════");
    println!("Test Matrix: Solid Red 256×256");
    println!("Testing: N ∈ [25, 64, 100, 225, 400, 625, 900, 1600, 2500]");
    println!("Finding: Optimal γ for each N");
    println!("═══════════════════════════════════════════════════════════\n");

    let gaussian_counts = [25, 64, 100, 225, 400, 625, 900, 1600, 2500];

    let mut results = Vec::new();

    for &N in &gaussian_counts {
        let grid_size = (N as f32).sqrt() as u32;

        println!("─────────────────────────────────────────────────");
        println!("N = {} Gaussians ({}×{} grid)", N, grid_size, grid_size);
        println!("─────────────────────────────────────────────────");

        // Test multiple γ values
        let gammas = if N < 100 {
            vec![0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
        } else if N < 500 {
            vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
        } else {
            vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        };

        let mut best_gamma = 0.0;
        let mut best_psnr = 0.0;

        for &gamma in &gammas {
            let psnr = test_gamma(N, grid_size, gamma);

            if psnr > best_psnr {
                best_psnr = psnr;
                best_gamma = gamma;
            }

            let status = if psnr >= 99.0 {
                "✅"
            } else if psnr >= 90.0 {
                "✓"
            } else if psnr >= 80.0 {
                "⚠️"
            } else {
                "❌"
            };

            println!("  γ={:.2}: PSNR={:6.2} dB  {}", gamma, psnr, status);
        }

        println!("\n  Best γ for N={}: {:.2} (PSNR={:.2} dB)", N, best_gamma, best_psnr);
        println!();

        results.push((N, best_gamma, best_psnr));
    }

    // Analyze results
    println!("\n═══════════════════════════════════════════════════");
    println!("SUMMARY: Optimal γ for Each N");
    println!("═══════════════════════════════════════════════════");
    println!("\n| N | Optimal γ | PSNR | √N | γ×√N |");
    println!("|---|-----------|------|-----|------|");

    for (n, gamma, psnr) in &results {
        let sqrt_n = (*n as f32).sqrt();
        let product = gamma * sqrt_n;
        println!("| {} | {:.2} | {:.1} dB | {:.1} | {:.1} |", n, gamma, psnr, sqrt_n, product);
    }

    // Try to fit formula
    println!("\n═══════════════════════════════════════════════════");
    println!("FORMULA FITTING:");
    println!("═══════════════════════════════════════════════════");

    // Check if γ × √N is constant
    let products: Vec<f32> = results.iter().map(|(n, g, _)| g * (*n as f32).sqrt()).collect();
    let avg_product: f32 = products.iter().sum::<f32>() / products.len() as f32;
    let std_dev: f32 = (products.iter()
        .map(|p| (p - avg_product).powi(2))
        .sum::<f32>() / products.len() as f32).sqrt();

    println!("Testing: γ × √N = constant?");
    println!("  Mean(γ×√N) = {:.2}", avg_product);
    println!("  StdDev = {:.2}", std_dev);
    println!("  CoV = {:.1}%\n", 100.0 * std_dev / avg_product);

    if std_dev / avg_product < 0.2 {
        println!("✅ γ × √N is approximately constant!");
        println!("Recommended formula:");
        println!("  γ(N) = {:.1} / √N", avg_product);
    } else {
        println!("⚠️  γ × √N varies significantly");
        println!("Need more complex formula (power law, piecewise, etc.)");
    }

    // Alternative: try γ ∝ N^α
    println!("\nTesting: γ = k × N^α");
    // Simple linear regression on log-log scale
    let log_n: Vec<f32> = results.iter().map(|(n, _, _)| (*n as f32).ln()).collect();
    let log_gamma: Vec<f32> = results.iter().map(|(_, g, _)| g.ln()).collect();

    let n_mean: f32 = log_n.iter().sum::<f32>() / log_n.len() as f32;
    let g_mean: f32 = log_gamma.iter().sum::<f32>() / log_gamma.len() as f32;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..log_n.len() {
        numerator += (log_n[i] - n_mean) * (log_gamma[i] - g_mean);
        denominator += (log_n[i] - n_mean).powi(2);
    }

    let alpha = numerator / denominator;
    let k = (g_mean - alpha * n_mean).exp();

    println!("  Best fit: γ = {:.2} × N^({:.3})", k, alpha);
    println!();

    // Test fitted formula
    println!("Validation of γ = {:.2} × N^({:.3}):", k, alpha);
    for (n, gamma_actual, _) in &results {
        let gamma_predicted = k * (*n as f32).powf(alpha);
        let error = ((gamma_predicted - gamma_actual) / gamma_actual * 100.0).abs();
        println!("  N={:4}: γ_actual={:.2}, γ_pred={:.2}, error={:4.1}%",
            n, gamma_actual, gamma_predicted, error);
    }
}

fn test_gamma(N: usize, grid_size: u32, gamma: f32) -> f32 {
    // Create solid red target
    let mut target = ImageBuffer::new(256, 256);
    for pixel in &mut target.data {
        *pixel = Color4::new(1.0, 0.0, 0.0, 1.0);
    }

    // Compute sigma from gamma
    let sigma_base_px = gamma * ((256.0_f32 * 256.0) / N as f32).sqrt();
    let sigma_norm = sigma_base_px / 256.0;

    // Create Gaussians on grid
    let mut gaussians = Vec::new();
    let step = 256 / grid_size;

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let x = ((gx * step + step / 2) as f32).min(255.0) / 256.0;
            let y = ((gy * step + step / 2) as f32).min(255.0) / 256.0;

            gaussians.push(Gaussian2D::new(
                Vector2::new(x, y),
                Euler::isotropic(sigma_norm),
                Color4::new(1.0, 0.0, 0.0, 1.0),  // Correct color
                1.0,
            ));
        }
    }

    // Render and measure
    let rendered = RendererV2::render(&gaussians, 256, 256);
    compute_psnr(&target, &rendered)
}

fn compute_psnr(original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    let count = (original.width * original.height * 3) as f32;

    for (p1, p2) in original.data.iter().zip(rendered.data.iter()) {
        mse += (p1.r - p2.r).powi(2);
        mse += (p1.g - p2.g).powi(2);
        mse += (p1.b - p2.b).powi(2);
    }

    mse /= count;

    if mse < 1e-10 {
        100.0
    } else {
        20.0 * (1.0 / mse.sqrt()).log10()
    }
}
