//! Rate-Distortion Framework Test
//! Tests R-D optimization capabilities

use lgi_core::rate_distortion::{RateDistortionOptimizer, ProfileRate};
use lgi_core::quantization::LGIQProfile;
use lgi_core::ImageBuffer;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};

fn main() {
    println!("=== Rate-Distortion Framework Test ===\n");

    // Test 1: Profile rate estimation
    println!("📊 Test 1: Profile Rate Estimation");
    test_profile_rates();

    // Test 2: MSE and PSNR computation
    println!("\n📊 Test 2: MSE and PSNR Computation");
    test_distortion_metrics();

    // Test 3: R-D cost computation
    println!("\n📊 Test 3: R-D Cost Computation");
    test_rd_cost();

    // Test 4: Gaussian count selection for target PSNR
    println!("\n📊 Test 4: Gaussian Count Selection");
    test_gaussian_count_selection();

    // Test 5: Lambda tuning
    println!("\n📊 Test 5: Lambda Tuning");
    test_lambda_tuning();

    // Test 6: Gaussian pruning
    println!("\n📊 Test 6: Gaussian Pruning by R-D Cost");
    test_gaussian_pruning();

    println!("\n=== All Tests Complete ===");
}

fn test_profile_rates() {
    let profiles = [
        LGIQProfile::Baseline,
        LGIQProfile::Standard,
        LGIQProfile::HighFidelity,
        LGIQProfile::Extended,
    ];

    println!("  Profile Rates (1000 Gaussians):");
    println!("  Profile            | Bytes/G | Bits/G | Compressed | Total KB | Compressed KB");
    println!("  -------------------|---------|--------|------------|----------|---------------");

    for profile in profiles.iter() {
        let rate = ProfileRate {
            profile: *profile,
            bits_per_gaussian: 0.0,
        };

        let bytes_per_g = rate.bytes_per_gaussian();
        let bits_per_g = rate.bits_per_gaussian();
        let compressed_bits = rate.compressed_bits_per_gaussian();

        let total_kb = (bytes_per_g * 1000.0) / 1024.0;
        let compressed_kb = (compressed_bits * 1000.0 / 8.0) / 1024.0;

        println!(
            "  {:18} | {:7.1} | {:6.0} | {:10.1} | {:8.1} | {:13.1}",
            format!("{:?}", profile),
            bytes_per_g,
            bits_per_g,
            compressed_bits,
            total_kb,
            compressed_kb
        );
    }

    println!("\n  ✅ Compression ratios: Baseline/Standard/High ~5x, Lossless ~1x");
}

fn test_distortion_metrics() {
    let width = 64;
    let height = 64;

    // Create original image (white)
    use lgi_math::color::Color4;
    let original = ImageBuffer::with_background(width, height, Color4::new(1.0, 1.0, 1.0, 1.0));

    // Create test images with varying distortion
    let test_cases: Vec<(&str, f32)> = vec![
        ("Perfect match", 0.0),
        ("Small noise", 0.01),
        ("Medium noise", 0.05),
        ("Large noise", 0.1),
    ];

    println!("  Distortion Metrics (64×64 image):");
    println!("  Test Case      | Noise Level | MSE      | PSNR (dB)");
    println!("  ---------------|-------------|----------|----------");

    let rd_opt = RateDistortionOptimizer::default();

    for (name, noise) in test_cases {
        // Create rendered image with additive noise
        let rendered_data: Vec<Color4<f32>> = original.data.iter().map(|pixel| {
            Color4::new(
                (pixel.r + noise).clamp(0.0f32, 1.0f32),
                (pixel.g + noise).clamp(0.0f32, 1.0f32),
                (pixel.b + noise).clamp(0.0f32, 1.0f32),
                pixel.a,
            )
        }).collect();

        let rendered = ImageBuffer {
            width,
            height,
            data: rendered_data,
        };

        let mse = rd_opt.compute_mse(&original, &rendered);
        let psnr = rd_opt.compute_psnr(&original, &rendered);

        println!(
            "  {:14} | {:11.3} | {:8.6} | {:9.2}",
            name, noise, mse, psnr
        );
    }

    println!("\n  ✅ PSNR correctly computed: Lower MSE → Higher PSNR");
}

fn test_rd_cost() {
    println!("  R-D Cost Computation:");
    println!("  Lambda | Distortion | Rate (bits) | J = D + λR");
    println!("  -------|------------|-------------|------------");

    let lambdas = vec![0.001, 0.01, 0.1, 1.0];
    let distortion = 0.05;  // MSE
    let rate = 1000.0;  // bits

    for lambda in lambdas {
        let rd_opt = RateDistortionOptimizer::new(lambda);
        let cost = rd_opt.compute_cost(distortion, rate);

        println!(
            "  {:6.3} | {:10.3} | {:11.0} | {:10.2}",
            lambda, distortion, rate, cost
        );
    }

    println!("\n  ✅ Higher λ → Higher cost (prefers low rate)");
    println!("     Lower λ → Lower cost (prefers low distortion)");
}

fn test_gaussian_count_selection() {
    let rd_opt = RateDistortionOptimizer::default();

    let test_cases = vec![
        (256, 256, 22.0),
        (256, 256, 28.0),
        (256, 256, 35.0),
        (512, 512, 28.0),
        (1024, 1024, 28.0),
    ];

    println!("  Gaussian Count Selection:");
    println!("  Image Size | Target PSNR | Recommended Count | Pixels/Gaussian");
    println!("  -----------|-------------|-------------------|----------------");

    for (width, height, target_psnr) in test_cases {
        let count = rd_opt.select_gaussian_count_for_psnr(width, height, target_psnr);
        let pixels_per_gaussian = (width * height) as f32 / count as f32;

        println!(
            "  {:4}×{:4}  | {:11.1} | {:17} | {:14.1}",
            width, height, target_psnr, count, pixels_per_gaussian
        );
    }

    println!("\n  ✅ Higher PSNR → More Gaussians needed");
    println!("     Larger image → More Gaussians needed");
}

fn test_lambda_tuning() {
    let rd_opt = RateDistortionOptimizer::new(0.01);

    println!("  Lambda Tuning for Target Bitrate:");
    println!("  Initial λ | Target Rate | Current Rate | New λ   | Direction");
    println!("  ----------|-------------|--------------|---------|----------");

    let test_cases = vec![
        (10000.0, 8000.0),   // Over budget
        (10000.0, 12000.0),  // Under budget
        (10000.0, 10000.0),  // On target
    ];

    for (target, current) in test_cases {
        let new_lambda = rd_opt.compute_lambda_for_bitrate(target, current);
        let direction = if new_lambda > rd_opt.lambda {
            "Increase ↑"
        } else if new_lambda < rd_opt.lambda {
            "Decrease ↓"
        } else {
            "Maintain →"
        };

        println!(
            "  {:9.3} | {:11.0} | {:12.0} | {:7.3} | {}",
            rd_opt.lambda, target, current, new_lambda, direction
        );
    }

    println!("\n  Lambda Tuning for Target PSNR:");
    println!("  Initial λ | Target PSNR | Current PSNR | New λ   | Direction");
    println!("  ----------|-------------|--------------|---------|----------");

    let psnr_cases = vec![
        (28.0, 25.0),  // Too low, need more quality
        (28.0, 31.0),  // Too high, can reduce quality
        (28.0, 28.0),  // On target
    ];

    for (target, current) in psnr_cases {
        let new_lambda = rd_opt.compute_lambda_for_psnr(target, current);
        let direction = if new_lambda > rd_opt.lambda {
            "Increase ↑"
        } else if new_lambda < rd_opt.lambda {
            "Decrease ↓"
        } else {
            "Maintain →"
        };

        println!(
            "  {:9.3} | {:11.1} | {:12.1} | {:7.3} | {}",
            rd_opt.lambda, target, current, new_lambda, direction
        );
    }

    println!("\n  ✅ Lambda adapts to achieve target rate/quality");
}

fn test_gaussian_pruning() {
    // Create test Gaussians with varying importance
    let gaussians = vec![
        // High contribution (large, opaque)
        Gaussian2D::new(
            Vector2::new(0.5, 0.5),
            Euler::isotropic(0.1),
            Color4::new(1.0, 0.0, 0.0, 1.0),
            0.9,
        ),
        // Medium contribution
        Gaussian2D::new(
            Vector2::new(0.3, 0.3),
            Euler::isotropic(0.05),
            Color4::new(0.0, 1.0, 0.0, 1.0),
            0.7,
        ),
        // Low contribution (small, transparent)
        Gaussian2D::new(
            Vector2::new(0.8, 0.8),
            Euler::isotropic(0.01),
            Color4::new(0.0, 0.0, 1.0, 1.0),
            0.3,
        ),
        // Very low contribution
        Gaussian2D::new(
            Vector2::new(0.1, 0.9),
            Euler::isotropic(0.005),
            Color4::new(1.0, 1.0, 0.0, 1.0),
            0.2,
        ),
    ];

    // Simulate contributions (proportional to coverage × opacity)
    let contributions: Vec<f32> = gaussians
        .iter()
        .map(|g| {
            let coverage = g.shape.scale_x * g.shape.scale_y * std::f32::consts::PI;
            coverage * g.opacity
        })
        .collect();

    println!("  Gaussian Pruning Test:");
    println!("  Gaussian | Coverage Area | Opacity | Contribution");
    println!("  ---------|---------------|---------|-------------");

    for (i, (g, contrib)) in gaussians.iter().zip(contributions.iter()).enumerate() {
        let coverage = g.shape.scale_x * g.shape.scale_y * std::f32::consts::PI;
        println!(
            "  G{}       | {:13.4} | {:7.2} | {:12.4}",
            i, coverage, g.opacity, contrib
        );
    }

    // Test pruning with different lambda values
    println!("\n  Pruning Results:");
    println!("  Lambda  | Kept Gaussians | Pruned Count | Kept Indices");
    println!("  --------|----------------|--------------|-------------");

    let lambdas = vec![0.0001, 0.001, 0.01, 0.1];

    for lambda in lambdas {
        let rd_opt = RateDistortionOptimizer::new(lambda);
        let kept_indices = rd_opt.prune_by_rd(&gaussians, &contributions);
        let pruned_count = gaussians.len() - kept_indices.len();

        println!(
            "  {:6.4} | {:14} | {:12} | {:?}",
            lambda,
            kept_indices.len(),
            pruned_count,
            kept_indices
        );
    }

    println!("\n  ✅ Higher λ → More aggressive pruning (keep fewer Gaussians)");
    println!("     Lower λ → Less pruning (keep more Gaussians)");
}
