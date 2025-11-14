//! Module Toggle Test - Systematic Enable/Disable
//!
//! Test EACH module independently:
//! - Geodesic EDT clamping
//! - Structure tensor alignment
//! - Adaptive LR (density-based)
//! - Guided filter colors
//! - MS-SSIM loss
//! - Edge-weighted gradients
//!
//! Fixed N=100, 100 iterations, measure impact of each

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};
use lgi_core::ImageBuffer;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler};
use std::path::PathBuf;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("       MODULE TOGGLE TEST - Enable/Disable Each Feature        ");
    println!("═══════════════════════════════════════════════════════════════\n");

    let image = load_test_image();

    println!("Test: kodim02.png (768×512)");
    println!("Fixed N=100, 100 optimization iterations\n");

    // Test configurations
    let tests: Vec<(&str, fn(&ImageBuffer<f32>) -> (f32, f32))> = vec![
        ("Baseline (all default)", test_all_default as fn(&ImageBuffer<f32>) -> (f32, f32)),
        ("NO geodesic clamping", test_no_geodesic as fn(&ImageBuffer<f32>) -> (f32, f32)),
        ("NO structure tensor (isotropic only)", test_no_structure as fn(&ImageBuffer<f32>) -> (f32, f32)),
        ("NO adaptive LR", test_no_adaptive_lr as fn(&ImageBuffer<f32>) -> (f32, f32)),
        ("NO guided filter", test_no_guided as fn(&ImageBuffer<f32>) -> (f32, f32)),
        ("WITH MS-SSIM loss", test_with_msssim as fn(&ImageBuffer<f32>) -> (f32, f32)),
        ("WITH edge-weighted", test_with_edge_weight as fn(&ImageBuffer<f32>) -> (f32, f32)),
    ];

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║ Configuration                          │ Init   │ Final  │ Gain   │ Status ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");

    let mut results = Vec::new();

    for (name, test_fn) in tests {
        let (init_psnr, final_psnr) = test_fn(&image);
        let gain = final_psnr - init_psnr;
        let status = if gain > 0.3 { "✅" } else if gain > -0.5 { "⚠️ " } else { "❌" };

        println!("║ {:38} │ {:6.2} │ {:6.2} │ {:+6.2} │   {}   ║",
            name, init_psnr, final_psnr, gain, status);

        results.push((name, init_psnr, final_psnr, gain));
    }

    println!("╚═══════════════════════════════════════════════════════════════════════════╝");

    println!("\n📊 EMPIRICAL FINDINGS:\n");

    // Find best configuration
    let best = results.iter().max_by(|a, b| a.3.partial_cmp(&b.3).unwrap()).unwrap();
    let worst = results.iter().min_by(|a, b| a.3.partial_cmp(&b.3).unwrap()).unwrap();

    println!("  BEST: {} ({:+.2} dB gain)", best.0, best.3);
    println!("  WORST: {} ({:+.2} dB gain)", worst.0, worst.3);

    println!("\n  Module Impact:");
    let baseline_gain = results[0].3;
    for (name, _, _, gain) in &results[1..] {
        let diff = gain - baseline_gain;
        if diff.abs() > 0.1 {
            println!("    {}: {:+.2} dB vs baseline", name, diff);
        }
    }
}

fn test_all_default(image: &ImageBuffer<f32>) -> (f32, f32) {
    let encoder = EncoderV2::new(image.clone()).unwrap();
    let init = encoder.initialize_gaussians(10);
    let init_psnr = compute_psnr(image, &init);

    let mut gaussians = init.clone();
    let mut opt = OptimizerV2::default();
    opt.max_iterations = 100;
    opt.optimize(&mut gaussians, image);

    let final_psnr = compute_psnr(image, &gaussians);
    (init_psnr, final_psnr)
}

fn test_no_geodesic(_image: &ImageBuffer<f32>) -> (f32, f32) {
    // TODO: Need flag to disable geodesic clamping
    (0.0, 0.0)  // Placeholder
}

fn test_no_structure(image: &ImageBuffer<f32>) -> (f32, f32) {
    // Initialize with isotropic Gaussians only (no structure tensor alignment)
    let encoder = EncoderV2::new(image.clone()).unwrap();

    // Create isotropic grid manually
    let grid_size = 10u32;
    let sigma = 0.05;  // Fixed isotropic
    let mut gaussians = Vec::new();

    let step_x = image.width / grid_size;
    let step_y = image.height / grid_size;

    for gy in 0..grid_size {
        for gx in 0..grid_size {
            let x = (gx * step_x + step_x / 2).min(image.width - 1);
            let y = (gy * step_y + step_y / 2).min(image.height - 1);

            let position = lgi_math::vec::Vector2::new(
                x as f32 / image.width as f32,
                y as f32 / image.height as f32,
            );

            let color = image.get_pixel(x, y).unwrap();

            gaussians.push(lgi_math::gaussian::Gaussian2D::new(
                position,
                lgi_math::parameterization::Euler::isotropic(sigma),
                color,
                1.0,
            ));
        }
    }

    let init_psnr = compute_psnr(image, &gaussians);

    // Optimize
    let mut opt = OptimizerV2::default();
    opt.max_iterations = 100;
    opt.optimize(&mut gaussians, image);

    let final_psnr = compute_psnr(image, &gaussians);
    (init_psnr, final_psnr)
}

fn test_no_adaptive_lr(_image: &ImageBuffer<f32>) -> (f32, f32) {
    // TODO: Need flag to disable density_factor
    (0.0, 0.0)
}

fn test_no_guided(image: &ImageBuffer<f32>) -> (f32, f32) {
    // Use initialize_gaussians instead of initialize_gaussians_guided
    let encoder = EncoderV2::new(image.clone()).unwrap();
    let init = encoder.initialize_gaussians(10);  // No guided filter
    let init_psnr = compute_psnr(image, &init);

    let mut gaussians = init.clone();
    let mut opt = OptimizerV2::default();
    opt.max_iterations = 100;
    opt.optimize(&mut gaussians, image);

    let final_psnr = compute_psnr(image, &gaussians);
    (init_psnr, final_psnr)
}

fn test_with_msssim(image: &ImageBuffer<f32>) -> (f32, f32) {
    let encoder = EncoderV2::new(image.clone()).unwrap();
    let init = encoder.initialize_gaussians(10);
    let init_psnr = compute_psnr(image, &init);

    let mut gaussians = init.clone();
    let mut opt = OptimizerV2::default();
    opt.max_iterations = 100;
    opt.use_ms_ssim = true;  // Enable MS-SSIM
    opt.optimize(&mut gaussians, image);

    let final_psnr = compute_psnr(image, &gaussians);
    (init_psnr, final_psnr)
}

fn test_with_edge_weight(image: &ImageBuffer<f32>) -> (f32, f32) {
    let encoder = EncoderV2::new(image.clone()).unwrap();
    let init = encoder.initialize_gaussians(10);
    let init_psnr = compute_psnr(image, &init);

    let mut gaussians = init.clone();
    let mut opt = OptimizerV2::default();
    opt.max_iterations = 100;
    opt.use_edge_weighted = true;  // Enable edge weighting
    opt.optimize(&mut gaussians, image);

    let final_psnr = compute_psnr(image, &gaussians);
    (init_psnr, final_psnr)
}

fn compute_psnr(target: &ImageBuffer<f32>, gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> f32 {
    let rendered = RendererV2::render(gaussians, target.width, target.height);
    let mut mse = 0.0f32;
    let pixel_count = target.width * target.height;

    for y in 0..target.height {
        for x in 0..target.width {
            let t = target.get_pixel(x, y).unwrap();
            let r = rendered.get_pixel(x, y).unwrap();
            mse += (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2);
        }
    }

    mse /= (pixel_count * 3) as f32;
    if mse < 1e-10 { 100.0 } else { -10.0 * mse.log10() }
}

fn load_test_image() -> ImageBuffer<f32> {
    let path = PathBuf::from("../../kodak-dataset/kodim02.png");
    let img = image::open(&path).expect("Failed to load");
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let mut buffer = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);
            buffer.set_pixel(x, y, Color4::new(
                pixel[0] as f32 / 255.0,
                pixel[1] as f32 / 255.0,
                pixel[2] as f32 / 255.0,
                1.0
            ));
        }
    }
    buffer
}
