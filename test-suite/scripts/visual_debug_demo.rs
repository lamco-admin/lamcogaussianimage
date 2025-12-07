//! Visual Debug Demo
//!
//! Shows iteration-by-iteration:
//! - Rendered images
//! - Error heatmaps
//! - Gaussian positions
//! - Metrics CSV
//!
//! Output: debug_output/ folder with PNG sequence + metrics.csv

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2, debug_logger::{DebugLogger, DebugConfig}};
use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use std::path::PathBuf;
use std::time::Instant;

fn main() -> std::io::Result<()> {
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("           VISUAL DEBUG DEMO - See What's Happening            ");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Load test image
    let image = load_test_image();
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");

    // Initialize Gaussians
    let mut gaussians = encoder.initialize_gaussians(10);  // N=100

    println!("Test: kodim02.png (768×512)");
    println!("N = {}", gaussians.len());
    println!("Output: debug_output/\n");

    // Create debug logger
    let config = DebugConfig {
        output_dir: PathBuf::from("debug_output"),
        save_every_n_iters: 5,  // Save every 5 iterations
        save_rendered: true,
        save_error_maps: true,
        save_gaussian_viz: true,
        save_comparison: true,
        save_metrics_csv: true,
    };

    let mut debug_logger = DebugLogger::new(config)?;

    // Optimize with visual logging
    let mut opt = OptimizerV2::default();
    opt.max_iterations = 1;  // We'll manually loop to intercept each iteration

    println!("Running 50 iterations with visual logging...\n");

    for iter in 0..50 {
        let iter_start = Instant::now();

        // Render and compute metrics
        let rendered = RendererV2::render(&gaussians, image.width, image.height);
        let loss = compute_loss(&image, &rendered);
        let psnr = compute_psnr(&image, &gaussians);

        // Log to debug output
        debug_logger.log_iteration(
            iter,
            0,  // pass number
            &gaussians,
            &image,
            loss,
            psnr,
            iter_start.elapsed().as_millis() as u64,
        )?;

        // Progress
        if iter % 10 == 0 {
            println!("Iteration {}: PSNR = {:.2} dB, loss = {:.6}", iter, psnr, loss);
        }

        // Optimize one step
        opt.optimize(&mut gaussians, &image);
    }

    println!("\n✅ Complete!");
    println!("   Output directory: debug_output/");
    println!("   - iter_XXXX_rendered.png (rendered images)");
    println!("   - iter_XXXX_error.png (error heatmaps)");
    println!("   - iter_XXXX_gaussians.png (Gaussian positions)");
    println!("   - iter_XXXX_comparison.png (side-by-side grids)");
    println!("   - metrics.csv (iteration, N, loss, PSNR, time)");
    println!("\n   Use: ffmpeg -framerate 5 -i debug_output/iter_%04d_rendered.png output.mp4");
    println!("   To create animation of optimization process!");

    Ok(())
}

fn compute_loss(target: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0f32;
    let pixel_count = target.width * target.height;

    for (t, r) in target.data.iter().zip(rendered.data.iter()) {
        mse += (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2);
    }

    mse / (pixel_count * 3) as f32
}

fn compute_psnr(target: &ImageBuffer<f32>, gaussians: &[lgi_math::gaussian::Gaussian2D<f32, lgi_math::parameterization::Euler<f32>>]) -> f32 {
    let rendered = RendererV2::render(gaussians, target.width, target.height);
    let mse = compute_loss(target, &rendered);
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
