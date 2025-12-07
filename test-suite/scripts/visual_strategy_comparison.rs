//! Visual Strategy Comparison
//!
//! Generates debug output for multiple initialization strategies
//! Creates PNG sequences to SEE what each strategy does iteration-by-iteration
//!
//! Output: debug_output/{strategy_name}/ folders with frame sequences

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2, debug_logger::{DebugLogger, DebugConfig}};
use lgi_core::ImageBuffer;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> std::io::Result<()> {
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("        VISUAL STRATEGY COMPARISON - See What Happens           ");
    println!("═══════════════════════════════════════════════════════════════\n");

    let image = load_simple_kodak();
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");

    println!("Test Image: kodim01.png (simple scene)");
    println!("Output: debug_output/<strategy>/ folders\n");

    // Test multiple strategies visually
    let strategies = vec![
        ("grid_n100", init_grid(&encoder, 100)),
        ("grid_n500", init_grid(&encoder, 500)),
        ("entropy_auto", init_entropy(&encoder)),
        // K-means and others when ready
    ];

    for (name, gaussians) in strategies {
        println!("\n╔═══════════════════════════════════════════════════════════╗");
        println!("║ Strategy: {:47} ║", name);
        println!("╚═══════════════════════════════════════════════════════════╝");

        run_visual_optimization(name, gaussians, &image)?;
    }

    println!("\n✅ Complete! Check debug_output/ for PNG sequences");
    println!("   Create videos:");
    println!("   ffmpeg -framerate 5 -i debug_output/grid_n100/iter_%04d_rendered.png grid_n100.mp4");
    println!("   ffmpeg -framerate 5 -i debug_output/grid_n500/iter_%04d_rendered.png grid_n500.mp4");

    Ok(())
}

fn run_visual_optimization(
    strategy_name: &str,
    mut gaussians: Vec<Gaussian2D<f32, Euler<f32>>>,
    image: &ImageBuffer<f32>,
) -> std::io::Result<()> {
    let init_psnr = compute_psnr(image, &gaussians);
    println!("  Initial: N={}, PSNR={:.2} dB", gaussians.len(), init_psnr);

    // Create debug logger for this strategy
    let config = DebugConfig {
        output_dir: PathBuf::from(format!("debug_output/{}", strategy_name)),
        save_every_n_iters: 10,  // Every 10 iterations
        save_rendered: true,
        save_error_maps: true,
        save_gaussian_viz: true,
        save_comparison: true,
        save_metrics_csv: true,
    };

    let mut debug = DebugLogger::new(config)?;

    // Optimize with visual logging
    let mut opt = OptimizerV2::default();
    opt.max_iterations = 1;  // Manual loop

    println!("  Optimizing 100 iterations with visual logging...");

    for iter in 0..100 {
        let iter_start = Instant::now();

        // Compute metrics
        let rendered = RendererV2::render(&gaussians, image.width, image.height);
        let loss = compute_loss(image, &rendered);
        let psnr = compute_psnr(image, &gaussians);

        // Log to debug output
        debug.log_iteration(
            iter,
            0,
            &gaussians,
            image,
            loss,
            psnr,
            iter_start.elapsed().as_millis() as u64,
        )?;

        if iter % 10 == 0 {
            println!("    Iter {}: PSNR={:.2} dB, loss={:.6}", iter, psnr, loss);
        }

        // Optimize one step
        opt.optimize(&mut gaussians, image);
    }

    let final_psnr = compute_psnr(image, &gaussians);
    println!("  Final: PSNR={:.2} dB ({:+.2} dB gain)", final_psnr, final_psnr - init_psnr);

    Ok(())
}

fn init_grid(encoder: &EncoderV2, n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let grid_size = (n as f32).sqrt().ceil() as u32;
    encoder.initialize_gaussians(grid_size)
}

fn init_entropy(encoder: &EncoderV2) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let n = encoder.auto_gaussian_count();
    let grid_size = (n as f32).sqrt().ceil() as u32;
    encoder.initialize_gaussians(grid_size)
}

fn compute_loss(target: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0f32;
    let pixel_count = target.width * target.height;

    for (t, r) in target.data.iter().zip(rendered.data.iter()) {
        mse += (t.r - r.r).powi(2) + (t.g - r.g).powi(2) + (t.b - r.b).powi(2);
    }

    mse / (pixel_count * 3) as f32
}

fn compute_psnr(target: &ImageBuffer<f32>, gaussians: &[Gaussian2D<f32, Euler<f32>>]) -> f32 {
    let rendered = RendererV2::render(gaussians, target.width, target.height);
    let mse = compute_loss(target, &rendered);
    if mse < 1e-10 { 100.0 } else { -10.0 * mse.log10() }
}

fn load_simple_kodak() -> ImageBuffer<f32> {
    // kodim01 is a simple scene (house, sky) - good for visualization
    let path = PathBuf::from("/home/greg/gaussian-image-projects/kodak-dataset/kodim01.png");
    let img = image::open(&path).expect("Failed to load kodim01.png");
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
