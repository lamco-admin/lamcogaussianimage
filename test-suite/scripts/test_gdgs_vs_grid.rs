//! GDGS vs Grid Initialization Comparison
//!
//! Tests if Gradient Domain (Laplacian peaks) gives better quality with fewer Gaussians
//!
//! Expected from research: 10-100× fewer Gaussians for same quality

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2, gdgs_init};
use lgi_core::ImageBuffer;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler};
use std::path::PathBuf;

fn main() {
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("         GDGS vs GRID INITIALIZATION - Efficiency Test         ");
    println!("═══════════════════════════════════════════════════════════════\n");

    let image = load_test_image();
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");

    println!("Test: kodim02.png (768×512)\n");

    // Test at multiple N values
    let target_n_values = vec![50, 100, 200, 500];

    println!("╔═══════════════════════════════════════════════════════════════════════════════╗");
    println!("║ Target N │ Grid N │ Grid PSNR │ GDGS N │ GDGS PSNR │ Efficiency │ Winner    ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════════╣");

    for target_n in target_n_values {
        // Grid initialization
        let grid_size = (target_n as f32).sqrt().ceil() as u32;
        let grid_gaussians = encoder.initialize_gaussians(grid_size);
        let grid_n = grid_gaussians.len();

        // Optimize grid
        let mut grid_opt = grid_gaussians.clone();
        let mut opt = OptimizerV2::default();
        opt.max_iterations = 100;
        opt.optimize(&mut grid_opt, &image);

        let grid_psnr = compute_psnr(&image, &grid_opt);

        // GDGS initialization (target same N)
        let gdgs_gaussians = encoder.initialize_gdgs(target_n);
        let gdgs_n = gdgs_gaussians.len();

        // Optimize GDGS
        let mut gdgs_opt = gdgs_gaussians.clone();
        let mut opt2 = OptimizerV2::default();
        opt2.max_iterations = 100;
        opt2.optimize(&mut gdgs_opt, &image);

        let gdgs_psnr = compute_psnr(&image, &gdgs_opt);

        // Efficiency: How much better is GDGS per Gaussian?
        let efficiency = (gdgs_psnr - grid_psnr) / (gdgs_n as f32 / grid_n as f32);

        let winner = if gdgs_psnr > grid_psnr + 0.5 {
            "GDGS ✅"
        } else if gdgs_psnr > grid_psnr {
            "GDGS ⚠️ "
        } else if grid_psnr > gdgs_psnr + 0.5 {
            "Grid ❌"
        } else {
            "Tie  ⚠️ "
        };

        println!("║ {:8} │ {:6} │  {:7.2}  │ {:6} │  {:7.2}  │   {:+6.2}   │ {} ║",
            target_n, grid_n, grid_psnr, gdgs_n, gdgs_psnr, efficiency, winner);
    }

    println!("╚═══════════════════════════════════════════════════════════════════════════════╝");

    println!("\n📊 ANALYSIS:");
    println!("  If GDGS wins consistently:");
    println!("    → Sparse placement at edges is better");
    println!("    → Use GDGS as default initialization");
    println!("  If Grid wins:");
    println!("    → Uniform coverage matters more than edge focus");
    println!("    → Stick with grid init");
    println!("  If tie:");
    println!("    → No clear winner, offer both as strategies");
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
    let path = PathBuf::from("/home/greg/gaussian-image-projects/kodak-dataset/kodim02.png");
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
