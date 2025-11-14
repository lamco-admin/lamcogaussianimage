//! Test Gradient Peak Initialization
//!
//! Hybrid: Gradient peaks (80%) + Background grid (20%)
//! vs pure Grid
//!
//! Should perform better than GDGS (doesn't neglect smooth regions)

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};
use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use std::path::PathBuf;

fn main() {
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("      GRADIENT PEAK (Hybrid) vs GRID Initialization Test       ");
    println!("═══════════════════════════════════════════════════════════════\n");

    let image = load_test_image();
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");

    println!("Test: kodim02.png (768×512)\n");

    let target_n_values = vec![100, 200, 500];

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║ N   │ Grid Init │ Grid Final │ Peak Init │ Peak Final │ Advantage ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");

    for target_n in target_n_values {
        // Grid
        let grid_size = (target_n as f32).sqrt().ceil() as u32;
        let grid_init = encoder.initialize_gaussians(grid_size);
        let grid_init_psnr = compute_psnr(&image, &grid_init);

        let mut grid_opt = grid_init.clone();
        let mut opt1 = OptimizerV2::default();
        opt1.max_iterations = 100;
        opt1.optimize(&mut grid_opt, &image);
        let grid_final_psnr = compute_psnr(&image, &grid_opt);

        // Gradient Peaks (hybrid)
        let peak_init = encoder.initialize_gradient_peaks(target_n);
        let peak_init_psnr = compute_psnr(&image, &peak_init);

        let mut peak_opt = peak_init.clone();
        let mut opt2 = OptimizerV2::default();
        opt2.max_iterations = 100;
        opt2.optimize(&mut peak_opt, &image);
        let peak_final_psnr = compute_psnr(&image, &peak_opt);

        let advantage = peak_final_psnr - grid_final_psnr;

        println!("║ {:3} │   {:6.2}  │   {:7.2}  │   {:6.2}  │   {:7.2}  │  {:+7.2}  ║",
            target_n,
            grid_init_psnr,
            grid_final_psnr,
            peak_init_psnr,
            peak_final_psnr,
            advantage);
    }

    println!("╚══════════════════════════════════════════════════════════════════════╝");

    println!("\n📊 VERDICT:");
    println!("  If advantage > 0.5 dB:");
    println!("    → Gradient peaks WORK - use as alternative init strategy");
    println!("  If advantage close to 0:");
    println!("    → Grid and peaks equivalent - offer both");
    println!("  If negative:");
    println!("    → Grid still better - gradient peaks don't help");
}

fn compute_psnr(target: &ImageBuffer<f32>, gaussians: &[lgi_math::gaussian::Gaussian2D<f32, lgi_math::parameterization::Euler<f32>>]) -> f32 {
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
