//! SLIC vs Grid Initialization Test
//!
//! Prerequisite: Run SLIC preprocessing first:
//! source .venv/bin/activate
//! python tools/slic_preprocess.py kodak-dataset/kodim02.png 500 /tmp/slic_kodim02.json

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};
use lgi_core::ImageBuffer;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler};
use std::path::PathBuf;

fn main() {
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("          SLIC vs GRID Initialization Test                     ");
    println!("═══════════════════════════════════════════════════════════════\n");

    let image = load_test_image();
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");

    println!("Test: kodim02.png (768×512)\n");

    // Grid N=500
    let grid_init = encoder.initialize_gaussians(23);  // 23×23 ≈ 529
    let grid_init_psnr = compute_psnr(&image, &grid_init);
    println!("[1/2] GRID (N={}):", grid_init.len());
    println!("  Init PSNR: {:.2} dB", grid_init_psnr);

    let mut grid_opt = grid_init.clone();
    let mut opt1 = OptimizerV2::default();
    opt1.max_iterations = 100;
    opt1.optimize(&mut grid_opt, &image);
    let grid_final_psnr = compute_psnr(&image, &grid_opt);
    println!("  Final PSNR: {:.2} dB ({:+.2} dB gain)\n", grid_final_psnr, grid_final_psnr - grid_init_psnr);

    // SLIC
    let slic_init = encoder.initialize_slic("/tmp/slic_kodim02.json");
    let slic_init_psnr = compute_psnr(&image, &slic_init);
    println!("[2/2] SLIC (N={}):", slic_init.len());
    println!("  Init PSNR: {:.2} dB", slic_init_psnr);

    let mut slic_opt = slic_init.clone();
    let mut opt2 = OptimizerV2::default();
    opt2.max_iterations = 100;
    opt2.optimize(&mut slic_opt, &image);
    let slic_final_psnr = compute_psnr(&image, &slic_opt);
    println!("  Final PSNR: {:.2} dB ({:+.2} dB gain)\n", slic_final_psnr, slic_final_psnr - slic_init_psnr);

    // Summary
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║ Strategy  │ N   │ Init PSNR │ Final PSNR │ Gain  │ Winner ║");
    println!("╠════════════════════════════════════════════════════════════╣");
    println!("║ Grid      │ {:3} │   {:6.2}  │   {:7.2}  │ {:+.2} │        ║",
        grid_init.len(), grid_init_psnr, grid_final_psnr, grid_final_psnr - grid_init_psnr);
    println!("║ SLIC      │ {:3} │   {:6.2}  │   {:7.2}  │ {:+.2} │        ║",
        slic_init.len(), slic_init_psnr, slic_final_psnr, slic_final_psnr - slic_init_psnr);
    println!("╚════════════════════════════════════════════════════════════╝");

    let advantage = slic_final_psnr - grid_final_psnr;
    println!("\n📊 RESULT: SLIC {:+.2} dB vs Grid", advantage);

    if advantage > 1.0 {
        println!("   ✅ SLIC WINS - Boundary-aware segmentation helps!");
    } else if advantage > 0.0 {
        println!("   ⚠️  SLIC slightly better - marginal benefit");
    } else if advantage > -1.0 {
        println!("   ⚠️  Grid slightly better - SLIC marginal loss");
    } else {
        println!("   ❌ GRID WINS - Uniform coverage still best");
    }
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
