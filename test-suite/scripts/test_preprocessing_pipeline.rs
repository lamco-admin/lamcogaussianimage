//! End-to-End Preprocessing Pipeline Test
//!
//! Validates complete workflow:
//! 1. Python preprocessing generates placement_map.npy
//! 2. Rust loads preprocessing and samples positions
//! 3. Initialize Gaussians from sampled positions
//! 4. Optimize and measure quality
//!
//! Prerequisite:
//!   source .venv/bin/activate
//!   python tools/preprocess_image_v2.py kodak-dataset/kodim01.png

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};
use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use std::path::PathBuf;

fn main() {
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("    END-TO-END PREPROCESSING PIPELINE TEST                      ");
    println!("═══════════════════════════════════════════════════════════════\n");

    let image_path = "/home/greg/gaussian-image-projects/kodak-dataset/kodim02.png";
    let json_path = "/home/greg/gaussian-image-projects/kodak-dataset/kodim02.json";

    // Check if preprocessing exists
    if !PathBuf::from(json_path).exists() {
        eprintln!("ERROR: Preprocessing not found!");
        eprintln!("Run: source .venv/bin/activate");
        eprintln!("     python tools/preprocess_image_v2.py {}", image_path);
        std::process::exit(1);
    }

    // Load image
    let image = load_image(image_path);
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");

    println!("Test: kodim02.png (768×512)\n");

    // Baseline: Grid N=500
    println!("[1/2] GRID BASELINE (N=500)");
    let grid_init = encoder.initialize_gaussians(23);  // ~529 Gaussians
    let grid_init_psnr = compute_psnr(&image, &grid_init);
    println!("  Init: N={}, PSNR={:.2} dB", grid_init.len(), grid_init_psnr);

    let mut grid_opt = grid_init.clone();
    let mut opt1 = OptimizerV2::default();
    opt1.max_iterations = 100;
    opt1.optimize(&mut grid_opt, &image);

    let grid_final_psnr = compute_psnr(&image, &grid_opt);
    println!("  Final: PSNR={:.2} dB ({:+.2} dB gain)\n", grid_final_psnr, grid_final_psnr - grid_init_psnr);

    // Preprocessing-guided
    println!("[2/2] PREPROCESSING-GUIDED (placement_map)");
    let prep_init = encoder.initialize_from_preprocessing(json_path, 500);
    let prep_init_psnr = compute_psnr(&image, &prep_init);
    println!("  Init: N={}, PSNR={:.2} dB", prep_init.len(), prep_init_psnr);

    let mut prep_opt = prep_init.clone();
    let mut opt2 = OptimizerV2::default();
    opt2.max_iterations = 100;
    opt2.optimize(&mut prep_opt, &image);

    let prep_final_psnr = compute_psnr(&image, &prep_opt);
    println!("  Final: PSNR={:.2} dB ({:+.2} dB gain)\n", prep_final_psnr, prep_final_psnr - prep_init_psnr);

    // Summary
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║ Method          │ N   │ Init PSNR │ Final PSNR │ Advantage  ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ Grid            │ {:3} │   {:6.2}  │   {:7.2}  │     —      ║",
        grid_init.len(), grid_init_psnr, grid_final_psnr);
    println!("║ Preprocessing   │ {:3} │   {:6.2}  │   {:7.2}  │  {:+7.2}  ║",
        prep_init.len(), prep_init_psnr, prep_final_psnr, prep_final_psnr - grid_final_psnr);
    println!("╚═══════════════════════════════════════════════════════════════╝");

    let advantage = prep_final_psnr - grid_final_psnr;

    println!("\n📊 RESULT: Preprocessing {:+.2} dB vs Grid", advantage);

    if advantage > 1.0 {
        println!("   ✅ PREPROCESSING WINS - Intelligent placement helps!");
        println!("      → Use preprocessing pipeline for production encoding");
    } else if advantage > 0.0 {
        println!("   ⚠️  PREPROCESSING slightly better");
        println!("      → Marginal benefit, grid adequate for simple use");
    } else {
        println!("   ❌ GRID STILL BETTER");
        println!("      → Preprocessing needs tuning or grid is already optimal");
    }
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

fn load_image(path: &str) -> ImageBuffer<f32> {
    let img = image::open(path).expect("Failed to load image");
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
