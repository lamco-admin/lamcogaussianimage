//! K-Means vs Grid Initialization Test
//!
//! Tests if clustering-based placement beats uniform grid

use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};
use lgi_core::ImageBuffer;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler};
use std::path::PathBuf;

fn main() {
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("          K-MEANS vs GRID Initialization Test                  ");
    println!("═══════════════════════════════════════════════════════════════\n");

    let image = load_test_image();
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");

    println!("Test: kodim02.png (768×512)\n");

    let target_n_values = vec![100, 200, 500];

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║ N   │ Grid Init │ Grid Final │ K-means Init │ K-means Final │ Advantage ║");
    println!("╠═══════════════════════════════════════════════════════════════════════════╣");

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

        // K-means
        let kmeans_init = encoder.initialize_kmeans(target_n);
        let kmeans_init_psnr = compute_psnr(&image, &kmeans_init);

        let mut kmeans_opt = kmeans_init.clone();
        let mut opt2 = OptimizerV2::default();
        opt2.max_iterations = 100;
        opt2.optimize(&mut kmeans_opt, &image);
        let kmeans_final_psnr = compute_psnr(&image, &kmeans_opt);

        let advantage = kmeans_final_psnr - grid_final_psnr;

        let status = if advantage > 0.5 { "K-means ✅" }
                     else if advantage > 0.0 { "K-means ⚠️ " }
                     else if advantage > -0.5 { "Tie ⚠️ " }
                     else { "Grid ❌" };

        println!("║ {:3} │   {:6.2}  │   {:7.2}  │    {:8.2}  │     {:9.2}  │  {:+7.2}  {} ║",
            target_n,
            grid_init_psnr,
            grid_final_psnr,
            kmeans_init_psnr,
            kmeans_final_psnr,
            advantage,
            status);
    }

    println!("╚═══════════════════════════════════════════════════════════════════════════╝");

    println!("\n📊 VERDICT:");
    println!("  If K-means wins:");
    println!("    → Clustering-based placement helps");
    println!("    → Add as init strategy option");
    println!("  If Grid wins:");
    println!("    → Uniform coverage still best");
    println!("    → Try SLIC or other strategies");
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
