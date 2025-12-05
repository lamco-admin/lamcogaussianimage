//! Test V2 with scaled learning rates to compensate for magnitude mismatch
//!
//! V2's gradients are ~226× smaller than finite-difference for position.
//! This test scales LRs to compensate.
//!
//! Run: cargo run --release --example test_v2_scaled_lr -p lgi-encoder-v2

use lgi_core::ImageBuffer;
use lgi_encoder_v2::optimizer_v2::OptimizerV2;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;

fn main() {
    println!("=== V2 SCALED LEARNING RATE TEST ===\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let img = image::open(image_path).expect("Failed to load");
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    let mut target = ImageBuffer::new(width, height);
    for (x, y, pixel) in rgb.enumerate_pixels() {
        target.set_pixel(x, y, Color4::new(
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
            1.0,
        ));
    }

    let grid_size = 16;
    let n_gaussians = grid_size * grid_size;
    println!("Image: {}x{}, Gaussians: {}", width, height, n_gaussians);

    // Learning rate scaling factors based on gradient magnitude mismatch
    let lr_configs = vec![
        ("Default (0.1, 0.6, 0.1)", 0.1, 0.6, 0.1),
        ("10× position (1.0, 0.6, 0.1)", 1.0, 0.6, 0.1),
        ("100× position (10.0, 0.6, 0.1)", 10.0, 0.6, 0.1),
        ("200× position (20.0, 0.6, 0.1)", 20.0, 0.6, 0.1),
        ("All scaled (2.0, 1.2, 2.0)", 2.0, 1.2, 2.0),
    ];

    for (name, lr_pos, lr_color, lr_scale) in &lr_configs {
        println!("\n--- {} ---", name);

        // Create fresh Gaussians
        let mut gaussians: Vec<Gaussian2D<f32, Euler<f32>>> = Vec::with_capacity(n_gaussians);
        for gy in 0..grid_size {
            for gx in 0..grid_size {
                let x = (gx as f32 + 0.5) / grid_size as f32;
                let y = (gy as f32 + 0.5) / grid_size as f32;
                let px = ((x * width as f32) as u32).min(width - 1);
                let py = ((y * height as f32) as u32).min(height - 1);
                let color = target.get_pixel(px, py).unwrap();
                let scale = 1.0 / grid_size as f32;
                gaussians.push(Gaussian2D::new(
                    Vector2::new(x, y),
                    Euler::new(scale, scale, 0.0),
                    color,
                    1.0,
                ));
            }
        }

        let initial_render = RendererV2::render(&gaussians, width, height);
        let initial_loss = compute_mse(&initial_render, &target);
        let initial_psnr = mse_to_psnr(initial_loss);

        let mut optimizer = OptimizerV2::default();
        optimizer.max_iterations = 100;
        optimizer.learning_rate_position = *lr_pos;
        optimizer.learning_rate_color = *lr_color;
        optimizer.learning_rate_scale = *lr_scale;

        let start = Instant::now();
        let _ = optimizer.optimize(&mut gaussians, &target);
        let elapsed = start.elapsed();

        let final_render = RendererV2::render(&gaussians, width, height);
        let final_loss = compute_mse(&final_render, &target);
        let final_psnr = mse_to_psnr(final_loss);

        println!("  Initial: {:.2} dB", initial_psnr);
        println!("  Final:   {:.2} dB", final_psnr);
        println!("  Change:  {:+.2} dB", final_psnr - initial_psnr);
        println!("  Time:    {:.1}s", elapsed.as_secs_f32());
    }

    println!("\n=== TEST COMPLETE ===");
}

fn compute_mse(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut sum = 0.0;
    let count = (rendered.width * rendered.height * 3) as f32;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        sum += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    sum / count
}

fn mse_to_psnr(mse: f32) -> f32 {
    if mse <= 0.0 { return 100.0; }
    10.0 * (1.0 / mse).log10()
}
