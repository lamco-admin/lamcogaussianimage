//! EXP-036: Extreme N Testing on Photos
//! Test N=2500, 4000, 6400 to find quality saturation point

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  EXP-036: Extreme N Testing on Photos                   ║");
    println!("║  Find quality saturation point                           ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let path = "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/133784383569199567.jpg";

    println!("Loading large photo...");
    let target = match ImageBuffer::load(path) {
        Ok(img) => {
            println!("Loaded: {}×{}", img.width, img.height);

            // Resize to 512×512 for extreme N testing
            let scale = 512.0 / img.width.max(img.height) as f32;
            let new_w = (img.width as f32 * scale) as u32;
            let new_h = (img.height as f32 * scale) as u32;
            println!("Resizing to {}×{} for testing\n", new_w, new_h);
            resize_bilinear(&img, new_w, new_h)
        }
        Err(e) => {
            println!("Failed: {}", e);
            return;
        }
    };

    // Test extreme N values
    let configs = vec![
        (50, "N=2500 (50×50)"),
        (64, "N=4096 (64×64)"),
        (80, "N=6400 (80×80)"),
    ];

    println!("Testing extreme Gaussian counts:\n");

    for (grid_size, desc) in configs {
        let n = grid_size * grid_size;

        println!("{}:", desc);

        let start_config = Instant::now();

        let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
        let mut gaussians = encoder.initialize_gaussians_guided(grid_size);

        let init_rendered = RendererV2::render(&gaussians, target.width, target.height);
        let init_psnr = compute_psnr(&target, &init_rendered);

        println!("  Init PSNR: {:.2} dB", init_psnr);
        println!("  Optimizing (max 300 iterations)...");

        let mut optimizer = OptimizerV2::default();
        optimizer.max_iterations = 300;
        let final_loss = optimizer.optimize(&mut gaussians, &target);

        let final_rendered = RendererV2::render(&gaussians, target.width, target.height);
        let final_psnr = compute_psnr(&target, &final_rendered);

        let time = start_config.elapsed().as_secs_f32();

        println!("  Final PSNR: {:.2} dB (Δ: {:+.2} dB)", final_psnr, final_psnr - init_psnr);
        println!("  Loss: {:.6}", final_loss);
        println!("  Time: {:.1}s", time);
        println!("  PSNR/N ratio: {:.6}\n", final_psnr / n as f32);
    }

    println!("═══════════════════════════════════════════════");
    println!("Analysis:");
    println!("  - If PSNR increases: Higher N helps");
    println!("  - If PSNR plateaus: Found saturation point");
    println!("  - Compare PSNR/N ratio for efficiency");
    println!("═══════════════════════════════════════════════");
}

fn resize_bilinear(img: &ImageBuffer<f32>, new_width: u32, new_height: u32) -> ImageBuffer<f32> {
    let mut resized = ImageBuffer::new(new_width, new_height);

    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = x as f32 * (img.width as f32 / new_width as f32);
            let src_y = y as f32 * (img.height as f32 / new_height as f32);

            let x0 = src_x.floor() as u32;
            let y0 = src_y.floor() as u32;
            let x1 = (x0 + 1).min(img.width - 1);
            let y1 = (y0 + 1).min(img.height - 1);

            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;

            if let (Some(c00), Some(c10), Some(c01), Some(c11)) = (
                img.get_pixel(x0, y0),
                img.get_pixel(x1, y0),
                img.get_pixel(x0, y1),
                img.get_pixel(x1, y1),
            ) {
                let r = (1.0 - fx) * (1.0 - fy) * c00.r +
                        fx * (1.0 - fy) * c10.r +
                        (1.0 - fx) * fy * c01.r +
                        fx * fy * c11.r;

                let g = (1.0 - fx) * (1.0 - fy) * c00.g +
                        fx * (1.0 - fy) * c10.g +
                        (1.0 - fx) * fy * c01.g +
                        fx * fy * c11.g;

                let b = (1.0 - fx) * (1.0 - fy) * c00.b +
                        fx * (1.0 - fy) * c10.b +
                        (1.0 - fx) * fy * c01.b +
                        fx * fy * c11.b;

                resized.set_pixel(x, y, lgi_math::color::Color4::new(r, g, b, 1.0));
            }
        }
    }

    resized
}

fn compute_psnr(original: &ImageBuffer<f32>, rendered: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    let count = (original.width * original.height * 3) as f32;

    for (p1, p2) in original.data.iter().zip(rendered.data.iter()) {
        mse += (p1.r - p2.r).powi(2);
        mse += (p1.g - p2.g).powi(2);
        mse += (p1.b - p2.b).powi(2);
    }

    mse /= count;
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}
