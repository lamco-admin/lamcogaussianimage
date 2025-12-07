//! Proper large photo testing - NO SHORTCUTS
//! Let processing run as long as needed

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  Large Photo Testing - Proper Long Test     ║");
    println!("║  NO TIMEOUTS - Let it run!                   ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // Test actual large photos
    let photos = vec![
        "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/133784383569199567.jpg",  // 1.2MB
        "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/133683084337742188.jpg",  // 2.1MB
    ];

    for (idx, photo_path) in photos.iter().enumerate() {
        println!("\n═══════════════════════════════════════════════════════════");
        println!("Photo {}: {}", idx + 1, photo_path.split('/').last().unwrap_or(photo_path));
        println!("═══════════════════════════════════════════════════════════");

        let start_load = Instant::now();
        let target = match ImageBuffer::load(photo_path) {
            Ok(img) => {
                println!("✓ Loaded: {}×{} ({:.1}s)", img.width, img.height, start_load.elapsed().as_secs_f32());
                img
            }
            Err(e) => {
                println!("✗ Failed to load: {}", e);
                continue;
            }
        };

        // Resize to manageable size but keep quality
        let target = if target.width > 768 || target.height > 768 {
            let scale = 768.0 / target.width.max(target.height) as f32;
            let new_w = (target.width as f32 * scale) as u32;
            let new_h = (target.height as f32 * scale) as u32;
            println!("Resizing to {}×{} for testing...", new_w, new_h);
            resize_bilinear(&target, new_w, new_h)
        } else {
            target
        };

        println!("Testing size: {}×{}\n", target.width, target.height);

        // Test with multiple configurations
        let configs = vec![
            (24, 300, "Low N, moderate iters"),
            (32, 300, "Medium N, moderate iters"),
            (40, 400, "High N, many iters"),
        ];

        for (grid_size, max_iters, desc) in configs {
            let n = grid_size * grid_size;
            println!("Config: {} (N={}, {} iterations)", desc, n, max_iters);

            let start_encode = Instant::now();

            let encoder = EncoderV2::new(target.clone()).expect("Encoder failed");
            let mut gaussians = encoder.initialize_gaussians_guided(grid_size);

            let init_rendered = RendererV2::render(&gaussians, target.width, target.height);
            let init_psnr = compute_psnr(&target, &init_rendered);

            println!("  Init PSNR: {:.2} dB", init_psnr);
            println!("  Optimizing (this may take several minutes)...");

            let mut optimizer = OptimizerV2::default();
            optimizer.max_iterations = max_iters;
            let final_loss = optimizer.optimize(&mut gaussians, &target);

            let final_rendered = RendererV2::render(&gaussians, target.width, target.height);
            let final_psnr = compute_psnr(&target, &final_rendered);

            let encode_time = start_encode.elapsed().as_secs_f32();

            println!("  Final PSNR: {:.2} dB (Δ: {:+.2} dB)", final_psnr, final_psnr - init_psnr);
            println!("  Loss: {:.6}", final_loss);
            println!("  Time: {:.1}s", encode_time);

            // Assessment
            if final_psnr >= 28.0 {
                println!("  ✅ GOOD - Photo quality acceptable");
            } else if final_psnr >= 22.0 {
                println!("  ⚠️  MARGINAL - Photo quality below target");
            } else {
                println!("  ❌ POOR - Need additional features");
            }

            // Save output for visual inspection
            let output_path = format!("/tmp/photo_{}_{}_n{}.png", idx + 1, desc.replace(" ", "_").replace(",", ""), n);
            if let Err(e) = final_rendered.save(&output_path) {
                println!("  Warning: Could not save output: {}", e);
            } else {
                println!("  Saved: {}", output_path);
            }

            println!();
        }
    }

    println!("═══════════════════════════════════════════════════════════");
    println!("Testing complete. Visual outputs in /tmp/photo_*.png");
    println!("Compare against targets to identify needed improvements.");
    println!("═══════════════════════════════════════════════════════════");
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
