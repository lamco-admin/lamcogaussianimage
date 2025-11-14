//! EXP-028: Real Image Testing
//! CRITICAL: Validate foundation on real photos

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2};

fn main() {
    println!("╔══════════════════════════════════════════════╗");
    println!("║  EXP-028: Real Image Testing - CRITICAL     ║");
    println!("╚══════════════════════════════════════════════╝\n");

    // Test images in order of size (small first to avoid timeout)
    let test_images = vec![
        "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/Lamco Head.jpg",
        "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/Robot Lamb.jpeg",
        "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/arulj.jpg",
    ];

    for image_path in test_images {
        test_image(image_path);
    }
}

fn test_image(path: &str) {
    println!("\n═══════════════════════════════════════════════");
    println!("Testing: {}", path.split('/').last().unwrap_or(path));
    println!("═══════════════════════════════════════════════");

    // Load and potentially resize
    let target = match ImageBuffer::load(path) {
        Ok(mut img) => {
            println!("  Loaded: {}×{}", img.width, img.height);

            // Resize if too large (for speed)
            if img.width > 512 || img.height > 512 {
                println!("  Resizing to max 512×512...");
                let scale = 512.0 / img.width.max(img.height) as f32;
                let new_width = (img.width as f32 * scale) as u32;
                let new_height = (img.height as f32 * scale) as u32;

                img = resize_image(&img, new_width, new_height);
                println!("  Resized to: {}×{}", img.width, img.height);
            }

            img
        }
        Err(e) => {
            println!("  ❌ Failed to load: {}", e);
            return;
        }
    };

    // Test with multiple N values
    for &grid_size in &[16, 20, 24, 32] {
        let num_gaussians = grid_size * grid_size;

        println!("\n  N={} ({}×{} grid):", num_gaussians, grid_size, grid_size);

        let encoder = match EncoderV2::new(target.clone()) {
            Ok(enc) => enc,
            Err(e) => {
                println!("    ❌ Encoder failed: {}", e);
                continue;
            }
        };

        let mut gaussians = encoder.initialize_gaussians(grid_size);

        let init_rendered = RendererV2::render(&gaussians, target.width, target.height);
        let init_psnr = compute_psnr(&target, &init_rendered);

        let mut optimizer = OptimizerV2::default();
        optimizer.max_iterations = 100;
        let final_loss = optimizer.optimize(&mut gaussians, &target);

        let final_rendered = RendererV2::render(&gaussians, target.width, target.height);
        let final_psnr = compute_psnr(&target, &final_rendered);

        println!("    Init: {:.2} dB → Final: {:.2} dB (Δ: {:+.2} dB, loss: {:.6})",
            init_psnr, final_psnr, final_psnr - init_psnr, final_loss);

        // Assessment
        if final_psnr >= 30.0 {
            println!("    ✅ Excellent quality");
        } else if final_psnr >= 25.0 {
            println!("    ✓ Good quality");
        } else if final_psnr >= 20.0 {
            println!("    ⚠️  Acceptable quality");
        } else {
            println!("    ❌ Poor quality - needs improvement");
        }
    }
}

fn resize_image(img: &ImageBuffer<f32>, new_width: u32, new_height: u32) -> ImageBuffer<f32> {
    let mut resized = ImageBuffer::new(new_width, new_height);

    for y in 0..new_height {
        for x in 0..new_width {
            // Simple bilinear interpolation
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
