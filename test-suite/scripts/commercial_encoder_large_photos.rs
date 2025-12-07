//! Commercial Encoder on Large Photos
//! Full test with N=1024, all features, proper long optimization

use lgi_core::ImageBuffer;
use lgi_encoder_v2::commercial_encoder::{CommercialEncoder, CommercialConfig};
use lgi_encoder_v2::renderer_v3_textured::RendererV3;
use std::time::Instant;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Commercial Encoder - Large Photo Production Test       ║");
    println!("║  Full feature set, no shortcuts                          ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let photos = vec![
        "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/133784383569199567.jpg",
    ];

    for (idx, path) in photos.iter().enumerate() {
        test_photo(idx, path);
    }
}

fn test_photo(idx: usize, path: &str) {
    println!("═══════════════════════════════════════════════════════════");
    println!("Photo {}: {}", idx + 1, path.split('/').last().unwrap_or(path));
    println!("═══════════════════════════════════════════════════════════\n");

    let start = Instant::now();

    let target = match ImageBuffer::load(path) {
        Ok(img) => {
            println!("Loaded: {}×{}", img.width, img.height);

            // Resize to 768 max dimension (keep quality reasonable)
            if img.width > 768 || img.height > 768 {
                let scale = 768.0 / img.width.max(img.height) as f32;
                let new_w = (img.width as f32 * scale) as u32;
                let new_h = (img.height as f32 * scale) as u32;
                println!("Resizing to {}×{}...", new_w, new_h);
                resize_bilinear(&img, new_w, new_h)
            } else {
                img
            }
        }
        Err(e) => {
            println!("Failed to load: {}", e);
            return;
        }
    };

    println!("Testing size: {}×{}\n", target.width, target.height);

    // Full commercial configuration
    let config = CommercialConfig {
        target_gaussian_count: 1600,  // High quality
        quality_target: 28.0,
        max_iterations: 500,           // Long optimization
        use_textures: true,
        texture_size: 16,              // Larger textures
        texture_threshold: 0.005,      // Sensitive
        use_residuals: true,
        residual_threshold: 0.03,
        use_guided_filter: true,
        use_triggers: true,
    };

    println!("Config: N={}, {} iterations, all features enabled",
        config.target_gaussian_count, config.max_iterations);

    let encoder = match CommercialEncoder::new(target.clone(), config) {
        Ok(enc) => enc,
        Err(e) => {
            println!("Encoder creation failed: {}", e);
            return;
        }
    };

    println!("\nEncoding (this will take several minutes)...\n");

    let (gaussians, residual) = encoder.encode();

    // Render final
    let mut final_rendered = RendererV3::render(&gaussians, target.width, target.height);

    if let Some(ref res) = residual {
        res.apply_to_image(&mut final_rendered);
    }

    let psnr = compute_psnr(&target, &final_rendered);
    let encode_time = start.elapsed().as_secs_f32();

    println!("\n═══════════════════════════════════════════════");
    println!("RESULTS");
    println!("═══════════════════════════════════════════════");
    println!("  Gaussians: {}", gaussians.len());
    println!("  Residual: {}", if residual.is_some() { "Yes" } else { "No" });
    println!("  Final PSNR: {:.2} dB", psnr);
    println!("  Encoding time: {:.1}s", encode_time);

    if psnr >= 28.0 {
        println!("  ✅ SUCCESS: Production quality achieved!");
    } else if psnr >= 25.0 {
        println!("  ✓ GOOD: Close to production quality");
    } else if psnr >= 22.0 {
        println!("  ⚠️  MARGINAL: Needs more tuning");
    } else {
        println!("  ❌ POOR: Significant work needed");
    }

    // Save output
    let output_path = format!("/tmp/commercial_photo_{}.png", idx + 1);
    if let Err(e) = final_rendered.save(&output_path) {
        println!("\nWarning: Could not save: {}", e);
    } else {
        println!("\nSaved: {}", output_path);
    }
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
