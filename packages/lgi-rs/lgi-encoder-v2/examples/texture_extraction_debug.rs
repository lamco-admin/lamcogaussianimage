//! Texture Extraction Debug - EXP-3-005
//! Validate texture extraction and visualize extracted textures
//! Diagnose why textures break quality (-7 dB instead of +3-5 dB)

use lgi_core::{ImageBuffer, texture_map::TextureMap};
use lgi_math::{color::Color4, vec::Vector2};

fn main() {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Texture Extraction Debug - EXP-3-005                    ║");
    println!("║  Validate extraction algorithm and visualize textures    ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Load test photo
    let path = "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/133784383569199567.jpg";

    let image = match ImageBuffer::load(path) {
        Ok(img) => {
            let scale = 768.0 / img.width.max(img.height) as f32;
            let new_w = (img.width as f32 * scale) as u32;
            let new_h = (img.height as f32 * scale) as u32;
            resize_bilinear(&img, new_w, new_h)
        }
        Err(e) => {
            println!("Failed to load: {}", e);
            return;
        }
    };

    println!("Image: {}×{}\n", image.width, image.height);

    // Test texture extraction at various locations
    let test_points = vec![
        (0.25, 0.25, "Sky/smooth region"),
        (0.5, 0.5, "Center/detail"),
        (0.75, 0.75, "Textured region"),
        (0.1, 0.9, "Corner"),
    ];

    let texture_sizes = vec![8, 16];

    for &texture_size in &texture_sizes {
        println!("═══════════════════════════════════════════");
        println!("Texture Size: {}×{}", texture_size, texture_size);
        println!("═══════════════════════════════════════════");

        for &(px, py, label) in &test_points {
            println!("\n  Location: ({:.2}, {:.2}) - {}", px, py, label);

            // Extract texture (assuming Gaussian scale ~0.01)
            let texture = TextureMap::extract_from_image(
                &image,
                Vector2::new(px, py),
                0.01,  // scale_x
                0.01,  // scale_y
                0.0,   // rotation
                texture_size,
            );

            // Compute texture statistics
            let mut min_r: f32 = 1.0;
            let mut max_r: f32 = 0.0;
            let mut mean_r: f32 = 0.0;
            let mut variance_sum = 0.0;

            for color in &texture.data {
                min_r = min_r.min(color.r);
                max_r = max_r.max(color.r);
                mean_r += color.r;
            }
            mean_r /= texture.data.len() as f32;

            for color in &texture.data {
                variance_sum += (color.r - mean_r).powi(2);
            }
            let variance = variance_sum / texture.data.len() as f32;

            println!("    R: min={:.3}, max={:.3}, mean={:.3}, var={:.6}", min_r, max_r, mean_r, variance);

            // Check if texture has actual variation
            if variance < 0.001 {
                println!("    ⚠️  LOW VARIANCE - nearly uniform!");
            } else if variance > 0.01 {
                println!("    ✅ GOOD VARIANCE - contains detail");
            }

            // Save texture visualization (first channel only)
            if label.contains("detail") {
                save_texture_visualization(&texture, texture_size, "/tmp/texture_debug.png");
                println!("    📁 Saved: /tmp/texture_debug.png");
            }
        }
    }

    println!("\n═══════════════════════════════════════════");
    println!("EXTRACTION VALIDATION");
    println!("═══════════════════════════════════════════");

    // Test different scale ranges
    println!("\nTesting extraction at different Gaussian scales:");

    let scales = vec![
        (0.005, "Very fine"),
        (0.01, "Fine"),
        (0.02, "Medium"),
        (0.05, "Coarse"),
    ];

    for &(scale, label) in &scales {
        let texture = TextureMap::extract_from_image(
            &image,
            Vector2::new(0.5, 0.5),  // Center
            scale,
            scale,
            0.0,
            16,
        );

        let variance = compute_variance(&texture);
        println!("  σ={:.3}: variance={:.6} ({})", scale, variance, label);
    }

    println!("\n✅ Extraction debug complete");
    println!("   Check variance values - should increase with smaller scales");
    println!("   Check saved texture visualization");
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
                let r = (1.0 - fx) * (1.0 - fy) * c00.r + fx * (1.0 - fy) * c10.r +
                        (1.0 - fx) * fy * c01.r + fx * fy * c11.r;
                let g = (1.0 - fx) * (1.0 - fy) * c00.g + fx * (1.0 - fy) * c10.g +
                        (1.0 - fx) * fy * c01.g + fx * fy * c11.g;
                let b = (1.0 - fx) * (1.0 - fy) * c00.b + fx * (1.0 - fy) * c10.b +
                        (1.0 - fx) * fy * c01.b + fx * fy * c11.b;

                resized.set_pixel(x, y, Color4::new(r, g, b, 1.0));
            }
        }
    }

    resized
}

fn compute_variance(texture: &TextureMap) -> f32 {
    let mut mean = 0.0;
    for color in &texture.data {
        mean += (color.r + color.g + color.b) / 3.0;
    }
    mean /= texture.data.len() as f32;

    let mut variance_sum = 0.0;
    for color in &texture.data {
        let val = (color.r + color.g + color.b) / 3.0;
        variance_sum += (val - mean).powi(2);
    }

    variance_sum / texture.data.len() as f32
}

fn save_texture_visualization(texture: &TextureMap, size: usize, path: &str) {
    // Create visualization image (upscale 10× for visibility)
    let scale = 10;
    let viz_width = size * scale;
    let viz_height = size * scale;
    let mut viz = ImageBuffer::new(viz_width as u32, viz_height as u32);

    for y in 0..size {
        for x in 0..size {
            let color = texture.get(x, y);

            // Fill 10×10 block
            for dy in 0..scale {
                for dx in 0..scale {
                    let vx = (x * scale + dx) as u32;
                    let vy = (y * scale + dy) as u32;
                    viz.set_pixel(vx, vy, color);
                }
            }
        }
    }

    if let Err(e) = viz.save(path) {
        println!("    ⚠️  Failed to save texture: {}", e);
    }
}
