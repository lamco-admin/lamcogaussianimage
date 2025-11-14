//! Analyze Test Image Corpus
//! Characterize all test images to guide testing strategy

use lgi_core::{ImageBuffer, StructureTensorField, content_detection};
use std::path::Path;

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Test Image Corpus Analysis                              ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let test_dirs = vec![
        "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/",
        "/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/new photos/",
    ];

    let mut all_images = Vec::new();

    for dir in &test_dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "JPG" || ext == "JPEG" || ext == "PNG" {
                        all_images.push(path);
                    }
                }
            }
        }
    }

    println!("Found {} test images\n", all_images.len());
    println!("Analyzing representative sample...\n");

    let mut results = Vec::new();

    // Analyze first 10 diverse images
    for (idx, path) in all_images.iter().take(10).enumerate() {
        let filename = path.file_name().unwrap().to_str().unwrap();

        println!("{}. {}", idx + 1, filename);

        let image = match ImageBuffer::load(path.to_str().unwrap()) {
            Ok(img) => {
                // Resize to standard test size
                let scale = 512.0 / img.width.max(img.height) as f32;
                let new_w = (img.width as f32 * scale) as u32;
                let new_h = (img.height as f32 * scale) as u32;
                resize_bilinear(&img, new_w, new_h)
            }
            Err(e) => {
                println!("   ❌ Failed to load: {}", e);
                continue;
            }
        };

        println!("   Resolution: {}×{}", image.width, image.height);

        // Compute statistics
        let (mean, variance) = compute_image_stats(&image);
        let gradient_strength = compute_mean_gradient(&image);

        // Structure analysis
        let tensor_field = match StructureTensorField::compute(&image, 1.2, 1.0) {
            Ok(tf) => tf,
            Err(_) => {
                println!("   ⚠️  Structure tensor failed");
                continue;
            }
        };

        let coherence_stats = analyze_coherence(&tensor_field);

        // Content type
        let content_type = content_detection::ContentAnalyzer::detect_content_type(&image, &tensor_field);

        println!("   Mean intensity: {:.3}", mean);
        println!("   Variance: {:.6}", variance);
        println!("   Gradient strength: {:.4}", gradient_strength);
        println!("   Content type: {:?}", content_type);
        println!("   Coherence - mean: {:.3}, high%: {:.1}%",
                 coherence_stats.0, coherence_stats.1 * 100.0);

        results.push((
            filename.to_string(),
            image.width * image.height,
            variance,
            gradient_strength,
            coherence_stats.1,
            content_type,
        ));
        println!();
    }

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║  Analysis Summary                                        ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!("Image                      | Pixels  | Variance | Gradient | Edges% | Type");
    println!("---------------------------|---------|----------|----------|--------|------");
    for (name, pixels, var, grad, edge_pct, ctype) in &results {
        println!("{:27}| {:7} | {:.6} | {:.4}   | {:5.1}% | {:?}",
                 &name[..name.len().min(26)], pixels, var, grad, edge_pct * 100.0, ctype);
    }

    println!("\n✅ Corpus analysis complete");
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
                img.get_pixel(x0, y0), img.get_pixel(x1, y0),
                img.get_pixel(x0, y1), img.get_pixel(x1, y1),
            ) {
                let r = (1.0-fx)*(1.0-fy)*c00.r + fx*(1.0-fy)*c10.r + (1.0-fx)*fy*c01.r + fx*fy*c11.r;
                let g = (1.0-fx)*(1.0-fy)*c00.g + fx*(1.0-fy)*c10.g + (1.0-fx)*fy*c01.g + fx*fy*c11.g;
                let b = (1.0-fx)*(1.0-fy)*c00.b + fx*(1.0-fy)*c10.b + (1.0-fx)*fy*c01.b + fx*fy*c11.b;
                resized.set_pixel(x, y, lgi_math::color::Color4::new(r, g, b, 1.0));
            }
        }
    }
    resized
}

fn compute_image_stats(image: &ImageBuffer<f32>) -> (f32, f32) {
    let mut mean = 0.0;
    for pixel in &image.data {
        mean += (pixel.r + pixel.g + pixel.b) / 3.0;
    }
    mean /= image.data.len() as f32;

    let mut variance = 0.0;
    for pixel in &image.data {
        let val = (pixel.r + pixel.g + pixel.b) / 3.0;
        variance += (val - mean).powi(2);
    }
    variance /= image.data.len() as f32;

    (mean, variance)
}

fn compute_mean_gradient(image: &ImageBuffer<f32>) -> f32 {
    let mut grad_sum = 0.0;
    let mut count = 0;

    for y in 1..(image.height - 1) {
        for x in 1..(image.width - 1) {
            if let (Some(c), Some(right), Some(down)) = (
                image.get_pixel(x, y),
                image.get_pixel(x + 1, y),
                image.get_pixel(x, y + 1),
            ) {
                let gx = ((right.r - c.r).abs() + (right.g - c.g).abs() + (right.b - c.b).abs()) / 3.0;
                let gy = ((down.r - c.r).abs() + (down.g - c.g).abs() + (down.b - c.b).abs()) / 3.0;
                grad_sum += (gx * gx + gy * gy).sqrt();
                count += 1;
            }
        }
    }

    grad_sum / count as f32
}

fn analyze_coherence(tensor_field: &StructureTensorField) -> (f32, f32) {
    let mut coherence_sum = 0.0;
    let mut high_coherence_count = 0;
    let total = (tensor_field.width * tensor_field.height) as f32;

    for y in 0..tensor_field.height {
        for x in 0..tensor_field.width {
            let c = tensor_field.get(x, y).coherence;
            coherence_sum += c;
            if c > 0.5 {
                high_coherence_count += 1;
            }
        }
    }

    let mean_coherence = coherence_sum / total;
    let high_coherence_fraction = high_coherence_count as f32 / total;

    (mean_coherence, high_coherence_fraction)
}
