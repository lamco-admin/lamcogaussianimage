//! Test LOD System quality at different levels
//!
//! The LOD system classifies Gaussians into Coarse/Medium/Fine bands.
//! This test measures:
//! - Quality contribution of each band
//! - Progressive rendering quality
//! - Sub-pixel culling effectiveness
//!
//! Run: cargo run --release --example test_lod_system -p lgi-encoder-v2

use lgi_core::{ImageBuffer, lod_system::{LODSystem, LODBand}};
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};
use std::time::Instant;
use serde_json::json;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   LOD SYSTEM TEST                                            ║");
    println!("║   Measuring quality at different LOD levels                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load image");
    let (width, height) = (target.width, target.height);

    println!("Image: {}x{}", width, height);

    // First, create and optimize a good set of Gaussians with varied scales
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Step 1: Creating multi-scale Gaussian set");
    println!("═══════════════════════════════════════════════════════════════");

    let mut gaussians = create_multiscale_grid(&target, 512);
    println!("  Created {} Gaussians with varied scales", gaussians.len());

    // Optimize
    println!("  Optimizing with Adam (100 iterations)...");
    let mut adam = AdamOptimizer::new();
    adam.max_iterations = 100;

    for iter in 0..100 {
        if iter % 25 == 0 {
            let rendered = RendererV2::render(&gaussians, width, height);
            let psnr = compute_psnr(&rendered, &target);
            println!("    [{:3}/100] PSNR={:.2}dB", iter, psnr);
        }
        adam.max_iterations = 1;
        adam.optimize(&mut gaussians, &target);
    }

    let full_render = RendererV2::render(&gaussians, width, height);
    let full_psnr = compute_psnr(&full_render, &target);
    println!("  Full quality PSNR: {:.2}dB (N={})", full_psnr, gaussians.len());

    // Classify into LOD bands
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Step 2: LOD Classification");
    println!("═══════════════════════════════════════════════════════════════");

    let lod = LODSystem::classify(&gaussians);
    lod.print_stats();

    // Measure quality at each LOD level
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Step 3: Quality at Each LOD Level");
    println!("═══════════════════════════════════════════════════════════════");

    let mut results = Vec::new();

    // Test: Coarse only
    {
        let coarse = lod.get_band(LODBand::Coarse);
        if !coarse.is_empty() {
            let render = RendererV2::render(coarse, width, height);
            let psnr = compute_psnr(&render, &target);
            let quality_ratio = psnr / full_psnr * 100.0;
            println!("  Coarse only: {:.2}dB ({:.1}% of full) - {} Gaussians",
                     psnr, quality_ratio, coarse.len());
            results.push(("Coarse", coarse.len(), psnr));
        } else {
            println!("  Coarse: No Gaussians in this band");
        }
    }

    // Test: Coarse + Medium
    {
        let mut coarse_medium: Vec<Gaussian2D<f32, Euler<f32>>> = Vec::new();
        coarse_medium.extend(lod.get_band(LODBand::Coarse).iter().cloned());
        coarse_medium.extend(lod.get_band(LODBand::Medium).iter().cloned());

        if !coarse_medium.is_empty() {
            let render = RendererV2::render(&coarse_medium, width, height);
            let psnr = compute_psnr(&render, &target);
            let quality_ratio = psnr / full_psnr * 100.0;
            println!("  Coarse+Medium: {:.2}dB ({:.1}% of full) - {} Gaussians",
                     psnr, quality_ratio, coarse_medium.len());
            results.push(("Coarse+Medium", coarse_medium.len(), psnr));
        }
    }

    // Test: Medium only
    {
        let medium = lod.get_band(LODBand::Medium);
        if !medium.is_empty() {
            let render = RendererV2::render(medium, width, height);
            let psnr = compute_psnr(&render, &target);
            println!("  Medium only: {:.2}dB - {} Gaussians", psnr, medium.len());
            results.push(("Medium", medium.len(), psnr));
        }
    }

    // Test: Fine only
    {
        let fine = lod.get_band(LODBand::Fine);
        if !fine.is_empty() {
            let render = RendererV2::render(fine, width, height);
            let psnr = compute_psnr(&render, &target);
            println!("  Fine only: {:.2}dB - {} Gaussians", psnr, fine.len());
            results.push(("Fine", fine.len(), psnr));
        }
    }

    // Test: Medium + Fine (zoomed in view)
    {
        let mut medium_fine: Vec<Gaussian2D<f32, Euler<f32>>> = Vec::new();
        medium_fine.extend(lod.get_band(LODBand::Medium).iter().cloned());
        medium_fine.extend(lod.get_band(LODBand::Fine).iter().cloned());

        if !medium_fine.is_empty() {
            let render = RendererV2::render(&medium_fine, width, height);
            let psnr = compute_psnr(&render, &target);
            let quality_ratio = psnr / full_psnr * 100.0;
            println!("  Medium+Fine: {:.2}dB ({:.1}% of full) - {} Gaussians",
                     psnr, quality_ratio, medium_fine.len());
            results.push(("Medium+Fine", medium_fine.len(), psnr));
        }
    }

    // Test: All (full)
    results.push(("Full", gaussians.len(), full_psnr));

    // Zoom-based selection test
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Step 4: Zoom-Based Selection");
    println!("═══════════════════════════════════════════════════════════════");

    let zoom_levels = vec![0.25, 0.5, 1.0, 2.0, 4.0];

    for zoom in &zoom_levels {
        let selected = lod.select_for_zoom(*zoom);
        let selected_vec: Vec<Gaussian2D<f32, Euler<f32>>> = selected.iter().map(|g| (*g).clone()).collect();

        if !selected_vec.is_empty() {
            let render = RendererV2::render(&selected_vec, width, height);
            let psnr = compute_psnr(&render, &target);
            println!("  Zoom {:.2}x: {} Gaussians → {:.2}dB",
                     zoom, selected_vec.len(), psnr);
        }
    }

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("{:<20} {:>10} {:>12} {:>12}",
             "LOD Level", "N Gauss", "PSNR", "% of Full");
    println!("{:-<20} {:-<10} {:-<12} {:-<12}", "", "", "", "");

    for (name, n, psnr) in &results {
        let percent = psnr / full_psnr * 100.0;
        println!("{:<20} {:>10} {:>11.2}dB {:>11.1}%",
                 name, n, psnr, percent);
    }

    // Save result
    let mut result = TestResult::new("lod_system_quality", image_path);
    result.optimizer = "Adam + LOD Analysis".to_string();
    result.image_dimensions = (width, height);
    result.n_gaussians = gaussians.len();
    result.iterations = 100;
    result.final_psnr = full_psnr;
    result.extra_metrics = Some(json!({
        "coarse_count": lod.coarse_count,
        "medium_count": lod.medium_count,
        "fine_count": lod.fine_count,
        "lod_results": results.iter().map(|(name, n, psnr)| {
            json!({ "level": name, "n_gaussians": n, "psnr": psnr })
        }).collect::<Vec<_>>(),
    }));
    let _ = result.save(RESULTS_DIR);

    println!("\nOBSERVATIONS:");
    println!("- LOD system enables progressive rendering");
    println!("- Coarse band alone should provide ~60% quality (preview)");
    println!("- Coarse+Medium should provide ~85% quality (usable)");
}

/// Create a grid with varied scales (for LOD testing)
fn create_multiscale_grid(target: &ImageBuffer<f32>, n: usize) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    let grid_size = (n as f32).sqrt().ceil() as usize;
    let (width, height) = (target.width, target.height);
    let mut gaussians = Vec::with_capacity(n);

    for i in 0..n {
        let gx = i % grid_size;
        let gy = i / grid_size;

        // Jitter
        let jx = ((i * 7 + 3) % 100) as f32 / 100.0 - 0.5;
        let jy = ((i * 11 + 7) % 100) as f32 / 100.0 - 0.5;

        let x = ((gx as f32 + 0.5 + jx * 0.4) / grid_size as f32).clamp(0.01, 0.99);
        let y = ((gy as f32 + 0.5 + jy * 0.4) / grid_size as f32).clamp(0.01, 0.99);

        let px = ((x * width as f32) as u32).min(width - 1);
        let py = ((y * height as f32) as u32).min(height - 1);
        let color = target.get_pixel(px, py).unwrap();

        // Varied scales to populate all LOD bands:
        // - 20% coarse (scale > 0.2)
        // - 30% medium (scale 0.1-0.2)
        // - 50% fine (scale < 0.1)
        let scale = match i % 10 {
            0 | 1 => 0.25,       // Coarse (det > 0.04)
            2 | 3 | 4 => 0.15,   // Medium (det 0.01-0.04)
            _ => 0.08,           // Fine (det < 0.01)
        };

        gaussians.push(Gaussian2D::new(
            Vector2::new(x, y),
            Euler::new(scale, scale, 0.0),
            color,
            1.0,
        ));
    }

    gaussians
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let count = (rendered.width * rendered.height * 3) as f32;
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse /= count;
    if mse <= 0.0 { 100.0 } else { 10.0 * (1.0 / mse).log10() }
}
