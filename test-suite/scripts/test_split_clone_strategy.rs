//! Test Split/Clone Strategy vs Add-at-Hotspots
//!
//! Compare:
//! 1. Current (broken): Add new Gaussians at hotspots
//! 2. 3D Splatting style: Split large, clone small
//!
//! Hypothesis: Split/clone preserves quality better than adding

use lgi_encoder_v2::{
    EncoderV2, renderer_v2::RendererV2, optimizer_v2::OptimizerV2, adam_optimizer::AdamOptimizer,
    adaptive_densification::{AdaptiveDensifier, apply_densification}
};
use lgi_core::ImageBuffer;
use lgi_math::{color::Color4, gaussian::Gaussian2D, parameterization::Euler};
use std::path::PathBuf;

fn main() {
    println!("═══════════════════════════════════════════════════════════════");
    println!("      SPLIT/CLONE vs ADD-AT-HOTSPOTS Strategy Test             ");
    println!("═══════════════════════════════════════════════════════════════\n");

    let image = load_test_image();

    println!("Test: kodim02.png (768×512)\n");

    // Baseline
    let encoder = EncoderV2::new(image.clone()).expect("Failed to create encoder");
    let baseline = encoder.initialize_gaussians(10);
    let baseline_psnr = compute_psnr(&image, &baseline);
    println!("BASELINE (N=100, no opt): {:.2} dB\n", baseline_psnr);

    // Test 1: Current broken method (add at hotspots)
    println!("[1/2] CURRENT METHOD: Add at hotspots (known broken)");
    let current = encoder.encode_error_driven_adam(31, 124);
    let current_psnr = compute_psnr(&image, &current);
    println!("  Final: N={}, PSNR={:.2} dB ({:+.2} dB)\n", current.len(), current_psnr, current_psnr - baseline_psnr);

    // Test 2: Split/clone method
    println!("[2/2] NEW METHOD: Split/clone strategy");
    let split_clone = encode_with_split_clone(&encoder, &image, 31, 124);
    let split_clone_psnr = compute_psnr(&image, &split_clone);
    println!("  Final: N={}, PSNR={:.2} dB ({:+.2} dB)\n", split_clone.len(), split_clone_psnr, split_clone_psnr - baseline_psnr);

    // Summary
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║ Strategy           │ Final PSNR │ vs Baseline │ Result       ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ Baseline (no opt)  │   {:6.2}   │      —      │      —       ║", baseline_psnr);
    println!("║ Add at hotspots    │   {:6.2}   │   {:+6.2}   │ {}      ║",
        current_psnr, current_psnr - baseline_psnr,
        if current_psnr > baseline_psnr { "✅ Works" } else { "❌ Broken" });
    println!("║ Split/clone        │   {:6.2}   │   {:+6.2}   │ {}      ║",
        split_clone_psnr, split_clone_psnr - baseline_psnr,
        if split_clone_psnr > baseline_psnr { "✅ Works" } else { "❌ Broken" });
    println!("╚═══════════════════════════════════════════════════════════════╝");

    println!("\n📊 VERDICT:");
    if split_clone_psnr > current_psnr + 1.0 {
        println!("  ✅ SPLIT/CLONE IS BETTER by {:.2} dB!", split_clone_psnr - current_psnr);
        println!("     → Replace error-driven with adaptive densification");
    } else if split_clone_psnr > baseline_psnr {
        println!("  ✅ SPLIT/CLONE WORKS (+{:.2} dB)", split_clone_psnr - baseline_psnr);
        println!("     → Still better than add-at-hotspots");
    } else {
        println!("  ❌ SPLIT/CLONE ALSO BROKEN ({:+.2} dB)", split_clone_psnr - baseline_psnr);
        println!("     → Problem is deeper than strategy choice");
    }
}

/// Encode using split/clone adaptive densification
fn encode_with_split_clone(
    encoder: &EncoderV2,
    image: &ImageBuffer<f32>,
    initial_n: usize,
    max_n: usize,
) -> Vec<Gaussian2D<f32, Euler<f32>>> {
    // Initialize
    let grid_size = (initial_n as f32).sqrt().ceil() as u32;
    let mut gaussians = encoder.initialize_gaussians(grid_size);

    let densifier = AdaptiveDensifier::default();
    let mut optimizer = OptimizerV2::default();
    optimizer.max_iterations = 100;

    println!("  Initial N={}", gaussians.len());

    for pass in 0..5 {  // Max 5 densification cycles
        // Optimize current set
        let _loss = optimizer.optimize(&mut gaussians, image);

        let psnr = compute_psnr(image, &gaussians);
        println!("    Pass {}: N={}, PSNR={:.2} dB", pass, gaussians.len(), psnr);

        if gaussians.len() >= max_n {
            break;
        }

        // Compute per-Gaussian gradients
        let grad_mags = densifier.compute_gaussian_gradients(&gaussians, image);

        // Adaptive densification (split/clone)
        let (to_add, to_remove) = densifier.densify(&gaussians, &grad_mags);

        if to_add.is_empty() && to_remove.is_empty() {
            println!("    No densification needed");
            break;
        }

        println!("    Densifying: +{} Gaussians, -{} removed", to_add.len(), to_remove.len());

        // Apply operations
        apply_densification(&mut gaussians, to_add, to_remove);

        // Limit to max_n
        if gaussians.len() > max_n {
            gaussians.truncate(max_n);
        }
    }

    gaussians
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
    let path = PathBuf::from("../../kodak-dataset/kodim02.png");
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
