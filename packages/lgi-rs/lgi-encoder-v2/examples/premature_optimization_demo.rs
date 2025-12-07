//! PREMATURE OPTIMIZATION DEMONSTRATION
//!
//! This experiment demonstrates why declaring an "optimal configuration"
//! is meaningless without comprehensive micro-level testing.
//!
//! We'll process kodim03 with our "best guess" and then exhaustively
//! document all the unknowns and untested alternatives.
//!
//! Run: cargo run --release --example premature_optimization_demo -p lgi-encoder-v2

use lgi_core::{ImageBuffer, initializer::{Initializer, InitStrategy}};
use lgi_encoder_v2::adam_optimizer::AdamOptimizer;
use lgi_encoder_v2::renderer_v2::RendererV2;
use lgi_encoder_v2::test_results::TestResult;
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2, color::Color4};
use std::time::Instant;

const RESULTS_DIR: &str = "lgi-encoder-v2/test-results";

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║   PREMATURE OPTIMIZATION DEMONSTRATION                                    ║");
    println!("║   Showing why 'optimal configuration' claims are currently meaningless   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    let image_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    let target = ImageBuffer::load(image_path).expect("Failed to load image");
    let (width, height) = (target.width, target.height);

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("PART 1: IMAGE ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    // Analyze image characteristics
    analyze_image(&target);

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("PART 2: 'OPTIMAL' CONFIGURATION (BEST GUESS)");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    // Document what we're using and why (with caveats)
    println!("Configuration choices and their UNCERTAINTY LEVELS:\n");

    println!("  1. Initialization: Grid");
    println!("     - WHY: Simple, predictable coverage");
    println!("     - UNTESTED ALTERNATIVES: Jitter grid, SLIC, Gradient, PPM sampling,");
    println!("       Structure tensor placement, Laplacian peak detection, K-means");
    println!("     - UNCERTAINTY: HIGH - Grid may be suboptimal for this image\n");

    println!("  2. Gaussian count: 512");
    println!("     - WHY: Balance between quality and efficiency");
    println!("     - UNTESTED: Adaptive N based on image complexity, content-aware density");
    println!("     - UNCERTAINTY: HIGH - Optimal N unknown for this specific image\n");

    println!("  3. Initial scale: 0.02 (isotropic)");
    println!("     - WHY: Historical default, seems reasonable");
    println!("     - UNTESTED: Coverage-based (0.7/√N), anisotropic shapes,");
    println!("       structure tensor eigenanalysis, content-adaptive scaling");
    println!("     - UNCERTAINTY: HIGH - Scale affects convergence significantly\n");

    println!("  4. Optimizer: Adam with per-parameter LR");
    println!("     - WHY: Best tested optimizer so far (+2.25 dB avg)");
    println!("     - UNTESTED: Adam + MS-SSIM hybrid, L-BFGS transitions,");
    println!("       Levenberg-Marquardt, GPU-accelerated Adam, adaptive LR schedules");
    println!("     - UNCERTAINTY: MEDIUM - Adam works but may not be optimal\n");

    println!("  5. Parameterization: Euler (scale_x, scale_y, rotation)");
    println!("     - WHY: Intuitive, only one tested");
    println!("     - UNTESTED: Cholesky (numerical stability), LogRadius (compression),");
    println!("       InverseCovariance (fast rendering)");
    println!("     - UNCERTAINTY: HIGH - Other parameterizations completely untested\n");

    println!("  6. Loss function: MSE");
    println!("     - WHY: Simple, differentiable");
    println!("     - UNTESTED: MS-SSIM, perceptual loss (LPIPS), hybrid losses,");
    println!("       edge-weighted MSE, frequency-domain loss");
    println!("     - UNCERTAINTY: HIGH - MSE != perceptual quality\n");

    println!("  7. Color initialization: Point sampling");
    println!("     - WHY: Simple, fast");
    println!("     - UNTESTED: Guided filter, closed-form least squares,");
    println!("       neighborhood averaging, bilateral filter");
    println!("     - UNCERTAINTY: MEDIUM\n");

    println!("  8. Densification/Pruning: None");
    println!("     - WHY: Testing base case first");
    println!("     - UNTESTED: Gradient-based densification, residual-guided splitting,");
    println!("       opacity pruning, hybrid pruning");
    println!("     - UNCERTAINTY: HIGH - Densification shown to help significantly\n");

    // Run the "optimal" configuration
    println!("Running 'optimal' configuration...\n");

    let start = Instant::now();

    // Initialize
    let initializer = Initializer::new(InitStrategy::Grid)
        .with_scale(0.02)
        .with_seed(42);

    let mut gaussians = initializer.initialize(&target, 512).expect("Init failed");
    let init_render = RendererV2::render(&gaussians, width, height);
    let init_psnr = compute_psnr(&init_render, &target);

    println!("  Initial PSNR: {:.2} dB", init_psnr);

    // Optimize with Adam
    let mut adam = AdamOptimizer::new();
    adam.max_iterations = 200;

    // Track progress
    for iter in 0..=200 {
        if iter % 50 == 0 {
            let current_render = RendererV2::render(&gaussians, width, height);
            let current_psnr = compute_psnr(&current_render, &target);
            println!("  Iteration {:3}: {:.2} dB", iter, current_psnr);
        }
        if iter < 200 {
            adam.max_iterations = 1;
            adam.optimize(&mut gaussians, &target);
        }
    }

    let final_render = RendererV2::render(&gaussians, width, height);
    let final_psnr = compute_psnr(&final_render, &target);
    let elapsed = start.elapsed().as_secs_f32();

    println!("\n  Final PSNR: {:.2} dB", final_psnr);
    println!("  Improvement: {:+.2} dB", final_psnr - init_psnr);
    println!("  Time: {:.2}s", elapsed);

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("PART 3: WHY THIS ANALYSIS IS MEANINGLESS");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    println!("PROBLEM 1: Combinatorial Explosion of Untested Configurations");
    println!("─────────────────────────────────────────────────────────────────────────\n");
    println!("  Initialization strategies:      8+ options");
    println!("  Gaussian counts:                10+ reasonable values");
    println!("  Scale configurations:           5+ strategies");
    println!("  Optimizers:                     6+ options");
    println!("  Parameterizations:              4 options");
    println!("  Loss functions:                 5+ options");
    println!("  Color init methods:             4+ options");
    println!("  Densification strategies:       5+ options");
    println!("  ──────────────────────────────────────────");
    println!("  TOTAL COMBINATIONS:             8×10×5×6×4×5×4×5 = 960,000+\n");
    println!("  We've tested maybe 50 configurations. That's 0.005% of the space.\n");

    println!("PROBLEM 2: Image-Specific Optimality");
    println!("─────────────────────────────────────────────────────────────────────────\n");
    println!("  kodim03 characteristics:");
    println!("    - Fine text ('Bahamas' script) → needs small, precise Gaussians");
    println!("    - Wood grain texture → needs oriented, anisotropic Gaussians");
    println!("    - Smooth sky gradients → needs large, overlapping Gaussians");
    println!("    - Sharp cap edges → needs edge-aware placement");
    println!("    - Neon colors → needs accurate color representation\n");
    println!("  Different images have different optimal configurations!");
    println!("  We've only tested on kodim03 extensively.\n");

    println!("PROBLEM 3: Untested Image Interpretation Modules");
    println!("─────────────────────────────────────────────────────────────────────────\n");
    println!("  These lgi-core modules exist but are UNTESTED with optimization:\n");
    println!("    • structure_tensor.rs      - Edge orientation → anisotropic shapes");
    println!("    • frangi_vesselness.rs     - Tubular structure detection");
    println!("    • geodesic_edt.rs          - Anti-bleeding sigma clamping");
    println!("    • content_detection.rs     - Auto-tuning based on content type");
    println!("    • saliency_detection.rs    - Importance weighting");
    println!("    • text_stroke_detection.rs - Special handling for text");
    println!("    • specular_detection.rs    - Highlight handling");
    println!("    • perceptual_weighting.rs  - Edge-weighted loss\n");
    println!("  Any of these could significantly improve results!\n");

    println!("PROBLEM 4: No Perceptual Quality Metrics");
    println!("─────────────────────────────────────────────────────────────────────────\n");
    println!("  We measured: PSNR = {:.2} dB", final_psnr);
    println!("  We did NOT measure:");
    println!("    - MS-SSIM (structural similarity)");
    println!("    - LPIPS (perceptual similarity)");
    println!("    - Text legibility score");
    println!("    - Edge sharpness");
    println!("    - Color accuracy (CIEDE2000)");
    println!("    - Artifact visibility");
    println!("  PSNR often doesn't correlate with perceived quality!\n");

    println!("PROBLEM 5: Single Random Seed");
    println!("─────────────────────────────────────────────────────────────────────────\n");
    println!("  Used seed: 42");
    println!("  Results may vary significantly with different seeds.");
    println!("  Proper evaluation requires multiple runs + statistics.\n");

    println!("PROBLEM 6: No Comparison with State of the Art");
    println!("─────────────────────────────────────────────────────────────────────────\n");
    println!("  We don't know how {:.2} dB compares to:", final_psnr);
    println!("    - JPEG at similar file size");
    println!("    - WebP at similar file size");
    println!("    - AVIF at similar file size");
    println!("    - Other Gaussian splatting implementations");
    println!("  Without baselines, the number is meaningless.\n");

    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("CONCLUSION: WHAT WE ACTUALLY NEED");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    println!("1. SYSTEMATIC MICRO-LEVEL TESTING");
    println!("   - Test each module in isolation");
    println!("   - Understand individual contributions");
    println!("   - Build a toolkit of validated techniques\n");

    println!("2. COMBINATORIAL EXPLORATION");
    println!("   - Test promising combinations");
    println!("   - Use statistical methods to navigate the space");
    println!("   - Identify synergies and conflicts\n");

    println!("3. MULTI-IMAGE VALIDATION");
    println!("   - Test across all 24 Kodak images");
    println!("   - Test on different image types (text, natural, synthetic)");
    println!("   - Identify image-specific vs universal techniques\n");

    println!("4. PERCEPTUAL EVALUATION");
    println!("   - Implement MS-SSIM, LPIPS metrics");
    println!("   - Consider human evaluation");
    println!("   - Balance multiple quality dimensions\n");

    println!("5. COMPETITIVE BASELINES");
    println!("   - Compare with JPEG, WebP, AVIF at matched bitrates");
    println!("   - Establish meaningful quality targets\n");

    println!("─────────────────────────────────────────────────────────────────────────");
    println!("Until we do all of the above, any claim of 'optimal configuration'");
    println!("is PREMATURE and potentially MISLEADING.");
    println!("─────────────────────────────────────────────────────────────────────────\n");

    // Save result and output image
    let mut result = TestResult::new("premature_optimization_demo", image_path);
    result.optimizer = "DEMO: Premature 'Optimal' Configuration".to_string();
    result.n_gaussians = 512;
    result.iterations = 200;
    result.initial_psnr = init_psnr;
    result.final_psnr = final_psnr;
    result.improvement_db = final_psnr - init_psnr;
    result.elapsed_seconds = elapsed;
    result.image_dimensions = (width, height);

    // Document all the unknowns
    result.set_param("init_strategy", "Grid");
    result.set_param("init_scale", 0.02);
    result.set_param("optimizer", "Adam");
    result.set_param("parameterization", "Euler");
    result.set_param("loss_function", "MSE");
    result.set_param("densification", "None");
    result.set_param("UNTESTED_init_strategies", 7);
    result.set_param("UNTESTED_parameterizations", 3);
    result.set_param("UNTESTED_loss_functions", 4);
    result.set_param("UNTESTED_densification_methods", 4);
    result.set_param("UNTESTED_image_interpretation_modules", 8);
    result.set_param("ESTIMATED_untested_combinations", 960000);
    result.set_param("TESTED_combinations_approx", 50);
    result.set_param("COVERAGE_percent", 0.005);

    result.complete();
    let _ = result.save_with_image(&final_render, RESULTS_DIR);

    println!("✓ Demo complete. Results saved to {}/", RESULTS_DIR);
    println!("  - Review the output image to see current quality");
    println!("  - Compare with the input to see what's missing");
    println!("  - Consider all the untested alternatives!\n");
}

fn analyze_image(img: &ImageBuffer<f32>) {
    let (width, height) = (img.width, img.height);

    // Basic statistics
    let mut r_sum = 0.0f64;
    let mut g_sum = 0.0f64;
    let mut b_sum = 0.0f64;
    let mut r_min = 1.0f32;
    let mut r_max = 0.0f32;
    let mut g_min = 1.0f32;
    let mut g_max = 0.0f32;
    let mut b_min = 1.0f32;
    let mut b_max = 0.0f32;

    for pixel in &img.data {
        r_sum += pixel.r as f64;
        g_sum += pixel.g as f64;
        b_sum += pixel.b as f64;
        r_min = r_min.min(pixel.r);
        r_max = r_max.max(pixel.r);
        g_min = g_min.min(pixel.g);
        g_max = g_max.max(pixel.g);
        b_min = b_min.min(pixel.b);
        b_max = b_max.max(pixel.b);
    }

    let n = img.data.len() as f64;

    println!("Image: {}x{} ({} pixels)", width, height, img.data.len());
    println!("\nColor Statistics:");
    println!("  R: mean={:.3}, range=[{:.3}, {:.3}]", r_sum/n, r_min, r_max);
    println!("  G: mean={:.3}, range=[{:.3}, {:.3}]", g_sum/n, g_min, g_max);
    println!("  B: mean={:.3}, range=[{:.3}, {:.3}]", b_sum/n, b_min, b_max);

    // Estimate gradient magnitude (edge content)
    let mut gradient_sum = 0.0f64;
    for y in 1..height-1 {
        for x in 1..width-1 {
            let idx = (y * width + x) as usize;
            let idx_left = (y * width + x - 1) as usize;
            let idx_right = (y * width + x + 1) as usize;
            let idx_up = ((y - 1) * width + x) as usize;
            let idx_down = ((y + 1) * width + x) as usize;

            let gx = (img.data[idx_right].r - img.data[idx_left].r).abs()
                   + (img.data[idx_right].g - img.data[idx_left].g).abs()
                   + (img.data[idx_right].b - img.data[idx_left].b).abs();
            let gy = (img.data[idx_down].r - img.data[idx_up].r).abs()
                   + (img.data[idx_down].g - img.data[idx_up].g).abs()
                   + (img.data[idx_down].b - img.data[idx_up].b).abs();

            gradient_sum += (gx * gx + gy * gy).sqrt() as f64;
        }
    }

    let avg_gradient = gradient_sum / ((width - 2) * (height - 2)) as f64;

    println!("\nComplexity Estimate:");
    println!("  Average gradient magnitude: {:.4}", avg_gradient);
    if avg_gradient > 0.1 {
        println!("  → HIGH edge content (textures, details)");
    } else if avg_gradient > 0.05 {
        println!("  → MEDIUM edge content (natural scene)");
    } else {
        println!("  → LOW edge content (smooth regions)");
    }

    println!("\nVisible Content (from image inspection):");
    println!("  • 6 colorful baseball caps ('Bahamas' text)");
    println!("  • Neon yellow, orange, green, pink, blue colors");
    println!("  • Cursive text embroidery (fine details)");
    println!("  • Wood grain texture (high frequency)");
    println!("  • Dramatic cloudy sky (smooth gradients)");
    println!("  • Strong cast shadows (smooth transitions)");
    println!("  • Sharp silhouette edges");
}

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mse: f32 = rendered
        .data
        .iter()
        .zip(target.data.iter())
        .map(|(r, t)| {
            let dr = r.r - t.r;
            let dg = r.g - t.g;
            let db = r.b - t.b;
            (dr * dr + dg * dg + db * db) / 3.0
        })
        .sum::<f32>()
        / rendered.data.len() as f32;

    if mse < 1e-10 {
        100.0
    } else {
        10.0 * (1.0 / mse).log10()
    }
}
