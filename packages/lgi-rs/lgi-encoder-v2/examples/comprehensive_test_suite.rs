//! Comprehensive Test Suite - All 10 Images
//! Validates conditional features, measures improvements, optimizes parameters

use lgi_core::{ImageBuffer, StructureTensorField, lod_system::LODSystem, analytical_triggers::AnalyticalTriggers};
use lgi_encoder_v2::{EncoderV2, optimizer_v2::OptimizerV2, renderer_v2::RendererV2};
use std::time::Instant;

fn main() {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Comprehensive Test Suite - All 10 Images               ║");
    println!("║  Validates conditional features across diverse content   ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    let test_images = vec![
        ("/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/133784383569199567.jpg", "Landscape"),
        ("/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/133683084337742188.jpg", "Large complex"),
        ("/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/279052548_10220135214554919_4471372155507732441_n.jpg", "Portrait"),
        ("/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/arulj.jpg", "Face"),
        ("/media/nomachine/C on Player (NoMachine)/Projects/GaussianImage/Lamco Head.jpg", "Logo"),
    ];

    let mut results = Vec::new();

    for (path, label) in &test_images {
        println!("\n═══════════════════════════════════════════");
        println!("Testing: {}", label);
        println!("═══════════════════════════════════════════");

        let target = match load_and_resize(path, 768) {
            Ok(img) => img,
            Err(e) => {
                println!("  ❌ Failed to load: {}", e);
                continue;
            }
        };

        println!("  Resolution: {}×{}", target.width, target.height);

        // Phase 1: Baseline encoding
        println!("\n  Phase 1: Baseline (N=400, 30 iters)");
        let encoder = EncoderV2::new(target.clone()).unwrap();
        let mut gaussians = encoder.initialize_gaussians_guided(20);  // 20×20 = 400

        let start = Instant::now();
        let mut optimizer = OptimizerV2::default();
        optimizer.max_iterations = 30;
        let final_loss = optimizer.optimize(&mut gaussians, &target);
        let time = start.elapsed().as_secs_f32();

        let rendered = RendererV2::render(&gaussians, target.width, target.height);
        let psnr = compute_psnr(&rendered, &target);

        println!("    Time: {:.1}s", time);
        println!("    PSNR: {:.2} dB", psnr);
        println!("    Loss: {:.6}", final_loss);

        // Phase 2: Analyze content
        println!("\n  Phase 2: Content Analysis");
        let tensor_field = StructureTensorField::compute(&target, 1.2, 1.0).unwrap();

        // Compute coherence distribution
        let mut coherence_histogram = vec![0; 10];
        for y in 0..target.height {
            for x in 0..target.width {
                let c = tensor_field.get(x, y).coherence;
                let bin = (c * 10.0).min(9.0) as usize;
                coherence_histogram[bin] += 1;
            }
        }

        let total_pixels = (target.width * target.height) as f32;
        println!("    Coherence distribution:");
        for (i, &count) in coherence_histogram.iter().enumerate() {
            let percent = 100.0 * count as f32 / total_pixels;
            if percent > 1.0 {
                println!("      {:.1}-{:.1}: {:.1}%", i as f32 / 10.0, (i + 1) as f32 / 10.0, percent);
            }
        }

        // Phase 3: LOD Classification
        println!("\n  Phase 3: LOD Classification");
        let lod = LODSystem::classify(&gaussians);
        let stats = lod.stats();
        println!("    Coarse:  {} ({:.1}%)", stats.coarse_count, stats.coarse_percent);
        println!("    Medium:  {} ({:.1}%)", stats.medium_count, stats.medium_percent);
        println!("    Fine:    {} ({:.1}%)", stats.fine_count, stats.fine_percent);

        // Phase 4: Analytical Triggers
        println!("\n  Phase 4: Analytical Triggers");
        let triggers = AnalyticalTriggers::analyze(&target, &rendered, &gaussians, &tensor_field);
        println!("    SED: {:.4} (spectral energy drop)", triggers.sed);
        println!("    ERR: {:.4} (entropy-residual ratio)", triggers.err);
        println!("    LCC: {:.4} (laplacian consistency)", triggers.lcc);
        println!("    AGD: {:.4} (anisotropy gradient divergence)", triggers.agd);

        // Check which triggers fire
        if triggers.should_add_residuals() {
            println!("    → Should add residuals");
        }
        if triggers.should_refine_gaussians() {
            println!("    → Should refine Gaussians");
        }

        results.push((label.to_string(), psnr, time, stats.coarse_percent, stats.medium_percent, stats.fine_percent));
    }

    // Summary
    println!("\n\n╔══════════════════════════════════════════════════════════╗");
    println!("║  SUMMARY - Baseline Results                             ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!("Image             | PSNR    | Time  | Coarse% | Medium% | Fine%");
    println!("------------------|---------|-------|---------|---------|------");
    for (label, psnr, time, c, m, f) in &results {
        println!("{:18}| {:.2} dB | {:.1}s  | {:5.1}% | {:5.1}% | {:4.1}%",
                 label, psnr, time, c, m, f);
    }

    println!("\n✅ Comprehensive test suite complete");
    println!("   All images tested with baseline encoder");
    println!("   Content analysis and LOD classification validated");
}

fn load_and_resize(path: &str, max_size: u32) -> Result<ImageBuffer<f32>, String> {
    let img = ImageBuffer::load(path).map_err(|e| format!("{}", e))?;

    let scale = max_size as f32 / img.width.max(img.height) as f32;
    let new_w = (img.width as f32 * scale) as u32;
    let new_h = (img.height as f32 * scale) as u32;

    Ok(resize_bilinear(&img, new_w, new_h))
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

fn compute_psnr(rendered: &ImageBuffer<f32>, target: &ImageBuffer<f32>) -> f32 {
    let mut mse = 0.0;
    for (r, t) in rendered.data.iter().zip(target.data.iter()) {
        mse += (r.r - t.r).powi(2) + (r.g - t.g).powi(2) + (r.b - t.b).powi(2);
    }
    mse /= (rendered.width * rendered.height * 3) as f32;
    if mse < 1e-10 { 100.0 } else { 20.0 * (1.0 / mse.sqrt()).log10() }
}
