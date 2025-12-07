//! Test Isotropic vs Anisotropic Edge Gaussians
//!
//! Tests the quantum discovery that isotropic Gaussians work better at edges
//! than the traditional anisotropic (elongated) approach.
//!
//! Quantum channels 3, 4, 7 showed σ_x ≈ σ_y despite high coherence (>0.96)

use lgi_core::ImageBuffer;
use lgi_encoder_v2::{
    EncoderV2,
    renderer_v2::RendererV2,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  Isotropic vs Anisotropic Edge Gaussian Test                 ║");
    println!("║  Testing quantum discovery: isotropic edges may work better  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load Kodak image
    let kodak_path = "/home/greg/gaussian-image-projects/lgi-project/test-data/kodak-dataset/kodim03.png";
    println!("Loading {}...", kodak_path);

    let target = match ImageBuffer::load(kodak_path) {
        Ok(img) => {
            println!("  Loaded: {}x{}", img.width, img.height);
            img
        }
        Err(e) => {
            eprintln!("ERROR: Could not load image: {}", e);
            return;
        }
    };

    let initial_n = 400;
    let max_n = 800;

    println!("\nUsing initial {} Gaussians, max {}", initial_n, max_n);

    // Test 1: Standard anisotropic (current approach)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 1: ANISOTROPIC edges (standard - elongated along edges)");
    println!("═══════════════════════════════════════════════════════════════");

    let encoder1 = EncoderV2::new(target.clone()).expect("Encoder failed");

    let start1 = std::time::Instant::now();
    let gaussians_aniso = encoder1.encode_error_driven_adam(initial_n, max_n);
    let time1 = start1.elapsed();

    let rendered_aniso = RendererV2::render(&gaussians_aniso, target.width, target.height);
    let psnr_aniso = compute_psnr(&target, &rendered_aniso);

    println!("  Final Gaussians: {}", gaussians_aniso.len());
    println!("  PSNR: {:.2} dB", psnr_aniso);
    println!("  Time: {:.2}s", time1.as_secs_f32());

    // Test 2: Isotropic edges (quantum-guided)
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Test 2: ISOTROPIC edges (quantum-guided - circular at edges)");
    println!("═══════════════════════════════════════════════════════════════");

    let encoder2 = EncoderV2::new(target.clone()).expect("Encoder failed");

    let start2 = std::time::Instant::now();
    let gaussians_iso = encoder2.encode_error_driven_adam_isotropic(initial_n, max_n);
    let time2 = start2.elapsed();

    let rendered_iso = RendererV2::render(&gaussians_iso, target.width, target.height);
    let psnr_iso = compute_psnr(&target, &rendered_iso);

    println!("  Final Gaussians: {}", gaussians_iso.len());
    println!("  PSNR: {:.2} dB", psnr_iso);
    println!("  Time: {:.2}s", time2.as_secs_f32());

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                        SUMMARY                                ");
    println!("═══════════════════════════════════════════════════════════════");
    println!("Anisotropic (standard):  {:.2} dB  ({} Gaussians, {:.2}s)",
             psnr_aniso, gaussians_aniso.len(), time1.as_secs_f32());
    println!("Isotropic (quantum):     {:.2} dB  ({} Gaussians, {:.2}s)",
             psnr_iso, gaussians_iso.len(), time2.as_secs_f32());
    println!("");

    let diff = psnr_iso - psnr_aniso;
    if diff > 0.1 {
        println!("ISOTROPIC WINS by:       {:+.2} dB  *** QUANTUM VALIDATED ***", diff);
    } else if diff < -0.1 {
        println!("ANISOTROPIC WINS by:     {:+.2} dB", -diff);
    } else {
        println!("Essentially TIED (diff: {:+.2} dB)", diff);
    }
    println!("═══════════════════════════════════════════════════════════════");
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
