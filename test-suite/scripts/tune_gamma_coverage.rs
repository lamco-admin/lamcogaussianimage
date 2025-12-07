//! Tune Gamma (Coverage Factor) - Find Optimal Coverage Scale
//!
//! Test: γ × √(W×H/N) with different γ values
//! Goal: Find what gives best PSNR for different content types

use lgi_core::ImageBuffer;
use lgi_math::color::Color4;
use lgi_encoder_v2::{renderer_v2::RendererV2};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler, vec::Vector2};

fn main() {
    println!("╔═══════════════════════════════════════════════════╗");
    println!("║     Gamma (Coverage Factor) Tuning Experiment     ║");
    println!("║  Testing: γ in σ = γ × √(W×H/N)                  ║");
    println!("╚═══════════════════════════════════════════════════╝\n");

    // Test on solid color first (simple case)
    test_gamma_on_solid_color();

    println!("\n");

    // Test on gradient
    test_gamma_on_gradient();
}

fn test_gamma_on_solid_color() {
    println!("Test A: Solid Red 256×256 with 64 Gaussians (8×8)");
    println!("═══════════════════════════════════════════════════");

    let mut target = ImageBuffer::new(256, 256);
    for pixel in &mut target.data {
        *pixel = Color4::new(1.0, 0.0, 0.0, 1.0);
    }

    let gammas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0];

    for &gamma in &gammas {
        let sigma_base_px = gamma * ((256.0_f32 * 256.0) / 64.0).sqrt();
        let sigma_norm = sigma_base_px / 256.0;

        // Create isotropic Gaussians on grid
        let mut gaussians = Vec::new();
        for gy in 0..8 {
            for gx in 0..8 {
                let x = (gx * 32 + 16) as f32 / 256.0;
                let y = (gy * 32 + 16) as f32 / 256.0;

                gaussians.push(Gaussian2D::new(
                    Vector2::new(x, y),
                    Euler::isotropic(sigma_norm),
                    Color4::new(1.0, 0.0, 0.0, 1.0),
                    1.0,
                ));
            }
        }

        let rendered = RendererV2::render(&gaussians, 256, 256);
        let psnr = compute_psnr(&target, &rendered);

        println!("γ = {:.2}: σ_base = {:5.1} px ({:.4} norm) → PSNR = {:6.2} dB",
            gamma, sigma_base_px, sigma_norm, psnr);
    }
}

fn test_gamma_on_gradient() {
    println!("Test B: Linear Gradient 256×256 with 400 Gaussians (20×20)");
    println!("═══════════════════════════════════════════════════");

    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let t = x as f32 / 255.0;
            target.set_pixel(x, y, Color4::new(t, 0.0, 1.0 - t, 1.0));
        }
    }

    let gammas = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0];

    for &gamma in &gammas {
        let sigma_base_px = gamma * ((256.0_f32 * 256.0) / 400.0).sqrt();
        let sigma_norm = sigma_base_px / 256.0;

        // Create isotropic Gaussians matching gradient colors
        let mut gaussians = Vec::new();
        for gy in 0..20 {
            for gx in 0..20 {
                let x = (gx * 12 + 6) as f32 / 256.0;
                let y = (gy * 12 + 6) as f32 / 256.0;

                // Sample gradient color
                let t = x;
                let color = Color4::new(t, 0.0, 1.0 - t, 1.0);

                gaussians.push(Gaussian2D::new(
                    Vector2::new(x, y),
                    Euler::isotropic(sigma_norm),
                    color,
                    1.0,
                ));
            }
        }

        let rendered = RendererV2::render(&gaussians, 256, 256);
        let psnr = compute_psnr(&target, &rendered);

        println!("γ = {:.2}: σ_base = {:5.1} px ({:.4} norm) → PSNR = {:6.2} dB",
            gamma, sigma_base_px, sigma_norm, psnr);
    }
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

    if mse < 1e-10 {
        100.0
    } else {
        20.0 * (1.0 / mse.sqrt()).log10()
    }
}
