//! Debug text detection - why no strokes found?

use lgi_core::{ImageBuffer, StructureTensorField};
use lgi_math::color::Color4;

fn main() {
    println!("Debug: Text Detection Issues\n");

    // Create text pattern
    let mut target = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let is_stroke = (x >= 60 && x <= 68) ||
                           (x >= 124 && x <= 132) ||
                           (x >= 188 && x <= 196);
            let val = if is_stroke { 1.0 } else { 0.0 };
            target.set_pixel(x, y, Color4::new(val, val, val, 1.0));
        }
    }

    // Compute structure tensor with different σ_smooth values
    for &sigma_smooth in &[0.3, 0.5, 1.0, 1.5, 2.0] {
        println!("σ_smooth = {}:", sigma_smooth);

        let tensor_field = StructureTensorField::compute(&target, 1.2, sigma_smooth)
            .expect("Structure tensor failed");

        // Sample at stroke center (x=64)
        let tensor = tensor_field.get(64, 128);

        println!("  At stroke (64, 128):");
        println!("    Coherence: {:.4}", tensor.coherence);
        println!("    λ1: {:.4}, λ2: {:.4}", tensor.eigenvalue_major, tensor.eigenvalue_minor);
        println!("    Gradient mag: {:.4}", tensor.eigenvalue_major.sqrt());

        // Count high-coherence pixels
        let mut high_coherence_count = 0;
        let mut high_gradient_count = 0;

        for y in 0..256 {
            for x in 0..256 {
                let t = tensor_field.get(x, y);
                if t.coherence > 0.8 {
                    high_coherence_count += 1;
                }
                if t.eigenvalue_major.sqrt() > 0.2 {
                    high_gradient_count += 1;
                }
            }
        }

        println!("    Pixels with coherence > 0.8: {}", high_coherence_count);
        println!("    Pixels with gradient > 0.2: {}", high_gradient_count);
        println!();
    }

    println!("Diagnosis:");
    println!("  - If coherence low: σ_smooth blurring strokes");
    println!("  - If gradient low: contrast threshold too high");
    println!("  - Need both high coherence AND high gradient");
}
