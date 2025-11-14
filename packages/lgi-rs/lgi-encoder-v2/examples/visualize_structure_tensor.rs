//! Visualize structure tensor to understand edge detection
//! Saves coherence map and orientation field as images

use lgi_core::{ImageBuffer, StructureTensorField};
use lgi_math::color::Color4;
use std::f32::consts::PI;

fn main() {
    println!("Structure Tensor Visualization");
    println!("==============================\n");

    // Test on vertical edge
    println!("Creating vertical edge test image...");
    let mut edge_image = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let val = if x < 128 { 0.0 } else { 1.0 };
            edge_image.set_pixel(x, y, Color4::new(val, val, val, 1.0));
        }
    }

    // Compute structure tensor
    println!("Computing structure tensor (σ_gradient=1.2, σ_smooth=1.0)...");
    let tensor_field = StructureTensorField::compute(&edge_image, 1.2, 1.0)
        .expect("Structure tensor computation failed");

    // Analyze coherence values
    println!("\nCoherence Statistics:");
    let mut coherences: Vec<f32> = Vec::new();
    for y in 0..256 {
        for x in 0..256 {
            let tensor = tensor_field.get(x, y);
            coherences.push(tensor.coherence);
        }
    }

    coherences.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = coherences[0];
    let p25 = coherences[coherences.len() / 4];
    let median = coherences[coherences.len() / 2];
    let p75 = coherences[3 * coherences.len() / 4];
    let max = coherences[coherences.len() - 1];

    println!("  Min:    {:.4}", min);
    println!("  25th:   {:.4}", p25);
    println!("  Median: {:.4}", median);
    println!("  75th:   {:.4}", p75);
    println!("  Max:    {:.4}", max);

    // Check coherence at edge (x=128)
    println!("\nCoherence at edge (x=128):");
    for y in [64, 128, 192] {
        let tensor = tensor_field.get(128, y);
        println!("  y={}: coherence={:.4}, λ1={:.2}, λ2={:.2}",
            y, tensor.coherence, tensor.eigenvalue_major, tensor.eigenvalue_minor);
    }

    // Check coherence in flat regions
    println!("\nCoherence in flat regions:");
    for (x, y) in [(64, 128), (192, 128)] {
        let tensor = tensor_field.get(x, y);
        println!("  ({}, {}): coherence={:.4}, λ1={:.4}, λ2={:.4}",
            x, y, tensor.coherence, tensor.eigenvalue_major, tensor.eigenvalue_minor);
    }

    // Save coherence map
    println!("\nSaving coherence map...");
    let mut coherence_map = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let c = tensor_field.get(x, y).coherence;
            coherence_map.set_pixel(x, y, Color4::new(c, c, c, 1.0));
        }
    }

    if let Err(e) = coherence_map.save("/tmp/coherence_map.png") {
        println!("Warning: Could not save coherence map: {}", e);
    } else {
        println!("  Saved: /tmp/coherence_map.png");
    }

    // Save orientation field visualization
    println!("Saving orientation field...");
    let mut orientation_map = ImageBuffer::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            let tensor = tensor_field.get(x, y);

            // Encode orientation as hue, coherence as saturation
            let angle = tensor.eigenvector_major.y.atan2(tensor.eigenvector_major.x);
            let hue = (angle + PI) / (2.0 * PI);  // Map [-π, π] to [0, 1]
            let sat = tensor.coherence;
            let val = 1.0;

            // HSV to RGB (simple)
            let rgb = hsv_to_rgb(hue, sat, val);
            orientation_map.set_pixel(x, y, rgb);
        }
    }

    if let Err(e) = orientation_map.save("/tmp/orientation_map.png") {
        println!("Warning: Could not save orientation map: {}", e);
    } else {
        println!("  Saved: /tmp/orientation_map.png");
    }

    // Interpretation
    println!("\nInterpretation:");
    println!("  - Hair toolkit expects: coherence > 0.6 for oriented structures");
    println!("  - At edge (x=128): coherence should be 0.8-0.9");
    println!("  - In flat regions: coherence should be ~0.0");
    println!("\nIf edge coherence < 0.6: Structure tensor not detecting edges properly");
    println!("If edge coherence > 0.6: Anisotropy should help (but data shows it doesn't)");
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Color4<f32> {
    let c = v * s;
    let h_prime = h * 6.0;
    let x = c * (1.0 - ((h_prime % 2.0) - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match h_prime as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    Color4::new(r + m, g + m, b + m, 1.0)
}
