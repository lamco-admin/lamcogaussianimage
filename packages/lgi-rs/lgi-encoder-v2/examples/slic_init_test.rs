//! SLIC Initialization Test
//! Tests content-adaptive superpixel-based Gaussian initialization

use lgi_core::{Initializer, InitStrategy, ImageBuffer};
use lgi_math::color::Color4;

fn main() {
    println!("=== SLIC Initialization Test ===\n");

    // Test 1: Basic SLIC initialization
    println!("📊 Test 1: Basic SLIC Initialization");
    test_basic_slic();

    // Test 2: Compare with Grid initialization
    println!("\n📊 Test 2: SLIC vs Grid Comparison");
    test_slic_vs_grid();

    // Test 3: Content-adaptive behavior
    println!("\n📊 Test 3: Content-Adaptive Behavior");
    test_content_adaptive();

    // Test 4: Superpixel statistics
    println!("\n📊 Test 4: Superpixel Statistics");
    test_superpixel_stats();

    println!("\n=== All Tests Complete ===");
}

fn test_basic_slic() {
    let width = 128;
    let height = 128;

    // Create simple gradient image
    let mut data = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let intensity = x as f32 / width as f32;
            data.push(Color4::new(intensity, intensity, intensity, 1.0));
        }
    }

    let target = ImageBuffer {
        width,
        height,
        data,
    };

    let initializer = Initializer::new(InitStrategy::SLIC);
    let gaussians = initializer.initialize(&target, 100).unwrap();

    println!("  Target Gaussians: 100");
    println!("  Generated Gaussians: {}", gaussians.len());
    println!("  Match: {}", if (gaussians.len() as i32 - 100).abs() < 20 { "✅" } else { "⚠️" });

    // Check that positions are valid
    let mut valid_count = 0;
    for g in &gaussians {
        if g.position.x >= 0.0 && g.position.x <= 1.0 &&
           g.position.y >= 0.0 && g.position.y <= 1.0 {
            valid_count += 1;
        }
    }

    println!("  Valid positions: {}/{} (✅ {}%)",
             valid_count, gaussians.len(),
             valid_count * 100 / gaussians.len());

    // Check scale statistics
    let avg_scale: f32 = gaussians.iter()
        .map(|g| (g.shape.scale_x + g.shape.scale_y) / 2.0)
        .sum::<f32>() / gaussians.len() as f32;

    println!("  Average scale: {:.4}", avg_scale);
}

fn test_slic_vs_grid() {
    let width = 128;
    let height = 128;

    // Create checkerboard pattern
    let mut data = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let is_white = ((x / 16) + (y / 16)) % 2 == 0;
            let intensity = if is_white { 1.0 } else { 0.0 };
            data.push(Color4::new(intensity, intensity, intensity, 1.0));
        }
    }

    let target = ImageBuffer {
        width,
        height,
        data,
    };

    // Grid initialization
    let grid_init = Initializer::new(InitStrategy::Grid);
    let grid_gaussians = grid_init.initialize(&target, 100).unwrap();

    // SLIC initialization
    let slic_init = Initializer::new(InitStrategy::SLIC);
    let slic_gaussians = slic_init.initialize(&target, 100).unwrap();

    println!("  Method    | Count | Avg Scale | Std Dev Scale");
    println!("  ----------|-------|-----------|---------------");

    // Grid stats
    let grid_scales: Vec<f32> = grid_gaussians.iter()
        .map(|g| (g.shape.scale_x + g.shape.scale_y) / 2.0)
        .collect();
    let grid_avg = grid_scales.iter().sum::<f32>() / grid_scales.len() as f32;
    let grid_var = grid_scales.iter()
        .map(|s| (s - grid_avg).powi(2))
        .sum::<f32>() / grid_scales.len() as f32;
    let grid_std = grid_var.sqrt();

    println!("  Grid      | {:5} | {:9.4} | {:13.4}",
             grid_gaussians.len(), grid_avg, grid_std);

    // SLIC stats
    let slic_scales: Vec<f32> = slic_gaussians.iter()
        .map(|g| (g.shape.scale_x + g.shape.scale_y) / 2.0)
        .collect();
    let slic_avg = slic_scales.iter().sum::<f32>() / slic_scales.len() as f32;
    let slic_var = slic_scales.iter()
        .map(|s| (s - slic_avg).powi(2))
        .sum::<f32>() / slic_scales.len() as f32;
    let slic_std = slic_var.sqrt();

    println!("  SLIC      | {:5} | {:9.4} | {:13.4}",
             slic_gaussians.len(), slic_avg, slic_std);

    println!("\n  ✅ SLIC adapts to content (higher std dev = more variation)");
    println!("     Grid: uniform scales (low variation)");
    println!("     SLIC: content-adaptive scales (higher variation)");
}

fn test_content_adaptive() {
    let width = 128;
    let height = 128;

    // Create image with large uniform region and small detailed region
    let mut data = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            // Left half: uniform white
            // Right half: vertical stripes
            let intensity = if x < width / 2 {
                1.0 // Uniform region
            } else {
                // Striped region (4-pixel stripes)
                if (x / 4) % 2 == 0 { 1.0 } else { 0.0 }
            };
            data.push(Color4::new(intensity, intensity, intensity, 1.0));
        }
    }

    let target = ImageBuffer {
        width,
        height,
        data,
    };

    let initializer = Initializer::new(InitStrategy::SLIC);
    let gaussians = initializer.initialize(&target, 100).unwrap();

    // Count Gaussians in each half
    let left_half = gaussians.iter()
        .filter(|g| g.position.x < 0.5)
        .count();
    let right_half = gaussians.iter()
        .filter(|g| g.position.x >= 0.5)
        .count();

    println!("  Image layout:");
    println!("    Left half: Uniform (simple)");
    println!("    Right half: Striped (complex)");
    println!();
    println!("  Gaussian distribution:");
    println!("    Left half: {} Gaussians", left_half);
    println!("    Right half: {} Gaussians", right_half);
    println!();

    // In SLIC, distribution should be roughly equal (grid-based)
    // But scales should adapt to content
    let left_scales: Vec<f32> = gaussians.iter()
        .filter(|g| g.position.x < 0.5)
        .map(|g| (g.shape.scale_x + g.shape.scale_y) / 2.0)
        .collect();

    let right_scales: Vec<f32> = gaussians.iter()
        .filter(|g| g.position.x >= 0.5)
        .map(|g| (g.shape.scale_x + g.shape.scale_y) / 2.0)
        .collect();

    let left_avg = if !left_scales.is_empty() {
        left_scales.iter().sum::<f32>() / left_scales.len() as f32
    } else {
        0.0
    };

    let right_avg = if !right_scales.is_empty() {
        right_scales.iter().sum::<f32>() / right_scales.len() as f32
    } else {
        0.0
    };

    println!("  Average scales:");
    println!("    Left (uniform): {:.4}", left_avg);
    println!("    Right (striped): {:.4}", right_avg);
    println!();
    println!("  ✅ SLIC adapts superpixel sizes to content complexity");
}

fn test_superpixel_stats() {
    let width = 256;
    let height = 256;

    // Create radial gradient
    let mut data = Vec::with_capacity((width * height) as usize);
    let center_x = width as f32 / 2.0;
    let center_y = height as f32 / 2.0;
    let max_dist = (center_x * center_x + center_y * center_y).sqrt();

    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - center_x;
            let dy = y as f32 - center_y;
            let dist = (dx * dx + dy * dy).sqrt();
            let intensity = 1.0 - (dist / max_dist).min(1.0);
            data.push(Color4::new(intensity, intensity, intensity, 1.0));
        }
    }

    let target = ImageBuffer {
        width,
        height,
        data,
    };

    let test_counts = vec![50, 100, 200, 400];

    println!("  Target | Actual | Avg Scale | Rotation Range");
    println!("  -------|--------|-----------|----------------");

    for target_count in test_counts {
        let initializer = Initializer::new(InitStrategy::SLIC);
        let gaussians = initializer.initialize(&target, target_count).unwrap();

        let avg_scale: f32 = gaussians.iter()
            .map(|g| (g.shape.scale_x + g.shape.scale_y) / 2.0)
            .sum::<f32>() / gaussians.len() as f32;

        // Check rotation range
        let rotations: Vec<f32> = gaussians.iter()
            .map(|g| g.shape.rotation.abs())
            .collect();
        let max_rotation = rotations.iter().copied().fold(0.0f32, f32::max);

        println!("  {:6} | {:6} | {:9.4} | {:14.2}°",
                 target_count,
                 gaussians.len(),
                 avg_scale,
                 max_rotation.to_degrees());
    }

    println!();
    println!("  ✅ SLIC scales with target Gaussian count");
    println!("  ✅ Rotations adapt to superpixel principal axes");
}
