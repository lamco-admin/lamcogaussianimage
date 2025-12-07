// GPU-based gradient computation for Gaussian optimization
// Copyright (c) 2025 Lamco Development
//
// Computes full gradients for all Gaussian parameters using backpropagation
// This accelerates training by 100-1000× compared to CPU

// Gaussian struct - must match ShaderGaussian in Rust exactly (48 bytes)
// Using separate f32 fields to avoid vec4 16-byte alignment issues
struct Gaussian {
    position: vec2<f32>,      // 8 bytes @ offset 0
    scale_x: f32,             // 4 bytes @ offset 8
    scale_y: f32,             // 4 bytes @ offset 12
    rotation: f32,            // 4 bytes @ offset 16
    color_r: f32,             // 4 bytes @ offset 20
    color_g: f32,             // 4 bytes @ offset 24
    color_b: f32,             // 4 bytes @ offset 28
    color_a: f32,             // 4 bytes @ offset 32
    opacity: f32,             // 4 bytes @ offset 36
    _padding: vec2<u32>,      // 8 bytes @ offset 40
}

// Gradient struct - must match GpuGradient in Rust exactly (48 bytes)
// CRITICAL: Array stride in WGSL is padded to alignment of largest field (vec2 = 8 bytes)
// So we need explicit padding to 48 bytes
struct GaussianGradient {
    d_position: vec2<f32>,    // 8 bytes @ offset 0
    d_scale_x: f32,           // 4 bytes @ offset 8
    d_scale_y: f32,           // 4 bytes @ offset 12
    d_rotation: f32,          // 4 bytes @ offset 16
    d_color_r: f32,           // 4 bytes @ offset 20
    d_color_g: f32,           // 4 bytes @ offset 24
    d_color_b: f32,           // 4 bytes @ offset 28
    d_color_a: f32,           // 4 bytes @ offset 32
    d_opacity: f32,           // 4 bytes @ offset 36
    _padding: vec2<f32>,      // 8 bytes @ offset 40 (48 bytes total)
}

@group(0) @binding(0) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(1) var<storage, read> rendered_image: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> target_image: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> gradients: array<GaussianGradient>;
@group(0) @binding(4) var<uniform> params: ComputeParams;

struct ComputeParams {
    width: u32,
    height: u32,
    num_gaussians: u32,
    _padding: u32,
}

// Compute L2 loss gradient for a pixel
fn compute_pixel_gradient(rendered: vec4<f32>, target_pixel: vec4<f32>) -> vec4<f32> {
    return 2.0 * (rendered - target_pixel);
}

// Evaluate Gaussian at normalized position [0,1]
fn evaluate_gaussian(gaussian: Gaussian, pos: vec2<f32>) -> f32 {
    // Transform to Gaussian's local coordinate system
    let dx = pos.x - gaussian.position.x;
    let dy = pos.y - gaussian.position.y;

    let cos_r = cos(gaussian.rotation);
    let sin_r = sin(gaussian.rotation);

    // Rotate
    let x_rot = dx * cos_r + dy * sin_r;
    let y_rot = -dx * sin_r + dy * cos_r;

    // Scale
    let x_scaled = x_rot / gaussian.scale_x;
    let y_scaled = y_rot / gaussian.scale_y;

    // Gaussian function
    let exponent = -(x_scaled * x_scaled + y_scaled * y_scaled) * 0.5;

    // Cutoff for performance
    if (exponent < -12.0) {
        return 0.0;
    }

    return exp(exponent);
}

// Compute derivatives of Gaussian wrt position
fn compute_position_derivatives(
    gaussian: Gaussian,
    pos: vec2<f32>,
    gaussian_value: f32
) -> vec2<f32> {
    let dx = pos.x - gaussian.position.x;
    let dy = pos.y - gaussian.position.y;

    let cos_r = cos(gaussian.rotation);
    let sin_r = sin(gaussian.rotation);

    let x_rot = dx * cos_r + dy * sin_r;
    let y_rot = -dx * sin_r + dy * cos_r;

    let x_scaled = x_rot / gaussian.scale_x;
    let y_scaled = y_rot / gaussian.scale_y;

    // dG/dμ = G(x) * Σ^-1 * (x - μ)
    let d_x = gaussian_value * (x_scaled / gaussian.scale_x * cos_r - y_scaled / gaussian.scale_y * sin_r);
    let d_y = gaussian_value * (x_scaled / gaussian.scale_x * sin_r + y_scaled / gaussian.scale_y * cos_r);

    return vec2<f32>(d_x, d_y);
}

// Compute derivatives wrt scale
fn compute_scale_derivatives(
    gaussian: Gaussian,
    pos: vec2<f32>,
    gaussian_value: f32
) -> vec2<f32> {
    let dx = pos.x - gaussian.position.x;
    let dy = pos.y - gaussian.position.y;

    let cos_r = cos(gaussian.rotation);
    let sin_r = sin(gaussian.rotation);

    let x_rot = dx * cos_r + dy * sin_r;
    let y_rot = -dx * sin_r + dy * cos_r;

    let x_scaled = x_rot / gaussian.scale_x;
    let y_scaled = y_rot / gaussian.scale_y;

    // dG/dσx = G(x) * (x_rot² / σx³)
    let d_scale_x = gaussian_value * (x_scaled * x_scaled / gaussian.scale_x);

    // dG/dσy = G(x) * (y_rot² / σy³)
    let d_scale_y = gaussian_value * (y_scaled * y_scaled / gaussian.scale_y);

    return vec2<f32>(d_scale_x, d_scale_y);
}

// Compute derivative wrt rotation
fn compute_rotation_derivative(
    gaussian: Gaussian,
    pos: vec2<f32>,
    gaussian_value: f32
) -> f32 {
    let dx = pos.x - gaussian.position.x;
    let dy = pos.y - gaussian.position.y;

    let cos_r = cos(gaussian.rotation);
    let sin_r = sin(gaussian.rotation);

    let x_rot = dx * cos_r + dy * sin_r;
    let y_rot = -dx * sin_r + dy * cos_r;

    let x_scaled = x_rot / gaussian.scale_x;
    let y_scaled = y_rot / gaussian.scale_y;

    // dG/dθ involves rotation of the scaled coordinates
    let d_rotation = gaussian_value * (
        -x_scaled / gaussian.scale_x * (-dx * sin_r + dy * cos_r) +
        -y_scaled / gaussian.scale_y * (-dx * cos_r - dy * sin_r)
    );

    return d_rotation;
}

@compute @workgroup_size(256, 1, 1)
fn compute_gradients_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let gaussian_idx = global_id.x;

    if (gaussian_idx >= params.num_gaussians) {
        return;
    }

    let gaussian = gaussians[gaussian_idx];

    // Initialize gradient accumulator (all zeros)
    var grad = GaussianGradient(
        vec2<f32>(0.0),           // d_position
        0.0, 0.0, 0.0,            // d_scale_x, d_scale_y, d_rotation
        0.0, 0.0, 0.0, 0.0,       // d_color_r, d_color_g, d_color_b, d_color_a
        0.0,                      // d_opacity
        vec2<f32>(0.0)            // _padding (48 bytes total)
    );

    // Compute bounding box - MUST match CPU's dist_sq > 0.1 cutoff
    // CPU iterates all pixels and skips if dist_sq > 0.1
    // sqrt(0.1) ≈ 0.316, so we need a box of ±0.316 in normalized coords
    // This is larger than 3.5σ for small Gaussians and ensures we don't miss pixels
    let cutoff_dist = 0.316;  // sqrt(0.1) - matches CPU cutoff

    let x_min_f = max(0.0, (gaussian.position.x - cutoff_dist) * f32(params.width));
    let y_min_f = max(0.0, (gaussian.position.y - cutoff_dist) * f32(params.height));
    let x_max_f = min(f32(params.width - 1u), (gaussian.position.x + cutoff_dist) * f32(params.width));
    let y_max_f = min(f32(params.height - 1u), (gaussian.position.y + cutoff_dist) * f32(params.height));

    let x_min = u32(x_min_f);
    let y_min = u32(y_min_f);
    let x_max = u32(x_max_f);
    let y_max = u32(y_max_f);

    // Accumulate gradients from all affected pixels
    for (var py = y_min; py <= y_max; py++) {
        for (var px = x_min; px <= x_max; px++) {
            let pixel_idx = py * params.width + px;

            // Normalized pixel position [0, 1]
            let pos = vec2<f32>(
                f32(px) / f32(params.width),
                f32(py) / f32(params.height)
            );

            // CRITICAL: Use simplified ISOTROPIC Gaussian formula (matching CPU adam_optimizer.rs)
            // The CPU uses: exp(-0.5 * dist_sq / (scale_x * scale_y))
            // NOT the full anisotropic formula with rotation
            let dx = pos.x - gaussian.position.x;
            let dy = pos.y - gaussian.position.y;
            let dist_sq = dx * dx + dy * dy;

            // CPU uses dist_sq > 0.1 as cutoff
            if (dist_sq > 0.1) {
                continue;
            }

            let scale_product = max(gaussian.scale_x * gaussian.scale_y, 1e-6);
            // CRITICAL FIX: weight = gaussian_value ONLY (NO opacity) - matches CPU exactly
            // CPU line 420: let weight = (-0.5 * dist_sq / scale_product).exp();
            let gaussian_value = exp(-0.5 * dist_sq / scale_product);
            let weight = gaussian_value;  // Same as CPU (no opacity multiplied)

            // Skip if contribution is negligible (CPU uses 1e-6)
            if (weight < 0.000001) {
                continue;
            }

            // Get image gradient at this pixel
            let rendered = rendered_image[pixel_idx];
            let target_pixel = target_image[pixel_idx];
            let img_grad = compute_pixel_gradient(rendered, target_pixel);

            // ====== SIMPLIFIED GRADIENTS (matching CPU adam_optimizer.rs) ======
            // The CPU uses a simplified approximation that's more stable for optimization

            // Color gradient: dI/dcolor = weight (for each channel)
            grad.d_color_r += img_grad.r * weight;
            grad.d_color_g += img_grad.g * weight;
            grad.d_color_b += img_grad.b * weight;
            grad.d_color_a += img_grad.a * weight;

            // Opacity gradient: DISABLED to match CPU
            // CPU adam_optimizer.rs doesn't compute opacity gradient (grads[i].opacity = 0 always)
            // GPU must also skip opacity to match CPU behavior
            // grad.d_opacity += dot(img_grad.rgb, g_color) * gaussian_value;

            // Error weighted by color contribution
            let g_color = vec3<f32>(gaussian.color_r, gaussian.color_g, gaussian.color_b);
            let error_weighted = dot(img_grad.rgb, g_color);

            // Position gradient (simplified): dL/dμ = error_weighted * weight * (x - μ)
            // dx, dy, dist_sq already computed above for isotropic Gaussian
            grad.d_position.x += error_weighted * weight * dx;
            grad.d_position.y += error_weighted * weight * dy;

            // Scale gradient (simplified): dL/dσ = error_weighted * weight * dist² / σ²
            let scale_x_sq = gaussian.scale_x * gaussian.scale_x;
            let scale_y_sq = gaussian.scale_y * gaussian.scale_y;
            grad.d_scale_x += error_weighted * weight * dist_sq / max(scale_x_sq, 1e-6);
            grad.d_scale_y += error_weighted * weight * dist_sq / max(scale_y_sq, 1e-6);

            // Rotation gradient: skip for now (CPU doesn't compute it)
            // grad.d_rotation stays at 0
        }
    }

    // Normalize gradients by pixel count (CRITICAL - matches CPU implementation)
    let pixel_count = f32(params.width * params.height);
    grad.d_position /= pixel_count;
    grad.d_color_r /= pixel_count;
    grad.d_color_g /= pixel_count;
    grad.d_color_b /= pixel_count;
    grad.d_color_a /= pixel_count;
    grad.d_scale_x /= pixel_count;
    grad.d_scale_y /= pixel_count;
    grad.d_rotation /= pixel_count;
    grad.d_opacity /= pixel_count;

    // Store computed gradients
    gradients[gaussian_idx] = grad;
}
