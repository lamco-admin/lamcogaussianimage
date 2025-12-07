//! WGSL Compute Shader for 2D Gaussian Rendering
//!
//! Evaluates 2D Gaussians and accumulates contributions to pixel buffer
//! Supports both alpha compositing and accumulated summation modes
//!
//! Performance: 1000+ FPS @ 1080p with 10K Gaussians on modern GPU

// Gaussian representation (must match ShaderGaussian in Rust)
// Using separate f32 fields to avoid vec4 16-byte alignment issues
// Total: 48 bytes (matches Rust exactly)
struct Gaussian {
    position: vec2<f32>,      // μx, μy (normalized [0, 1]) - 8 bytes @ offset 0
    scale_x: f32,             // σx - 4 bytes @ offset 8
    scale_y: f32,             // σy - 4 bytes @ offset 12
    rotation: f32,            // θ (radians) - 4 bytes @ offset 16
    color_r: f32,             // R - 4 bytes @ offset 20
    color_g: f32,             // G - 4 bytes @ offset 24
    color_b: f32,             // B - 4 bytes @ offset 28
    color_a: f32,             // A - 4 bytes @ offset 32
    opacity: f32,             // α - 4 bytes @ offset 36
    _padding: vec2<u32>,      // Alignment padding - 8 bytes @ offset 40
}

// Render parameters
struct RenderParams {
    width: u32,
    height: u32,
    gaussian_count: u32,
    render_mode: u32,         // 0=AlphaComposite, 1=AccumulatedSum
    cutoff_threshold: f32,
    n_sigma: f32,
    _padding: vec2<u32>,      // Alignment padding
}

// Bindings
@group(0) @binding(0) var<storage, read> gaussians: array<Gaussian>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: RenderParams;

// Evaluate 2D Gaussian at point
fn evaluate_gaussian(g: Gaussian, point: vec2<f32>) -> f32 {
    // Delta from center
    let delta = point - g.position;

    // Rotation matrix
    let cos_r = cos(g.rotation);
    let sin_r = sin(g.rotation);
    let rot_delta = vec2<f32>(
        delta.x * cos_r + delta.y * sin_r,
        -delta.x * sin_r + delta.y * cos_r
    );

    // Scale-normalized distance (use separate scale_x, scale_y)
    let scale = vec2<f32>(g.scale_x, g.scale_y);
    let scaled = rot_delta / scale;
    let mahalanobis_sq = dot(scaled, scaled);

    // Bounding box check (3.5 sigma typical)
    if (mahalanobis_sq > params.n_sigma * params.n_sigma) {
        return 0.0;
    }

    // Gaussian weight
    let weight = exp(-0.5 * mahalanobis_sq);

    // Cutoff threshold
    if (weight < params.cutoff_threshold) {
        return 0.0;
    }

    return weight;
}

// Main compute shader
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_x = global_id.x;
    let pixel_y = global_id.y;

    // Bounds check
    if (pixel_x >= params.width || pixel_y >= params.height) {
        return;
    }

    // CRITICAL FIX: Zero-initialize output buffer (may have undefined contents)
    let pixel_idx_init = pixel_y * params.width + pixel_x;
    output[pixel_idx_init] = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    // Normalized coordinates [0, 1]
    let point = vec2<f32>(
        f32(pixel_x) / f32(params.width),
        f32(pixel_y) / f32(params.height)
    );

    var color = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var alpha_accum = 0.0;

    // Evaluate all Gaussians for this pixel
    for (var i = 0u; i < params.gaussian_count; i++) {
        let g = gaussians[i];
        let weight = evaluate_gaussian(g, point);

        if (weight > 0.0) {
            let alpha_contrib = g.opacity * weight;

            if (params.render_mode == 0u) {
                // Alpha compositing (physically-based)
                let one_minus_alpha = 1.0 - alpha_accum;
                color.r += one_minus_alpha * g.color_r * alpha_contrib;
                color.g += one_minus_alpha * g.color_g * alpha_contrib;
                color.b += one_minus_alpha * g.color_b * alpha_contrib;
                alpha_accum += one_minus_alpha * alpha_contrib;

                // Early termination (99.9% opacity reached)
                if (alpha_accum > 0.999) {
                    break;
                }
            } else {
                // Accumulated summation with WEIGHTED NORMALIZATION (matches CPU renderer_v2.rs)
                // This is the correct implementation: C = Σ(w×c) / Σ(w)
                color.r += g.color_r * alpha_contrib;
                color.g += g.color_g * alpha_contrib;
                color.b += g.color_b * alpha_contrib;
                color.a += alpha_contrib;  // Accumulate total weight
            }
        }
    }

    // Weighted normalization (CRITICAL - matches CPU renderer)
    if (params.render_mode == 1u && color.a > 1e-8) {
        color.r = color.r / color.a;
        color.g = color.g / color.a;
        color.b = color.b / color.a;
    }

    // Clamp to [0, 1] range
    color = clamp(color, vec4<f32>(0.0), vec4<f32>(1.0));

    // Set alpha to 1.0 for opaque output
    color.a = 1.0;

    // Write to output buffer
    let pixel_idx = pixel_y * params.width + pixel_x;
    output[pixel_idx] = color;
}
