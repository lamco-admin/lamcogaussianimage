// GPU-based L-BFGS parameter update
// Copyright (c) 2025 Lamco Development
//
// Applies search direction: x = x + step_size * direction
// with parameter clamping.
//
// CRITICAL: Struct layouts must match Rust exactly (48-byte stride for arrays)

// Gaussian parameter layout - must match ShaderGaussian in Rust (48 bytes)
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
    _padding: vec2<u32>,      // 8 bytes @ offset 40 (48 bytes total)
}

// Search direction layout - must match SearchDirection in Rust (48 bytes)
// CRITICAL: vec2<f32> alignment means 8-byte array stride
struct Direction {
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

// Update parameters
struct LbfgsParams {
    step_size: f32,
    num_gaussians: u32,
    position_min: f32,
    position_max: f32,
    color_min: f32,
    color_max: f32,
    scale_min: f32,
    scale_max: f32,
    opacity_min: f32,
    opacity_max: f32,
    _padding: vec2<u32>,      // Pad to 48 bytes
}

// Buffer bindings
@group(0) @binding(0) var<storage, read_write> gaussians: array<Gaussian>;
@group(0) @binding(1) var<storage, read> directions: array<Direction>;
@group(0) @binding(2) var<uniform> params: LbfgsParams;

@compute @workgroup_size(256, 1, 1)
fn lbfgs_update_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.num_gaussians) {
        return;
    }

    // Load current state and direction
    var gaussian = gaussians[idx];
    let dir = directions[idx];

    // Apply update: x = x + step_size * direction
    // with clamping to valid ranges

    // Position update
    gaussian.position = clamp(
        gaussian.position + params.step_size * dir.d_position,
        vec2<f32>(params.position_min),
        vec2<f32>(params.position_max)
    );

    // Scale update
    gaussian.scale_x = clamp(
        gaussian.scale_x + params.step_size * dir.d_scale_x,
        params.scale_min,
        params.scale_max
    );
    gaussian.scale_y = clamp(
        gaussian.scale_y + params.step_size * dir.d_scale_y,
        params.scale_min,
        params.scale_max
    );

    // Rotation update (wrap to [-π, π])
    var new_rotation = gaussian.rotation + params.step_size * dir.d_rotation;
    let pi = 3.14159265358979323846;
    if (new_rotation > pi) {
        new_rotation = new_rotation - 2.0 * pi;
    } else if (new_rotation < -pi) {
        new_rotation = new_rotation + 2.0 * pi;
    }
    gaussian.rotation = new_rotation;

    // Color update
    gaussian.color_r = clamp(
        gaussian.color_r + params.step_size * dir.d_color_r,
        params.color_min,
        params.color_max
    );
    gaussian.color_g = clamp(
        gaussian.color_g + params.step_size * dir.d_color_g,
        params.color_min,
        params.color_max
    );
    gaussian.color_b = clamp(
        gaussian.color_b + params.step_size * dir.d_color_b,
        params.color_min,
        params.color_max
    );
    gaussian.color_a = clamp(
        gaussian.color_a + params.step_size * dir.d_color_a,
        params.color_min,
        params.color_max
    );

    // Opacity update
    gaussian.opacity = clamp(
        gaussian.opacity + params.step_size * dir.d_opacity,
        params.opacity_min,
        params.opacity_max
    );

    // Store updated state
    gaussians[idx] = gaussian;
}
