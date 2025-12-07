// GPU-based Adam optimizer update for Gaussian parameters
// Copyright (c) 2025 Lamco Development
//
// Performs in-place Adam updates on all Gaussian parameters.
// This eliminates GPU→CPU→GPU round-trips during optimization.
//
// Algorithm: θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)
// Where:
//   m_t = β1 * m_{t-1} + (1 - β1) * g_t
//   v_t = β2 * v_{t-1} + (1 - β2) * g_t²
//   m̂_t = m_t / (1 - β1^t)
//   v̂_t = v_t / (1 - β2^t)

// Gaussian parameter layout - must match ShaderGaussian in Rust exactly (48 bytes)
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

// Gradient layout - must match GpuGradient in Rust exactly (48 bytes)
// CRITICAL: WGSL pads structs in arrays to alignment of largest field (vec2 = 8 bytes)
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

// Adam moment buffers - must match AdamMoment in Rust exactly (48 bytes)
// CRITICAL: WGSL pads structs in arrays to alignment of largest field (vec2 = 8 bytes)
struct AdamMoment {
    m_position: vec2<f32>,    // 8 bytes @ offset 0
    m_scale_x: f32,           // 4 bytes @ offset 8
    m_scale_y: f32,           // 4 bytes @ offset 12
    m_rotation: f32,          // 4 bytes @ offset 16
    m_color_r: f32,           // 4 bytes @ offset 20
    m_color_g: f32,           // 4 bytes @ offset 24
    m_color_b: f32,           // 4 bytes @ offset 28
    m_color_a: f32,           // 4 bytes @ offset 32
    m_opacity: f32,           // 4 bytes @ offset 36
    _padding: vec2<f32>,      // 8 bytes @ offset 40 (48 bytes total)
}

// Second moment - must match AdamVariance in Rust exactly (48 bytes)
// CRITICAL: WGSL pads structs in arrays to alignment of largest field (vec2 = 8 bytes)
struct AdamVariance {
    v_position: vec2<f32>,    // 8 bytes @ offset 0
    v_scale_x: f32,           // 4 bytes @ offset 8
    v_scale_y: f32,           // 4 bytes @ offset 12
    v_rotation: f32,          // 4 bytes @ offset 16
    v_color_r: f32,           // 4 bytes @ offset 20
    v_color_g: f32,           // 4 bytes @ offset 24
    v_color_b: f32,           // 4 bytes @ offset 28
    v_color_a: f32,           // 4 bytes @ offset 32
    v_opacity: f32,           // 4 bytes @ offset 36
    _padding: vec2<f32>,      // 8 bytes @ offset 40 (48 bytes total)
}

// Optimizer hyperparameters
struct AdamParams {
    // Per-parameter learning rates
    lr_position: f32,
    lr_color: f32,
    lr_scale: f32,
    lr_opacity: f32,
    lr_rotation: f32,

    // Adam hyperparameters
    beta1: f32,           // Default: 0.9
    beta2: f32,           // Default: 0.999
    epsilon: f32,         // Default: 1e-8

    // Bias correction factors (precomputed on CPU)
    bias_correction1: f32,  // 1 - β1^t
    bias_correction2: f32,  // 1 - β2^t

    // Clamp ranges
    position_min: f32,
    position_max: f32,
    color_min: f32,
    color_max: f32,
    scale_min: f32,
    scale_max: f32,
    opacity_min: f32,
    opacity_max: f32,

    num_gaussians: u32,
    _padding: vec3<u32>,
}

// Buffer bindings
@group(0) @binding(0) var<storage, read_write> gaussians: array<Gaussian>;
@group(0) @binding(1) var<storage, read> gradients: array<GaussianGradient>;
@group(0) @binding(2) var<storage, read_write> m: array<AdamMoment>;
@group(0) @binding(3) var<storage, read_write> v: array<AdamVariance>;
@group(0) @binding(4) var<uniform> params: AdamParams;

// Helper: Adam update for a scalar value
fn adam_update_scalar(
    param: f32,
    grad: f32,
    m_old: f32,
    v_old: f32,
    lr: f32,
) -> vec3<f32> {
    // Update first moment
    let m_new = params.beta1 * m_old + (1.0 - params.beta1) * grad;

    // Update second moment
    let v_new = params.beta2 * v_old + (1.0 - params.beta2) * grad * grad;

    // Bias-corrected estimates
    let m_hat = m_new / params.bias_correction1;
    let v_hat = v_new / params.bias_correction2;

    // Adam update
    let update = lr * m_hat / (sqrt(v_hat) + params.epsilon);
    let param_new = param - update;

    // Return: (new_param, new_m, new_v)
    return vec3<f32>(param_new, m_new, v_new);
}

// Helper: Adam update for vec2
fn adam_update_vec2(
    param: vec2<f32>,
    grad: vec2<f32>,
    m_old: vec2<f32>,
    v_old: vec2<f32>,
    lr: f32,
) -> array<vec2<f32>, 3> {
    // Update first moment
    let m_new = params.beta1 * m_old + (1.0 - params.beta1) * grad;

    // Update second moment
    let v_new = params.beta2 * v_old + (1.0 - params.beta2) * grad * grad;

    // Bias-corrected estimates
    let m_hat = m_new / params.bias_correction1;
    let v_hat = v_new / params.bias_correction2;

    // Adam update
    let update = lr * m_hat / (sqrt(v_hat) + vec2<f32>(params.epsilon));
    let param_new = param - update;

    // Return: [new_param, new_m, new_v]
    return array<vec2<f32>, 3>(param_new, m_new, v_new);
}

// Helper: Adam update for vec4 (color)
fn adam_update_vec4(
    param: vec4<f32>,
    grad: vec4<f32>,
    m_old: vec4<f32>,
    v_old: vec4<f32>,
    lr: f32,
) -> array<vec4<f32>, 3> {
    // Update first moment
    let m_new = params.beta1 * m_old + (1.0 - params.beta1) * grad;

    // Update second moment
    let v_new = params.beta2 * v_old + (1.0 - params.beta2) * grad * grad;

    // Bias-corrected estimates
    let m_hat = m_new / params.bias_correction1;
    let v_hat = v_new / params.bias_correction2;

    // Adam update
    let update = lr * m_hat / (sqrt(v_hat) + vec4<f32>(params.epsilon));
    let param_new = param - update;

    return array<vec4<f32>, 3>(param_new, m_new, v_new);
}

@compute @workgroup_size(256, 1, 1)
fn adam_update_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.num_gaussians) {
        return;
    }

    // Load current state
    var gaussian = gaussians[idx];
    let grad = gradients[idx];
    var moment = m[idx];
    var variance = v[idx];

    // === Position update ===
    let pos_result = adam_update_vec2(
        gaussian.position,
        grad.d_position,
        moment.m_position,
        variance.v_position,
        params.lr_position
    );
    gaussian.position = clamp(pos_result[0],
        vec2<f32>(params.position_min),
        vec2<f32>(params.position_max));
    moment.m_position = pos_result[1];
    variance.v_position = pos_result[2];

    // === Color update (all 4 channels) ===
    // Reconstruct vec4s from separate fields
    let g_color = vec4<f32>(gaussian.color_r, gaussian.color_g, gaussian.color_b, gaussian.color_a);
    let d_color = vec4<f32>(grad.d_color_r, grad.d_color_g, grad.d_color_b, grad.d_color_a);
    let m_color = vec4<f32>(moment.m_color_r, moment.m_color_g, moment.m_color_b, moment.m_color_a);
    let v_color = vec4<f32>(variance.v_color_r, variance.v_color_g, variance.v_color_b, variance.v_color_a);

    let color_result = adam_update_vec4(g_color, d_color, m_color, v_color, params.lr_color);
    let new_color = clamp(color_result[0], vec4<f32>(params.color_min), vec4<f32>(params.color_max));

    // Store back to separate fields
    gaussian.color_r = new_color.r;
    gaussian.color_g = new_color.g;
    gaussian.color_b = new_color.b;
    gaussian.color_a = new_color.a;
    moment.m_color_r = color_result[1].r;
    moment.m_color_g = color_result[1].g;
    moment.m_color_b = color_result[1].b;
    moment.m_color_a = color_result[1].a;
    variance.v_color_r = color_result[2].r;
    variance.v_color_g = color_result[2].g;
    variance.v_color_b = color_result[2].b;
    variance.v_color_a = color_result[2].a;

    // === Scale X update ===
    let scale_x_result = adam_update_scalar(
        gaussian.scale_x,
        grad.d_scale_x,
        moment.m_scale_x,
        variance.v_scale_x,
        params.lr_scale
    );
    gaussian.scale_x = clamp(scale_x_result.x, params.scale_min, params.scale_max);
    moment.m_scale_x = scale_x_result.y;
    variance.v_scale_x = scale_x_result.z;

    // === Scale Y update ===
    let scale_y_result = adam_update_scalar(
        gaussian.scale_y,
        grad.d_scale_y,
        moment.m_scale_y,
        variance.v_scale_y,
        params.lr_scale
    );
    gaussian.scale_y = clamp(scale_y_result.x, params.scale_min, params.scale_max);
    moment.m_scale_y = scale_y_result.y;
    variance.v_scale_y = scale_y_result.z;

    // === Rotation update ===
    let rotation_result = adam_update_scalar(
        gaussian.rotation,
        grad.d_rotation,
        moment.m_rotation,
        variance.v_rotation,
        params.lr_rotation
    );
    // Wrap rotation to [-π, π]
    var new_rotation = rotation_result.x;
    let pi = 3.14159265358979323846;
    if (new_rotation > pi) {
        new_rotation = new_rotation - 2.0 * pi;
    } else if (new_rotation < -pi) {
        new_rotation = new_rotation + 2.0 * pi;
    }
    gaussian.rotation = new_rotation;
    moment.m_rotation = rotation_result.y;
    variance.v_rotation = rotation_result.z;

    // === Opacity update ===
    let opacity_result = adam_update_scalar(
        gaussian.opacity,
        grad.d_opacity,
        moment.m_opacity,
        variance.v_opacity,
        params.lr_opacity
    );
    gaussian.opacity = clamp(opacity_result.x, params.opacity_min, params.opacity_max);
    moment.m_opacity = opacity_result.y;
    variance.v_opacity = opacity_result.z;

    // Store updated state
    gaussians[idx] = gaussian;
    m[idx] = moment;
    v[idx] = variance;
}

// Optional: Reset moment buffers (call before starting optimization)
@compute @workgroup_size(256, 1, 1)
fn reset_moments(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.num_gaussians) {
        return;
    }

    m[idx] = AdamMoment(
        vec2<f32>(0.0),           // m_position
        0.0, 0.0, 0.0,            // m_scale_x, m_scale_y, m_rotation
        0.0, 0.0, 0.0, 0.0,       // m_color_r, m_color_g, m_color_b, m_color_a
        0.0,                      // m_opacity
        vec2<f32>(0.0)            // _padding (8 bytes for 48-byte stride)
    );

    v[idx] = AdamVariance(
        vec2<f32>(0.0),           // v_position
        0.0, 0.0, 0.0,            // v_scale_x, v_scale_y, v_rotation
        0.0, 0.0, 0.0, 0.0,       // v_color_r, v_color_g, v_color_b, v_color_a
        0.0,                      // v_opacity
        vec2<f32>(0.0)            // _padding (8 bytes for 48-byte stride)
    );
}
