// Levenberg-Marquardt parameter update shader
//
// Applies the computed delta to Gaussian parameters with clamping

struct ShaderGaussian {
    position: vec2<f32>,        // 8 bytes @ offset 0
    scale_x: f32,               // 4 bytes @ offset 8
    scale_y: f32,               // 4 bytes @ offset 12
    rotation: f32,              // 4 bytes @ offset 16
    color: vec4<f32>,           // 16 bytes @ offset 20
    opacity: f32,               // 4 bytes @ offset 36
    _padding: vec2<f32>,        // 8 bytes @ offset 40 (48 bytes total)
}

struct Params {
    width: u32,
    height: u32,
    num_gaussians: u32,
    num_params: u32,
}

const PARAMS_PER_GAUSSIAN: u32 = 10u;

@group(0) @binding(0) var<storage, read_write> gaussians: array<ShaderGaussian>;
@group(0) @binding(1) var<storage, read> delta: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let g_idx = global_id.x;

    if (g_idx >= params.num_gaussians) {
        return;
    }

    let base = g_idx * PARAMS_PER_GAUSSIAN;

    // Read current Gaussian
    var g = gaussians[g_idx];

    // Apply deltas with clamping
    // Order: x, y, scale_x, scale_y, rotation, r, g, b, a, opacity

    // Position (clamp to [0, 1])
    g.position.x = clamp(g.position.x + delta[base + 0u], 0.0, 1.0);
    g.position.y = clamp(g.position.y + delta[base + 1u], 0.0, 1.0);

    // Scale (clamp to [0.001, 0.5])
    g.scale_x = clamp(g.scale_x + delta[base + 2u], 0.001, 0.5);
    g.scale_y = clamp(g.scale_y + delta[base + 3u], 0.001, 0.5);

    // Rotation (wrap to [-pi, pi])
    g.rotation = g.rotation + delta[base + 4u];
    // Simple wrap
    if (g.rotation > 3.14159265359) {
        g.rotation -= 6.28318530718;
    } else if (g.rotation < -3.14159265359) {
        g.rotation += 6.28318530718;
    }

    // Color (clamp to [0, 1])
    g.color.r = clamp(g.color.r + delta[base + 5u], 0.0, 1.0);
    g.color.g = clamp(g.color.g + delta[base + 6u], 0.0, 1.0);
    g.color.b = clamp(g.color.b + delta[base + 7u], 0.0, 1.0);
    g.color.a = clamp(g.color.a + delta[base + 8u], 0.0, 1.0);

    // Opacity (clamp to [0.01, 1.0])
    g.opacity = clamp(g.opacity + delta[base + 9u], 0.01, 1.0);

    // Write back
    gaussians[g_idx] = g;
}
