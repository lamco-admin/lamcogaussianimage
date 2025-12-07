// Levenberg-Marquardt Jacobian computation shader
//
// For each pixel, computes:
// 1. Residual r = rendered - target (3 values: R, G, B)
// 2. Jacobian row J[pixel, :] = d(residual)/d(params) for all params
// 3. Accumulates into J^T J and J^T r using atomics
//
// This computes the FULL Jacobian implicitly by accumulating outer products.
// NOTE: WGSL doesn't allow passing storage pointers to functions, so atomics are inlined.

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

// Parameters per Gaussian: x, y, scale_x, scale_y, rotation, r, g, b, a, opacity
const PARAMS_PER_GAUSSIAN: u32 = 10u;

@group(0) @binding(0) var<storage, read> gaussians: array<ShaderGaussian>;
@group(0) @binding(1) var<storage, read> target_image: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> rendered: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> jtj: array<atomic<u32>>;  // Stored as fixed-point
@group(0) @binding(4) var<storage, read_write> jtr: array<atomic<u32>>;  // Stored as fixed-point
@group(0) @binding(5) var<uniform> params: Params;

// Fixed-point scale for atomic accumulation (allows ~6 decimal places)
const FIXED_SCALE: f32 = 1000000.0;

// Evaluate 2D Gaussian at a point
fn eval_gaussian(g: ShaderGaussian, px: f32, py: f32) -> f32 {
    let dx = px - g.position.x;
    let dy = py - g.position.y;

    // Compute rotation
    let cos_r = cos(g.rotation);
    let sin_r = sin(g.rotation);

    // Rotate delta
    let rx = cos_r * dx + sin_r * dy;
    let ry = -sin_r * dx + cos_r * dy;

    // Scale
    let sx = rx / max(g.scale_x, 0.001);
    let sy = ry / max(g.scale_y, 0.001);

    // Gaussian
    let exponent = -0.5 * (sx * sx + sy * sy);
    return exp(exponent);
}

// Compute gradient of Gaussian value w.r.t. parameters
// Returns: [d/d_px, d/d_py, d/d_sx, d/d_sy, d/d_rot]
fn grad_gaussian_shape(g: ShaderGaussian, px: f32, py: f32, gauss_val: f32) -> array<f32, 5> {
    let dx = px - g.position.x;
    let dy = py - g.position.y;

    let cos_r = cos(g.rotation);
    let sin_r = sin(g.rotation);

    let rx = cos_r * dx + sin_r * dy;
    let ry = -sin_r * dx + cos_r * dy;

    let sx = max(g.scale_x, 0.001);
    let sy = max(g.scale_y, 0.001);

    let nx = rx / sx;  // normalized x
    let ny = ry / sy;  // normalized y

    // d/d_position_x
    let d_exp_d_px = -nx * (-cos_r / sx) + (-ny) * (sin_r / sy);
    let d_gauss_d_px = gauss_val * d_exp_d_px;

    // d/d_position_y
    let d_exp_d_py = -nx * (-sin_r / sx) + (-ny) * (-cos_r / sy);
    let d_gauss_d_py = gauss_val * d_exp_d_py;

    // d/d_scale_x
    let d_exp_d_sx = -nx * (-rx / (sx * sx));
    let d_gauss_d_sx = gauss_val * d_exp_d_sx;

    // d/d_scale_y
    let d_exp_d_sy = -ny * (-ry / (sy * sy));
    let d_gauss_d_sy = gauss_val * d_exp_d_sy;

    // d/d_rotation
    let d_rx_d_rot = -sin_r * dx + cos_r * dy;
    let d_ry_d_rot = -cos_r * dx - sin_r * dy;
    let d_exp_d_rot = -nx * (d_rx_d_rot / sx) + (-ny) * (d_ry_d_rot / sy);
    let d_gauss_d_rot = gauss_val * d_exp_d_rot;

    return array<f32, 5>(d_gauss_d_px, d_gauss_d_py, d_gauss_d_sx, d_gauss_d_sy, d_gauss_d_rot);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let pixel_idx = y * params.width + x;
    let px = (f32(x) + 0.5) / f32(params.width);
    let py = (f32(y) + 0.5) / f32(params.height);

    // Get residual (rendered - target)
    let r_color = rendered[pixel_idx];
    let t_color = target_image[pixel_idx];
    let residual = vec3<f32>(r_color.r - t_color.r, r_color.g - t_color.g, r_color.b - t_color.b);

    // For each Gaussian, compute contribution to this pixel's Jacobian row
    for (var g_idx = 0u; g_idx < params.num_gaussians; g_idx++) {
        let g = gaussians[g_idx];
        let gauss_val = eval_gaussian(g, px, py);

        // Skip if Gaussian has negligible contribution
        if (gauss_val < 1e-6) {
            continue;
        }

        let shape_grads = grad_gaussian_shape(g, px, py, gauss_val);
        let base_param = g_idx * PARAMS_PER_GAUSSIAN;

        let opacity = g.opacity;
        let color = g.color;

        // Compute Jacobian entries for each color channel
        // Shape gradients multiplied by opacity
        let jac_shape_0 = shape_grads[0] * opacity;
        let jac_shape_1 = shape_grads[1] * opacity;
        let jac_shape_2 = shape_grads[2] * opacity;
        let jac_shape_3 = shape_grads[3] * opacity;
        let jac_shape_4 = shape_grads[4] * opacity;
        let jac_color = gauss_val * opacity;
        let jac_opacity = gauss_val;

        // R channel Jacobians
        let jr_px = jac_shape_0 * color.r;
        let jr_py = jac_shape_1 * color.r;
        let jr_sx = jac_shape_2 * color.r;
        let jr_sy = jac_shape_3 * color.r;
        let jr_rot = jac_shape_4 * color.r;
        let jr_r = jac_color;
        let jr_opacity = jac_opacity * color.r;

        // G channel Jacobians
        let jg_px = jac_shape_0 * color.g;
        let jg_py = jac_shape_1 * color.g;
        let jg_sx = jac_shape_2 * color.g;
        let jg_sy = jac_shape_3 * color.g;
        let jg_rot = jac_shape_4 * color.g;
        let jg_g = jac_color;
        let jg_opacity = jac_opacity * color.g;

        // B channel Jacobians
        let jb_px = jac_shape_0 * color.b;
        let jb_py = jac_shape_1 * color.b;
        let jb_sx = jac_shape_2 * color.b;
        let jb_sy = jac_shape_3 * color.b;
        let jb_rot = jac_shape_4 * color.b;
        let jb_b = jac_color;
        let jb_opacity = jac_opacity * color.b;

        let r_residual = residual.r;
        let g_residual = residual.g;
        let b_residual = residual.b;

        // Accumulate J^T r (inline atomic adds)
        // Position x
        let jtr_val_0 = (jr_px * r_residual + jg_px * g_residual + jb_px * b_residual) * FIXED_SCALE;
        atomicAdd(&jtr[base_param + 0u], bitcast<u32>(i32(jtr_val_0)));

        // Position y
        let jtr_val_1 = (jr_py * r_residual + jg_py * g_residual + jb_py * b_residual) * FIXED_SCALE;
        atomicAdd(&jtr[base_param + 1u], bitcast<u32>(i32(jtr_val_1)));

        // Scale x
        let jtr_val_2 = (jr_sx * r_residual + jg_sx * g_residual + jb_sx * b_residual) * FIXED_SCALE;
        atomicAdd(&jtr[base_param + 2u], bitcast<u32>(i32(jtr_val_2)));

        // Scale y
        let jtr_val_3 = (jr_sy * r_residual + jg_sy * g_residual + jb_sy * b_residual) * FIXED_SCALE;
        atomicAdd(&jtr[base_param + 3u], bitcast<u32>(i32(jtr_val_3)));

        // Rotation
        let jtr_val_4 = (jr_rot * r_residual + jg_rot * g_residual + jb_rot * b_residual) * FIXED_SCALE;
        atomicAdd(&jtr[base_param + 4u], bitcast<u32>(i32(jtr_val_4)));

        // Color R
        let jtr_val_5 = jr_r * r_residual * FIXED_SCALE;
        atomicAdd(&jtr[base_param + 5u], bitcast<u32>(i32(jtr_val_5)));

        // Color G
        let jtr_val_6 = jg_g * g_residual * FIXED_SCALE;
        atomicAdd(&jtr[base_param + 6u], bitcast<u32>(i32(jtr_val_6)));

        // Color B
        let jtr_val_7 = jb_b * b_residual * FIXED_SCALE;
        atomicAdd(&jtr[base_param + 7u], bitcast<u32>(i32(jtr_val_7)));

        // Color A (unused in simplified model)
        // jtr[base_param + 8] stays 0

        // Opacity
        let jtr_val_9 = (jr_opacity * r_residual + jg_opacity * g_residual + jb_opacity * b_residual) * FIXED_SCALE;
        atomicAdd(&jtr[base_param + 9u], bitcast<u32>(i32(jtr_val_9)));

        // Accumulate J^T J (block diagonal approximation - only within this Gaussian)
        // IMPORTANT: J^T J = Jr^T*Jr + Jg^T*Jg + Jb^T*Jb (sum of outer products)
        // NOT: (Jr+Jg+Jb)^T * (Jr+Jg+Jb) (outer product of sums) - that's wrong!

        // Build Jacobian vectors for each color channel
        var jr = array<f32, 10>();
        jr[0] = jr_px; jr[1] = jr_py; jr[2] = jr_sx; jr[3] = jr_sy; jr[4] = jr_rot;
        jr[5] = jr_r; jr[6] = 0.0; jr[7] = 0.0; jr[8] = 0.0; jr[9] = jr_opacity;

        var jg = array<f32, 10>();
        jg[0] = jg_px; jg[1] = jg_py; jg[2] = jg_sx; jg[3] = jg_sy; jg[4] = jg_rot;
        jg[5] = 0.0; jg[6] = jg_g; jg[7] = 0.0; jg[8] = 0.0; jg[9] = jg_opacity;

        var jb = array<f32, 10>();
        jb[0] = jb_px; jb[1] = jb_py; jb[2] = jb_sx; jb[3] = jb_sy; jb[4] = jb_rot;
        jb[5] = 0.0; jb[6] = 0.0; jb[7] = jb_b; jb[8] = 0.0; jb[9] = jb_opacity;

        let n_params = params.num_params;
        for (var i = 0u; i < PARAMS_PER_GAUSSIAN; i++) {
            for (var j = 0u; j < PARAMS_PER_GAUSSIAN; j++) {
                let idx = (base_param + i) * n_params + (base_param + j);
                // Sum of outer products: Jr^T*Jr + Jg^T*Jg + Jb^T*Jb
                let jtj_val = (jr[i] * jr[j] + jg[i] * jg[j] + jb[i] * jb[j]) * FIXED_SCALE;
                atomicAdd(&jtj[idx], bitcast<u32>(i32(jtj_val)));
            }
        }
    }
}
