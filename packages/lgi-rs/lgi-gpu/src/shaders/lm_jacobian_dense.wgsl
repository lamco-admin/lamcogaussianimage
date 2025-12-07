// Full-Dense Levenberg-Marquardt Jacobian computation shader
//
// DIFFERENCE FROM BLOCK-DIAGONAL:
// This shader computes ALL cross-Gaussian J^T J blocks, not just diagonal.
// For each pixel, for ALL Gaussian pairs (gi, gj), we accumulate:
//   J_gi^T * J_gj into the (gi, gj) block of J^T J
//
// This is O(G²) per pixel instead of O(G), but captures all interactions
// between overlapping Gaussians.
//
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

// Minimum Gaussian contribution to consider (for efficiency)
// Increased threshold to reduce numerical noise from tiny accumulations
const MIN_CONTRIBUTION: f32 = 1e-4;

// Minimum squared contribution for cross-Gaussian blocks
// This prevents accumulating noise from distant Gaussian pairs
const MIN_CROSS_CONTRIBUTION: f32 = 1e-6;

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

// Build full Jacobian row for one Gaussian at this pixel
// Returns 10-element array: [d/d_px, d/d_py, d/d_sx, d/d_sy, d/d_rot, d/d_r, d/d_g, d/d_b, d/d_a, d/d_opacity]
// For each color channel (R, G, B)
struct JacobianRow {
    jr: array<f32, 10>,  // Jacobian for R residual
    jg: array<f32, 10>,  // Jacobian for G residual
    jb: array<f32, 10>,  // Jacobian for B residual
}

fn compute_jacobian_row(g: ShaderGaussian, px: f32, py: f32) -> JacobianRow {
    let gauss_val = eval_gaussian(g, px, py);

    var result: JacobianRow;

    // Initialize to zero
    for (var k = 0u; k < 10u; k++) {
        result.jr[k] = 0.0;
        result.jg[k] = 0.0;
        result.jb[k] = 0.0;
    }

    // Skip if Gaussian has negligible contribution
    if (gauss_val < MIN_CONTRIBUTION) {
        return result;
    }

    let shape_grads = grad_gaussian_shape(g, px, py, gauss_val);
    let opacity = g.opacity;
    let color = g.color;

    // Shape gradients multiplied by opacity
    let jac_shape_0 = shape_grads[0] * opacity;
    let jac_shape_1 = shape_grads[1] * opacity;
    let jac_shape_2 = shape_grads[2] * opacity;
    let jac_shape_3 = shape_grads[3] * opacity;
    let jac_shape_4 = shape_grads[4] * opacity;
    let jac_color = gauss_val * opacity;
    let jac_opacity = gauss_val;

    // R channel Jacobians
    result.jr[0] = jac_shape_0 * color.r;  // d/d_px
    result.jr[1] = jac_shape_1 * color.r;  // d/d_py
    result.jr[2] = jac_shape_2 * color.r;  // d/d_sx
    result.jr[3] = jac_shape_3 * color.r;  // d/d_sy
    result.jr[4] = jac_shape_4 * color.r;  // d/d_rot
    result.jr[5] = jac_color;              // d/d_r
    result.jr[6] = 0.0;                    // d/d_g
    result.jr[7] = 0.0;                    // d/d_b
    result.jr[8] = 0.0;                    // d/d_a
    result.jr[9] = jac_opacity * color.r;  // d/d_opacity

    // G channel Jacobians
    result.jg[0] = jac_shape_0 * color.g;
    result.jg[1] = jac_shape_1 * color.g;
    result.jg[2] = jac_shape_2 * color.g;
    result.jg[3] = jac_shape_3 * color.g;
    result.jg[4] = jac_shape_4 * color.g;
    result.jg[5] = 0.0;
    result.jg[6] = jac_color;
    result.jg[7] = 0.0;
    result.jg[8] = 0.0;
    result.jg[9] = jac_opacity * color.g;

    // B channel Jacobians
    result.jb[0] = jac_shape_0 * color.b;
    result.jb[1] = jac_shape_1 * color.b;
    result.jb[2] = jac_shape_2 * color.b;
    result.jb[3] = jac_shape_3 * color.b;
    result.jb[4] = jac_shape_4 * color.b;
    result.jb[5] = 0.0;
    result.jb[6] = 0.0;
    result.jb[7] = jac_color;
    result.jb[8] = 0.0;
    result.jb[9] = jac_opacity * color.b;

    return result;
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

    let n_params = params.num_params;
    let num_g = params.num_gaussians;

    // FULL DENSE: For each Gaussian pair (gi, gj), compute J_gi^T * J_gj
    // This is O(G²) per pixel, but with early exit for negligible contributions!

    for (var gi = 0u; gi < num_g; gi++) {
        let g_i = gaussians[gi];

        // EARLY EXIT: Check if this Gaussian has any contribution at this pixel
        let gauss_val_i = eval_gaussian(g_i, px, py);
        if (gauss_val_i < MIN_CONTRIBUTION) {
            continue;  // Skip this Gaussian entirely - no contribution here
        }

        let jac_i = compute_jacobian_row(g_i, px, py);
        let base_i = gi * PARAMS_PER_GAUSSIAN;

        // Accumulate J^T r for this Gaussian
        // J^T r = sum over channels: J_channel^T * r_channel
        for (var k = 0u; k < PARAMS_PER_GAUSSIAN; k++) {
            let jtr_val = (jac_i.jr[k] * residual.r + jac_i.jg[k] * residual.g + jac_i.jb[k] * residual.b) * FIXED_SCALE;
            atomicAdd(&jtr[base_i + k], bitcast<u32>(i32(jtr_val)));
        }

        // Compute J^T J for pairs (gi, gj) where BOTH have significant contribution
        for (var gj = 0u; gj < num_g; gj++) {
            let g_j = gaussians[gj];

            // EARLY EXIT: Check if this pair has meaningful overlap
            let gauss_val_j = eval_gaussian(g_j, px, py);
            if (gauss_val_j < MIN_CONTRIBUTION) {
                continue;  // Skip - gj has no contribution at this pixel
            }

            // Additional check for cross-Gaussian blocks: require stronger overlap
            // This prevents accumulating tiny numerical noise from distant pairs
            if (gi != gj && gauss_val_i * gauss_val_j < MIN_CROSS_CONTRIBUTION) {
                continue;  // Combined contribution too small
            }

            let jac_j = compute_jacobian_row(g_j, px, py);
            let base_j = gj * PARAMS_PER_GAUSSIAN;

            // Compute outer product: J_gi^T * J_gj
            // Sum of outer products for each color channel: Jr_i^T*Jr_j + Jg_i^T*Jg_j + Jb_i^T*Jb_j
            for (var i = 0u; i < PARAMS_PER_GAUSSIAN; i++) {
                for (var j = 0u; j < PARAMS_PER_GAUSSIAN; j++) {
                    let idx = (base_i + i) * n_params + (base_j + j);
                    let jtj_val = (
                        jac_i.jr[i] * jac_j.jr[j] +
                        jac_i.jg[i] * jac_j.jg[j] +
                        jac_i.jb[i] * jac_j.jb[j]
                    ) * FIXED_SCALE;
                    atomicAdd(&jtj[idx], bitcast<u32>(i32(jtj_val)));
                }
            }
        }
    }
}
