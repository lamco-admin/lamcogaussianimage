// Sparse Levenberg-Marquardt Jacobian computation shader
//
// DIFFERENCE FROM FULL-DENSE:
// Only computes J_i^T × J_j for Gaussian pairs that actually overlap spatially.
// Overlap is determined by comparing center distance to combined effective radius.
//
// This is O(G×k) per pixel where k is avg neighbors, vs O(G²) for full-dense.
// Should capture meaningful interactions while avoiding ill-conditioning from distant pairs.

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
    overlap_sigma: f32,         // How many sigma to consider overlap (default: 3.0)
    _padding: vec3<f32>,
}

// Parameters per Gaussian: x, y, scale_x, scale_y, rotation, r, g, b, a, opacity
const PARAMS_PER_GAUSSIAN: u32 = 10u;

@group(0) @binding(0) var<storage, read> gaussians: array<ShaderGaussian>;
@group(0) @binding(1) var<storage, read> target_image: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> rendered: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> jtj: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> jtr: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: Params;

// Fixed-point scale for atomic accumulation
const FIXED_SCALE: f32 = 1000000.0;

// Minimum contribution threshold
const MIN_CONTRIBUTION: f32 = 1e-4;

// Check if two Gaussians overlap (within n sigma)
fn gaussians_overlap(gi: u32, gj: u32) -> bool {
    let g_i = gaussians[gi];
    let g_j = gaussians[gj];

    // Distance between centers
    let dx = g_i.position.x - g_j.position.x;
    let dy = g_i.position.y - g_j.position.y;
    let dist_sq = dx * dx + dy * dy;

    // Effective radius of each Gaussian (max of scale_x, scale_y)
    let r_i = max(g_i.scale_x, g_i.scale_y) * params.overlap_sigma;
    let r_j = max(g_j.scale_x, g_j.scale_y) * params.overlap_sigma;

    // Sum of radii squared (for comparison without sqrt)
    let sum_r = r_i + r_j;
    let sum_r_sq = sum_r * sum_r;

    return dist_sq < sum_r_sq;
}

// Evaluate 2D Gaussian at a point
fn eval_gaussian(g: ShaderGaussian, px: f32, py: f32) -> f32 {
    let dx = px - g.position.x;
    let dy = py - g.position.y;

    let cos_r = cos(g.rotation);
    let sin_r = sin(g.rotation);

    let rx = cos_r * dx + sin_r * dy;
    let ry = -sin_r * dx + cos_r * dy;

    let sx = rx / max(g.scale_x, 0.001);
    let sy = ry / max(g.scale_y, 0.001);

    let exponent = -0.5 * (sx * sx + sy * sy);
    return exp(exponent);
}

// Compute gradient of Gaussian value w.r.t. parameters
fn grad_gaussian_shape(g: ShaderGaussian, px: f32, py: f32, gauss_val: f32) -> array<f32, 5> {
    let dx = px - g.position.x;
    let dy = py - g.position.y;

    let cos_r = cos(g.rotation);
    let sin_r = sin(g.rotation);

    let rx = cos_r * dx + sin_r * dy;
    let ry = -sin_r * dx + cos_r * dy;

    let sx = max(g.scale_x, 0.001);
    let sy = max(g.scale_y, 0.001);

    let nx = rx / sx;
    let ny = ry / sy;

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

// Jacobian row for one Gaussian at this pixel
struct JacobianRow {
    jr: array<f32, 10>,
    jg: array<f32, 10>,
    jb: array<f32, 10>,
}

fn compute_jacobian_row(g: ShaderGaussian, px: f32, py: f32) -> JacobianRow {
    let gauss_val = eval_gaussian(g, px, py);

    var result: JacobianRow;
    for (var k = 0u; k < 10u; k++) {
        result.jr[k] = 0.0;
        result.jg[k] = 0.0;
        result.jb[k] = 0.0;
    }

    if (gauss_val < MIN_CONTRIBUTION) {
        return result;
    }

    let shape_grads = grad_gaussian_shape(g, px, py, gauss_val);
    let opacity = g.opacity;
    let color = g.color;

    let jac_shape_0 = shape_grads[0] * opacity;
    let jac_shape_1 = shape_grads[1] * opacity;
    let jac_shape_2 = shape_grads[2] * opacity;
    let jac_shape_3 = shape_grads[3] * opacity;
    let jac_shape_4 = shape_grads[4] * opacity;
    let jac_color = gauss_val * opacity;
    let jac_opacity = gauss_val;

    // R channel
    result.jr[0] = jac_shape_0 * color.r;
    result.jr[1] = jac_shape_1 * color.r;
    result.jr[2] = jac_shape_2 * color.r;
    result.jr[3] = jac_shape_3 * color.r;
    result.jr[4] = jac_shape_4 * color.r;
    result.jr[5] = jac_color;
    result.jr[6] = 0.0;
    result.jr[7] = 0.0;
    result.jr[8] = 0.0;
    result.jr[9] = jac_opacity * color.r;

    // G channel
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

    // B channel
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

    // SPARSE: For each Gaussian pair (gi, gj), only compute if they overlap
    for (var gi = 0u; gi < num_g; gi++) {
        let g_i = gaussians[gi];

        // Check if this Gaussian contributes at this pixel
        let gauss_val_i = eval_gaussian(g_i, px, py);
        if (gauss_val_i < MIN_CONTRIBUTION) {
            continue;
        }

        let jac_i = compute_jacobian_row(g_i, px, py);
        let base_i = gi * PARAMS_PER_GAUSSIAN;

        // Accumulate J^T r for this Gaussian
        for (var k = 0u; k < PARAMS_PER_GAUSSIAN; k++) {
            let jtr_val = (jac_i.jr[k] * residual.r + jac_i.jg[k] * residual.g + jac_i.jb[k] * residual.b) * FIXED_SCALE;
            atomicAdd(&jtr[base_i + k], bitcast<u32>(i32(jtr_val)));
        }

        // Compute J^T J for pairs (gi, gj) where BOTH overlap AND contribute
        for (var gj = 0u; gj < num_g; gj++) {
            // SPARSE CHECK: Skip if Gaussians don't overlap (unless diagonal block)
            if (gi != gj && !gaussians_overlap(gi, gj)) {
                continue;
            }

            let g_j = gaussians[gj];
            let gauss_val_j = eval_gaussian(g_j, px, py);
            if (gauss_val_j < MIN_CONTRIBUTION) {
                continue;
            }

            let jac_j = compute_jacobian_row(g_j, px, py);
            let base_j = gj * PARAMS_PER_GAUSSIAN;

            // Compute outer product: J_gi^T * J_gj
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
