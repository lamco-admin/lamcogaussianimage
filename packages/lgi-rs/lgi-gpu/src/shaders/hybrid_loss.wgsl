// GPU-based hybrid loss computation for Gaussian optimization
// Copyright (c) 2025 Lamco Development
//
// Implements the hybrid loss from the 3D Gaussian Splatting paper:
//   Loss = α·L2 + (1-α)·D-SSIM
//   where D-SSIM = (1 - SSIM) / 2
//
// This shader computes the per-pixel loss gradient for backpropagation.
// The gradient is: dL/dI = α·2(I-T) + (1-α)·(-dSSIM/dI)
//
// SSIM gradient derivation:
//   SSIM(x,y) = L(x,y) · C(x,y) · S(x,y)
//   where L = luminance, C = contrast, S = structure
//   dSSIM/dI involves local window statistics

struct HybridLossParams {
    width: u32,
    height: u32,
    alpha: f32,              // MSE weight (0.84 recommended)
    window_size: u32,        // SSIM window size (typically 11, we use 8)
    c1: f32,                 // SSIM constant: (k1*L)^2, L=1, k1=0.01 → 0.0001
    c2: f32,                 // SSIM constant: (k2*L)^2, L=1, k2=0.03 → 0.0009
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> rendered_image: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> target_image: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> loss_gradient: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> loss_value: array<f32>;  // Single element for reduction
@group(0) @binding(4) var<uniform> params: HybridLossParams;

// Shared memory for workgroup reduction (SSIM window statistics)
var<workgroup> shared_rendered: array<vec3<f32>, 256>;
var<workgroup> shared_target: array<vec3<f32>, 256>;

// Compute MSE loss gradient for a pixel: dL/dI = 2(I - T)
fn compute_mse_gradient(rendered: vec4<f32>, target: vec4<f32>) -> vec4<f32> {
    return 2.0 * (rendered - target);
}

// Compute local SSIM and its gradient for a pixel
// This is a simplified version using a small local window
fn compute_ssim_at_pixel(
    px: u32,
    py: u32,
    channel: u32,
) -> vec2<f32> {  // Returns (ssim_value, gradient)
    let half_w = i32(params.window_size / 2u);
    let width = i32(params.width);
    let height = i32(params.height);

    var sum_r = 0.0;
    var sum_t = 0.0;
    var sum_r2 = 0.0;
    var sum_t2 = 0.0;
    var sum_rt = 0.0;
    var count = 0.0;

    // Collect window statistics
    for (var dy = -half_w; dy <= half_w; dy++) {
        for (var dx = -half_w; dx <= half_w; dx++) {
            let nx = i32(px) + dx;
            let ny = i32(py) + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                let idx = u32(ny * width + nx);
                var r: f32;
                var t: f32;

                // Extract channel
                if (channel == 0u) {
                    r = rendered_image[idx].x;
                    t = target_image[idx].x;
                } else if (channel == 1u) {
                    r = rendered_image[idx].y;
                    t = target_image[idx].y;
                } else {
                    r = rendered_image[idx].z;
                    t = target_image[idx].z;
                }

                sum_r += r;
                sum_t += t;
                sum_r2 += r * r;
                sum_t2 += t * t;
                sum_rt += r * t;
                count += 1.0;
            }
        }
    }

    if (count < 1.0) {
        return vec2<f32>(1.0, 0.0);  // Edge case: no valid pixels
    }

    // Compute statistics
    let mean_r = sum_r / count;
    let mean_t = sum_t / count;
    let var_r = (sum_r2 / count) - (mean_r * mean_r);
    let var_t = (sum_t2 / count) - (mean_t * mean_t);
    let cov_rt = (sum_rt / count) - (mean_r * mean_t);

    // SSIM formula
    let numerator = (2.0 * mean_r * mean_t + params.c1) * (2.0 * cov_rt + params.c2);
    let denominator = (mean_r * mean_r + mean_t * mean_t + params.c1) * (var_r + var_t + params.c2);

    let ssim = numerator / max(denominator, 0.0001);

    // SSIM gradient w.r.t. rendered pixel at (px, py)
    // Using chain rule: dSSIM/dI = dSSIM/dμr · dμr/dI + dSSIM/dσr² · dσr²/dI + dSSIM/dσrt · dσrt/dI
    //
    // dμr/dI = 1/N (for pixel in window)
    // dσr²/dI = 2(I - μr)/N
    // dσrt/dI = (T - μt)/N

    let n = count;

    // Gradient components (simplified analytical form)
    let a = 2.0 * mean_t + params.c1 / max(mean_r, 0.0001);
    let b = 2.0 * cov_rt + params.c2;
    let c = mean_r * mean_r + mean_t * mean_t + params.c1;
    let d = var_r + var_t + params.c2;

    let pixel_idx = py * params.width + px;
    var current_pixel: f32;
    if (channel == 0u) {
        current_pixel = rendered_image[pixel_idx].x;
    } else if (channel == 1u) {
        current_pixel = rendered_image[pixel_idx].y;
    } else {
        current_pixel = rendered_image[pixel_idx].z;
    }

    var target_pixel: f32;
    if (channel == 0u) {
        target_pixel = target_image[pixel_idx].x;
    } else if (channel == 1u) {
        target_pixel = target_image[pixel_idx].y;
    } else {
        target_pixel = target_image[pixel_idx].z;
    }

    // Gradient approximation using quotient rule
    // d(num/den)/dI = (den·dnum/dI - num·dden/dI) / den²
    let dnum_dI = (2.0 * mean_t / n) * (2.0 * cov_rt + params.c2) +
                  (2.0 * mean_r * mean_t + params.c1) * (2.0 * (target_pixel - mean_t) / n);

    let dden_dI = (2.0 * mean_r / n) * (var_r + var_t + params.c2) +
                  (mean_r * mean_r + mean_t * mean_t + params.c1) * (2.0 * (current_pixel - mean_r) / n);

    let grad = (denominator * dnum_dI - numerator * dden_dI) / max(denominator * denominator, 0.0001);

    return vec2<f32>(ssim, grad);
}

@compute @workgroup_size(16, 16, 1)
fn compute_hybrid_gradient(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let px = global_id.x;
    let py = global_id.y;

    if (px >= params.width || py >= params.height) {
        return;
    }

    let pixel_idx = py * params.width + px;
    let rendered = rendered_image[pixel_idx];
    let target = target_image[pixel_idx];

    // MSE gradient: dL_mse/dI = 2(I - T)
    let mse_grad = compute_mse_gradient(rendered, target);

    // SSIM gradient for each channel
    let ssim_r = compute_ssim_at_pixel(px, py, 0u);
    let ssim_g = compute_ssim_at_pixel(px, py, 1u);
    let ssim_b = compute_ssim_at_pixel(px, py, 2u);

    // D-SSIM gradient: dL_dssim/dI = -dSSIM/dI (since D-SSIM = 1 - SSIM)
    let dssim_grad = vec4<f32>(-ssim_r.y, -ssim_g.y, -ssim_b.y, 0.0);

    // Hybrid gradient: α·MSE_grad + (1-α)·DSSIM_grad
    let hybrid_grad = params.alpha * mse_grad + (1.0 - params.alpha) * dssim_grad;

    loss_gradient[pixel_idx] = hybrid_grad;
}

// Simplified version: MSE-only gradient (faster, for comparison)
@compute @workgroup_size(16, 16, 1)
fn compute_mse_only_gradient(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let px = global_id.x;
    let py = global_id.y;

    if (px >= params.width || py >= params.height) {
        return;
    }

    let pixel_idx = py * params.width + px;
    let rendered = rendered_image[pixel_idx];
    let target = target_image[pixel_idx];

    // MSE gradient: dL/dI = 2(I - T)
    loss_gradient[pixel_idx] = 2.0 * (rendered - target);
}

// Compute scalar loss value (reduction to single value)
// This is separate from gradient computation for efficiency
@compute @workgroup_size(256, 1, 1)
fn compute_loss_value(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let idx = global_id.x;
    let local_idx = local_id.x;

    var mse: f32 = 0.0;
    var ssim_sum: f32 = 0.0;

    if (idx < params.width * params.height) {
        let rendered = rendered_image[idx];
        let target = target_image[idx];

        // MSE per pixel
        let diff = rendered - target;
        mse = (diff.x * diff.x + diff.y * diff.y + diff.z * diff.z) / 3.0;

        // For SSIM, we'd need a separate pass with window operations
        // Here we just track MSE for now
    }

    // Use workgroup reduction for MSE sum
    // (Full reduction would require multiple passes or atomics)
    // For simplicity, store per-pixel loss and reduce on CPU

    if (idx < params.width * params.height) {
        // Store per-pixel MSE for CPU reduction
        // A proper implementation would use atomic add
        loss_value[idx] = mse;
    }
}
