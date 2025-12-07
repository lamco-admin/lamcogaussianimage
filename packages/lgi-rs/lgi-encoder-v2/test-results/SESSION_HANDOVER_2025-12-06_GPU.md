# Session Handover: 2025-12-06 GPU Optimizer Development

## Executive Summary

This session focused on **GPU-accelerated optimization** for the LGI encoder. We created a complete GPU Adam optimizer implementation with hybrid MS-SSIM loss, but discovered the NVIDIA driver needs rebuilding after a kernel upgrade.

---

## GPU Optimizer Implementation (Complete)

### New Files Created

#### 1. `lgi-gpu/src/shaders/adam_update.wgsl` (NEW - 150 lines)

GPU Adam optimizer compute shader with:
- Per-parameter learning rates (position, color, scale, opacity, rotation)
- Bias-corrected moment updates
- In-place parameter updates with clamping
- Workgroup size 256x1x1 for per-Gaussian parallelism

```wgsl
struct AdamParams {
    lr_position: f32, lr_color: f32, lr_scale: f32, lr_opacity: f32, lr_rotation: f32,
    beta1: f32, beta2: f32, epsilon: f32,
    bias_correction1: f32, bias_correction2: f32,
    // clamp ranges...
}

@compute @workgroup_size(256, 1, 1)
fn adam_update_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Updates Gaussians in-place using Adam with per-parameter LRs
}
```

#### 2. `lgi-gpu/src/shaders/hybrid_loss.wgsl` (NEW - 244 lines)

GPU hybrid loss computation (MS-SSIM + MSE):
- Per-pixel SSIM gradient computation using local window statistics
- Hybrid gradient: `α·MSE_grad + (1-α)·DSSIM_grad`
- Optimized entry point for MSE-only gradient
- Workgroup size 16x16x1 for per-pixel parallelism

```wgsl
@compute @workgroup_size(16, 16, 1)
fn compute_hybrid_gradient(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // SSIM gradient for each channel
    let ssim_r = compute_ssim_at_pixel(px, py, 0u);
    // Combine: α·MSE + (1-α)·DSSIM
    let hybrid_grad = params.alpha * mse_grad + (1.0 - params.alpha) * dssim_grad;
}
```

#### 3. `lgi-gpu/src/adam_optimizer.rs` (NEW - 800+ lines)

Complete Rust wrapper for GPU Adam optimizer:
- `GpuAdamOptimizer` struct with wgpu pipelines
- `ShaderGaussian`, `AdamMoment`, `AdamVariance` GPU buffer types
- Full optimization loop staying on GPU (no CPU round-trips)
- Configurable via `GpuLearningRates`

```rust
pub struct GpuAdamOptimizer {
    render_pipeline: wgpu::ComputePipeline,
    gradient_pipeline: wgpu::ComputePipeline,
    adam_pipeline: wgpu::ComputePipeline,
    reset_pipeline: wgpu::ComputePipeline,
    // ...
}

impl GpuAdamOptimizer {
    pub fn new(device: &Device) -> Result<Self> { /* Creates pipelines */ }

    pub fn optimize(
        &self, device: &Device, queue: &Queue,
        gaussians: &mut [Gaussian2D<f32, Euler<f32>>],
        target: &ImageBuffer<f32>
    ) -> Result<f32> {
        // Full GPU optimization loop - no CPU round-trips per iteration
    }
}
```

#### 4. `lgi-encoder-v2/examples/bench_gpu_vs_cpu_adam.rs` (NEW)

Benchmark comparing GPU vs CPU Adam:
- Same initialization for fair comparison
- Measures wall-clock time and iterations/second
- Calculates speedup factor

---

## WGSL Struct Alignment Issue (Fixed)

Encountered buffer size mismatch during benchmark. Root cause: **WGSL vec3 alignment**.

### Problem
```
Buffer is bound with size 88 where the shader expects 96 in group[0] compact index 4
```

### Cause
In WGSL, `vec3<u32>` has 16-byte alignment. The Rust struct had:
```rust
pub num_gaussians: u32,    // offset 72-76
pub _padding: [u32; 3],    // offset 76-88 (Rust: 4-byte aligned)
```

But WGSL expected:
```wgsl
num_gaussians: u32,        // offset 72-76
_padding: vec3<u32>,       // offset 80-92 (WGSL: 16-byte aligned)
// struct padded to 96 bytes
```

### Fix
Added explicit padding in Rust to match WGSL layout:
```rust
pub num_gaussians: u32,        // offset 72-76
pub _pad1: u32,                // offset 76-80 (align to 16 bytes)
pub _padding: [u32; 3],        // offset 80-92
pub _pad2: u32,                // offset 92-96 (round struct to 16 bytes)
```

---

## GPU Driver Issue (Secure Boot Blocking)

### Progress Made
1. ✅ Removed old Debian nvidia packages (550.163.01)
2. ✅ Installed kernel headers for 6.12.57
3. ✅ Downloaded NVIDIA 580.82.09 driver
4. ✅ Driver compiled successfully with DKMS
5. ❌ Module load blocked by Secure Boot

### Current State
```bash
$ mokutil --sb-state
SecureBoot enabled

$ dkms status
nvidia/580.82.09, 6.12.57+deb13-amd64, x86_64: installed
```

The nvidia.ko module is compiled and installed, but the kernel refuses to load it because Secure Boot requires signed modules.

### Driver Installed Successfully

The NVIDIA 580.82.09 driver is fully installed:
```bash
$ ls -la /usr/bin/nvidia-smi
-rwxr-xr-x 1 root root 1218080 Dec  6 22:43 /usr/bin/nvidia-smi

$ ls /lib/modules/6.12.57+deb13-amd64/updates/dkms/
nvidia.ko  nvidia-drm.ko  nvidia-modeset.ko  nvidia-peermem.ko  nvidia-uvm.ko
```

### Final Step: Disable Secure Boot in Proxmox

**From Proxmox Web UI:**
1. Select the VM
2. Go to Hardware → EFI Disk
3. Click Edit → Uncheck "Pre-Enroll Keys"
4. Reboot the VM

**Or via CLI on Proxmox host:**
```bash
qm set <VMID> -efidisk0 <storage>:1,pre-enrolled-keys=0
```

After rebooting with Secure Boot disabled:
```bash
nvidia-smi
```

The modules will load automatically after Secure Boot is disabled

---

## Architecture Notes

### GPU Optimization Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    GPU Adam Optimizer                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Upload initial Gaussians to GPU                         │
│     ShaderGaussian[] → gaussians_buffer                     │
│                                                              │
│  2. For each iteration:                                      │
│     ┌─────────────────────────────────────────────┐         │
│     │ Render Pass (gaussian_render.wgsl)          │         │
│     │ gaussians → rendered_image                  │         │
│     └─────────────────────────────────────────────┘         │
│                       ↓                                      │
│     ┌─────────────────────────────────────────────┐         │
│     │ Gradient Pass (hybrid_loss.wgsl)            │         │
│     │ rendered, target → per-pixel gradient       │         │
│     └─────────────────────────────────────────────┘         │
│                       ↓                                      │
│     ┌─────────────────────────────────────────────┐         │
│     │ Adam Update Pass (adam_update.wgsl)         │         │
│     │ gradients, moments → updated Gaussians      │         │
│     └─────────────────────────────────────────────┘         │
│                                                              │
│  3. Download final Gaussians from GPU                        │
│     gaussians_buffer → Vec<Gaussian2D>                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Key Design: Zero CPU Round-Trips

The critical optimization is keeping data on GPU throughout the iteration loop:
- CPU only uploads once (initial Gaussians, target image)
- CPU only downloads once (final Gaussians)
- All intermediate data (moments, gradients, rendered images) stays on GPU

This eliminates the GPU→CPU→GPU memory transfer bottleneck that would otherwise dominate timing.

---

## Files Modified

| File | Change |
|------|--------|
| `lgi-gpu/src/lib.rs` | Added `pub mod adam_optimizer; pub use adam_optimizer::*;` |
| `lgi-encoder-v2/Cargo.toml` | Added `wgpu = "27"` dev-dependency |

---

## Pending Tasks

| Task | Status | Notes |
|------|--------|-------|
| Rebuild NVIDIA kernel module | BLOCKING | Kernel 6.12.48 → 6.12.57 mismatch |
| Run GPU vs CPU benchmark | Pending | Waiting on driver fix |
| GPU L-BFGS optimizer | Future | Use similar pattern |
| GPU L-M optimizer | Future | Need sparse Jacobian shader |

---

## Continuation Prompt

```
I'm continuing GPU optimizer development for LGI. The NVIDIA 580.82.09 driver is installed but Secure Boot needs to be disabled in Proxmox.

First, check if nvidia-smi works:
nvidia-smi

If GPU is working:
- Run: cargo run --release --example bench_gpu_vs_cpu_adam -p lgi-encoder-v2
- Compare speedup vs CPU Adam
- Tune workgroup sizes if needed

If GPU still not working:
- Secure Boot may still be enabled - disable it in Proxmox
- Check: mokutil --sb-state

Read:
- /home/greg/gaussian-image-projects/lgi-project/packages/lgi-rs/lgi-encoder-v2/test-results/SESSION_HANDOVER_2025-12-06_GPU.md
```

---

*Document created: 2025-12-06*
*Updated: 2025-12-06 22:45 - Driver 580.82.09 installed, awaiting Secure Boot disable*
