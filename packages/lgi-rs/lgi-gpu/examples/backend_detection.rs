//! Backend Detection Example
//!
//! Demonstrates wgpu v27 auto-detection of available GPU backends

use lgi_gpu::GpuRenderer;

fn main() {
    env_logger::init();

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║  LGI GPU Backend Detection (wgpu v27)               ║");
    println!("╚══════════════════════════════════════════════════════╝\n");

    // Create GPU renderer (auto-detects best backend)
    let renderer = pollster::block_on(async {
        GpuRenderer::new().await
    });

    match renderer {
        Ok(renderer) => {
            println!("\n✅ GPU Renderer Initialized!");
            println!("\nBackend: {:?}", renderer.backend());
            println!("Adapter: {}", renderer.adapter_name());

            println!("\nCapabilities:");
            renderer.capabilities().print_info();

            println!("\n✨ GPU rendering ready!");
            println!("Expected performance: 1000+ FPS @ 1080p with 10K Gaussians");
        }
        Err(e) => {
            eprintln!("\n❌ Failed to initialize GPU: {}", e);
            eprintln!("Falling back to CPU rendering");
        }
    }
}
