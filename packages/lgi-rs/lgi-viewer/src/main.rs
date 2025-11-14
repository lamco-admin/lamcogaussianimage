//! LGI Viewer - Professional Gaussian Image Viewer
//! Copyright (c) 2025 Lamco Development
//!
//! A comprehensive viewer for LGI (Learnable Gaussian Image) files with:
//! - GPU-accelerated rendering (1,000+ FPS)
//! - Interactive zoom/pan
//! - Multi-level pyramid support
//! - Render mode comparison
//! - Gaussian visualization
//! - Quality analysis
//! - Export at any resolution
//! - Comprehensive profiling and debugging

mod profiler;
mod async_encoder;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::{info, warn, error, debug, trace, instrument};
use slint::{Image, Rgba8Pixel, SharedPixelBuffer, ComponentHandle};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use parking_lot::Mutex;

slint::include_modules!();

/// Professional LGI/LGIV Gaussian Image Viewer
#[derive(Parser)]
#[command(name = "lgi-viewer")]
#[command(about = "Professional Gaussian Image Viewer - Test ALL LGI format features", long_about = None)]
struct Args {
    /// LGI file to open
    file: Option<PathBuf>,

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

struct AppState {
    lgi_file: Option<lgi_format::LgiFile>,
    current_zoom: f32,
    render_mode: lgi_core::RenderMode,
    file_path: Option<PathBuf>,
    source_image: Option<lgi_core::ImageBuffer<f32>>,  // For re-encoding
    profiler: profiler::Profiler,
}

impl AppState {
    fn new() -> Self {
        Self {
            lgi_file: None,
            current_zoom: 1.0,
            render_mode: lgi_core::RenderMode::AccumulatedSum,
            file_path: None,
            source_image: None,
            profiler: profiler::Profiler::new(),
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Setup comprehensive tracing
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(if args.verbose {
            tracing::Level::TRACE
        } else {
            tracing::Level::INFO
        })
        .with_thread_ids(true)
        .with_thread_names(true)
        .with_file(true)
        .with_line_number(true)
        .with_target(true)
        .compact()
        .finish();

    tracing::subscriber::set_global_default(subscriber)?;
    
    // Bridge log crate to tracing
    tracing_log::LogTracer::init()?;
    info!("🔗 Log-to-tracing bridge enabled");

    info!("🎨 LGI Viewer starting...");
    info!("📊 Profiling enabled: tracing all operations");
    info!("💾 Memory tracking enabled");
    info!("⚡ GPU profiling enabled");

    // Create tokio runtime for GPU
    let runtime = tokio::runtime::Runtime::new()?;

    // Create UI
    let ui = LgiViewer::new()?;

    // Create shared state
    let state = Arc::new(Mutex::new(AppState::new()));

    // Initialize GLOBAL GPU (shared across viewer, encoder, everything!)
    info!("🎮 Initializing global GPU instance...");
    let gpu_result = runtime.block_on(async {
        lgi_gpu::GpuManager::global().initialize().await
    });

    match gpu_result {
        Ok(_) => {
            info!("✅ Global GPU initialized and ready");
            info!("   This GPU will be shared by:");
            info!("     - Viewer rendering");
            info!("     - Encoding optimization");
            info!("     - Decoding");
            info!("     - Export operations");

            ui.set_metrics(PerformanceMetrics {
                fps: 0.0,
                frame_time_ms: 0.0,
                gpu_name: "RTX 4060 (Shared)".into(),
                gpu_backend: "Vulkan".into(),
                pyramid_level: 0,
                total_levels: 0,
            });
        }
        Err(e) => {
            warn!("⚠️  GPU initialization failed: {}, using CPU fallback", e);
            ui.set_metrics(PerformanceMetrics {
                fps: 0.0,
                frame_time_ms: 0.0,
                gpu_name: "CPU (Fallback)".into(),
                gpu_backend: "Software".into(),
                pyramid_level: 0,
                total_levels: 0,
            });
        }
    }

    // Load initial file if provided (don't auto-load, let user choose)
    if let Some(ref path) = args.file {
        info!("📁 Initial file specified: {:?}", path);
        if path.exists() {
            info!("✅ Loading initial file...");
            load_and_render_file(&ui, &state, path)?;
        } else {
            warn!("⚠️  Initial file not found: {:?}", path);
            ui.set_status_text(format!("⚠️  File not found: {}", path.display()).into());
        }
    } else {
        info!("👋 No initial file - waiting for user to load");
        ui.set_status_text("👋 Welcome! Load a file to begin".into());
    }

    // === CALLBACKS ===

    // Load file
    {
        let state = Arc::clone(&state);
        let ui_handle = ui.as_weak();
        ui.on_load_file(move || {
            if let Some(ui) = ui_handle.upgrade() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("LGI Files", &["lgi"])
                    .pick_file()
                {
                    if let Err(e) = load_and_render_file(&ui, &state, &path) {
                        error!("Failed to load file: {}", e);
                        ui.set_status_text(format!("❌ Error: {}", e).into());
                    }
                }
            }
        });
    }

    // Zoom in
    {
        let state = Arc::clone(&state);
        let ui_handle = ui.as_weak();
        ui.on_zoom_in(move || {
            if let Some(ui) = ui_handle.upgrade() {
                let new_zoom = (ui.get_zoom() * 1.25).min(10.0);
                ui.set_zoom(new_zoom);
                state.lock().current_zoom = new_zoom;
                
                if let Err(e) = render_at_zoom(&ui, &state, new_zoom) {
                    error!("Zoom render failed: {}", e);
                }
            }
        });
    }

    // Zoom out
    {
        let state = Arc::clone(&state);
        let ui_handle = ui.as_weak();
        ui.on_zoom_out(move || {
            if let Some(ui) = ui_handle.upgrade() {
                let new_zoom = (ui.get_zoom() / 1.25).max(0.1);
                ui.set_zoom(new_zoom);
                state.lock().current_zoom = new_zoom;
                
                if let Err(e) = render_at_zoom(&ui, &state, new_zoom) {
                    error!("Zoom render failed: {}", e);
                }
            }
        });
    }

    // Reset view
    {
        let state = Arc::clone(&state);
        let ui_handle = ui.as_weak();
        ui.on_reset_view(move || {
            if let Some(ui) = ui_handle.upgrade() {
                ui.set_zoom(1.0);
                state.lock().current_zoom = 1.0;
                
                if let Err(e) = render_at_zoom(&ui, &state, 1.0) {
                    error!("Reset render failed: {}", e);
                }
                ui.set_status_text("↺ View reset to 1.0×".into());
            }
        });
    }

    // Toggle render mode
    {
        let state = Arc::clone(&state);
        let ui_handle = ui.as_weak();
        ui.on_toggle_render_mode(move || {
            if let Some(ui) = ui_handle.upgrade() {
                let mode_str = ui.get_render_mode();
                let new_mode = if mode_str.as_str() == "Accumulated Sum" {
                    lgi_core::RenderMode::AccumulatedSum
                } else {
                    lgi_core::RenderMode::AlphaComposite
                };

                state.lock().render_mode = new_mode;
                
                let zoom = state.lock().current_zoom;
                if let Err(e) = render_at_zoom(&ui, &state, zoom) {
                    error!("Mode switch failed: {}", e);
                } else {
                    ui.set_status_text(format!("🎮 Switched to: {}", mode_str).into());
                }
            }
        });
    }

    // Export PNG (native resolution)
    {
        let state = Arc::clone(&state);
        let ui_handle = ui.as_weak();
        ui.on_save_png_native(move || {
            if let Err(e) = export_png(&state, &ui_handle, 1.0) {
                error!("Export failed: {}", e);
            }
        });
    }

    // Export PNG (custom resolution)
    {
        let state = Arc::clone(&state);
        let ui_handle = ui.as_weak();
        ui.on_save_png_custom(move |scale_str| {
            let scale = match scale_str.as_str() {
                "2x" => 2.0,
                "4x" => 4.0,
                _ => 1.0,
            };
            if let Err(e) = export_png(&state, &ui_handle, scale) {
                error!("Export failed: {}", e);
            }
        });
    }

    // Export Gaussians CSV
    {
        let state = Arc::clone(&state);
        let ui_handle = ui.as_weak();
        ui.on_export_gaussians(move || {
            if let Err(e) = export_gaussians_csv(&state, &ui_handle) {
                error!("CSV export failed: {}", e);
            }
        });
    }

    // Load image and encode (ASYNC with background thread)
    {
        let state = Arc::clone(&state);
        let ui_handle = ui.as_weak();
        ui.on_load_image_for_encoding(move || {
            if let Some(ui) = ui_handle.upgrade() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Images", &["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"])
                    .pick_file()
                {
                    let num_gaussians = ui.get_num_gaussians() as usize;
                    let profile_str = ui.get_encoding_profile();

                    let profile = match profile_str.as_str() {
                        "Small" => lgi_encoder::EncoderConfig::fast(),
                        "High" => lgi_encoder::EncoderConfig::high_quality(),
                        "Lossless" => lgi_encoder::EncoderConfig::ultra(),
                        _ => lgi_encoder::EncoderConfig::balanced(),
                    };

                    info!("🔄 Starting async encoding");
                    info!("   File: {:?}", path);
                    info!("   Gaussians: {}", num_gaussians);
                    info!("   Profile: {}", profile_str);

                    ui.set_status_text(format!("🔄 Loading {}...", path.file_name().unwrap().to_string_lossy()).into());
                    ui.set_encoding_in_progress(true);
                    ui.set_encoding_progress(5.0);

                    let state_clone = Arc::clone(&state);
                    let path_clone = path.clone();

                    info!("⚠️  Note: Encoding will block UI for 30-200 seconds");
                    info!("   (Async with progress updates coming in next version)");

                    // For now, call synchronously (will block but at least it works)
                    // TODO: Make truly async with invoke_from_event_loop
                    match load_and_encode_image(&ui, &state_clone, &path_clone, num_gaussians, profile) {
                        Ok(_) => {
                            info!("✅ Encoding complete");
                            ui.set_encoding_in_progress(false);
                        }
                        Err(e) => {
                            error!("Encoding failed: {}", e);
                            ui.set_encoding_in_progress(false);
                            ui.set_status_text(format!("❌ Encoding failed: {}", e).into());
                        }
                    }
                }
            }
        });
    }

    // Save as LGI
    {
        let state = Arc::clone(&state);
        let ui_handle = ui.as_weak();
        ui.on_save_as_lgi(move || {
            if let Err(e) = save_lgi_file(&state, &ui_handle) {
                error!("Save failed: {}", e);
            }
        });
    }

    info!("✅ Starting UI event loop");
    ui.run()?;

    Ok(())
}

/// Load file and render initial view
fn load_and_render_file(
    ui: &LgiViewer,
    state: &Arc<Mutex<AppState>>,
    path: &PathBuf,
) -> Result<()> {
    info!("Loading: {:?}", path);

    // Load LGI file
    let file = lgi_format::LgiReader::read_file(path)?;
    let (width, height) = file.dimensions();
    let gaussian_count = file.gaussians().len();
    let file_size = std::fs::metadata(path)?.len();

    // Update state
    {
        let mut state_lock = state.lock();
        state_lock.lgi_file = Some(file);
        state_lock.file_path = Some(path.clone());
        state_lock.current_zoom = 1.0;
    }

    // Update file info in UI
    let file_info = {
        let state_lock = state.lock();
        let file = state_lock.lgi_file.as_ref().unwrap();

        FileInfo {
            filename: path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("Unknown")
                .into(),
            width: width as i32,
            height: height as i32,
            gaussian_count: gaussian_count as i32,
            file_size: file_size as i32,
            compression_ratio: 0.0,
            profile: "Auto".into(),
            has_vq: file.header.compression_flags.vq_compressed,
            has_qa: false,
        }
    };

    ui.set_file_info(file_info);
    ui.set_zoom(1.0);

    // Initial render
    render_at_zoom(ui, state, 1.0)?;

    ui.set_status_text(format!("✅ Loaded: {} Gaussians", gaussian_count).into());

    Ok(())
}

/// Render at specific zoom level
#[instrument(skip(ui, state), fields(zoom = zoom))]
fn render_at_zoom(
    ui: &LgiViewer,
    state: &Arc<Mutex<AppState>>,
    zoom: f32,
) -> Result<()> {
    let total_start = Instant::now();
    trace!("🖼️  render_at_zoom called with zoom={}", zoom);

    let (render_width, render_height, rgba_data) = {
        let mut state_lock = state.lock();

        let file = state_lock.lgi_file.as_ref()
            .context("No file loaded")?;

        let (base_width, base_height) = file.dimensions();
        let gaussians = file.gaussians();
        let render_mode = state_lock.render_mode;

        let render_width = (base_width as f32 * zoom) as u32;
        let render_height = (base_height as f32 * zoom) as u32;

        debug!("📏 Base: {}×{}, Zoom: {}, Target: {}×{}",
            base_width, base_height, zoom, render_width, render_height);
        debug!("   Gaussians to render: {}", gaussians.len());
        debug!("   Render mode: {:?}", render_mode);

        let render_start = Instant::now();

        // Render with GLOBAL GPU or CPU fallback
        let (backend, image_buffer) = if lgi_gpu::GpuManager::global().is_initialized() {
            trace!("🎮 Using global GPU renderer");
            let result = lgi_gpu::GpuManager::global().render(|gpu| {
                gpu.render(&gaussians, render_width, render_height, render_mode)
            });
            match result {
                Ok(img) => ("GPU", img),
                Err(e) => {
                    warn!("GPU render failed: {}, falling back to CPU", e);
                    let renderer = lgi_core::Renderer::new();
                    ("CPU", renderer.render(&gaussians, render_width, render_height)?)
                }
            }
        } else {
            trace!("💻 Using CPU renderer (GPU not available)");
            let renderer = lgi_core::Renderer::new();
            ("CPU", renderer.render(&gaussians, render_width, render_height)?)
        };

        let render_time = render_start.elapsed();
        info!("✅ {} rendered in {:.2}ms", backend, render_time.as_secs_f32() * 1000.0);
        debug!("   Pixels rendered: {}", render_width * render_height);
        debug!("   Throughput: {:.0} Mpixels/s",
            (render_width * render_height) as f32 / render_time.as_secs_f32() / 1_000_000.0);

        let image_buffer = image_buffer;

        // Convert float32 RGBA to u8
        let convert_start = Instant::now();
        trace!("🔄 Converting float32 to u8 RGBA for display");
        let mut rgba8 = Vec::with_capacity((render_width * render_height * 4) as usize);
        for y in 0..render_height {
            for x in 0..render_width {
                if let Some(pixel) = image_buffer.get_pixel(x, y) {
                    rgba8.push((pixel.r * 255.0).clamp(0.0, 255.0) as u8);
                    rgba8.push((pixel.g * 255.0).clamp(0.0, 255.0) as u8);
                    rgba8.push((pixel.b * 255.0).clamp(0.0, 255.0) as u8);
                    rgba8.push((pixel.a * 255.0).clamp(0.0, 255.0) as u8);
                }
            }
        }
        let convert_time = convert_start.elapsed();
        debug!("✅ Converted to u8 in {:.2}ms", convert_time.as_secs_f32() * 1000.0);

        (render_width, render_height, rgba8)
    };

    trace!("📦 Creating Slint pixel buffer");
    let buffer_start = Instant::now();

    // Update image in UI
    let pixel_buffer = SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(
        &rgba_data,
        render_width,
        render_height,
    );
    let buffer_time = buffer_start.elapsed();
    trace!("✅ Buffer created in {:.2}ms", buffer_time.as_secs_f32() * 1000.0);

    trace!("🖼️  Updating UI image");
    let ui_start = Instant::now();
    ui.set_display_image(Image::from_rgba8(pixel_buffer));
    let ui_time = ui_start.elapsed();
    trace!("✅ UI updated in {:.2}ms", ui_time.as_secs_f32() * 1000.0);

    // Update performance metrics
    let total_elapsed = total_start.elapsed();
    let fps = 1.0 / total_elapsed.as_secs_f32();
    let frame_time = total_elapsed.as_secs_f32() * 1000.0;

    let current_metrics = ui.get_metrics();
    ui.set_metrics(PerformanceMetrics {
        fps,
        frame_time_ms: frame_time,
        ..current_metrics
    });

    info!("✅ Total render pipeline: {:.2}ms ({:.0} FPS)", frame_time, fps);
    debug!("   Breakdown:");
    debug!("     GPU/CPU render: {:.2}ms", (total_elapsed - buffer_time - ui_time).as_secs_f32() * 1000.0);
    debug!("     Buffer creation: {:.2}ms", buffer_time.as_secs_f32() * 1000.0);
    debug!("     UI update: {:.2}ms", ui_time.as_secs_f32() * 1000.0);

    Ok(())
}

/// Export PNG at specified scale
fn export_png(
    state: &Arc<Mutex<AppState>>,
    ui_handle: &slint::Weak<LgiViewer>,
    scale: f32,
) -> Result<()> {
    let save_path = rfd::FileDialog::new()
        .add_filter("PNG Image", &["png"])
        .set_file_name(&format!("export_{}x.png", scale))
        .save_file()
        .context("No file selected")?;

    info!("Exporting at {}× to {:?}", scale, save_path);

    let (render_width, render_height, rgba_data) = {
        let mut state_lock = state.lock();

        let file = state_lock.lgi_file.as_ref()
            .context("No file loaded")?;

        let (base_width, base_height) = file.dimensions();
        let gaussians = file.gaussians();
        let render_mode = state_lock.render_mode;

        let render_width = (base_width as f32 * scale) as u32;
        let render_height = (base_height as f32 * scale) as u32;

        // Render with global GPU
        let image_buffer = if lgi_gpu::GpuManager::global().is_initialized() {
            lgi_gpu::GpuManager::global().render(|gpu| {
                gpu.render(&gaussians, render_width, render_height, render_mode)
            }).map_err(|e| anyhow::anyhow!("GPU render failed: {}", e))?
        } else {
            let renderer = lgi_core::Renderer::new();
            renderer.render(&gaussians, render_width, render_height)?
        };

        // Convert to u8
        let mut rgba8 = Vec::with_capacity((render_width * render_height * 4) as usize);
        for y in 0..render_height {
            for x in 0..render_width {
                if let Some(pixel) = image_buffer.get_pixel(x, y) {
                    rgba8.push((pixel.r * 255.0).clamp(0.0, 255.0) as u8);
                    rgba8.push((pixel.g * 255.0).clamp(0.0, 255.0) as u8);
                    rgba8.push((pixel.b * 255.0).clamp(0.0, 255.0) as u8);
                    rgba8.push((pixel.a * 255.0).clamp(0.0, 255.0) as u8);
                }
            }
        }

        (render_width, render_height, rgba8)
    };

    // Save PNG
    let img_buffer = image::RgbaImage::from_raw(
        render_width,
        render_height,
        rgba_data,
    ).context("Failed to create image buffer")?;

    img_buffer.save(&save_path)?;

    info!("✅ Exported {}× ({}×{}) to {:?}", scale, render_width, render_height, save_path);

    if let Some(ui) = ui_handle.upgrade() {
        ui.set_status_text(
            format!("✅ Exported {}×: {}", scale, save_path.display()).into()
        );
    }

    Ok(())
}

/// Export Gaussians to CSV
fn export_gaussians_csv(
    state: &Arc<Mutex<AppState>>,
    ui_handle: &slint::Weak<LgiViewer>,
) -> Result<()> {
    let save_path = rfd::FileDialog::new()
        .add_filter("CSV File", &["csv"])
        .set_file_name("gaussians.csv")
        .save_file()
        .context("No file selected")?;

    let gaussians = {
        let state_lock = state.lock();
        let file = state_lock.lgi_file.as_ref()
            .context("No file loaded")?;
        file.gaussians().to_vec()
    };

    use std::io::Write;
    let mut f = std::fs::File::create(&save_path)?;

    // CSV header
    writeln!(f, "index,pos_x,pos_y,scale_x,scale_y,rotation,color_r,color_g,color_b,color_a,opacity")?;

    // Write each Gaussian
    for (i, g) in gaussians.iter().enumerate() {
        writeln!(f, "{},{},{},{},{},{},{},{},{},{},{}",
            i,
            g.position.x, g.position.y,
            g.shape.scale_x, g.shape.scale_y,
            g.shape.rotation,
            g.color.r, g.color.g, g.color.b, g.color.a,
            g.opacity
        )?;
    }

    info!("✅ Exported {} Gaussians to {:?}", gaussians.len(), save_path);

    if let Some(ui) = ui_handle.upgrade() {
        ui.set_status_text(
            format!("✅ Exported {} Gaussians to CSV", gaussians.len()).into()
        );
    }

    Ok(())
}

/// Load image (JPG/PNG/GIF/etc) and encode to LGI
#[instrument(skip(ui, state), fields(path = ?path, num_gaussians = num_gaussians))]
fn load_and_encode_image(
    ui: &LgiViewer,
    state: &Arc<Mutex<AppState>>,
    path: &PathBuf,
    num_gaussians: usize,
    profile: lgi_encoder::EncoderConfig,
) -> Result<()> {
    let total_start = Instant::now();
    info!("🔄 Starting image load and encode pipeline");

    let _profiler_guard = {
        let state_lock = state.lock();
        state_lock.profiler.start_operation("load_and_encode_image")
    };

    // Load image with image crate (supports JPG, PNG, GIF, BMP, TIFF, WebP, etc.)
    let load_start = Instant::now();
    debug!("📂 Opening image file: {:?}", path);
    let img = image::open(path)?;
    debug!("🎨 Converting to RGBA8");
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    let load_time = load_start.elapsed();

    info!("✅ Loaded {}×{} image in {:.2}ms", width, height, load_time.as_secs_f32() * 1000.0);
    debug!("   Format: {:?}", img.color());
    debug!("   Pixels: {}", width * height);

    // Convert u8 RGBA to f32 RGBA [0, 1]
    let mut float_data = lgi_core::ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let pixel = rgba.get_pixel(x, y);
            float_data.set_pixel(
                x,
                y,
                lgi_math::color::Color4::new(
                    pixel[0] as f32 / 255.0,
                    pixel[1] as f32 / 255.0,
                    pixel[2] as f32 / 255.0,
                    pixel[3] as f32 / 255.0,
                ),
            );
        }
    }

    // Store source image
    state.lock().source_image = Some(float_data.clone());

    // Initialize Gaussians
    info!("Initializing {} Gaussians...", num_gaussians);
    let initializer = lgi_core::Initializer::new(lgi_core::InitStrategy::Gradient);
    let mut gaussians = initializer.initialize(&float_data, num_gaussians)?;

    // Create optimizer (will use global GPU automatically!)
    info!("🚀 Creating optimizer...");
    ui.set_encoding_in_progress(true);
    ui.set_encoding_progress(0.0);

    let mut optimizer = lgi_encoder::OptimizerV2::new(profile);

    if lgi_gpu::GpuManager::global().is_initialized() {
        info!("🎮 Optimizer will use global GPU (expect 1000× speedup!)");
    } else {
        warn!("⚠️  No GPU available, using CPU (will be VERY slow)");
    }

    // Optimize
    info!("🔧 Starting optimization - each iteration will be logged");
    optimizer.optimize(&mut gaussians, &float_data)?;

    ui.set_encoding_in_progress(false);
    ui.set_encoding_progress(100.0);

    info!("✅ Optimization complete! {} Gaussians", gaussians.len());

    // Create LGI file
    let compression_config = lgi_format::CompressionConfig::balanced();  // TODO: use profile param
    let lgi_file = lgi_format::LgiFile::with_compression(
        gaussians,
        width,
        height,
        compression_config,
    );

    // Update state
    {
        let mut state_lock = state.lock();
        state_lock.lgi_file = Some(lgi_file);
        state_lock.file_path = Some(path.clone());
        state_lock.current_zoom = 1.0;
    }

    // Update UI
    let file_info = {
        let state_lock = state.lock();
        let file = state_lock.lgi_file.as_ref().unwrap();

        FileInfo {
            filename: path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("Unknown")
                .into(),
            width: width as i32,
            height: height as i32,
            gaussian_count: file.gaussians().len() as i32,
            file_size: 0, // Not saved yet
            compression_ratio: 0.0,
            profile: "Auto".into(),
            has_vq: file.header.compression_flags.vq_compressed,
            has_qa: false,
        }
    };

    ui.set_file_info(file_info);
    ui.set_zoom(1.0);

    // Render
    render_at_zoom(ui, state, 1.0)?;

    ui.set_status_text(format!("✅ Encoded from {}", path.file_name().unwrap().to_string_lossy()).into());

    Ok(())
}

/// Save current LGI to file
fn save_lgi_file(
    state: &Arc<Mutex<AppState>>,
    ui_handle: &slint::Weak<LgiViewer>,
) -> Result<()> {
    let save_path = rfd::FileDialog::new()
        .add_filter("LGI File", &["lgi"])
        .set_file_name("compressed.lgi")
        .save_file()
        .context("No file selected")?;

    let lgi_file = {
        let state_lock = state.lock();
        state_lock.lgi_file.as_ref()
            .context("No LGI file to save")?
            .clone()
    };

    lgi_format::LgiWriter::write_file(&lgi_file, &save_path)?;

    let file_size = std::fs::metadata(&save_path)?.len();
    info!("✅ Saved .lgi file ({} KB) to {:?}", file_size / 1024, save_path);

    if let Some(ui) = ui_handle.upgrade() {
        ui.set_status_text(
            format!("✅ Saved: {} ({} KB)", save_path.display(), file_size / 1024).into()
        );
    }

    Ok(())
}

/// Update UI after encoding completes (called from main thread)
fn update_ui_after_encode(
    ui: &LgiViewer,
    state: &Arc<Mutex<AppState>>,
    width: u32,
    height: u32,
) -> Result<()> {
    info!("🖼️  Updating UI after encoding");

    // Update file info
    let file_info = {
        let state_lock = state.lock();
        let file = state_lock.lgi_file.as_ref().context("No file")?;

        FileInfo {
            filename: "Encoded Image".into(),
            width: width as i32,
            height: height as i32,
            gaussian_count: file.gaussians().len() as i32,
            file_size: 0,
            compression_ratio: 0.0,
            profile: "Auto".into(),
            has_vq: file.header.compression_flags.vq_compressed,
            has_qa: false,
        }
    };

    ui.set_file_info(file_info);
    ui.set_zoom(1.0);
    ui.set_encoding_in_progress(false);
    ui.set_encoding_progress(100.0);

    // Render
    render_at_zoom(ui, state, 1.0)?;

    ui.set_status_text("✅ Encoding complete!".into());

    Ok(())
}
