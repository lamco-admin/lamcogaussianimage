//! # LGI C FFI Library
//!
//! C-compatible Foreign Function Interface for LGI codec.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_uint, c_float};
use std::ptr;
use std::slice;
use std::sync::Mutex;

use once_cell::sync::Lazy;

use lgi_core::{ImageBuffer, Initializer, Renderer, RenderMode};
use lgi_encoder::{EncoderConfig, OptimizerV2};
use lgi_format::{LgiFile, LgiWriter, LgiReader, CompressionConfig, QuantizationProfile};
use lgi_math::{gaussian::Gaussian2D, parameterization::Euler};

// Global GPU renderer singleton
static GPU_RENDERER: Lazy<Mutex<Option<(tokio::runtime::Runtime, lgi_gpu::GpuRenderer)>>> = Lazy::new(|| {
    // Try to initialize GPU once at startup
    let result = tokio::runtime::Runtime::new()
        .ok()
        .and_then(|rt| {
            let gpu = rt.block_on(async {
                lgi_gpu::GpuRenderer::new().await.ok()
            })?;
            Some((rt, gpu))
        });
    Mutex::new(result)
});

/// Error codes
#[repr(C)]
pub enum LgiErrorCode {
    Success = 0,
    InvalidParameter = 1,
    FileNotFound = 2,
    EncodingFailed = 3,
    DecodingFailed = 4,
    OutOfMemory = 5,
    InvalidFormat = 6,
    IoError = 7,
}

/// Compression profile selection
#[repr(C)]
#[derive(Copy, Clone)]
pub enum LgiProfile {
    Balanced = 0,
    Small = 1,
    High = 2,
    Lossless = 3,
}

/// Opaque encoder handle
#[repr(C)]
pub struct LgiEncoder { _private: [u8; 0] }

/// Opaque decoder handle
#[repr(C)]
pub struct LgiDecoder { _private: [u8; 0] }

struct EncoderContext {
    config: EncoderConfig,
    target: Option<ImageBuffer<f32>>,
    gaussians: Option<Vec<Gaussian2D<f32, Euler<f32>>>>,
}

struct DecoderContext {
    file: Option<LgiFile>,
    image: Option<ImageBuffer<f32>>,
}

/// Create encoder
#[no_mangle]
pub unsafe extern "C" fn lgi_encoder_create(profile: LgiProfile) -> *mut LgiEncoder {
    let config = match profile {
        LgiProfile::Balanced => EncoderConfig::balanced(),
        LgiProfile::Small => EncoderConfig::fast(),
        LgiProfile::High => EncoderConfig::high_quality(),
        LgiProfile::Lossless => EncoderConfig::ultra(),
    };

    let ctx = Box::new(EncoderContext {
        config,
        target: None,
        gaussians: None,
    });

    Box::into_raw(ctx) as *mut LgiEncoder
}

/// Set input image
#[no_mangle]
pub unsafe extern "C" fn lgi_encoder_set_image(
    encoder: *mut LgiEncoder,
    width: c_uint,
    height: c_uint,
    data: *const c_float,
) -> LgiErrorCode {
    if encoder.is_null() || data.is_null() {
        return LgiErrorCode::InvalidParameter;
    }

    let ctx = &mut *(encoder as *mut EncoderContext);
    let pixel_count = (width * height) as usize;
    let data_slice = slice::from_raw_parts(data, pixel_count * 4);

    let mut image = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            image.set_pixel(x, y, lgi_math::color::Color4::new(
                data_slice[idx],
                data_slice[idx + 1],
                data_slice[idx + 2],
                data_slice[idx + 3],
            ));
        }
    }

    ctx.target = Some(image);
    LgiErrorCode::Success
}

/// Encode to Gaussians
#[no_mangle]
pub unsafe extern "C" fn lgi_encoder_encode(
    encoder: *mut LgiEncoder,
    num_gaussians: c_uint,
) -> LgiErrorCode {
    if encoder.is_null() {
        return LgiErrorCode::InvalidParameter;
    }

    let ctx = &mut *(encoder as *mut EncoderContext);

    let target = match &ctx.target {
        Some(t) => t,
        None => return LgiErrorCode::InvalidParameter,
    };

    let initializer = Initializer::new(ctx.config.init_strategy)
        .with_scale(ctx.config.initial_scale);

    let mut gaussians = match initializer.initialize(target, num_gaussians as usize) {
        Ok(g) => g,
        Err(_) => return LgiErrorCode::EncodingFailed,
    };

    let optimizer = OptimizerV2::new(ctx.config.clone());
    match optimizer.optimize_with_metrics(&mut gaussians, target) {
        Ok(_) => {
            ctx.gaussians = Some(gaussians);
            LgiErrorCode::Success
        }
        Err(_) => LgiErrorCode::EncodingFailed,
    }
}

/// Save to file
#[no_mangle]
pub unsafe extern "C" fn lgi_encoder_save(
    encoder: *mut LgiEncoder,
    filename: *const c_char,
    profile: LgiProfile,
) -> LgiErrorCode {
    if encoder.is_null() || filename.is_null() {
        return LgiErrorCode::InvalidParameter;
    }

    let ctx = &*(encoder as *const EncoderContext);

    let gaussians = match &ctx.gaussians {
        Some(g) => g,
        None => return LgiErrorCode::EncodingFailed,
    };

    let target = match &ctx.target {
        Some(t) => t,
        None => return LgiErrorCode::InvalidParameter,
    };

    let path_str = match CStr::from_ptr(filename).to_str() {
        Ok(s) => s,
        Err(_) => return LgiErrorCode::InvalidParameter,
    };

    let compression = match profile {
        LgiProfile::Balanced => CompressionConfig::balanced(),
        LgiProfile::Small => CompressionConfig::small(),
        LgiProfile::High => CompressionConfig::high_quality(),
        LgiProfile::Lossless => CompressionConfig::lossless(),
    };

    let file = LgiFile::with_compression(
        gaussians.clone(),
        target.width,
        target.height,
        compression,
    );

    match LgiWriter::write_file(&file, path_str) {
        Ok(_) => LgiErrorCode::Success,
        Err(_) => LgiErrorCode::IoError,
    }
}

/// Save to memory buffer (for streaming/blob operations)
#[no_mangle]
pub unsafe extern "C" fn lgi_encoder_save_to_buffer(
    encoder: *mut LgiEncoder,
    profile: LgiProfile,
    data_out: *mut *mut u8,
    size_out: *mut usize,
) -> LgiErrorCode {
    if encoder.is_null() || data_out.is_null() || size_out.is_null() {
        return LgiErrorCode::InvalidParameter;
    }

    let ctx = &*(encoder as *const EncoderContext);

    let gaussians = match &ctx.gaussians {
        Some(g) => g,
        None => return LgiErrorCode::EncodingFailed,
    };

    let target = match &ctx.target {
        Some(t) => t,
        None => return LgiErrorCode::InvalidParameter,
    };

    let compression = match profile {
        LgiProfile::Balanced => CompressionConfig::balanced(),
        LgiProfile::Small => CompressionConfig::small(),
        LgiProfile::High => CompressionConfig::high_quality(),
        LgiProfile::Lossless => CompressionConfig::lossless(),
    };

    let file = LgiFile::with_compression(
        gaussians.clone(),
        target.width,
        target.height,
        compression,
    );

    // Write to memory buffer
    let mut buffer = Vec::new();
    match LgiWriter::write(&mut buffer, &file) {
        Ok(_) => {
            // Allocate buffer and copy data
            *size_out = buffer.len();
            let boxed_slice = buffer.into_boxed_slice();
            *data_out = Box::into_raw(boxed_slice) as *mut u8;
            LgiErrorCode::Success
        }
        Err(_) => LgiErrorCode::IoError,
    }
}

/// Free buffer allocated by encoder
#[no_mangle]
pub unsafe extern "C" fn lgi_free_buffer(data: *mut u8, size: usize) {
    if !data.is_null() && size > 0 {
        let _ = Box::from_raw(slice::from_raw_parts_mut(data, size));
    }
}

/// Destroy encoder
#[no_mangle]
pub unsafe extern "C" fn lgi_encoder_destroy(encoder: *mut LgiEncoder) {
    if !encoder.is_null() {
        let _ = Box::from_raw(encoder as *mut EncoderContext);
    }
}

/// Create decoder
#[no_mangle]
pub unsafe extern "C" fn lgi_decoder_create() -> *mut LgiDecoder {
    let ctx = Box::new(DecoderContext {
        file: None,
        image: None,
    });

    Box::into_raw(ctx) as *mut LgiDecoder
}

/// Load file from path
#[no_mangle]
pub unsafe extern "C" fn lgi_decoder_load(
    decoder: *mut LgiDecoder,
    filename: *const c_char,
) -> LgiErrorCode {
    if decoder.is_null() || filename.is_null() {
        return LgiErrorCode::InvalidParameter;
    }

    let ctx = &mut *(decoder as *mut DecoderContext);

    let path_str = match CStr::from_ptr(filename).to_str() {
        Ok(s) => s,
        Err(_) => return LgiErrorCode::InvalidParameter,
    };

    match LgiReader::read_file(path_str) {
        Ok(file) => {
            ctx.file = Some(file);
            LgiErrorCode::Success
        }
        Err(_) => LgiErrorCode::FileNotFound,
    }
}

/// Load from memory buffer (for streaming/blob operations)
#[no_mangle]
pub unsafe extern "C" fn lgi_decoder_load_from_buffer(
    decoder: *mut LgiDecoder,
    data: *const u8,
    size: usize,
) -> LgiErrorCode {
    if decoder.is_null() || data.is_null() || size == 0 {
        return LgiErrorCode::InvalidParameter;
    }

    let ctx = &mut *(decoder as *mut DecoderContext);

    // Create slice from raw buffer
    let buffer = slice::from_raw_parts(data, size);

    // Use Cursor to provide Read trait
    let mut cursor = std::io::Cursor::new(buffer);

    match LgiReader::read(&mut cursor) {
        Ok(file) => {
            ctx.file = Some(file);
            LgiErrorCode::Success
        }
        Err(_) => LgiErrorCode::InvalidFormat,
    }
}

/// Decode to image
#[no_mangle]
pub unsafe extern "C" fn lgi_decoder_decode(decoder: *mut LgiDecoder) -> LgiErrorCode {
    if decoder.is_null() {
        return LgiErrorCode::InvalidParameter;
    }

    let ctx = &mut *(decoder as *mut DecoderContext);

    let file = match &ctx.file {
        Some(f) => f,
        None => return LgiErrorCode::InvalidParameter,
    };

    let gaussians = file.gaussians();
    let (width, height) = file.dimensions();

    // Try GPU first (using singleton), fallback to CPU
    let result = {
        // Check if GPU is available
        let has_gpu = GPU_RENDERER.lock().unwrap().is_some();

        if has_gpu {
            // GPU available, use it
            let mut gpu_lock = GPU_RENDERER.lock().unwrap();
            let (_, ref mut gpu_renderer) = gpu_lock.as_mut().unwrap();
            gpu_renderer.render(&gaussians, width, height, RenderMode::AccumulatedSum)
                .map_err(|_| lgi_core::LgiError::ImageError("GPU rendering failed".to_string()))
        } else {
            // No GPU, use CPU
            let renderer = Renderer::new();
            renderer.render(&gaussians, width, height)
        }
    };

    match result {
        Ok(image) => {
            ctx.image = Some(image);
            LgiErrorCode::Success
        }
        Err(_) => LgiErrorCode::DecodingFailed,
    }
}

/// Get dimensions
#[no_mangle]
pub unsafe extern "C" fn lgi_decoder_get_dimensions(
    decoder: *const LgiDecoder,
    width: *mut c_uint,
    height: *mut c_uint,
) -> LgiErrorCode {
    if decoder.is_null() || width.is_null() || height.is_null() {
        return LgiErrorCode::InvalidParameter;
    }

    let ctx = &*(decoder as *const DecoderContext);

    let image = match &ctx.image {
        Some(i) => i,
        None => return LgiErrorCode::InvalidParameter,
    };

    *width = image.width;
    *height = image.height;
    LgiErrorCode::Success
}

/// Get image data
#[no_mangle]
pub unsafe extern "C" fn lgi_decoder_get_data(
    decoder: *const LgiDecoder,
    data: *mut c_float,
) -> LgiErrorCode {
    if decoder.is_null() || data.is_null() {
        return LgiErrorCode::InvalidParameter;
    }

    let ctx = &*(decoder as *const DecoderContext);

    let image = match &ctx.image {
        Some(i) => i,
        None => return LgiErrorCode::InvalidParameter,
    };

    let data_slice = slice::from_raw_parts_mut(data, (image.width * image.height * 4) as usize);

    for y in 0..image.height {
        for x in 0..image.width {
            if let Some(pixel) = image.get_pixel(x, y) {
                let idx = ((y * image.width + x) * 4) as usize;
                data_slice[idx] = pixel.r;
                data_slice[idx + 1] = pixel.g;
                data_slice[idx + 2] = pixel.b;
                data_slice[idx + 3] = pixel.a;
            }
        }
    }

    LgiErrorCode::Success
}

/// Destroy decoder
#[no_mangle]
pub unsafe extern "C" fn lgi_decoder_destroy(decoder: *mut LgiDecoder) {
    if !decoder.is_null() {
        let _ = Box::from_raw(decoder as *mut DecoderContext);
    }
}

/// Get version
#[no_mangle]
pub unsafe extern "C" fn lgi_version() -> *const c_char {
    static VERSION: &str = concat!("LGI v", env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr() as *const c_char
}

/// Check GPU support
#[no_mangle]
pub extern "C" fn lgi_has_gpu_support() -> c_int {
    #[cfg(feature = "gpu")]
    return 1;

    #[cfg(not(feature = "gpu"))]
    return 0;
}
