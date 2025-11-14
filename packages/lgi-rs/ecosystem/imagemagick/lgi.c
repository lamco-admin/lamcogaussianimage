/*
 * LGI coder for ImageMagick
 * Copyright (c) 2025 LGI Project
 *
 * Read/write support for LGI (Lamco Gaussian Image) format
 */

#include "MagickCore/studio.h"
#include "MagickCore/blob.h"
#include "MagickCore/cache.h"
#include "MagickCore/colorspace.h"
#include "MagickCore/exception.h"
#include "MagickCore/image.h"
#include "MagickCore/list.h"
#include "MagickCore/magick.h"
#include "MagickCore/memory_.h"
#include "MagickCore/quantum-private.h"
#include "MagickCore/static.h"
#include "MagickCore/string_.h"
#include "MagickCore/module.h"
#include <lgi.h>

/*
  Forward declarations
*/
static MagickBooleanType WriteLGIImage(const ImageInfo *, Image *, ExceptionInfo *);

/*
 * Read LGI image
 */
static Image *ReadLGIImage(const ImageInfo *image_info, ExceptionInfo *exception)
{
    Image *image;
    LgiDecoder *decoder;
    unsigned int width, height;
    float *data;
    ssize_t y;

    // Open image
    image = AcquireImage(image_info, exception);
    if (OpenBlob(image_info, image, ReadBinaryBlobMode, exception) == MagickFalse)
        return (Image *) NULL;

    // Create decoder
    decoder = lgi_decoder_create();
    if (!decoder) {
        (void) ThrowMagickException(exception, GetMagickModule(), CoderError,
            "Failed to create LGI decoder", "`%s'", image->filename);
        return (Image *) NULL;
    }

    // Load from file
    if (lgi_decoder_load(decoder, image->filename) != 0) {
        lgi_decoder_destroy(decoder);
        (void) ThrowMagickException(exception, GetMagickModule(), CorruptImageError,
            "Failed to load LGI file", "`%s'", image->filename);
        return (Image *) NULL;
    }

    // Decode
    if (lgi_decoder_decode(decoder) != 0) {
        lgi_decoder_destroy(decoder);
        (void) ThrowMagickException(exception, GetMagickModule(), CorruptImageError,
            "Failed to decode LGI file", "`%s'", image->filename);
        return (Image *) NULL;
    }

    // Get dimensions
    lgi_decoder_get_dimensions(decoder, &width, &height);

    // Set image properties
    image->columns = width;
    image->rows = height;
    image->depth = 32;  // Float32
    image->colorspace = sRGBColorspace;

    // Allocate data buffer
    data = (float *) AcquireMagickMemory(width * height * 4 * sizeof(float));
    if (!data) {
        lgi_decoder_destroy(decoder);
        ThrowReaderException(ResourceLimitError, "MemoryAllocationFailed");
    }

    // Get decoded data
    lgi_decoder_get_data(decoder, data);

    // Convert to ImageMagick pixels
    for (y = 0; y < (ssize_t) height; y++) {
        Quantum *q = QueueAuthenticPixels(image, 0, y, width, 1, exception);
        if (!q)
            break;

        for (ssize_t x = 0; x < (ssize_t) width; x++) {
            size_t idx = (y * width + x) * 4;
            SetPixelRed(image, ClampToQuantum(QuantumRange * data[idx]), q);
            SetPixelGreen(image, ClampToQuantum(QuantumRange * data[idx + 1]), q);
            SetPixelBlue(image, ClampToQuantum(QuantumRange * data[idx + 2]), q);
            SetPixelAlpha(image, ClampToQuantum(QuantumRange * data[idx + 3]), q);
            q += GetPixelChannels(image);
        }

        if (SyncAuthenticPixels(image, exception) == MagickFalse)
            break;
    }

    // Cleanup
    RelinquishMagickMemory(data);
    lgi_decoder_destroy(decoder);
    CloseBlob(image);

    return image;
}

/*
 * Write LGI image
 */
static MagickBooleanType WriteLGIImage(const ImageInfo *image_info,
    Image *image, ExceptionInfo *exception)
{
    LgiEncoder *encoder;
    float *data;
    const Quantum *p;
    ssize_t y;
    unsigned int width = image->columns;
    unsigned int height = image->rows;

    // Open output
    if (OpenBlob(image_info, image, WriteBinaryBlobMode, exception) == MagickFalse)
        return MagickFalse;

    // Create encoder
    encoder = lgi_encoder_create(LgiProfile_Balanced);  // TODO: Make configurable
    if (!encoder) {
        (void) ThrowMagickException(exception, GetMagickModule(), CoderError,
            "Failed to create LGI encoder", "`%s'", image->filename);
        return MagickFalse;
    }

    // Allocate data buffer (RGBA float32)
    data = (float *) AcquireMagickMemory(width * height * 4 * sizeof(float));
    if (!data) {
        lgi_encoder_destroy(encoder);
        ThrowWriterException(ResourceLimitError, "MemoryAllocationFailed");
    }

    // Convert from ImageMagick pixels
    for (y = 0; y < (ssize_t) height; y++) {
        p = GetVirtualPixels(image, 0, y, width, 1, exception);
        if (!p)
            break;

        for (ssize_t x = 0; x < (ssize_t) width; x++) {
            size_t idx = (y * width + x) * 4;
            data[idx] = (float) GetPixelRed(image, p) / QuantumRange;
            data[idx + 1] = (float) GetPixelGreen(image, p) / QuantumRange;
            data[idx + 2] = (float) GetPixelBlue(image, p) / QuantumRange;
            data[idx + 3] = (float) GetPixelAlpha(image, p) / QuantumRange;
            p += GetPixelChannels(image);
        }
    }

    // Set image
    lgi_encoder_set_image(encoder, width, height, data);

    // Encode (use adaptive Gaussian count based on image size)
    unsigned int num_gaussians = (width * height) / 64;  // ~1.5% of pixels
    if (num_gaussians < 100) num_gaussians = 100;
    if (num_gaussians > 10000) num_gaussians = 10000;

    if (lgi_encoder_encode(encoder, num_gaussians) != 0) {
        RelinquishMagickMemory(data);
        lgi_encoder_destroy(encoder);
        return MagickFalse;
    }

    // Save to file
    if (lgi_encoder_save(encoder, image->filename, LgiProfile_Balanced) != 0) {
        RelinquishMagickMemory(data);
        lgi_encoder_destroy(encoder);
        (void) ThrowMagickException(exception, GetMagickModule(), CoderError,
            "Failed to save LGI file", "`%s'", image->filename);
        return MagickFalse;
    }

    // Cleanup
    RelinquishMagickMemory(data);
    lgi_encoder_destroy(encoder);
    CloseBlob(image);

    return MagickTrue;
}

/*
 * Module registration
 */
ModuleExport size_t RegisterLGIImage(void)
{
    MagickInfo *entry;

    entry = AcquireMagickInfo("LGI", "LGI", "Lamco Gaussian Image");
    entry->decoder = (DecodeImageHandler *) ReadLGIImage;
    entry->encoder = (EncodeImageHandler *) WriteLGIImage;
    entry->magick = (IsImageFormatHandler *) IsLGI;
    entry->flags |= CoderDecoderSeekableStreamFlag;
    entry->flags ^= CoderAdjoinFlag;

    (void) RegisterMagickInfo(entry);

    return MagickImageCoderSignature;
}

ModuleExport void UnregisterLGIImage(void)
{
    (void) UnregisterMagickInfo("LGI");
}

/* Check if file is LGI format */
static MagickBooleanType IsLGI(const unsigned char *magick, const size_t length)
{
    if (length < 4)
        return MagickFalse;

    // Check magic number "LGI\0"
    if (magick[0] == 'L' && magick[1] == 'G' && magick[2] == 'I' && magick[3] == 0)
        return MagickTrue;

    return MagickFalse;
}
