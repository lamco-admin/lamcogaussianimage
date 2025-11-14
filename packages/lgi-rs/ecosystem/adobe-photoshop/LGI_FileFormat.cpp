/*
 * Adobe Photoshop LGI File Format Plugin
 * Copyright (c) 2025 LGI Project
 *
 * Photoshop SDK plugin for reading/writing LGI files
 */

#include "PIDefines.h"
#include "PITypes.h"
#include "PIFormat.h"
#include "FileUtilities.h"

extern "C" {
    #include <lgi.h>
}

// Plugin entry points
DLLExport MACPASCAL void PluginMain(const int16 selector,
                                    FormatRecordPtr formatRecord,
                                    intptr_t *data,
                                    int16 *result);

// Resource definitions
#define plugInClassID       'lgiP'
#define plugInEventID       '----'
#define vendorName          "LGI Project"
#define plugInName          "LGI Format"
#define plugInCopyrightYear "2025"

// Plugin data structure
typedef struct {
    LgiDecoder *decoder;
    LgiEncoder *encoder;
    float *imageData;
    int32 width;
    int32 height;
} LGIData;

// Forward declarations
static void DoReadPrepare(FormatRecordPtr formatRecord, LGIData *data, int16 *result);
static void DoReadStart(FormatRecordPtr formatRecord, LGIData *data, int16 *result);
static void DoReadContinue(FormatRecordPtr formatRecord, LGIData *data, int16 *result);
static void DoReadFinish(FormatRecordPtr formatRecord, LGIData *data, int16 *result);

static void DoWritePrepare(FormatRecordPtr formatRecord, LGIData *data, int16 *result);
static void DoWriteStart(FormatRecordPtr formatRecord, LGIData *data, int16 *result);
static void DoWriteContinue(FormatRecordPtr formatRecord, LGIData *data, int16 *result);
static void DoWriteFinish(FormatRecordPtr formatRecord, LGIData *data, int16 *result);

// Main plugin entry point
DLLExport MACPASCAL void PluginMain(const int16 selector,
                                    FormatRecordPtr formatRecord,
                                    intptr_t *data,
                                    int16 *result)
{
    // Allocate data structure on first call
    if (selector == formatSelectorAbout) {
        // Show about dialog
        // TODO: Implement about dialog
        *result = noErr;
        return;
    }

    if (*data == 0) {
        *data = (intptr_t)new LGIData();
        if (*data == 0) {
            *result = memFullErr;
            return;
        }
        LGIData *pluginData = (LGIData*)*data;
        pluginData->decoder = nullptr;
        pluginData->encoder = nullptr;
        pluginData->imageData = nullptr;
    }

    LGIData *pluginData = (LGIData*)*data;

    // Handle selectors
    switch (selector) {
        case formatSelectorReadPrepare:
            DoReadPrepare(formatRecord, pluginData, result);
            break;

        case formatSelectorReadStart:
            DoReadStart(formatRecord, pluginData, result);
            break;

        case formatSelectorReadContinue:
            DoReadContinue(formatRecord, pluginData, result);
            break;

        case formatSelectorReadFinish:
            DoReadFinish(formatRecord, pluginData, result);
            break;

        case formatSelectorWritePrepare:
            DoWritePrepare(formatRecord, pluginData, result);
            break;

        case formatSelectorWriteStart:
            DoWriteStart(formatRecord, pluginData, result);
            break;

        case formatSelectorWriteContinue:
            DoWriteContinue(formatRecord, pluginData, result);
            break;

        case formatSelectorWriteFinish:
            DoWriteFinish(formatRecord, pluginData, result);
            break;

        default:
            *result = formatBadParameters;
    }

    // Cleanup on finish
    if (selector == formatSelectorReadFinish || selector == formatSelectorWriteFinish) {
        delete pluginData;
        *data = 0;
    }
}

// Read implementation
static void DoReadPrepare(FormatRecordPtr formatRecord, LGIData *data, int16 *result)
{
    // Check if file has LGI magic number
    unsigned char magic[4];
    int32 count = 4;

    *result = PSSDKRead(formatRecord->dataFork, &count, magic);
    if (*result != noErr) return;

    // Reset file position
    *result = PSSDKSetFPos(formatRecord->dataFork, fsFromStart, 0);
    if (*result != noErr) return;

    // Check magic "LGI\0"
    if (magic[0] != 'L' || magic[1] != 'G' || magic[2] != 'I' || magic[3] != 0) {
        *result = formatCannotRead;
        return;
    }

    // Get file path
    // Create decoder
    data->decoder = lgi_decoder_create();
    if (!data->decoder) {
        *result = memFullErr;
        return;
    }

    *result = noErr;
}

static void DoReadStart(FormatRecordPtr formatRecord, LGIData *data, int16 *result)
{
    // Load and decode file
    // (Photoshop provides file handle, we need to save to temp file)
    // TODO: Implement file reading from handle

    char tmpfile[] = "/tmp/lgi_ps_XXXXXX";
    int fd = mkstemp(tmpfile);
    if (fd < 0) {
        *result = ioErr;
        return;
    }

    // Copy data from Photoshop file to temp file
    // TODO: Implement proper file copying

    close(fd);

    // Load LGI file
    if (lgi_decoder_load(data->decoder, tmpfile) != Success) {
        unlink(tmpfile);
        *result = formatCannotRead;
        return;
    }

    // Decode
    if (lgi_decoder_decode(data->decoder) != Success) {
        unlink(tmpfile);
        *result = formatCannotRead;
        return;
    }

    // Get dimensions
    unsigned int width, height;
    lgi_decoder_get_dimensions(data->decoder, &width, &height);

    data->width = width;
    data->height = height;

    // Set format record dimensions
    formatRecord->imageSize.h = width;
    formatRecord->imageSize.v = height;
    formatRecord->depth = 32;  // 32-bit float
    formatRecord->imageMode = plugInModeRGBColor;
    formatRecord->planes = 3;  // RGB

    // Allocate image data
    size_t data_size = width * height * 4 * sizeof(float);
    data->imageData = (float*)malloc(data_size);
    if (!data->imageData) {
        unlink(tmpfile);
        *result = memFullErr;
        return;
    }

    // Get image data
    lgi_decoder_get_data(data->decoder, data->imageData);
    unlink(tmpfile);

    *result = noErr;
}

static void DoReadContinue(FormatRecordPtr formatRecord, LGIData *data, int16 *result)
{
    // Transfer data to Photoshop
    // TODO: Implement scanline transfer to Photoshop buffer

    *result = noErr;
}

static void DoReadFinish(FormatRecordPtr formatRecord, LGIData *data, int16 *result)
{
    // Cleanup
    if (data->imageData) {
        free(data->imageData);
        data->imageData = nullptr;
    }

    if (data->decoder) {
        lgi_decoder_destroy(data->decoder);
        data->decoder = nullptr;
    }

    *result = noErr;
}

// Write implementation (similar structure)
static void DoWritePrepare(FormatRecordPtr formatRecord, LGIData *data, int16 *result)
{
    // Prepare for writing
    data->width = formatRecord->imageSize.h;
    data->height = formatRecord->imageSize.v;

    data->encoder = lgi_encoder_create(Balanced);
    if (!data->encoder) {
        *result = memFullErr;
        return;
    }

    *result = noErr;
}

static void DoWriteStart(FormatRecordPtr formatRecord, LGIData *data, int16 *result)
{
    // Allocate buffer
    size_t data_size = data->width * data->height * 4 * sizeof(float);
    data->imageData = (float*)malloc(data_size);
    if (!data->imageData) {
        *result = memFullErr;
        return;
    }

    // TODO: Read image data from Photoshop

    *result = noErr;
}

static void DoWriteContinue(FormatRecordPtr formatRecord, LGIData *data, int16 *result)
{
    // Continue receiving scanlines from Photoshop
    // TODO: Implement scanline reception

    *result = noErr;
}

static void DoWriteFinish(FormatRecordPtr formatRecord, LGIData *data, int16 *result)
{
    // Encode and save
    if (lgi_encoder_set_image(data->encoder, data->width, data->height, data->imageData) != Success) {
        *result = formatCannotWrite;
        return;
    }

    int num_gaussians = (data->width * data->height) / 64;
    if (lgi_encoder_encode(data->encoder, num_gaussians) != Success) {
        *result = formatCannotWrite;
        return;
    }

    // TODO: Get filename from formatRecord and save

    // Cleanup
    if (data->imageData) {
        free(data->imageData);
        data->imageData = nullptr;
    }

    if (data->encoder) {
        lgi_encoder_destroy(data->encoder);
        data->encoder = nullptr;
    }

    *result = noErr;
}
