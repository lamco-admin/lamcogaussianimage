/*
 * Test C FFI library
 * Compile: gcc -o test_ffi test_ffi.c -L../../target/release -llgi_ffi -lm
 * Run: LD_LIBRARY_PATH=../../target/release ./test_ffi
 */

#include "../lgi-ffi/include/lgi.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("=================================================\n");
    printf("  LGI C FFI Test\n");
    printf("=================================================\n\n");

    // Check version
    const char *version = lgi_version();
    printf("LGI Version: %s\n", version);
    printf("GPU Support: %s\n\n", lgi_has_gpu_support() ? "Yes" : "No");

    // Test encoding workflow
    printf("Testing Encoder...\n");

    // Create test image (256×256 gradient)
    unsigned int width = 256;
    unsigned int height = 256;
    float *image_data = (float*)malloc(width * height * 4 * sizeof(float));

    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            unsigned int idx = (y * width + x) * 4;
            float val = (float)x / width;
            image_data[idx] = val;     // R
            image_data[idx + 1] = 0.5f;  // G
            image_data[idx + 2] = 0.5f;  // B
            image_data[idx + 3] = 1.0f;  // A
        }
    }

    // Create encoder
    struct LgiEncoder *encoder = lgi_encoder_create(Balanced);
    if (!encoder) {
        printf("❌ Failed to create encoder\n");
        return 1;
    }
    printf("✅ Encoder created\n");

    // Set image
    enum LgiErrorCode err = lgi_encoder_set_image(encoder, width, height, image_data);
    if (err != Success) {
        printf("❌ Failed to set image: %d\n", err);
        return 1;
    }
    printf("✅ Image set (%u×%u)\n", width, height);

    // Encode (fast for testing - 200 Gaussians)
    printf("Encoding with 200 Gaussians (this may take 10-30 seconds)...\n");
    err = lgi_encoder_encode(encoder, 200);
    if (err != Success) {
        printf("❌ Encoding failed: %d\n", err);
        return 1;
    }
    printf("✅ Encoding complete\n");

    // Save
    const char *output_file = "/tmp/test_ffi.lgi";
    err = lgi_encoder_save(encoder, output_file, Balanced);
    if (err != Success) {
        printf("❌ Save failed: %d\n", err);
        return 1;
    }
    printf("✅ Saved to %s\n", output_file);

    // Cleanup encoder
    lgi_encoder_destroy(encoder);

    // Test decoding workflow
    printf("\nTesting Decoder...\n");

    // Create decoder
    struct LgiDecoder *decoder = lgi_decoder_create();
    if (!decoder) {
        printf("❌ Failed to create decoder\n");
        return 1;
    }
    printf("✅ Decoder created\n");

    // Load file
    err = lgi_decoder_load(decoder, output_file);
    if (err != Success) {
        printf("❌ Load failed: %d\n", err);
        return 1;
    }
    printf("✅ File loaded\n");

    // Decode
    err = lgi_decoder_decode(decoder);
    if (err != Success) {
        printf("❌ Decode failed: %d\n", err);
        return 1;
    }
    printf("✅ Decoded\n");

    // Get dimensions
    unsigned int dec_width, dec_height;
    err = lgi_decoder_get_dimensions(decoder, &dec_width, &dec_height);
    if (err != Success) {
        printf("❌ Get dimensions failed: %d\n", err);
        return 1;
    }
    printf("✅ Dimensions: %u×%u\n", dec_width, dec_height);

    // Get data
    float *decoded_data = (float*)malloc(dec_width * dec_height * 4 * sizeof(float));
    err = lgi_decoder_get_data(decoder, decoded_data);
    if (err != Success) {
        printf("❌ Get data failed: %d\n", err);
        return 1;
    }
    printf("✅ Data retrieved\n");

    // Verify dimensions match
    if (dec_width != width || dec_height != height) {
        printf("❌ Dimension mismatch: %u×%u vs %u×%u\n",
            dec_width, dec_height, width, height);
        return 1;
    }
    printf("✅ Dimensions match\n");

    // Cleanup
    free(image_data);
    free(decoded_data);
    lgi_decoder_destroy(decoder);

    printf("\n=================================================\n");
    printf("  ✅ All C FFI tests passed!\n");
    printf("=================================================\n\n");

    printf("Next steps:\n");
    printf("  1. This library can now be used by FFmpeg\n");
    printf("  2. This library can now be used by ImageMagick\n");
    printf("  3. This library can now be used by GIMP\n");
    printf("  4. Any C/C++ application can use LGI codec\n\n");

    return 0;
}
