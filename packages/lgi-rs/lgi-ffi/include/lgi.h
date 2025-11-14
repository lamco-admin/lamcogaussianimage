#ifndef LGI_H
#define LGI_H

#pragma once

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

/**
 * Error codes
 */
typedef enum LgiErrorCode {
  Success = 0,
  InvalidParameter = 1,
  FileNotFound = 2,
  EncodingFailed = 3,
  DecodingFailed = 4,
  OutOfMemory = 5,
  InvalidFormat = 6,
  IoError = 7,
} LgiErrorCode;

/**
 * Compression profile selection
 */
typedef enum LgiProfile {
  Balanced = 0,
  Small = 1,
  High = 2,
  Lossless = 3,
} LgiProfile;

/**
 * Opaque encoder handle
 */
typedef struct LgiEncoder {
  uint8_t _private[0];
} LgiEncoder;

/**
 * Opaque decoder handle
 */
typedef struct LgiDecoder {
  uint8_t _private[0];
} LgiDecoder;

/**
 * Create encoder
 */
struct LgiEncoder *lgi_encoder_create(enum LgiProfile profile);

/**
 * Set input image
 */
enum LgiErrorCode lgi_encoder_set_image(struct LgiEncoder *encoder,
                                        unsigned int width,
                                        unsigned int height,
                                        const float *data);

/**
 * Encode to Gaussians
 */
enum LgiErrorCode lgi_encoder_encode(struct LgiEncoder *encoder, unsigned int num_gaussians);

/**
 * Save to file
 */
enum LgiErrorCode lgi_encoder_save(struct LgiEncoder *encoder,
                                   const char *filename,
                                   enum LgiProfile profile);

/**
 * Save to memory buffer (for streaming/blob operations)
 */
enum LgiErrorCode lgi_encoder_save_to_buffer(struct LgiEncoder *encoder,
                                             enum LgiProfile profile,
                                             uint8_t **data_out,
                                             uintptr_t *size_out);

/**
 * Free buffer allocated by encoder
 */
void lgi_free_buffer(uint8_t *data, uintptr_t size);

/**
 * Destroy encoder
 */
void lgi_encoder_destroy(struct LgiEncoder *encoder);

/**
 * Create decoder
 */
struct LgiDecoder *lgi_decoder_create(void);

/**
 * Load file from path
 */
enum LgiErrorCode lgi_decoder_load(struct LgiDecoder *decoder, const char *filename);

/**
 * Load from memory buffer (for streaming/blob operations)
 */
enum LgiErrorCode lgi_decoder_load_from_buffer(struct LgiDecoder *decoder,
                                               const uint8_t *data,
                                               uintptr_t size);

/**
 * Decode to image
 */
enum LgiErrorCode lgi_decoder_decode(struct LgiDecoder *decoder);

/**
 * Get dimensions
 */
enum LgiErrorCode lgi_decoder_get_dimensions(const struct LgiDecoder *decoder,
                                             unsigned int *width,
                                             unsigned int *height);

/**
 * Get image data
 */
enum LgiErrorCode lgi_decoder_get_data(const struct LgiDecoder *decoder, float *data);

/**
 * Destroy decoder
 */
void lgi_decoder_destroy(struct LgiDecoder *decoder);

/**
 * Get version
 */
const char *lgi_version(void);

/**
 * Check GPU support
 */
int lgi_has_gpu_support(void);

#endif  /* LGI_H */
