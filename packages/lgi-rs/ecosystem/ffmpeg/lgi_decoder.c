/*
 * LGI decoder for FFmpeg
 * Copyright (c) 2025 LGI Project
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 */

/**
 * @file
 * LGI (Lamco Gaussian Image) decoder
 */

#include "avcodec.h"
#include "codec_internal.h"
#include "libavutil/imgutils.h"
#include <lgi.h>

typedef struct LGIContext {
    LgiDecoder *decoder;
    AVFrame *frame;
} LGIContext;

static av_cold int lgi_decode_init(AVCodecContext *avctx)
{
    LGIContext *s = avctx->priv_data;

    // Create LGI decoder
    s->decoder = lgi_decoder_create();
    if (!s->decoder)
        return AVERROR(ENOMEM);

    // Set pixel format to RGB float
    avctx->pix_fmt = AV_PIX_FMT_RGBF32LE;

    s->frame = av_frame_alloc();
    if (!s->frame)
        return AVERROR(ENOMEM);

    return 0;
}

static int lgi_decode_frame(AVCodecContext *avctx, AVFrame *frame,
                            int *got_frame, AVPacket *avpkt)
{
    LGIContext *s = avctx->priv_data;
    int ret;

    // Write packet data to temporary file
    // (FFmpeg provides data as buffer, LGI API uses files)
    // TODO: Add buffer-based API to lgi-ffi
    char tmpfile[] = "/tmp/lgi_decode_XXXXXX";
    int fd = mkstemp(tmpfile);
    if (fd < 0)
        return AVERROR(EIO);

    write(fd, avpkt->data, avpkt->size);
    close(fd);

    // Load .lgi file
    if (lgi_decoder_load(s->decoder, tmpfile) != 0) {
        unlink(tmpfile);
        return AVERROR_INVALIDDATA;
    }

    // Decode
    if (lgi_decoder_decode(s->decoder) != 0) {
        unlink(tmpfile);
        return AVERROR(EINVAL);
    }

    // Get dimensions
    unsigned int width, height;
    lgi_decoder_get_dimensions(s->decoder, &width, &height);

    // Set frame properties
    frame->width = width;
    frame->height = height;
    frame->format = AV_PIX_FMT_RGBF32LE;

    // Allocate frame buffer
    ret = av_frame_get_buffer(frame, 0);
    if (ret < 0) {
        unlink(tmpfile);
        return ret;
    }

    // Get decoded data
    lgi_decoder_get_data(s->decoder, (float*)frame->data[0]);

    unlink(tmpfile);

    *got_frame = 1;
    return avpkt->size;
}

static av_cold int lgi_decode_close(AVCodecContext *avctx)
{
    LGIContext *s = avctx->priv_data;

    if (s->decoder)
        lgi_decoder_destroy(s->decoder);

    av_frame_free(&s->frame);

    return 0;
}

const FFCodec ff_lgi_decoder = {
    .p.name         = "lgi",
    .p.long_name    = NULL_IF_CONFIG_SMALL("LGI (Lamco Gaussian Image)"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_LGI,
    .priv_data_size = sizeof(LGIContext),
    .init           = lgi_decode_init,
    FF_CODEC_DECODE_CB(lgi_decode_frame),
    .close          = lgi_decode_close,
    .p.capabilities = AV_CODEC_CAP_DR1,
};
