/*
 * LGI encoder for FFmpeg
 * Copyright (c) 2025 LGI Project
 *
 * This file is part of FFmpeg.
 */

/**
 * @file
 * LGI (Lamco Gaussian Image) encoder
 */

#include "avcodec.h"
#include "codec_internal.h"
#include "encode.h"
#include "libavutil/opt.h"
#include "libavutil/imgutils.h"
#include <lgi.h>

typedef struct LGIEncContext {
    AVClass *class;
    LgiEncoder *encoder;

    /* Encoding options */
    int num_gaussians;
    int quality;  // 0=small, 1=balanced, 2=high, 3=lossless
    int enable_qa_training;
} LGIEncContext;

static av_cold int lgi_encode_init(AVCodecContext *avctx)
{
    LGIEncContext *s = avctx->priv_data;

    /* Map quality setting to LGI profile */
    LgiProfile profile;
    switch (s->quality) {
        case 0: profile = Small; break;
        case 1: profile = Balanced; break;
        case 2: profile = High; break;
        case 3: profile = Lossless; break;
        default: profile = Balanced;
    }

    /* Create encoder */
    s->encoder = lgi_encoder_create(profile);
    if (!s->encoder)
        return AVERROR(ENOMEM);

    /* Auto-determine Gaussian count if not set */
    if (s->num_gaussians <= 0) {
        /* Use 1.5% of pixels as default */
        s->num_gaussians = (avctx->width * avctx->height) / 64;
        if (s->num_gaussians < 100) s->num_gaussians = 100;
        if (s->num_gaussians > 10000) s->num_gaussians = 10000;
    }

    av_log(avctx, AV_LOG_INFO, "LGI Encoder: %dx%d, %d Gaussians, profile=%d\n",
           avctx->width, avctx->height, s->num_gaussians, s->quality);

    return 0;
}

static int lgi_encode_frame(AVCodecContext *avctx, AVPacket *pkt,
                            const AVFrame *frame, int *got_packet)
{
    LGIEncContext *s = avctx->priv_data;
    int ret;

    /* Convert frame to float RGBA */
    size_t data_size = avctx->width * avctx->height * 4 * sizeof(float);
    float *rgba_data = av_malloc(data_size);
    if (!rgba_data)
        return AVERROR(ENOMEM);

    /* Convert from AVFrame to float RGBA */
    /* TODO: Handle different pixel formats properly */
    for (int y = 0; y < avctx->height; y++) {
        for (int x = 0; x < avctx->width; x++) {
            int src_idx = y * frame->linesize[0] + x * 3;
            int dst_idx = (y * avctx->width + x) * 4;

            /* Simple RGB -> float conversion (assuming RGB24 input) */
            rgba_data[dst_idx + 0] = frame->data[0][src_idx + 0] / 255.0f;
            rgba_data[dst_idx + 1] = frame->data[0][src_idx + 1] / 255.0f;
            rgba_data[dst_idx + 2] = frame->data[0][src_idx + 2] / 255.0f;
            rgba_data[dst_idx + 3] = 1.0f;
        }
    }

    /* Set image data */
    if (lgi_encoder_set_image(s->encoder, avctx->width, avctx->height, rgba_data) != Success) {
        av_free(rgba_data);
        return AVERROR(EINVAL);
    }

    /* Encode to Gaussians */
    av_log(avctx, AV_LOG_INFO, "Encoding with %d Gaussians (this may take 30-60s)...\n",
           s->num_gaussians);
    if (lgi_encoder_encode(s->encoder, s->num_gaussians) != Success) {
        av_free(rgba_data);
        return AVERROR(EINVAL);
    }

    /* Save to temporary file, then read into packet */
    char tmpfile[] = "/tmp/lgi_encode_XXXXXX";
    int fd = mkstemp(tmpfile);
    if (fd < 0) {
        av_free(rgba_data);
        return AVERROR(EIO);
    }
    close(fd);

    LgiProfile save_profile = (s->quality == 3) ? Lossless : Balanced;
    if (lgi_encoder_save(s->encoder, tmpfile, save_profile) != Success) {
        unlink(tmpfile);
        av_free(rgba_data);
        return AVERROR(EIO);
    }

    /* Read file into packet */
    FILE *fp = fopen(tmpfile, "rb");
    if (!fp) {
        unlink(tmpfile);
        av_free(rgba_data);
        return AVERROR(EIO);
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    ret = ff_get_encode_buffer(avctx, pkt, file_size, 0);
    if (ret < 0) {
        fclose(fp);
        unlink(tmpfile);
        av_free(rgba_data);
        return ret;
    }

    fread(pkt->data, 1, file_size, fp);
    fclose(fp);
    unlink(tmpfile);
    av_free(rgba_data);

    pkt->flags |= AV_PKT_FLAG_KEY;  /* All frames are keyframes for now */
    *got_packet = 1;

    av_log(avctx, AV_LOG_INFO, "Encoded to %ld bytes (%.1fx compression)\n",
           file_size, (float)(avctx->width * avctx->height * 3) / file_size);

    return 0;
}

static av_cold int lgi_encode_close(AVCodecContext *avctx)
{
    LGIEncContext *s = avctx->priv_data;

    if (s->encoder)
        lgi_encoder_destroy(s->encoder);

    return 0;
}

#define OFFSET(x) offsetof(LGIEncContext, x)
#define VE AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM
static const AVOption lgi_options[] = {
    { "gaussians", "Number of Gaussians", OFFSET(num_gaussians), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 50000, VE },
    { "quality", "Quality level (0=small, 1=balanced, 2=high, 3=lossless)", OFFSET(quality), AV_OPT_TYPE_INT, { .i64 = 1 }, 0, 3, VE },
    { "qa_training", "Enable Quantization-Aware training", OFFSET(enable_qa_training), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, VE },
    { NULL }
};

static const AVClass lgi_enc_class = {
    .class_name = "LGI encoder",
    .item_name  = av_default_item_name,
    .option     = lgi_options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const FFCodec ff_lgi_encoder = {
    .p.name         = "lgi",
    .p.long_name    = NULL_IF_CONFIG_SMALL("LGI (Lamco Gaussian Image)"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_LGI,
    .priv_data_size = sizeof(LGIEncContext),
    .init           = lgi_encode_init,
    FF_CODEC_ENCODE_CB(lgi_encode_frame),
    .close          = lgi_encode_close,
    .p.capabilities = AV_CODEC_CAP_DR1,
    .p.priv_class   = &lgi_enc_class,
    .p.pix_fmts     = (const enum AVPixelFormat[]) {
        AV_PIX_FMT_RGB24,
        AV_PIX_FMT_RGBA,
        AV_PIX_FMT_RGBF32LE,
        AV_PIX_FMT_NONE
    },
};
