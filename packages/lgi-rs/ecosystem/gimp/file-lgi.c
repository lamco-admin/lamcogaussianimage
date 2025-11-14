/*
 * GIMP LGI file format plugin
 * Copyright (c) 2025 LGI Project
 *
 * Load and save LGI (Lamco Gaussian Image) files in GIMP
 */

#include <libgimp/gimp.h>
#include <libgimp/gimpui.h>
#include <lgi.h>

#define PLUG_IN_BINARY "file-lgi"
#define PLUG_IN_ROLE   "gimp-file-lgi"

typedef struct {
    gint num_gaussians;
    gint quality;  /* 0=small, 1=balanced, 2=high, 3=lossless */
    gboolean qa_training;
} LGISaveOptions;

/* Default save options */
static LGISaveOptions lgi_defaults = {
    .num_gaussians = 1000,
    .quality = 1,  /* Balanced */
    .qa_training = TRUE
};

/* Function prototypes */
static void query(void);
static void run(const gchar *name, gint nparams, const GimpParam *param,
                gint *nreturn_vals, GimpParam **return_vals);
static gint32 load_image(const gchar *filename, GimpRunMode run_mode);
static gboolean save_image(const gchar *filename, gint32 image_id,
                           gint32 drawable_id, GimpRunMode run_mode);
static gboolean save_dialog(void);

/* GIMP plugin structure */
const GimpPlugIn PLUG_IN_INFO = {
    NULL,  /* init_proc */
    NULL,  /* quit_proc */
    query, /* query_proc */
    run    /* run_proc */
};

MAIN()

static void query(void)
{
    static const GimpParamDef load_args[] = {
        { GIMP_PDB_INT32, "run-mode", "The run mode { RUN-INTERACTIVE (0), RUN-NONINTERACTIVE (1) }" },
        { GIMP_PDB_STRING, "filename", "The name of the file to load" },
        { GIMP_PDB_STRING, "raw-filename", "The name entered" }
    };

    static const GimpParamDef load_return_vals[] = {
        { GIMP_PDB_IMAGE, "image", "Output image" }
    };

    static const GimpParamDef save_args[] = {
        { GIMP_PDB_INT32, "run-mode", "The run mode { RUN-INTERACTIVE (0), RUN-NONINTERACTIVE (1) }" },
        { GIMP_PDB_IMAGE, "image", "Input image" },
        { GIMP_PDB_DRAWABLE, "drawable", "Drawable to save" },
        { GIMP_PDB_STRING, "filename", "The name of the file to save" },
        { GIMP_PDB_STRING, "raw-filename", "The name entered" },
        { GIMP_PDB_INT32, "num-gaussians", "Number of Gaussians (0=auto)" },
        { GIMP_PDB_INT32, "quality", "Quality (0=small, 1=balanced, 2=high, 3=lossless)" }
    };

    gimp_install_procedure(
        "file-lgi-load",
        "Loads LGI (Gaussian Image) files",
        "Loads Lamco Gaussian Image format files",
        "LGI Project",
        "LGI Project",
        "2025",
        "LGI Image",
        NULL,
        GIMP_PLUGIN,
        G_N_ELEMENTS(load_args),
        G_N_ELEMENTS(load_return_vals),
        load_args,
        load_return_vals);

    gimp_register_file_handler_mime("file-lgi-load", "image/x-lgi");
    gimp_register_load_handler("file-lgi-load", "lgi", "");

    gimp_install_procedure(
        "file-lgi-save",
        "Saves files in LGI format",
        "Saves files in Lamco Gaussian Image format with compression",
        "LGI Project",
        "LGI Project",
        "2025",
        "LGI Image",
        "RGB*, GRAY*",
        GIMP_PLUGIN,
        G_N_ELEMENTS(save_args),
        0,
        save_args,
        NULL);

    gimp_register_file_handler_mime("file-lgi-save", "image/x-lgi");
    gimp_register_save_handler("file-lgi-save", "lgi", "");
}

static void run(const gchar *name, gint nparams, const GimpParam *param,
                gint *nreturn_vals, GimpParam **return_vals)
{
    static GimpParam values[2];
    GimpRunMode run_mode;
    GimpPDBStatusType status = GIMP_PDB_SUCCESS;
    gint32 image_ID;

    run_mode = param[0].data.d_int32;

    *nreturn_vals = 1;
    *return_vals = values;

    values[0].type = GIMP_PDB_STATUS;
    values[0].data.d_status = GIMP_PDB_EXECUTION_ERROR;

    if (strcmp(name, "file-lgi-load") == 0) {
        image_ID = load_image(param[1].data.d_string, run_mode);

        if (image_ID != -1) {
            *nreturn_vals = 2;
            values[0].data.d_status = GIMP_PDB_SUCCESS;
            values[1].type = GIMP_PDB_IMAGE;
            values[1].data.d_image = image_ID;
        } else {
            status = GIMP_PDB_EXECUTION_ERROR;
        }
    } else if (strcmp(name, "file-lgi-save") == 0) {
        gint32 drawable_ID;

        image_ID = param[1].data.d_int32;
        drawable_ID = param[2].data.d_int32;

        /* Get save parameters */
        if (nparams >= 6) {
            lgi_defaults.num_gaussians = param[5].data.d_int32;
        }
        if (nparams >= 7) {
            lgi_defaults.quality = param[6].data.d_int32;
        }

        if (save_image(param[3].data.d_string, image_ID, drawable_ID, run_mode)) {
            values[0].data.d_status = GIMP_PDB_SUCCESS;
        } else {
            status = GIMP_PDB_EXECUTION_ERROR;
        }
    }

    values[0].data.d_status = status;
}

static gint32 load_image(const gchar *filename, GimpRunMode run_mode)
{
    gint32 image_ID, layer_ID;
    LgiDecoder *decoder;
    unsigned int width, height;
    float *data;
    GimpPixelRgn pixel_rgn;
    GimpDrawable *drawable;
    guchar *row_data;

    /* Create decoder */
    decoder = lgi_decoder_create();
    if (!decoder) {
        g_message("Failed to create LGI decoder");
        return -1;
    }

    /* Load file */
    if (lgi_decoder_load(decoder, filename) != Success) {
        lgi_decoder_destroy(decoder);
        g_message("Failed to load LGI file: %s", filename);
        return -1;
    }

    /* Decode */
    if (lgi_decoder_decode(decoder) != Success) {
        lgi_decoder_destroy(decoder);
        g_message("Failed to decode LGI file");
        return -1;
    }

    /* Get dimensions */
    lgi_decoder_get_dimensions(decoder, &width, &height);

    /* Create GIMP image */
    image_ID = gimp_image_new(width, height, GIMP_RGB);
    gimp_image_set_filename(image_ID, filename);

    layer_ID = gimp_layer_new(image_ID, "Background", width, height,
                              GIMP_RGBA_IMAGE, 100, GIMP_NORMAL_MODE);
    gimp_image_insert_layer(image_ID, layer_ID, -1, 0);

    /* Get drawable */
    drawable = gimp_drawable_get(layer_ID);

    /* Allocate data */
    data = g_new(float, width * height * 4);
    lgi_decoder_get_data(decoder, data);

    /* Convert float RGBA to GIMP format */
    row_data = g_new(guchar, width * 4);

    gimp_pixel_rgn_init(&pixel_rgn, drawable, 0, 0, width, height, TRUE, FALSE);

    for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
            unsigned int src_idx = (y * width + x) * 4;
            unsigned int dst_idx = x * 4;

            row_data[dst_idx + 0] = (guchar)(data[src_idx + 0] * 255.0f);
            row_data[dst_idx + 1] = (guchar)(data[src_idx + 1] * 255.0f);
            row_data[dst_idx + 2] = (guchar)(data[src_idx + 2] * 255.0f);
            row_data[dst_idx + 3] = (guchar)(data[src_idx + 3] * 255.0f);
        }
        gimp_pixel_rgn_set_row(&pixel_rgn, row_data, 0, y, width);
    }

    gimp_drawable_flush(drawable);
    gimp_drawable_detach(drawable);

    /* Cleanup */
    g_free(data);
    g_free(row_data);
    lgi_decoder_destroy(decoder);

    return image_ID;
}

static gboolean save_image(const gchar *filename, gint32 image_ID,
                           gint32 drawable_ID, GimpRunMode run_mode)
{
    GimpPixelRgn pixel_rgn;
    GimpDrawable *drawable;
    LgiEncoder *encoder;
    float *data;
    guchar *row_data;
    gint width, height;
    LgiProfile profile;

    /* Show save dialog in interactive mode */
    if (run_mode == GIMP_RUN_INTERACTIVE) {
        if (!save_dialog())
            return FALSE;
    }

    /* Get drawable */
    drawable = gimp_drawable_get(drawable_ID);
    width = drawable->width;
    height = drawable->height;

    /* Allocate buffers */
    data = g_new(float, width * height * 4);
    row_data = g_new(guchar, width * 4);

    /* Read pixels from GIMP */
    gimp_pixel_rgn_init(&pixel_rgn, drawable, 0, 0, width, height, FALSE, FALSE);

    for (gint y = 0; y < height; y++) {
        gimp_pixel_rgn_get_row(&pixel_rgn, row_data, 0, y, width);

        for (gint x = 0; x < width; x++) {
            gint src_idx = x * 4;
            gint dst_idx = (y * width + x) * 4;

            data[dst_idx + 0] = row_data[src_idx + 0] / 255.0f;
            data[dst_idx + 1] = row_data[src_idx + 1] / 255.0f;
            data[dst_idx + 2] = row_data[src_idx + 2] / 255.0f;
            data[dst_idx + 3] = row_data[src_idx + 3] / 255.0f;
        }
    }

    /* Create encoder */
    switch (lgi_defaults.quality) {
        case 0: profile = Small; break;
        case 1: profile = Balanced; break;
        case 2: profile = High; break;
        case 3: profile = Lossless; break;
        default: profile = Balanced;
    }

    encoder = lgi_encoder_create(profile);
    if (!encoder) {
        g_free(data);
        g_free(row_data);
        gimp_drawable_detach(drawable);
        g_message("Failed to create LGI encoder");
        return FALSE;
    }

    /* Set image */
    if (lgi_encoder_set_image(encoder, width, height, data) != Success) {
        lgi_encoder_destroy(encoder);
        g_free(data);
        g_free(row_data);
        gimp_drawable_detach(drawable);
        g_message("Failed to set image data");
        return FALSE;
    }

    /* Auto Gaussian count if not set */
    gint num_gaussians = lgi_defaults.num_gaussians;
    if (num_gaussians <= 0) {
        num_gaussians = (width * height) / 64;  /* 1.5% of pixels */
        if (num_gaussians < 100) num_gaussians = 100;
        if (num_gaussians > 10000) num_gaussians = 10000;
    }

    /* Show progress */
    gimp_progress_init_printf("Encoding %s...", filename);

    /* Encode */
    if (lgi_encoder_encode(encoder, num_gaussians) != Success) {
        lgi_encoder_destroy(encoder);
        g_free(data);
        g_free(row_data);
        gimp_drawable_detach(drawable);
        g_message("Encoding failed");
        return FALSE;
    }

    gimp_progress_update(0.9);

    /* Save */
    if (lgi_encoder_save(encoder, filename, profile) != Success) {
        lgi_encoder_destroy(encoder);
        g_free(data);
        g_free(row_data);
        gimp_drawable_detach(drawable);
        g_message("Failed to save file: %s", filename);
        return FALSE;
    }

    gimp_progress_update(1.0);

    /* Cleanup */
    g_free(data);
    g_free(row_data);
    lgi_encoder_destroy(encoder);
    gimp_drawable_detach(drawable);

    return TRUE;
}

static gboolean save_dialog(void)
{
    GtkWidget *dialog;
    GtkWidget *vbox;
    GtkWidget *table;
    GtkWidget *spinbutton;
    GtkWidget *combo;
    GtkWidget *checkbutton;
    gboolean run;

    gimp_ui_init(PLUG_IN_BINARY, FALSE);

    dialog = gimp_export_dialog_new("LGI", PLUG_IN_BINARY, PLUG_IN_ROLE);

    vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 12);
    gtk_container_set_border_width(GTK_CONTAINER(vbox), 12);
    gtk_box_pack_start(GTK_BOX(gimp_export_dialog_get_content_area(dialog)),
                      vbox, TRUE, TRUE, 0);
    gtk_widget_show(vbox);

    /* Gaussian count */
    table = gtk_table_new(3, 2, FALSE);
    gtk_table_set_col_spacings(GTK_TABLE(table), 6);
    gtk_table_set_row_spacings(GTK_TABLE(table), 6);
    gtk_box_pack_start(GTK_BOX(vbox), table, FALSE, FALSE, 0);
    gtk_widget_show(table);

    spinbutton = gimp_spin_button_new(100, 50000, 1,
                                     &lgi_defaults.num_gaussians,
                                     1, 100, 1000, 0, 0);
    gimp_table_attach_aligned(GTK_TABLE(table), 0, 0,
                             "Number of Gaussians:", 0.0, 0.5,
                             spinbutton, 1, FALSE);

    /* Quality preset */
    const gchar *quality_labels[] = {
        "Small (max compression)",
        "Balanced (recommended)",
        "High (best quality)",
        "Lossless (bit-exact)"
    };

    combo = gimp_int_combo_box_new_array(4, quality_labels);
    gimp_int_combo_box_set_active(GIMP_INT_COMBO_BOX(combo), lgi_defaults.quality);
    gimp_table_attach_aligned(GTK_TABLE(table), 0, 1,
                             "Quality:", 0.0, 0.5,
                             combo, 1, FALSE);

    /* QA training checkbox */
    checkbutton = gtk_check_button_new_with_label("Enable QA training (better compression)");
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(checkbutton),
                                lgi_defaults.qa_training);
    gtk_box_pack_start(GTK_BOX(vbox), checkbutton, FALSE, FALSE, 0);
    gtk_widget_show(checkbutton);

    gtk_widget_show(dialog);

    run = (gimp_dialog_run(GIMP_DIALOG(dialog)) == GTK_RESPONSE_OK);

    gtk_widget_destroy(dialog);

    return run;
}
