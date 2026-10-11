/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

/* halation: simulates the warm reddish glow that bleeds out of the bright
   parts of an image, like the halation of analog film where light passes
   through the emulsion, reflects off the film base and re-exposes mostly the
   red-sensitive layer.

   Works on scene-referred linear RGB, before the tone mapping: the light
   exceeding the threshold is gathered, spread with a gaussian blur and added
   back to the image as tinted light, so the halo behaves like real light. */

#include "bauhaus/bauhaus.h"
#include "common/colorspaces_inline_conversions.h"
#include "common/gaussian.h"
#include "common/imagebuf.h"
#include "common/iop_profile.h"
#include "common/math.h"
#include "common/opencl.h"
#include "control/control.h"
#include "develop/develop.h"
#include "develop/imageop.h"
#include "develop/imageop_math.h"
#include "develop/imageop_gui.h"
#include "develop/tiling.h"
#include "gui/gtk.h"
#include "iop/iop_api.h"

#include <gtk/gtk.h>
#include <stdlib.h>
#include <string.h>

#define MIDDLE_GREY 0.1845f

DT_MODULE_INTROSPECTION(2, dt_iop_halation_params_t)

typedef struct dt_iop_halation_params_t
{
  float size;       // $MIN: 0.0 $MAX: 5.0 $DEFAULT: 0.5 $DESCRIPTION: "size"
  float threshold;  // $MIN: -4.0 $MAX: 8.0 $DEFAULT: 1.5 $DESCRIPTION: "threshold"
  float strength;   // $MIN: 0.0 $MAX: 400.0 $DEFAULT: 50.0 $DESCRIPTION: "strength"
  float hue;        // $MIN: 0.0 $MAX: 360.0 $DEFAULT: 15.0 $DESCRIPTION: "hue"
  float saturation; // $MIN: 0.0 $MAX: 100.0 $DEFAULT: 80.0 $DESCRIPTION: "saturation"
} dt_iop_halation_params_t;

typedef struct dt_iop_halation_gui_data_t
{
  GtkWidget *size;
  GtkWidget *threshold;
  GtkWidget *strength;
  GtkWidget *hue;
  GtkWidget *saturation;
} dt_iop_halation_gui_data_t;

typedef struct dt_iop_halation_data_t
{
  float size;             // sigma of the halo, as a fraction of the longest image side
  float threshold;        // linear luminance above which light bleeds
  dt_aligned_pixel_t tint; // halo color in pipeline RGB, already scaled by strength
} dt_iop_halation_data_t;

typedef struct dt_iop_halation_global_data_t
{
  int kernel_halation_threshold;
  int kernel_halation_mix;
} dt_iop_halation_global_data_t;

const char *name()
{
  return _("halation");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self, _("add a warm reddish glow bleeding out of the highlights, like analog film halation"),
                                      _("creative"),
                                      _("linear, RGB, scene-referred"),
                                      _("linear, RGB"),
                                      _("linear, RGB, scene-referred"));
}

int flags()
{
  return IOP_FLAGS_INCLUDE_IN_STYLES | IOP_FLAGS_SUPPORTS_BLENDING;
}

int default_group()
{
  return IOP_GROUP_EFFECT | IOP_GROUP_EFFECTS;
}

dt_iop_colorspace_type_t default_colorspace(dt_iop_module_t *self,
                                            dt_dev_pixelpipe_t *pipe,
                                            dt_dev_pixelpipe_iop_t *piece)
{
  return IOP_CS_RGB;
}

int legacy_params(dt_iop_module_t *self,
                  const void *const old_params,
                  const int old_version,
                  void **new_params,
                  int32_t *new_params_size,
                  int *new_version)
{
  if(old_version == 1)
  {
    // v1 worked in Lab after the tone mapping, with a box blur sized in
    // pixels. There is no exact equivalent, so keep the look approximately.
    typedef struct dt_iop_halation_params_v1_t
    {
      float size;       // 0..100, box radius = 2.56 px per unit
      float threshold;  // 0..100, Lab lightness
      float strength;
      float hue;
      float saturation;
    } dt_iop_halation_params_v1_t;

    const dt_iop_halation_params_v1_t *old = old_params;
    dt_iop_halation_params_t *new = malloc(sizeof(dt_iop_halation_params_t));

    // radius in pixels -> % of the longest side, assuming a 6000 px image
    new->size = CLAMP(old->size * 2.56f / 6000.0f * 100.0f, 0.0f, 5.0f);

    // Lab lightness -> EV relative to middle grey
    const float L = CLAMP(old->threshold, 0.0f, 100.0f);
    const float fy = (L + 16.0f) / 116.0f;
    const float Y = (L > 8.0f) ? fy * fy * fy : L / 903.3f;
    new->threshold = CLAMP(log2f(fmaxf(Y, 1e-3f) / MIDDLE_GREY), -4.0f, 8.0f);

    new->strength = old->strength;
    new->hue = old->hue;
    new->saturation = old->saturation;

    *new_params = new;
    *new_params_size = sizeof(dt_iop_halation_params_t);
    *new_version = 2;
    return 0;
  }
  return 1;
}

/* gaussian sigma of the halo in pixels at the current pipe scale */
static inline float _halo_sigma(const dt_iop_halation_data_t *const d,
                                const dt_dev_pixelpipe_iop_t *const piece,
                                const dt_iop_roi_t *const roi_in)
{
  const float longest = MAX(piece->buf_in.width, piece->buf_in.height);
  return fmaxf(0.1f, d->size * longest * roi_in->scale / piece->iscale);
}

/* coefficients giving the luminance of a pipeline RGB pixel */
static inline void _luminance_coeffs(const dt_dev_pixelpipe_t *const pipe,
                                     dt_aligned_pixel_t coeffs)
{
  const dt_iop_order_iccprofile_info_t *const work_profile = dt_ioppr_get_pipe_work_profile_info(pipe);

  // the working profile is assumed linear, which is the case for all the
  // profiles darktable offers as working space
  if(work_profile)
  {
    for_three_channels(c) coeffs[c] = work_profile->matrix_in[1][c];
  }
  else
  {
    // Rec2020
    coeffs[0] = 0.2627f;
    coeffs[1] = 0.6780f;
    coeffs[2] = 0.0593f;
  }
  coeffs[3] = 0.0f;
}

void process(dt_iop_module_t *self,
             dt_dev_pixelpipe_iop_t *piece,
             const void *const ivoid,
             void *const ovoid,
             const dt_iop_roi_t *const roi_in,
             const dt_iop_roi_t *const roi_out)
{
  const dt_iop_halation_data_t *const data = piece->data;
  if(!dt_iop_have_required_input_format(4 /*we need full-color pixels*/,
                                        self,
                                        piece->colors,
                                        ivoid,
                                        ovoid,
                                        roi_in,
                                        roi_out))
    // image has been copied through to output and module's trouble flag has been updated
    return;

  float *restrict halo;
  if(!dt_iop_alloc_image_buffers(self, roi_in, roi_out, 1, &halo, 0))
  {
    // out of memory, so just copy image through to output
    dt_iop_copy_image_roi(ovoid, ivoid, piece->colors, roi_in, roi_out);
    return;
  }

  const float *const restrict in = DT_IS_ALIGNED((float *)ivoid);
  float *const restrict out = DT_IS_ALIGNED((float *)ovoid);
  const size_t npixels = (size_t)roi_out->width * roi_out->height;

  const float sigma = _halo_sigma(data, piece, roi_in);
  const float threshold = data->threshold;

  dt_aligned_pixel_t lum;
  _luminance_coeffs(piece->pipe, lum);

  /* gather the light exceeding the threshold into the single-channel halo buffer */
  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float Y = in[4 * k] * lum[0] + in[4 * k + 1] * lum[1] + in[4 * k + 2] * lum[2];
    halo[k] = fmaxf(Y - threshold, 0.0f);
  }

  /* spread the gathered light out into a soft halo */
  dt_gaussian_mean_blur(halo, roi_out->width, roi_out->height, 1, sigma);

  /* add the halo back as tinted light */
  const dt_aligned_pixel_t tint = { data->tint[0], data->tint[1], data->tint[2], 0.0f };
  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float h = fmaxf(halo[k], 0.0f);
    for_four_channels(c)
      out[4 * k + c] = in[4 * k + c] + h * tint[c];
  }
  dt_free_align(halo);
}

#ifdef HAVE_OPENCL
int process_cl(dt_iop_module_t *self,
               dt_dev_pixelpipe_iop_t *piece,
               cl_mem dev_in,
               cl_mem dev_out,
               const dt_iop_roi_t *const roi_in,
               const dt_iop_roi_t *const roi_out)
{
  const dt_iop_halation_data_t *d = piece->data;
  const dt_iop_halation_global_data_t *gd = self->global_data;

  cl_int err = DT_OPENCL_DEFAULT_ERROR;
  cl_mem dev_halo = NULL;
  dt_gaussian_cl_t *g = NULL;

  const int devid = piece->pipe->devid;
  const int width = roi_in->width;
  const int height = roi_in->height;

  const float sigma = _halo_sigma(d, piece, roi_in);
  const float threshold = d->threshold;

  dt_aligned_pixel_t lum;
  _luminance_coeffs(piece->pipe, lum);
  const dt_aligned_pixel_t tint = { d->tint[0], d->tint[1], d->tint[2], 0.0f };

  dev_halo = dt_opencl_alloc_device(devid, width, height, sizeof(float));
  if(dev_halo == NULL)
  {
    err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    goto error;
  }

  /* gather the light exceeding the threshold */
  err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_halation_threshold, width, height,
                                         CLARG(dev_in), CLARG(dev_halo),
                                         CLARG(width), CLARG(height),
                                         CLFLARRAY(4, lum), CLARG(threshold));
  if(err != CL_SUCCESS) goto error;

  /* spread the gathered light out into a soft halo */
  const float range = 1.0e9f;
  const float max[] = { range };
  const float min[] = { -range };
  g = dt_gaussian_init_cl(devid, width, height, 1, max, min, sigma, DT_IOP_GAUSSIAN_ZERO);
  if(!g)
  {
    err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    goto error;
  }
  err = dt_gaussian_blur_cl(g, dev_halo, dev_halo);
  if(err != CL_SUCCESS) goto error;

  /* add the halo back as tinted light */
  err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_halation_mix, width, height,
                                         CLARG(dev_in), CLARG(dev_halo), CLARG(dev_out),
                                         CLARG(width), CLARG(height), CLFLARRAY(4, tint));

error:
  if(g) dt_gaussian_free_cl(g);
  dt_opencl_release_mem_object(dev_halo);
  return err;
}
#endif

void tiling_callback(dt_iop_module_t *self,
                     dt_dev_pixelpipe_iop_t *piece,
                     const dt_iop_roi_t *roi_in,
                     const dt_iop_roi_t *roi_out,
                     dt_develop_tiling_t *tiling)
{
  const dt_iop_halation_data_t *d = piece->data;

  const int width = roi_in->width;
  const int height = roi_in->height;
  const float sigma = _halo_sigma(d, piece, roi_in);

  const size_t basebuffer = sizeof(float) * 4 * width * height;

  // in + out + single-channel halo + gaussian temporaries
  tiling->factor = 2.25f + (float)dt_gaussian_memory_use(width, height, 1) / basebuffer;
#ifdef HAVE_OPENCL
  tiling->factor_cl = 2.25f + (float)dt_gaussian_memory_use_cl(width, height, 1) / basebuffer;
#endif
  tiling->maxbuf = fmaxf(1.0f, (float)dt_gaussian_singlebuffer_size(width, height, 1) / basebuffer);
  tiling->overhead = 0;
  tiling->overlap = ceilf(4 * sigma);
  tiling->align = 1;
  return;
}

void init_global(dt_iop_module_so_t *self)
{
  const int program = 100; // halation.cl, from programs.conf
  dt_iop_halation_global_data_t *gd = malloc(sizeof(dt_iop_halation_global_data_t));
  self->data = gd;

  gd->kernel_halation_threshold = dt_opencl_create_kernel(program, "halation_threshold");
  gd->kernel_halation_mix = dt_opencl_create_kernel(program, "halation_mix");
}

void cleanup_global(dt_iop_module_so_t *self)
{
  const dt_iop_halation_global_data_t *gd = self->data;
  dt_opencl_free_kernel(gd->kernel_halation_threshold);
  dt_opencl_free_kernel(gd->kernel_halation_mix);
  free(self->data);
  self->data = NULL;
}

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_halation_params_t *p = (dt_iop_halation_params_t *)p1;
  dt_iop_halation_data_t *d = piece->data;

  d->size = p->size / 100.0f;
  d->threshold = MIDDLE_GREY * exp2f(p->threshold);

  /* halo color: full-value HSV in the pipeline RGB, scaled by the strength */
  const dt_aligned_pixel_t HSV = { p->hue / 360.0f, p->saturation / 100.0f, 1.0f, 0.0f };
  dt_HSV_2_RGB(HSV, d->tint);
  for_three_channels(c) d->tint[c] *= p->strength / 100.0f;
  d->tint[3] = 0.0f;
}

void init_pipe(dt_iop_module_t *self,
               dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = dt_calloc1_align_type(dt_iop_halation_data_t);
}

void cleanup_pipe(dt_iop_module_t *self,
                  dt_dev_pixelpipe_t *pipe,
                  dt_dev_pixelpipe_iop_t *piece)
{
  dt_free_align(piece->data);
  piece->data = NULL;
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_halation_gui_data_t *g = IOP_GUI_ALLOC(halation);

  g->size = dt_bauhaus_slider_from_params(self, N_("size"));
  dt_bauhaus_slider_set_soft_range(g->size, 0.0f, 2.0f);
  dt_bauhaus_slider_set_digits(g->size, 2);
  dt_bauhaus_slider_set_format(g->size, "%");
  gtk_widget_set_tooltip_text(g->size, _("how far the halo bleeds out of the highlights,\n"
                                         "relative to the longest side of the image"));

  g->threshold = dt_bauhaus_slider_from_params(self, N_("threshold"));
  dt_bauhaus_slider_set_soft_range(g->threshold, -2.0f, 4.0f);
  dt_bauhaus_slider_set_format(g->threshold, _(" EV"));
  gtk_widget_set_tooltip_text(g->threshold, _("how bright a pixel must be to glow,\n"
                                              "relative to middle gray"));

  g->strength = dt_bauhaus_slider_from_params(self, N_("strength"));
  dt_bauhaus_slider_set_soft_range(g->strength, 0.0f, 100.0f);
  dt_bauhaus_slider_set_format(g->strength, "%");
  gtk_widget_set_tooltip_text(g->strength, _("how much of the light above the threshold bleeds into the halo"));

  g->hue = dt_bauhaus_slider_from_params(self, N_("hue"));
  dt_bauhaus_slider_set_feedback(g->hue, 0);
  dt_bauhaus_slider_set_stop(g->hue, 0.0f  , 1.0f, 0.0f, 0.0f);
  dt_bauhaus_slider_set_stop(g->hue, 0.166f, 1.0f, 1.0f, 0.0f);
  dt_bauhaus_slider_set_stop(g->hue, 0.322f, 0.0f, 1.0f, 0.0f);
  dt_bauhaus_slider_set_stop(g->hue, 0.498f, 0.0f, 1.0f, 1.0f);
  dt_bauhaus_slider_set_stop(g->hue, 0.664f, 0.0f, 0.0f, 1.0f);
  dt_bauhaus_slider_set_stop(g->hue, 0.830f, 1.0f, 0.0f, 1.0f);
  dt_bauhaus_slider_set_stop(g->hue, 1.0f  , 1.0f, 0.0f, 0.0f);
  dt_bauhaus_slider_set_format(g->hue, "\302\260");
  gtk_widget_set_tooltip_text(g->hue, _("the color of the halo (around 15\302\260 for film-like red-orange)"));

  g->saturation = dt_bauhaus_slider_from_params(self, N_("saturation"));
  dt_bauhaus_slider_set_format(g->saturation, "%");
  gtk_widget_set_tooltip_text(g->saturation, _("how strongly the halo is tinted"));
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
