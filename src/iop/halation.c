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
   red-sensitive layer.  Built on the same highlight-extract / blur / screen
   approach as the bloom module, with an added warm tint on the glow. */

#include "bauhaus/bauhaus.h"
#include "common/box_filters.h"
#include "common/imagebuf.h"
#include "common/math.h"
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

DT_MODULE_INTROSPECTION(1, dt_iop_halation_params_t)

typedef struct dt_iop_halation_params_t
{
  float size;       // $MIN: 0.0 $MAX: 100.0 $DEFAULT: 30.0 $DESCRIPTION: "size"
  float threshold;  // $MIN: 0.0 $MAX: 100.0 $DEFAULT: 80.0 $DESCRIPTION: "threshold"
  float strength;   // $MIN: 0.0 $MAX: 100.0 $DEFAULT: 40.0 $DESCRIPTION: "strength"
  float hue;        // $MIN: 0.0 $MAX: 360.0 $DEFAULT: 25.0 $DESCRIPTION: "hue"
  float saturation; // $MIN: 0.0 $MAX: 100.0 $DEFAULT: 50.0 $DESCRIPTION: "saturation"
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
  float size;
  float threshold;
  float strength;
  float hue;
  float saturation;
} dt_iop_halation_data_t;

const char *name()
{
  return _("halation");
}

const char **description(dt_iop_module_t *self)
{
  return dt_iop_set_description(self, _("add a warm reddish glow bleeding out of the highlights, like analog film halation"),
                                      _("creative"),
                                      _("non-linear, Lab, display-referred"),
                                      _("non-linear, Lab"),
                                      _("non-linear, Lab, display-referred"));
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
  return IOP_CS_LAB;
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

  /* blur radius scaled to the current preview/export resolution */
  const int rad = 256.0f * (fmin(100.0f, data->size + 1.0f) / 100.0f);
  const float _r = ceilf(rad * roi_in->scale / piece->iscale);
  const int radius = MIN(256.0f, _r);

  /* glow intensity, same curve as bloom: boosts the lights before threshold */
  const float scale = 1.0f / exp2f(-1.0f * (fmin(100.0f, data->strength + 1.0f) / 100.0f));
  const float threshold = data->threshold;

  /* warm tint direction in the Lab a/b plane (red-orange by default) */
  const float hue = deg2radf(data->hue);
  const float tint_a = data->saturation * cosf(hue);
  const float tint_b = data->saturation * sinf(hue);

  /* gather the bright pixels into the single-channel halo buffer */
  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float L = in[4 * k] * scale;
    halo[k] = (L > threshold) ? L : 0.0f;
  }

  /* spread the gathered light out into a soft halo */
  const int range = 2 * radius + 1;
  const int hr = range / 2;
  dt_box_mean(halo, roi_out->height, roi_out->width, 1, hr, BOX_ITERATIONS);

  /* screen-blend the halo back onto L and tint the bleed warm on a/b */
  DT_OMP_FOR()
  for(size_t k = 0; k < npixels; k++)
  {
    const float h = halo[k];
    const float w = fminf(1.0f, h / 100.0f); // halo coverage at this pixel, 0..1
    out[4 * k + 0] = 100.0f - (((100.0f - in[4 * k + 0]) * (100.0f - h)) / 100.0f); // screen blend
    out[4 * k + 1] = in[4 * k + 1] + tint_a * w;
    out[4 * k + 2] = in[4 * k + 2] + tint_b * w;
    out[4 * k + 3] = in[4 * k + 3];
  }
  dt_free_align(halo);
}

void tiling_callback(dt_iop_module_t *self,
                     dt_dev_pixelpipe_iop_t *piece,
                     const dt_iop_roi_t *roi_in,
                     const dt_iop_roi_t *roi_out,
                     dt_develop_tiling_t *tiling)
{
  const dt_iop_halation_data_t *d = piece->data;

  const int rad = 256.0f * (fmin(100.0f, d->size + 1.0f) / 100.0f);
  const float _r = ceilf(rad * roi_in->scale / piece->iscale);
  const int radius = MIN(256.0f, _r);

  tiling->factor = 2.0f + 0.25f + 0.05f; // in + out + halo + slice for dt_box_mean
  tiling->maxbuf = 1.0f;
  tiling->overhead = 0;
  tiling->overlap = 5 * radius;
  tiling->align = 1;
  return;
}

void commit_params(dt_iop_module_t *self,
                   dt_iop_params_t *p1,
                   dt_dev_pixelpipe_t *pipe,
                   dt_dev_pixelpipe_iop_t *piece)
{
  const dt_iop_halation_params_t *p = (dt_iop_halation_params_t *)p1;
  dt_iop_halation_data_t *d = piece->data;

  d->size = p->size;
  d->threshold = p->threshold;
  d->strength = p->strength;
  d->hue = p->hue;
  d->saturation = p->saturation;
}

void init_pipe(dt_iop_module_t *self,
               dt_dev_pixelpipe_t *pipe,
               dt_dev_pixelpipe_iop_t *piece)
{
  piece->data = calloc(1, sizeof(dt_iop_halation_data_t));
}

void cleanup_pipe(dt_iop_module_t *self,
                  dt_dev_pixelpipe_t *pipe,
                  dt_dev_pixelpipe_iop_t *piece)
{
  free(piece->data);
  piece->data = NULL;
}

void gui_init(dt_iop_module_t *self)
{
  dt_iop_halation_gui_data_t *g = IOP_GUI_ALLOC(halation);

  g->size = dt_bauhaus_slider_from_params(self, N_("size"));
  dt_bauhaus_slider_set_format(g->size, "%");
  gtk_widget_set_tooltip_text(g->size, _("the size of the halo bleeding out of the highlights"));

  g->threshold = dt_bauhaus_slider_from_params(self, N_("threshold"));
  dt_bauhaus_slider_set_format(g->threshold, "%");
  gtk_widget_set_tooltip_text(g->threshold, _("how bright a pixel must be to glow"));

  g->strength = dt_bauhaus_slider_from_params(self, N_("strength"));
  dt_bauhaus_slider_set_format(g->strength, "%");
  gtk_widget_set_tooltip_text(g->strength, _("the intensity of the glow"));

  g->hue = dt_bauhaus_slider_from_params(self, N_("hue"));
  dt_bauhaus_slider_set_format(g->hue, "\302\260");
  gtk_widget_set_tooltip_text(g->hue, _("the color of the halo (around 25\302\260 for film-like red-orange)"));

  g->saturation = dt_bauhaus_slider_from_params(self, N_("saturation"));
  dt_bauhaus_slider_set_format(g->saturation, "%");
  gtk_widget_set_tooltip_text(g->saturation, _("how strongly the halo is tinted"));
}

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on
