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

#include "common.h"

/* first step for halation module: gather the light exceeding the threshold */
kernel void
halation_threshold(read_only image2d_t in, write_only image2d_t out, const int width, const int height,
                   const float4 lum_coeffs, const float threshold)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if(x >= width || y >= height) return;

  const float4 pixel = read_imagef(in, sampleri, (int2)(x, y));
  const float Y = pixel.x * lum_coeffs.x + pixel.y * lum_coeffs.y + pixel.z * lum_coeffs.z;

  write_imagef(out, (int2)(x, y), fmax(Y - threshold, 0.0f));
}

/* final step for halation module: add the tinted, blurred halo to the image */
kernel void
halation_mix(read_only image2d_t in, read_only image2d_t halo, write_only image2d_t out,
             const int width, const int height, const float4 tint)
{
  const int x = get_global_id(0);
  const int y = get_global_id(1);

  if(x >= width || y >= height) return;

  float4 pixel = read_imagef(in, sampleri, (int2)(x, y));
  const float h = fmax(read_imagef(halo, sampleri, (int2)(x, y)).x, 0.0f);

  pixel.xyz += h * tint.xyz;

  write_imagef(out, (int2)(x, y), pixel);
}
