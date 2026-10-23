# Preprocessing API and feature contract

## Entry points

```python
preprocess_image(image, body_part=None, *, roi_mask=None, roi_bbox=None,
                 skin_mask=None, pre_cropped=False, config=DEFAULT_CONFIG)
preprocess_with_details(...)  # same arguments; includes stages and metadata
load_prepared_sample(row, dataset_root, *, cached=True)
```

`image` is a file path, PIL image or nonempty H×W×3 uint8 RGB array. NumPy BGR,
RGBA arrays, floats, normalized tensors and grayscale inputs are not silently
reinterpreted. Use a PIL image/path for alpha handling. EXIF orientation is
applied before mask/box validation. Array inputs are already oriented RGB.

Canonical body parts are `conjunctiva` and `fingernail`. The documented aliases
include `eye`, `eyes`, `conj`, `conjunctival`, `palpebral_conjunctiva`, `nail`,
`nails`, `finger`, `fingers`, and `fingernails`. Missing/unsupported values return
an invalid result; the unknown encoding remains `[0.5,0.5]` for structural
compatibility but must not reach inference.

Masks are H×W boolean or uint8 containing only 0/1/255. Boxes contain integer
`(x0,y0,x1,y1)` coordinates within the oriented image. A mask and box cannot
both be supplied. Alpha is intersected with an explicitly supplied ROI.

## Unchanged result fields

| Field | Contract |
|---|---|
| `raw_image_rgb` | Oriented full-source uint8 RGB copy; alpha is composited on white for previews |
| `normalized_image_rgb` | Cropped normalized ROI, H×W×3 uint8; dimensions may differ from raw |
| `feature_vector` | Exactly `(27,)` finite float32, ordered below |
| `body_part_encoding` | `(2,)` float32: eye `[1,0]`, nail `[0,1]` |
| `ita_angle` | Degrees; NaN when no identified skin or undefined calculation |
| `ita_category` | Six ITA categories or `Unknown` |
| `glare_fraction` | Fraction of tissue pixels detected as glare before inpainting |
| `is_valid` | False for unreadable/undersized images, unsupported input/body, invalid ROI or excessive glare |
| `quality_warnings` | List of `CODE: explanation` strings |

JSON exports convert missing ITA to `null` and never contain NaN/Infinity.
Invalid results have a zero 27-vector and a 224×224 zero normalized placeholder.
Consumers must check `is_valid`. Structural zeros are not imputed measurements.

## Exact 27-feature order

The feature image is the final normalized RGB ROI, resized to 224×224 using
OpenCV's default linear interpolation and converted to float32. Let
`Y = 0.299R + 0.587G + 0.114B`. Standard deviations use `ddof=0`.

| Index | Name | Calculation |
|---:|---|---|
| 0 | brightness_mean | mean(Y) |
| 1 | brightness_std | std(Y) |
| 2–6 | brightness_p10/p25/p50/p75/p90 | NumPy linear percentiles |
| 7–8 | redness_mean/std | mean/std of R/(R+G+B+1e-6) |
| 9 | white_pallor_ratio | fraction Y > 180 |
| 10 | pink_ratio | fraction R > 150 and G < 150 and B < 150 |
| 11–12 | red_mean/std | mean/std(R) |
| 13–14 | green_mean/std | mean/std(G) |
| 15–16 | blue_mean/std | mean/std(B) |
| 17 | red_green_ratio | (mean(R)+1)/(mean(G)+1) |
| 18 | red_blue_ratio | (mean(R)+1)/(mean(B)+1) |
| 19 | red_blue_difference | (mean(R)-mean(B))/255 |
| 20–21 | hb_color_proxy_mean/std | mean/std of the pixel map R/(G+B+1) |
| 22 | top_bottom_brightness | top third mean(Y) minus bottom third mean(Y) |
| 23 | center_global_brightness | central half-width/half-height rectangle mean(Y) minus global mean(Y) |
| 24–25 | gradient_mean/std | mean/std(sqrt(gx²+gy²)); gx,gy are absolute forward differences with the first column/row prepended |
| 26 | local_std_mean | mean(sqrt(max(uniform_filter((Y-uniform_filter(Y,7))²,7),0))); SciPy default reflect mode |

Brightness/channel/gradient features use 0–255-scale intensities; ratios are
dimensionless. The Hb color proxy is not a g/dL estimate. These are colorimetric
features, not a demonstrated guarantee of camera invariance. The recipe is
tested against the exact supplied Person 2 implementation, including its local
texture formula. No Sobel substitution or extra 28th feature is used.

## Processing and defaults

The source must be at least 50px in both dimensions before resizing. A thinner
isolated ROI from a sufficiently-sized source produces `SMALL_ROI` as a warning.
No upscaling is used to conceal an undersized source.

The ROI is tightly cropped. Outside-tissue pixels inside the rectangle are
extended from the nearest valid tissue pixel during boundary processing. After
CLAHE, those excluded pixels are set to neutral RGB `(127,127,127)` before model
input and feature extraction, so synthetic extension streaks cannot become
spatial features. The neutral margins are part of the documented feature image;
the 27-feature helper itself is unchanged. Glare and white-balance statistics
use the actual tissue mask. Glare: HSV S < 40 and V > 225;
Telea radius 3 when the detected fraction exceeds 0.001; warning above 0.15;
rejection at or above 0.50. The JSON separately records detected and inpainted
fractions. The rejection fraction is a configurable engineering threshold,
not a clinically validated distinction between glare and pallor.

Grey-World uses channel means plus 1e-6, clips to [0,255], and returns uint8.
CLAHE applies only to HSV V, with clip limit 2 and an 8×8 tile grid.
Stages are deterministic within the recorded runtime/configuration.

ITA uses raw identified surrounding skin in standard CIE LAB, not a
white-balanced tissue crop. It requires at least 16 selected skin pixels and
nonzero mean b*. Categories use >55 Very Light, >41 Light, >28 Intermediate,
>10 Tan, >−30 Brown, otherwise Dark. No skin mask means unavailable in the main
pipeline. The low-level `compute_ita(image)` helper assumes its entire argument
is an already-isolated skin patch.

The fingerprint covers configuration, core implementation bytes, feature
schema and core dependency versions. It is a compatibility marker, not a
clinical certification. Checkpoints and prepared data must agree with it.
