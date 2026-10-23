# Integration with Persons 2 and 3

## Person 2: final training and inference

Install the `vyra-preprocessing` package in the actual training environment.
The `data` extra is needed to prepare data, not to consume a prepared run.
Copy `integration/normalization_adapter.py` over the adapter in your own
`src/data/` directory after reviewing it. The original adapter supplied for
this task remains unchanged in the parent project and is preserved as a test
reference. All nine `PreprocessingResult` fields and helper imports remain.

The image-processing behavior has intentionally changed: EXIF is honored,
transparent/segmented backgrounds are excluded, source dimensions below 50px
are invalid, ROI inputs are explicit, and ITA requires identified skin.
Existing weights trained with the baseline are not certified for this pipeline.
Keep that run as a baseline and train the final candidate with the frozen package.

Prepare the datasets in the final model environment, then freeze the package,
dependency versions and manifests. If your Python/dependency versions differ
from the observed Python 3.14 environment, run the tests and create a new prepared
run; fingerprints are deliberately environment-sensitive.

### Read a training sample

```python
import csv
from pathlib import Path
from PIL import Image
from vyra_preprocessing import load_prepared_sample

dataset_root = Path("path/to/data/prepared")
with (dataset_root / "splits/train.csv").open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

row = rows[0]
prepared = load_prepared_sample(row, dataset_root)
if not prepared.is_valid:
    raise ValueError(prepared.quality_warnings)

pil_image = Image.fromarray(prepared.normalized_image_rgb)
color_features = prepared.feature_vector
body_encoding = prepared.body_part_encoding
has_hb = row["hb_available"] == "true"
hb_target = float(row["hb_g_dl"]) if has_hb else None
source_class = int(row["anemia_label"]) if row["anemia_label"] else None
```

Apply your model-specific resize, tensor conversion and ImageNet normalization
to `pil_image`. Keep the 27 color features separate. Fit any learned feature
scaler on training rows only and save it with the model. This package does not
fit a scaler or mutate the feature recipe.

The cached feature vector describes the normalized ROI before training-time
augmentation. If an augmentation changes its colors or spatial layout and you
need paired features, recompute the 27 values from that augmented uint8 RGB view
before tensor normalization. Do not silently mix a changed image with features
that are claimed to describe that changed image.

Cached normalized images and features are read together. Do not pass a cached
normalized PNG back through `preprocess_image`: that would normalize it twice.
`load_prepared_sample(..., cached=False)` reconstructs the full-source ROI mask
and recomputes the same result; the included consumer example compares both paths.

Nails have source binary class labels but no measured Hb. Mask out their
regression loss; never replace missing Hb with zero. Eye rows retain Hb, age and
sex, but have no invented binary class label. Derive eye class targets using
your documented clinical labeling policy. Empty CSV values mean missing, not
zero; parse booleans explicitly (`value == "true"`).

### Evaluation and checkpoint compatibility

Use the corresponding validation/test files without resplitting. For verified
patient-level evaluation, use `verified_splits/` (eyes only). Nail results from
`splits/` are exploratory source-group results and must be described accordingly.
Validation-fitted calibration and model metrics remain Person 2's responsibility.

Record these fields with every checkpoint:

- `preprocessing_version` and `preprocessing_fingerprint` from `run.json`;
- `preparation_fingerprint` to identify source archives, importer, seed and audit;
- the 27 ordered feature names, runtime versions and any learned feature scaler;
- your model-specific image transformations and clinical label policy.

At serving startup, compare the checkpoint preprocessing fingerprint with
`pipeline_fingerprint()`. Reject a mismatch instead of silently swapping inputs.
The package deliberately does not load or rewrite your checkpoint.

### Live capture

```python
from vyra_preprocessing import preprocess_with_details

details = preprocess_with_details(
    "capture.jpg", "conjunctiva", roi_bbox=(40, 60, 240, 160)
)
result = details.result
if not result.is_valid:
    # Return the warnings to the capture UI and request another image/ROI.
    raise ValueError(result.quality_warnings)
```

Use RGB uint8 input before any tensor normalization. A verified tissue-only crop
may use `pre_cropped=True`; transparency can supply the ROI automatically.
An ordinary opaque full-frame photo requires a mask/box. There is no anatomical
detector pretending to recognize eyes or nails from arbitrary images.

## Person 3: previews, warnings and coordinates

For a selected ROI, use `raw_roi.png` and `normalized.png` for the comparison
slider. `raw.png` is the full EXIF-oriented source. Other exports include glare
correction, white balance, the ROI mask and strict JSON metadata.

`raw_image_rgb` and `normalized_image_rgb` may have different dimensions because
the normalized image is the cropped ROI. `metadata.roi_bbox` is `[x0,y0,x1,y1]`
with exclusive upper bounds in the oriented source coordinates. Resize a model
Grad-CAM back to this ROI size before overlaying it at the box in `raw.png`.
The box does not refer to a separate, differently-sized original JPEG listed
in the dataset manifest. Use the exported raw/ROI pair to avoid that mismatch.

Show `quality_warnings`; when `is_valid` is false, block inference and show the
recapture/ROI message. Rejected results contain placeholder arrays only to keep
the return contract stable. They are not predictions. An invalid recapture
clears earlier normalized preview files to prevent stale comparisons.

Missing ITA is JSON `null`, category `Unknown`. Show “Skin tone unavailable”;
do not substitute zero or classify the tissue itself as surrounding skin.
Skin masks are optional, binary, in oriented-source coordinates and must not
overlap the eye/nail ROI. No ITA-dependent color scaling is applied.
