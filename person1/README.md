# Vyra — Person 1 data and preprocessing

Prepare eye/nail datasets and turn tissue photographs into normalized RGB images,
Person 2's exact **27 color features**, body-part encoding, and quality/ITA metadata.
The package includes an adapter for Person 2, preview exports for Person 3, tests,
and the observed dataset manifests. No model training or frontend is included.

## Install and test

Run these commands in the directory containing this `pyproject.toml` (`person1/`
in the shared project). Python 3.11+ is required. Local verification used Linux
and Python 3.14.6; the exact tested versions are recorded in
`requirements-tested-py314.txt`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[data,test]'
.venv/bin/python -m pytest -q
```

For the exact Python 3.14 environment, install
`requirements-tested-py314.txt` before the editable package. On Windows, use
`.venv\Scripts\python.exe` in place of `.venv/bin/python`.

RAR extraction requires the system libarchive library as well as the `data`
extra. It is present on the tested machine. If unavailable, extract the nail
RAR with your archive manager and pass its directory to `--nails`; inference
does not require archive tools or spreadsheet libraries.

## Prepare the supplied archives

From this folder, with the two original ZIPs in its parent directory:

```bash
.venv/bin/python -m vyra_preprocessing prepare \
  --eyes ../eyes-defy.zip \
  --nails '../Detection of Anemia using Colour of the Fingernails Image Datasets from Ghana.zip' \
  --output data/prepared

.venv/bin/python -m vyra_preprocessing verify --dataset-root data/prepared
.venv/bin/python examples/model_consumer.py --dataset-root data/prepared
.venv/bin/python tools/validate_real_data.py --dataset-root data/prepared \
  --output reports/observed-data/real_data_validation.json
```

`--eyes` also accepts an extracted eye directory; `--nails` accepts a RAR or
extracted directory. Repeat preparation with the same inputs to verify and reuse
the existing output. Changed inputs, code, configuration or core dependency
versions require a new output directory. Sources are never overwritten.

## Observed results from the supplied archives

| Dataset | Inventory assets | Included samples | Train / validation / test |
|---|---:|---:|---:|
| EYES-DEFY-ANEMIA | 862 | 209 | 147 / 21 / 41 |
| Ghana Fingernails | 4,260 | 1,347 | 943 / 135 / 269 |

Eye selection uses one palpebral crop per patient. Of 218 patients, 217 have
laboratory Hb; one crop failed the glare gate and seven cross-patient perceptual
candidates were quarantined. The nail pipeline excluded 1,218 undersized files,
202 glare-heavy files, and 1,493 eligible exact duplicates.

The split audit found **zero known patient, source-family or duplicate-group
overlap**. Nail patient independence remains **unverified**: that archive has
no patient mapping or laboratory Hb table. `verified_splits/` contains eyes only;
`splits/` additionally includes exploratory nail groups. Read the full observed
reports before interpreting results. Perceptual matches and glare exclusions
are conservative rules, not manually adjudicated clinical findings.

## Use with the model

```python
from vyra_preprocessing import load_prepared_sample, preprocess_image

# Training/evaluation: row is a CSV dictionary from the prepared splits.
result = load_prepared_sample(row, dataset_root)  # already-normalized cache
assert result.is_valid
image_rgb = result.normalized_image_rgb
features = result.feature_vector                 # (27,) float32
body = result.body_part_encoding                 # (2,) float32

# Live capture: box coordinates use the EXIF-oriented source image.
result = preprocess_image("capture.jpg", "conjunctiva", roi_bbox=(40, 60, 240, 160))
if not result.is_valid:
    print(result.quality_warnings)                # do not run the model
```

Do not normalize `normalized_path` a second time. To verify a cached sample
against its source, call `load_prepared_sample(..., cached=False)`.
The final model must be trained and served with the same preprocessing
fingerprint and partitions. Person 2's existing training run is a baseline.

## Handoff and GitHub

- [Person 2 and Person 3 integration](docs/INTEGRATION.md)
- [API and exact feature specification](docs/CONTRACT.md)
- [Dataset schema, exclusions and split policy](docs/DATA.md)
- [Source attribution](THIRD_PARTY_NOTICES.md)
- [Observed manifests and audit reports](reports/observed-data/README.md)
- [Ready-to-send teammate note](HANDOFF.md)

Upload the contents of this folder for a standalone repository, or keep it as
`person1/` in the team repository and install with `pip install -e './person1[data]'`.
The included GitHub Actions workflow is for a standalone package repository;
when nesting the package, move the workflow to the repository root and set its
working directory to `person1`.

Dataset archives, extracted photographs, local previews, environments, and build
outputs stay outside Git. The committed reports contain metadata/features only.
To regenerate that handoff after preparation:

```bash
.venv/bin/python -m vyra_preprocessing export-handoff \
  --dataset-root data/prepared --output reports/observed-data
```
