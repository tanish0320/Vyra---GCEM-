# Person 1 handoff

Person 1's implementation is complete: ROI-aware normalization, strict input
quality checks, the existing 27-feature contract, real dataset preparation,
duplicate auditing, grouped splits, local previews, tests and installable packaging.

## For Person 2

1. Install the package in the final training environment and run its tests.
2. Prepare the two supplied archives there; freeze `run.json` and its manifests.
3. Use `load_prepared_sample(row, dataset_root)` for training/evaluation. It reads
   normalized images and their 27 features together; do not normalize the cache again.
4. Use the supplied adapter wrapper and the same package for live preprocessing.
   Record and check the preprocessing fingerprint when saving/loading checkpoints.
5. Keep the current baseline run for comparison. Train the final model after
   switching to this frozen preprocessing and split set.
6. Use Hb loss only for rows with `hb_available == "true"`. Nails have source
   class labels but no measured Hb. Eye binary class policy remains with Person 2.

## For Person 3

Supply a tissue crop, mask or oriented-source ROI box. Check `is_valid` before
inference and display the warnings. Use the exported raw ROI and normalized ROI
for before/after previews. Missing ITA is `null`/`Unknown`, not a zero-degree
skin-tone result. Grad-CAM construction remains with the model/frontend work.

## Observed data

| Set | Train | Validation | Test | Included total |
|---|---:|---:|---:|---:|
| Eyes — source patient IDs verified | 147 | 21 | 41 | 209 |
| Nails — exploratory source groups | 943 | 135 | 269 | 1,347 |

All 5,122 source image assets are inventoried. Known patient/source/duplicate
overlap across partitions is zero. Nail patient independence cannot be certified
without a real patient mapping; nail Hb cannot be supplied without a laboratory
table. The report records every exclusion and conservative near-duplicate flag.
The prior Markdown's dataset counts are not used as evidence.

## Verification completed

- 65 automated tests passed locally and from the built wheel in a separate
  virtual environment, running outside the source checkout.
- All 5,122 source image assets and saved manifests/artifacts passed hash checks.
- All 1,556 included samples were recomputed from source using the installed
  wheel; normalized images, cached features and CSV feature rows matched exactly.
- The wheel's command-line entry point, a model-input consumer example, and
  dependency consistency checks passed. No actual model prediction is implied.
- Excluded background is neutral gray in final model inputs; nearest-tissue
  processing extensions are not exposed as artificial background texture.
- GitHub CI is configured for Python 3.11/3.14 and will run after upload; it has
  not been run on GitHub during this local task.

Evidence is in `reports/verification/package_validation.json` and
`reports/observed-data/real_data_validation.json`. The wheel remains in the
ignored local `dist/` folder; use its preprocessing fingerprint as the compatibility
marker because rebuilt ZIP/wheel timestamps can change binary archive hashes.

Read `docs/INTEGRATION.md` for the exact model and preview contracts and
`reports/observed-data/` for actual manifests, features, audit and verification.
Raw images/archives remain local under ignored data paths. No model performance
is claimed by this handoff.

## Upload

Upload this package folder, including source, integration wrapper, tests, docs,
the tested dependency file and the metadata-only reports. `.gitignore` excludes
photographs, environments and build outputs. The supplied parent-folder adapter
and Word documents were not edited.

`tools/build_github_bundle.py` creates a source/report ZIP after checking the
recorded verification and report checksums. It never overwrites an existing ZIP
and excludes data, environments, build outputs and photographs.
