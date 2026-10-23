# Dataset preparation and audit policy

## Source truth

The supplied archives contain 862 eye image assets and 4,260 nail image assets.
Metadata spreadsheets and documentation are retained under the ignored local
raw directory but are not counted as image assets. Full-eye images and alternate
eye segmentations are inventoried; only one canonical palpebral crop per patient
is selected for model input. Dataset file availability is not inferred from the
different counts claimed in the original Person 2 note.

EYES-DEFY IDs are qualified by country (`eda:india:1`, `eda:italy:1`) to avoid
collisions. IDs written as `1.0` are parsed as integers; Hb decimal commas are
parsed as decimal points. Italy patient 93's `_` Hb remains missing. Original
filenames remain unchanged; importer aliases cover `papebral`, `palplebral`,
and extra `(1)` crop copies. The input PNGs include 187 bad iCCP checksums;
only that ancillary checksum is repaired in memory, with the original pixel
payload and source file retained. Other bad PNG chunk checksums are rejected.

The nail filename parser recognizes Anemic-Fin, Anemic-FN, Anmeic-fn,
Non-anemic-Fin and Non-Anrmic-FN prefixes. Augmentation suffixes, repeated
numeric copy suffixes, and the observed FV/RA qualifiers are kept in conservative
label-qualified numeric source families. This can overgroup images; it is
deliberately not promoted to a verified patient ID. Nail age, sex, patient ID
and Hb remain missing because no mapping/table was supplied.

## Outputs and paths

All CSV image paths are relative to the preparation output root, such as
`data/prepared`, not to the directory containing a copied CSV. CSV booleans are
the strings `true` and `false`; blank values mean missing.

| Output | Contents |
|---|---|
| manifests/dataset_inventory.csv | Every source image, including alternate/excluded assets |
| manifests/sample_manifest.csv | Selected canonical eye crops and all nail inputs, with final status |
| manifests/duplicate_audit.csv | Byte/pixel duplicate events and perceptual candidate edges |
| manifests/features.csv | Included sample IDs and the 27 ordered feature columns |
| splits/{train,validation,test}.csv | Combined eye and exploratory nail partitions |
| verified_splits/{train,validation,test}.csv | Verified source patient IDs from the eye dataset only |
| reports/audit_summary.json | Observed counts, exclusions, split balance and limitations |
| run.json | Source hashes, preparation/pipeline fingerprints, runtime and artifact hashes |
| derived/ | Local normalized images, cropped tissue masks and metadata; ignored by Git |
| examples/ | Local real before/after stages and rejection cases; ignored by Git |

The selected sample's `image_path` is its source crop. `original_image_path`
links to the related full-eye JPEG for reference; its dimensions may differ.
`roi_mask_path` is a **cropped** mask. `load_prepared_sample(cached=False)`
reconstructs it in full source coordinates using the sidecar box.

Other important fields are `patient_id_verified`, `source_group_id`,
`duplicate_group_id`, `hb_available`, `anemia_label`, `inclusion_status`,
`exclusion_reason`, `split_basis`, and `preprocessing_fingerprint`.
Source binary labels use 0=non-anemic, 1=anemic; eye binary labels are left empty.

## Duplicate handling and partitioning

1. SHA-256 detects byte-identical and decoded RGB/mask-identical samples.
2. Known patient/source families and exact duplicate relationships are joined.
3. Conflicting ground truth within exact components is quarantined.
4. A 63-bit DCT pHash (32px input, 8×8 low frequencies excluding DC) checks eight
   rotations/reflections. Hamming distance at most six produces a candidate edge.
5. Unresolved cross-patient eye candidates are excluded from verified evaluation.
   Nail candidate families are conservatively joined for exploratory partitioning.
6. One otherwise-eligible representative of each exact decoded image is retained.

Byte and pixel audit events overlap; do not add their counts to obtain a number
of excluded images. The sample manifest's status/exclusion counts are definitive.
Perceptual candidates are not automatically asserted to be the same photograph
or the same person. Their files remain available for review.

Connected duplicate/source components are assigned together using seed 42,
targeting 70/10/20. The deterministic greedy assignment balances per-dataset
totals, cohort/Hb bands or source class strata, and global counts. Group integrity
takes precedence over exact percentages; nonempty dataset partitions are reserved
when at least three groups exist. Hb bands are balancing strata, not anemia labels.

Every included sample has valid preprocessing and an available supervised
target. Eye Hb is retained in g/dL; nail rows contribute only their supplied
classification label. All statuses—including too-small, glare-rejected,
unlabeled, duplicate and quarantined—remain visible.

## Reproduction and limitations

The initial source inventory totals 5,122 images. The observed included sets are
209 eye patients and 1,347 nail images. See the exported report for exact class,
cohort and partition counts. Quality gates can alter the class distribution;
the real-data validation report includes exclusions by source class.

The verification command rehashes all inventoried source images and derived
artifacts, checks manifest checksums, validates finite features, reconciles IDs
between tables, and checks known group overlap. It cannot prove identities absent
from the source data, find every possible transformed duplicate, validate a
clinical glare threshold, or establish model accuracy.

Grey-World equalizes channel means and can remove useful chromatic information.
The required 27-feature contract deliberately remains post-normalization, as in
Person 2's adapter. Raw ROI/stage exports are retained so Person 2 can evaluate
that tradeoff against laboratory Hb; preprocessing alone is not evidence of
improved screening performance.

CP-AnemiC is not supplied or silently substituted. Its official V2 page reported
author-requested removal during source inspection. Acquisition remains recorded
as unavailable; the pipeline and reports cover the two supplied datasets only.
