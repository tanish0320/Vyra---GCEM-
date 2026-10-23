# Sources and attribution

The feature recipe and compatibility interface are adapted from the project
teammate's supplied `normalization_adapter.py`. Its original SHA-256 is
`e5d0db9724822bd2a21f931ca3c2ca368ab4889014c10097f08e2b0ab02f4e67`.
An unchanged copy is kept under test fixtures solely for contract regression
tests. The original file in the parent project has not been replaced.

The project guide references the [JetX-GT nail detector](https://huggingface.co/JetX-GT/nail-anemia-detector)
for colorimetric feature engineering, [AnemiaVision](https://github.com/RAHULPATEL2002/anemia-detection)
for the application scaffold, and [PallorHb](https://github.com/Pai-88/pallor-hb)
for duplicate-audit methodology. Their full applications, weights, scalers and
datasets are not vendored here. The exact feature implementation used for
compatibility is the teammate-provided 27-value recipe, not the 28-value model
card example.

## Datasets

- Dimauro, Giovanni; Maglietta, Rosalia; Bai, Thulasi; Kasiviswanathan, Sivachandar
  (2022). *Eyes-defy-anemia*. IEEE DataPort, DOI
  [10.21227/t5s2-4j73](https://doi.org/10.21227/t5s2-4j73).
  The supplied archive also requests citations to its accompanying anemia and
  conjunctiva-segmentation publications; retain the included dataset documentation
  when preparing data.
- Asare, Justice Williams; Appiahene, Peter; Donkoh, Emmanuel (2022).
  *Detection of Anemia using Colour of the Fingernails Image Datasets from Ghana*.
  Mendeley Data V1, DOI [10.17632/2xx4j3kjg2.1](https://doi.org/10.17632/2xx4j3kjg2.1).

No dataset photographs are distributed in the GitHub source/report bundle.
Dataset terms and attribution remain those of the respective providers; this
package does not relicense the datasets. Python dependencies are installed from
their own distributions and retain their respective notices.
