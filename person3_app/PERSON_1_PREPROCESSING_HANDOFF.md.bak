# Person 1 Preprocessing & Data Handoff Note

**To:** Person 1 (Data, Computer Vision & Optical Normalization)  
**From:** Person 2 (AI/ML, Model Architecture & Evaluation)  
**Status:** Baseline adapter integrated; ready for final normalization module substitution.

---

## 1. Preprocessing Contract & Shared Signature

The AI/ML model consumes preprocessed visual tensors and colorimetric vectors through `src/data/normalization_adapter.py`. When your final `normalization.py` module is completed, it should satisfy:

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class PreprocessingResult:
    raw_image_rgb: np.ndarray          # (H, W, 3) uint8
    normalized_image_rgb: np.ndarray   # (H, W, 3) uint8, white-balanced and CLAHE-equalized
    feature_vector: np.ndarray         # (27,) float32 camera-invariant colorimetric ratios
    body_part_encoding: np.ndarray     # (2,) float32: [1.0, 0.0] for conjunctiva, [0.0, 1.0] for fingernails
    ita_angle: float                   # Individual Typology Angle in degrees
    ita_category: str                  # Very Light through Dark
    glare_fraction: float              # Fraction of specular glare pixels inpainted
    is_valid: bool                     # False if resolution < 50px or file unreadable
    quality_warnings: list[str] = field(default_factory=list)
```

---

## 2. Handcrafted Colorimetric Feature Order (27-dim Vector)

The neural network fusion layer expects the 27 features in this exact index order:

| Index | Feature Description | Formula / Source |
|---|---|---|
| `0` | Mean Brightness | $0.299 R + 0.587 G + 0.114 B$ |
| `1` | Std Brightness | Standard deviation of grayscale intensity |
| `2-6` | Brightness Percentiles | 10th, 25th, 50th, 75th, 90th percentiles |
| `7` | Mean Redness Index | $R / (R + G + B + \epsilon)$ |
| `8` | Std Redness Index | Standard deviation of redness |
| `9` | White Pallor Ratio | Fraction of pixels with brightness $> 180$ |
| `10` | Pink Ratio | Fraction of pixels with $R > 150, G < 150, B < 150$ |
| `11-16` | R, G, B Channel Statistics | Mean and std for Red, Green, and Blue channels |
| `17` | Red/Green Ratio | $(\bar{R} + 1) / (\bar{G} + 1)$ |
| `18` | Red/Blue Ratio | $(\bar{R} + 1) / (\bar{B} + 1)$ |
| `19` | Red-Blue Difference | $(\bar{R} - \bar{B}) / 255.0$ |
| `20-21` | Hemoglobin Proxy | $\bar{R} / (\bar{G} + \bar{B} + 1)$ mean and std |
| `22` | Vertical Spatial Gradient | Top 1/3 mean brightness minus Bottom 1/3 mean brightness |
| `23` | Center-Surround Contrast | Center crop brightness minus global mean brightness |
| `24-25` | Gradient Magnitude | Mean and std of spatial Sobel / diff gradient |
| `26` | Local Texture Variance | Mean local standard deviation across $7 \times 7$ neighborhood |

---

## 3. Data Leakage & Split Manifests

All three datasets have been reconciled into patient-level manifests with **zero patient overlap**:
- `data/manifests/dataset_inventory.csv`: Complete metadata for all 5,187 images.
- `data/manifests/duplicate_audit.csv`: 2,375 exact duplicates and 98 perceptual duplicates quarantined.
- `data/splits/train.csv`: 3,560 images (690 patient clusters; 652 with lab Hb).
- `data/splits/validation.csv`: 602 images (99 patient clusters; 100 with lab Hb).
- `data/splits/test.csv`: 1,025 images (196 patient clusters; 174 with lab Hb).
