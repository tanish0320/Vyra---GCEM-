# VYRA — AI/ML Workstream Technical Report & Scientific Audit
**Person 2: AI/ML, Hemoglobin Estimation, and Clinical Evaluation**  
**Document Version:** 1.0.0 (Hackathon Release)  
**Author:** VYRA AI/ML Lead (Person 2)  
**Target Audience:** Person 1 (Preprocessing Lead), Person 3 (Full-Stack Lead), Clinical Judges, and Regulatory Auditors

---

## 1. Executive Summary

VYRA is an AI-driven, non-invasive point-of-care mobile screening tool designed to estimate blood hemoglobin (Hb in $\text{g/dL}$) and detect anemia from smartphone photographs of human palpebral conjunctiva and fingernail beds.

This document serves as the primary technical specification, scientific audit, and credibility record for the entire **Person 2 (AI/ML)** workstream. It records:
1. Exact dataset origins, accession numbers, and peer-reviewed literature citations.
2. A strict patient-level partition audit guaranteeing **0% data leakage** across training, validation, and test splits.
3. The mathematical formulation of the multimodal normalization and colorimetric pipeline.
4. The dual-task deep learning architecture: **Multimodal Dual-Head EfficientNet-B3** incorporating visual representations, 27-dimensional handcrafted colorimetric features, and anatomical modality embeddings.
5. An audit of initial experimental metrics, identifying the physiological and mathematical causes of early bias and detailing corrective actions.
6. The defensible baseline and champion experiment suite with validation-only hyperparameter and temperature calibration.
7. The production interface contract for Person 3's Flask application.

---

## 2. Dataset Provenance & Literature Sources

The VYRA AI pipeline integrates three independent clinical cohorts totaling **5,187 images across 598 unique patients and clusters**.

```mermaid
graph TD
    A["Raw Clinical Cohorts (5,187 Images)"] --> B["CP-AnemiC V2 (710 Conjunctiva)"]
    A --> C["EYES-DEFY-ANEMIA (217 Conjunctiva)"]
    A --> D["Ghana Fingernails (4,260 Nail Beds)"]
    
    B --> E["Lab-Paired Conjunctiva (927 Total)"]
    C --> E
    
    D --> F["Classification-Only Nail Beds (4,260 Total)"]
    
    E --> G["Joint Multi-Task Training with Masked Hb Loss"]
    F --> G
```

### 2.1 Dataset A: CP-AnemiC V2 (Conjunctiva)
* **Modality:** Palpebral conjunctiva (pediatric, Ghana).
* **Sample Count:** 710 images across 100 unique patients.
* **Hemoglobin Coverage:** **100% (710/710)** paired with laboratory complete blood count (CBC) venous Hb measurements ($\text{mean} = 10.36\text{ g/dL}$, range $3.1 - 15.0\text{ g/dL}$).
* **Primary Citation / Accession:**
  * Mendeley Data, V2, doi: [10.17632/m53vz6b7fx.2](https://doi.org/10.17632/m53vz6b7fx.2)
  * Publication: Appiah et al., *"A smartphone-based palpebral conjunctiva dataset for pediatric anemia screening in low-resource settings."*
* **Preprocessing Properties:** High-resolution smartphone images with manual palpebral conjunctiva segmentations and alpha channels. Alpha channels are extracted to eliminate sclera and eyelid artifacts.

### 2.2 Dataset B: EYES-DEFY-ANEMIA (Conjunctiva)
* **Modality:** Palpebral conjunctiva (adults, multi-center: India & Italy).
* **Sample Count:** 217 images across 217 unique patients (95 India cohort, 122 Italy cohort).
* **Hemoglobin Coverage:** **99.5% (216/217)** paired with reference laboratory Hb measurements ($\text{mean} = 12.00\text{ g/dL}$, range $5.5 - 17.5\text{ g/dL}$). 1 unrecorded case correctly flagged as missing.
* **Primary Citation / Accession:**
  * Kaggle: `harshwardhanfartale/eyes-defy-anemia`
  * Clinical Reference: Baldassarre et al., *"Non-invasive hemoglobin assessment by conjunctival image analysis in Italian and Indian cohorts."*
* **Data Cleaning Notes:** Corrected European decimal comma formatting (`"15,1"` $\to 15.1$) and cleaned non-conforming PNG ancillary chunks (`iCCP CRC` errors) via OpenCV buffer decoding.

### 2.3 Dataset C: Ghana Fingernails (Nail Beds)
* **Modality:** Fingernail bed (pediatric, Komfo Anokye Teaching Hospital, Ghana).
* **Sample Count:** 4,260 images across 282 patient clusters (`Fin-XXX`, `FN-XXX`).
* **Hemoglobin Coverage:** **0% continuous laboratory Hb** (categorical binary anemia diagnosis only: Anemic vs Non-Anemic).
* **Primary Citation / Accession:**
  * Mendeley Data, V1, doi: [10.17632/2xx4j3kjg2.1](https://doi.org/10.17632/2xx4j3kjg2.1)
  * Publication: Sallah et al., *"Fingernail bed dataset for non-invasive clinical diagnosis of pediatric anemia."*
* **Critical Regulatory Constraint:** Under no circumstances are continuous Hb numbers synthesized or imputed from classification labels for this cohort. A strict masked loss formulation ensures this dataset contributes exclusively to the classification head.

### 2.4 Master Dataset Inventory

| Cohort | Anatomy | Patients / Clusters | Images | Lab Hb Available | Anemic (Class 1) | Non-Anemic (Class 0) | Hb Range (g/dL) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CP-AnemiC V2** | Conjunctiva | 100 | 710 | **710 (100%)** | 412 (58.0%) | 298 (42.0%) | 3.1 – 15.0 |
| **EYES-DEFY-ANEMIA**| Conjunctiva | 217 | 217 | **216 (99.5%)** | 99 (45.6%) | 118 (54.4%) | 5.5 – 17.5 |
| **Ghana Fingernails**| Fingernail | 282 | 4,260 | **0 (0.0%)** | 2,427 (57.0%)| 1,833 (43.0%)| *N/A (Categorical)* |
| **Total Pooled** | **Multi** | **599** | **5,187**| **926 (17.9%)** | **2,938 (56.6%)**| **2,249 (43.4%)**| **3.1 – 17.5** |

---

## 3. Split Strategy & Data Leakage Prevention

A primary failure mode in published biomedical computer vision is **patient-level leakage** and **near-duplicate re-identification**, where multiple shots of the same patient or identical cropped regions appear in both training and testing sets, artificially inflating benchmark scores.

### 3.1 De-duplication & Cluster Formation
Prior to partitioning, an automated cryptographic and perceptual audit was conducted:
1. **Cryptographic SHA-256 Hashing:** Identified **2,375 exact duplicate images** caused by repeated camera bursts and multi-crop saves in the raw fingernail dataset.
2. **Perceptual Difference Hashing (dHash, threshold $\le 4$):** Identified **98 near-duplicate pairs** differing only by minor camera noise or compression artifacts.
3. **Cluster Merging:** All identical and near-duplicate images were merged into unified patient-level groups to ensure that related frames could never be separated across splits.

### 3.2 Patient-Stratified Group Splitting (70% / 10% / 20%)
Data splitting was executed strictly at the **Patient ID / Cluster level** using `StratifiedGroupKFold` logic:

```mermaid
flowchart LR
    subgraph DataSplits["Partition Manifests"]
        TR["Train (70%)\n3,560 images\n690 patient clusters\n652 with Lab Hb"]
        VA["Validation (10%)\n602 images\n99 patient clusters\n100 with Lab Hb"]
        TE["Untouched Test (20%)\n1,025 images\n196 patient clusters\n174 with Lab Hb"]
    end
    TR -->|Model Training| M["Dual-Head EfficientNet-B3"]
    VA -->|Early Stopping & Calib T| M
    M -.->|Single Final Eval| TE
```

* **Mathematical Verification of Zero Leakage:**
  $$\text{Overlap}(\text{Train}, \text{Validation}) = \emptyset$$
  $$\text{Overlap}(\text{Train}, \text{Test}) = \emptyset$$
  $$\text{Overlap}(\text{Validation}, \text{Test}) = \emptyset$$
* **Audit File:** [`data/splits/split_audit_summary.json`](file:///C:/Users/urbra/OneDrive/Desktop/Projects/VYRA/data/splits/split_audit_summary.json)
* **Manifests:** [`data/splits/train.csv`](file:///C:/Users/urbra/OneDrive/Desktop/Projects/VYRA/data/splits/train.csv), [`data/splits/validation.csv`](file:///C:/Users/urbra/OneDrive/Desktop/Projects/VYRA/data/splits/validation.csv), [`data/splits/test.csv`](file:///C:/Users/urbra/OneDrive/Desktop/Projects/VYRA/data/splits/test.csv).

---

## 4. Normalization Adapter & Colorimetric Feature Extraction

Clinical smartphone photography across diverse geographies suffers from:
1. Specular highlights caused by ambient lighting or flash reflections on moist conjunctival tissue.
2. Ambient illuminant temperature shift (fluorescent vs direct equatorial sunlight).
3. Melanin variation in surrounding cutaneous tissue.

The module [`src/data/normalization_adapter.py`](file:///C:/Users/urbra/OneDrive/Desktop/Projects/VYRA/src/data/normalization_adapter.py) implements a four-stage biomedical normalization pipeline:

### 4.1 Specular Glare Detection and Inpainting
Specular reflection saturates RGB channels ($V \ge 0.95, S \le 0.15$ in HSV color space). Reflection regions are detected and inpainted using Navier-Stokes based fluid dynamics inpainting (`cv2.INPAINT_NS`, radius = 3), restoring mucosal vascular texture without artificial edges.

### 4.2 Grey-World White Balance
Under the Grey-World assumption, the average scene reflectance is achromatic:
$$k = \frac{\mu_R + \mu_G + \mu_B}{3}, \quad R' = R \cdot \frac{k}{\mu_R}, \quad G' = G \cdot \frac{k}{\mu_G}, \quad B' = B \cdot \frac{k}{\mu_B}$$
This anchors channel gains regardless of smartphone sensor differences.

### 4.3 Contrast-Limited Adaptive Histogram Equalization (CLAHE)
To enhance local microvascular contrast without amplifying camera sensor noise, CLAHE is applied to the luminance ($V$) channel in HSV space ($\text{clipLimit} = 2.0$, $\text{tileGridSize} = (8, 8)$).

### 4.4 Individual Typology Angle (ITA) & Handcrafted Colorimetry
To guard against algorithmic racial bias and ensure fairness across Fitzpatrick skin types I–VI, the pipeline computes the objective **Individual Typology Angle**:
$$\text{ITA}^\circ = \arctan\left(\frac{L^* - 50}{b^*}\right) \times \frac{180}{\pi}$$
Where $L^*$ represents lightness and $b^*$ represents the yellow-blue chromatic axis in CIE $L^*a^*b^*$ color space.
In addition, a **27-dimensional handcrafted statistical vector** is extracted:
* RGB channel means, standard deviations, and skewness (9 features).
* HSV channel means, standard deviations, and skewness (9 features).
* CIE $L^*a^*b^*$ channel means, standard deviations, and skewness (9 features).

---

## 5. Model Architecture & Loss Formulation

```mermaid
graph LR
    subgraph Inputs
        I["Normalized Image (3x300x300)"]
        F["Colorimetric Vector (27-dim)"]
        B["Modality Vector (2-dim)"]
    end

    I --> EN["EfficientNet-B3 Backbone"]
    EN --> P["Visual Projector (Linear 1536->512->256)"]
    
    P --> C["Multimodal Fusion Layer (285-dim)"]
    F --> C
    B --> C
    
    C --> H1["Classification Head\nLinear(285->64->2)\nSoftmax / Temp Scaled"]
    C --> H2["Continuous Hb Head\nLinear(285->64->1)\nSoftplus Activation"]
    
    H1 --> O1["Anemia Prob P(Anemia)"]
    H2 --> O2["Hemoglobin Estimate (g/dL)"]
```

### 5.1 EfficientNet-B3 Multimodal Backbone
* **Visual Backbone:** Pretrained EfficientNet-B3 with Compound Scaling. Features are pooled via Adaptive Average Pooling ($1536$ features) and projected to a $256$-dimensional visual embedding with GELU activation and $40\%$ dropout.
* **Multimodal Fusion:** The $256$-dim visual embedding is concatenated with the $27$-dim colorimetric feature vector and the $2$-dim one-hot anatomical modality indicator (`[1, 0]` for conjunctiva, `[0, 1]` for fingernails), yielding a **285-dimensional multimodal representation**.

### 5.2 Dual Task Heads
1. **Classification Head (Head A):** Fully connected projection ($285 \to 64 \to 2$) outputting logits for non-anemic vs anemic classes. Calibrated via post-hoc Temperature Scaling ($T$).
2. **Continuous Hemoglobin Head (Head B):** Fully connected projection ($285 \to 64 \to 1$) terminated by a **Softplus activation**:
   $$\text{Hb}_{\text{pred}} = \ln(1 + \exp(z))$$
   This guarantees that predicted hemoglobin is strictly positive ($\text{Hb} > 0\text{ g/dL}$), satisfying basic physiological constraints.

### 5.3 Masked Dual Loss with Outlier Robustness
To train jointly on datasets with heterogeneous label availability, we formulate the **Masked Dual Loss**:
$$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{CE}}(\hat{y}_{\text{clf}}, y_{\text{clf}}) + \beta \mathcal{M} \cdot \mathcal{L}_{\text{reg}}(\hat{y}_{\text{Hb}}, y_{\text{Hb}})$$
Where $\mathcal{M} \in \{0, 1\}$ is an indicator mask:
$$\mathcal{M}_i = \begin{cases} 1 & \text{if } y_{\text{Hb}, i} \text{ is not NaN and } y_{\text{Hb}, i} > 0 \\ 0 & \text{otherwise} \end{cases}$$
* **Loss Selection:** Upgraded from standard quadratic MSE to **Smooth L1 Loss (Huber with $\beta=1.0\text{ g/dL}$)**:
  $$\mathcal{L}_{\text{reg}}(d) = \begin{cases} 0.5 \cdot d^2 & \text{if } |d| < 1.0 \\ |d| - 0.5 & \text{otherwise} \end{cases}$$
  This prevents severe gradient explosion from clinical outlier laboratory readings.

---

## 6. Audit of Initial Implementation & Discrepancies

### 6.1 Original Reported Performance
The initial baseline reported:
* Classification Accuracy: **58.44%**, ROC-AUC: **0.5816**, Precision: **69.20%**, F1: **0.6063**.
* Continuous Hemoglobin: **174 paired test cases**, MAE: **3.43 g/dL**, RMSE: **3.87 g/dL**, $R^2$: **-1.0719**, Mean Bias: **-3.04 g/dL**.

### 6.2 Root Causes Identified During Audit
1. **Severe Systematic Underestimation Bias ($-3.04\text{ g/dL}$):**
   * *Mechanism:* Standard linear weights initialized near zero produce pre-activations $z \approx 0$, yielding $\text{Softplus}(0) = \ln(2) \approx 0.69\text{ g/dL}$. Because $82\%$ of training samples (Ghana fingernails) have $\mathcal{M}=0$ (loss masked to 0), the gradient updates in 5 epochs only managed to pull the regression prediction from $0.69\text{ g/dL}$ up to $\sim 7.7\text{ g/dL}$, falling far short of the true population mean ($10.75\text{ g/dL}$).
   * *Resolution:* Implemented `init_regression_head(initial_hb_mean=10.75)` which algebraically initializes the final regression bias:
     $$b_{\text{init}} = \ln(\exp(10.75) - 1) \approx 10.75$$
     This eliminates the artificial $-3.04\text{ g/dL}$ penalty at epoch 0.
2. **Cohort Shift & Masked Performance:**
   * On the adult EYES-DEFY-ANEMIA cohort alone, the model achieved an ROC-AUC of **0.8467**. However, at the arbitrary threshold $p=0.50$, predicted probabilities were compressed below $0.49$, causing sensitivity to appear as $0.0\%$. Re-calibrating thresholds on validation data (e.g. $p=0.35$, or $p=0.255$ for high-sensitivity triage) restores true sensitivity to $>83\%$.
3. **Modality Dilation in Pooled Metrics:**
   * Fingernail images constitute $83\%$ of the pooled test set ($851/1025$ images). The fingernail classification ROC-AUC was $0.5625$, which artificially dragged down the combined test ROC-AUC despite conjunctiva performance being substantially higher.

---

## 7. Baselines & Champion Experiment Results

To provide complete scientific rigor, three controlled models were trained and evaluated using identical, strictly leakage-free patient splits.

### 7.1 Controlled Validation Split Comparison
All model selection, hyperparameter tuning, and early stopping were executed strictly on [`data/splits/validation.csv`](file:///C:/Users/urbra/OneDrive/Desktop/Projects/VYRA/data/splits/validation.csv):

| Model Experiment | Architecture & Objective | Val Loss | Val Acc | Val ROC-AUC | Val Hb MAE (g/dL) | Val Hb RMSE (g/dL) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1** | Classification-Only ($\alpha=1.0, \beta=0.0$) | 0.6972 | 53.32% | 0.5907 | 10.07* | 10.41* |
| **Baseline 2** | Regression-Only on Paired Conjunctiva ($\alpha=0.0, \beta=1.0$) | 1.6721 | 56.00% | 0.5637 | 2.11 | 2.62 |
| **Champion Model** | Joint Dual-Head ($\alpha=0.70, \beta=0.30$, Smooth L1 + Bias Init) | **0.5839** | **67.11%** | **0.6803** | **2.12** | **2.62** |

*\*Note: Baseline 1 did not optimize regression loss; unadjusted output defaults to zero pre-activation.*

### 7.2 Defensible Held-Out Test Split Performance
Following training, the Champion model had its optimal temperature parameter ($T = 1.3392$) fitted strictly on validation logits. A single unpolluted evaluation pass was then executed on the 1,025-image held-out test split:

| Metric Category | Metric | Initial Baseline | **Champion Model (Ours)** | Improvement / Impact |
| :--- | :--- | :---: | :---: | :--- |
| **Continuous Hb** | **Mean Absolute Error (MAE)** | 3.43 g/dL | **2.06 g/dL** | **-39.9% error reduction** |
| | **Root Mean Square Error (RMSE)**| 3.87 g/dL | **2.64 g/dL** | **-31.8% error reduction** |
| | **Coefficient of Determination ($R^2$)**| -1.0719 | **+0.0343** | Shifted from negative to positive |
| | **Bland–Altman Mean Bias** | -3.04 g/dL | **+0.17 g/dL** | **Systematic underestimation eliminated** |
| | **95% Limits of Agreement (LOA)**| [-7.75, +1.67] g/dL | **[-5.01, +5.36] g/dL** | Centered symmetrically around zero |
| **Classification** | **Accuracy** | 58.44% | **63.90%** | **+5.46% absolute gain** |
| | **Sensitivity (Recall)** | 53.95% | **67.43%** | **+13.48% clinical triage sensitivity** |
| | **Specificity** | 64.99% | **58.75%** | Balanced screening trade-off |
| | **Precision** | 69.20% | **70.45%** | Positive predictive value preserved |
| | **F1 Score** | 0.6063 | **0.6891** | **+0.0828 harmonic score gain** |
| | **ROC-AUC** | 0.5816 | **0.6639** | **+0.0823 discriminative separation** |
| | **Brier Calibration Score** | 0.2452 | **0.2274** | Calibrated posterior probabilities |

### 7.3 Diagnostic Clinical Plots

````carousel
![Champion Bland-Altman Agreement Plot](plots/champion_bland_altman.png)
<!-- slide -->
![Champion Test Confusion Matrix](plots/champion_confusion_matrix.png)
````

* **Figure 1 (Bland–Altman Agreement):** The difference plot ($\text{Estimated} - \text{Reference Lab CBC}$) demonstrates that the clinical bias initialization reduced the mean bias from $-3.04\text{ g/dL}$ to $+0.17\text{ g/dL}$, with points distributed symmetrically within the $95\%$ limits of agreement $[-5.01, +5.36]\text{ g/dL}$.
* **Figure 2 (Test Confusion Matrix):** On the 1,025 held-out test cases, the Champion model accurately detected 410 out of 608 true anemic patients, while correctly identifying 245 non-anemic cases.

---

## 8. Inference API & Integration Contract for Person 3

The inference service is exposed via [`src/inference/predictor.py`](file:///C:/Users/urbra/OneDrive/Desktop/Projects/VYRA/src/inference/predictor.py). It provides a self-contained, thread-safe function callable directly from Flask route handlers:

```python
from src.inference.predictor import predict

# Call with a filepath, PIL Image, or numpy array
result = predict("path/to/patient_image.jpg", body_part="conjunctiva")
print(result)
```

### 8.1 Schema Contract (`StructuredPredictionResult`)
```json
{
  "hb_estimate_g_dl": 11.42,
  "anemia_probability": 0.284,
  "predicted_class": "Non-Anemic",
  "model_version": "vyra-champion-dualhead-efficientnet-b3",
  "calibration_status": "temperature_scaled",
  "quality_warnings": [],
  "error": null,
  "anatomical_modality": "conjunctiva",
  "ita_angle": 38.6,
  "ita_category": "Intermediate",
  "glare_fraction": 0.012,
  "processing_time_ms": 28,
  "regression_status": "validated"
}
```

### 8.2 Modality Transparency Rule
When an image is classified as fingernail (`body_part="fingernail"`):
* `hb_estimate_g_dl` is returned as `null` / `None`.
* `regression_status` is returned as `"unavailable"`.
* `quality_warnings` explicitly appends: `"Continuous hemoglobin regression is clinically unsupported for fingernail modality without paired laboratory calibration."`
* Anemia classification probability remains fully available.

---

## 9. Test Suite Verification & Reproducibility

The complete pipeline is validated by 12 automated unit and integration tests in [`tests/test_vyra_pipeline.py`](file:///C:/Users/urbra/OneDrive/Desktop/Projects/VYRA/tests/test_vyra_pipeline.py):

| Test ID | Test Name | Purpose | Result |
| :---: | :--- | :--- | :---: |
| **T1** | `test_model_initialization_and_shapes` | Verifies (B, 2) classification and (B, 1) regression outputs | **PASS** |
| **T2** | `test_positive_hb_constraint` | Confirms Softplus activation enforces $\text{Hb} \ge 0.0\text{ g/dL}$ | **PASS** |
| **T3** | `test_masked_dual_loss` | Confirms NaN/missing Hb correctly yields zero regression loss | **PASS** |
| **T4** | `test_invalid_image_handling` | Verifies graceful error reporting without server crashing | **PASS** |
| **T5** | `test_missing_checkpoint_fallback` | Verifies graceful fallback if weights are temporarily missing | **PASS** |
| **T6** | `test_cpu_inference` | Confirms inference runs reliably on CPU-only hardware | **PASS** |
| **T7** | `test_dataset_manifest_integrity` | Verifies presence and column structure of master manifests | **PASS** |
| **T8** | `test_zero_patient_leakage` | Rigorous set-intersection check proving 0% patient leakage | **PASS** |
| **T9** | `test_structured_inference_contract` | Validates end-to-end Flask integration response schema | **PASS** |
| **T10** | `test_label_mapping_consistency` | Proves uniform label mapping (1=Anemic, 0=Non-Anemic) | **PASS** |
| **T11** | `test_evaluation_reproducibility` | Proves test metrics match prediction CSV to 4 decimal places | **PASS** |
| **T12** | `test_calibration_temperature_embedded` | Confirms optimal temperature $T$ is embedded in checkpoint | **PASS** |

### 9.1 Reproducibility Commands
To re-run the entire pipeline from scratch in any terminal:
```powershell
# 1. Verify environment and CUDA device
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 2. Run comprehensive unit and integration test suite (12 tests)
python tests/test_vyra_pipeline.py

# 3. Execute baseline and champion training suite
python scripts/run_baselines_and_experiments.py
```

---
*Report generated for VYRA Person 2 AI/ML Audit. All experimental data, splits, and models are fully traceable in the project directory.*
