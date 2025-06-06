
# BreastDCEDL

**BreastDCEDL** is a curated collection of pretreatment 3D dynamic contrast-enhanced MRI (DCE-MRI) scans from **2,085 breast cancer patients**, assembled into a deep learning–ready dataset. It integrates data from three major clinical trials: **I-SPY2** (n = 982), **I-SPY1** (n = 173), and **Duke** (n = 920). The dataset, originally sourced from The Cancer Imaging Archive (TCIA), includes:

-   3D raw MRI scans converted to NIfTI format
-   Corresponding 3D tumor binary segmentation masks
-   Clinical and demographic metadata, including pCR, HER2, HR, age, and race

----------

## Benchmark Prediction Tasks

The dataset provides a standardized benchmark for three central classification tasks in breast cancer MRI:

-   **Pathological Complete Response (pCR):** A binary classification task predicting treatment response based on pretreatment imaging. Approximately 32.2% of patients (n = 317) achieved pCR, offering a moderately balanced class distribution. _pCR (pathologic complete response) refers to the complete disappearance of all invasive cancer cells in the breast and lymph nodes following neoadjuvant therapy and is considered a strong surrogate for favorable long-term prognosis._
    
-   **Hormone Receptor (HR) Status:** Classification of HR positivity (present in 54.5% of cases, n = 537) directly from imaging, assessing the link between MRI features and receptor expression.
    
-   **HER2 Status:** Prediction of HER2 expression (positive in 24.8%, n = 244) from imaging data, enabling evaluation of MRI-based biomarker inference.
    

Ground truth labels include HR, HER2, and pCR status, as well as molecular subtypes (HR+/HER2−, HER2+, Triple Negative) and MammaPrint risk categories. The dataset is split into training, validation, and test cohorts with preserved class distributions to ensure consistent and reproducible 
| Split      | pCR N | pCR+ | pCR− | HR N | HR+  | HR−  | HER2 N | HER2+ | HER2− |
|------------|-------|------|------|------|------|------|--------|-------|-------|
| Training   | 1099  | 324  | 775  | 1543 | 997  | 546  | 1542   | 349   | 1193  |
| Validation | 174   | 53   | 121  | 269  | 168  | 101  | 269    | 58    | 211   |
| Test       | 175   | 53   | 122  | 271  | 173  | 98   | 269    | 56    | 213   |
| **Total**  | 1448  | 430  | 1018 | 2083 | 1338 | 745  | 2080   | 463   | 1617  |



> _Note: `pCR N` refers to the number of patients with non-missing pCR labels; similarly, `HR N` and `HER2 N` indicate the number of patients with available HR and HER2 status, respectively. Class distributions are shown for each split._

----------

## DCE-MRI Clinical Background

Dynamic Contrast-Enhanced MRI (DCE-MRI) is a 3D imaging technique that captures a sequence of scans before and after the injection of a contrast agent (typically gadolinium). The contrast enhances visibility of blood vessels and tissue perfusion, allowing observation of how the agent accumulates and clears from tissues over time.

Tumors exhibit characteristic enhancement patterns: malignant lesions often enhance quickly and wash out, while benign lesions typically enhance more slowly or steadily. Radiologists assess these patterns by reviewing two or three key time points—commonly the pre-contrast image and one or two post-contrast phases (e.g., the 2nd, 3rd, or 4th scan in the series). This helps them distinguish between benign and malignant lesions and informs treatment decisions.

These enhancement dynamics are critical both for clinical evaluation and for machine learning models that aim to predict malignancy, treatment response, or other tumor characteristics.

----------

## Dataset Details

### 🧪 I-SPY2 Dataset

The **I-SPY2** trial (Li et al., 2022; Newitt et al., 2021) provides DCE-MRI scans for 982 patients acquired from 2010 to 2016 across over 22 clinical centers using a standardized imaging protocol.

-   Target cohort: Women with high-risk, locally advanced breast cancer
-   Clinical data: pCR, HR, HER2, MammaPrint (MP) scores, type of neoadjuvant therapy, age, and race

**🖼️ Imaging Details:**

-   Each MRI scan includes 3 to 12 time points (mostly 7)
-   Radiologists selected 3 time points for tumor segmentation: typically scans 0 (pre-contrast), 2 (early post-contrast), and 5 or 6 (late post-contrast). These selections are provided in the metadata under `pre`, `post_early`, and `post_late`.

----------

### 🧪 I-SPY1 Dataset

The **I-SPY1** dataset is a predecessor to I-SPY2 and contains similar imaging and clinical information, with slightly fewer patients and minor differences in acquisition protocols.

-   Patients: 173 with 3–5 usable DCE scans
-   Clinical data: pCR, HR, HER2, and other core biomarkers

----------

### 🧪 Duke Dataset

The **Duke Breast Cancer Dataset** consists of 920 patients with biopsy-confirmed invasive breast cancer, collected between 2000 and 2014.

-   Only 288 patients (31%) received neoadjuvant chemotherapy (NAC) and have annotated pCR values.
-   The rest underwent surgery first, followed by adjuvant therapy, and are not included in pCR analysis.
-   DCE-MRI scans include one pre-contrast and 2–4 post-contrast acquisitions, spaced 1–2 minutes apart.

**🖼️ Data Processing Notes:**

-   Bounding box annotations of the largest tumor are provided.
-   No full tumor segmentation masks are available for Duke.

----------

## 🔗 Source

All datasets were originally acquired from:

-   [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)
-   Monticciolo et al., 2018, _Journal of the American College of Radiology (JACR)_
-   ClinicalTrials.gov - I-SPY2 (NCT01042379)
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTY0MDcxODczOSwtMTM4MTMyNTczM119
-->