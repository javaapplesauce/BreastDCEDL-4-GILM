# Column Descriptions and Statistics

## Patient Demographics and Clinical Information
| Column | Description | Mean | Median | Std | Missing |
|--------|-------------|------|---------|-----|---------|
| pid | Unique patient identifier | N/A | N/A | N/A | 0 |
| age | Patient age (years) | 48.8 | 49.0 | 10.5 | 3 |
| menopause | Menopausal status (binary) | 0.4 | 0.0 | 0.5 | 38 |
| race_white | Race: White (binary) | 0.8 | 1.0 | 0.4 | 3 |
| race_black | Race: Black (binary) | 0.1 | 0.0 | 0.3 | 3 |
| e_hispanic_latino | Ethnicity: Hispanic/Latino (binary) | 0.1 | 0.0 | 0.3 | 1 |

## Hormone Status and Cancer Subtypes
| Column | Description | Mean | Median | Std | Missing |
|--------|-------------|------|---------|-----|---------|
| hormon_status | Hormonal status | N/A | N/A | N/A | 0 |
| HR | Hormone receptor status (binary) | 0.5 | 1.0 | 0.5 | 0 |
| HER2 | HER2 status (binary) | 0.2 | 0.0 | 0.4 | 0 |
| HR_HER2_STATUS | Combined HR and HER2 status | N/A | N/A | N/A | 0 |
| TripleNeg | Triple negative status (binary) | 0.4 | 0.0 | 0.5 | 0 |
| HER2pos | HER2 positive status (binary) | 0.2 | 0.0 | 0.4 | 0 |
| HRposHER2neg | HR positive/HER2 negative status (binary) | 0.4 | 0.0 | 0.5 | 0 |
| MP | MammaPrint risk score (0=low risk, 1=high risk) | 0.5 | 0.0 | 0.5 | 0 |

## Neoadjuvant Chemotherapy Drugs
| Column | Description | Mean | Median | Std | Missing |
|--------|-------------|------|---------|-----|---------|
| Ganitumab | Ganitumab treatment (binary) | 0.1 | 0.0 | 0.3 | 0 |
| Pembrolizumab | Pembrolizumab treatment (binary) | 0.1 | 0.0 | 0.3 | 0 |
| AMG 386 | AMG 386 treatment (binary) | 0.1 | 0.0 | 0.3 | 0 |
| Ganetespib | Ganetespib treatment (binary) | 0.1 | 0.0 | 0.3 | 0 |
| Carboplatin | Carboplatin treatment (binary) | 0.1 | 0.0 | 0.3 | 0 |
| MK-2206 | MK-2206 treatment (binary) | 0.1 | 0.0 | 0.3 | 0 |
| T-DM1 | T-DM1 treatment (binary) | 0.1 | 0.0 | 0.2 | 0 |
| Neratinib | Neratinib treatment (binary) | 0.1 | 0.0 | 0.3 | 0 |
| ABT 888 | ABT 888 treatment (binary) | 0.1 | 0.0 | 0.3 | 0 |
| Trastuzumab | Trastuzumab treatment (binary) | 0.1 | 0.0 | 0.3 | 0 |
| Paclitaxel | Paclitaxel treatment (binary) | 0.9 | 1.0 | 0.2 | 0 |
| Pertuzumab | Pertuzumab treatment (binary) | 0.1 | 0.0 | 0.3 | 0 |

## Imaging Parameters and Measurements
| Column | Description | Mean | Median | Std | Missing |
|--------|-------------|------|---------|-----|---------|
| n_xy | Number of pixels in x-y plane | 278.7 | 256.0 | 46.6 | 0 |
| n_z | Number of slices in z-direction | 105.9 | 80.0 | 41.2 | 0 |
| n_times | Number of time points | 7.2 | 7.0 | 0.8 | 0 |
| slice_thick | Slice thickness (mm) | 1.7 | 2.0 | 0.5 | 0 |
| xy_spacing | Pixel spacing in x-y plane (mm) | 0.7 | 0.7 | 0.1 | 0 |
| mask_start | Starting slice of tumor mask | 33.9 | 30.0 | 20.8 | 0 |
| mask_end | Ending slice of tumor mask | 71.4 | 64.0 | 29.7 | 0 |

## Study Design Parameters
| Column | Description | Mean | Median | Std | Missing |
|--------|-------------|------|---------|-----|---------|
| pre | Pre-treatment imaging indicator | 0.0 | 0.0 | 0.0 | 0 |
| post_early | Early post-treatment imaging | 2.0 | 2.0 | 0.0 | 0 |
| post_late | Late post-treatment imaging | 5.5 | 5.0 | 0.5 | 0 |
| pCR | Pathologic complete response status | 0.3 | 0.0 | 0.5 | 0 |
| test | Train/validation or test set indicator | 0.3 | 0.0 | 0.6 | 0 |

Notes:
1. Tumor volume has been converted from mm³ to cm³
2. Binary variables are coded as 0/1
3. Missing values are most prevalent in menopause status (38 missing)
4. Some demographic information has a few missing values (3-4 cases)
5. Most treatment indicators show low usage (~10% of patients) except Paclitaxel (94.7%)
6. Image dimensions and tumor volumes show considerable variation across patients