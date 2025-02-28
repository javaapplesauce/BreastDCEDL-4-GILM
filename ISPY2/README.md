## I-SPY-2 dataset

Notebook to view I-SPY-2 data samples [![ISPY2_view_data.ipynb Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naomifridman/BreastDCEDL/blob/main/ISPY2_view_data.ipynb)

The public Breast DCE- MRI dataset entitled I-SPY 2 trial (Li et al., 2022; Newitt et al., 2021) ,comprises DCE-MRI data from 982 patients, acquired during the period from 2010 to 2016 across more than 22 clinical centers, all following a standardized image acquisition protocol.
 For each patient, the dataset encompasses MRI examinations at four distinct time points: pre-treatment (T0) and three subsequent MRI sessions conducted during and after the course of Neoadjuvant Chemotherapy (NAC). For our study's purposes, we exclusively utilized the pre-treatment MRI scans. Furthermore, the dataset includes derived maps and segmentations generated from the DCE acquisitions. Each patient's record is enriched with histopathologic information, encompassing post-treatment functional tumor volume, pathological complete response (pCR) status, hormone receptor (HR) status, human epidermal growth factor receptor 2 (HER2) status (categorized as positive or negative), MammaPrint (MP) risk level, and patient’s age at screening. Remarkably, 313 patients, constituting 32% of the cohort, achieved a pCR in response to NAC.
From a clinical perspective, the cohort demonstrates important molecular subtype distributions: 54.5% (n=537) of patients are Hormone Receptor-positive (HR+), and 24.8% (n=244) are HER2-positive. The pathologic complete response (pCR) rate, a crucial measure of treatment effectiveness, was observed in 32.2% (n=317) of patients. 

### Metadata files from TCIA - https://www.cancerimagingarchive.net/collection/ispy2/
* **ISPY2/ACRIN 6698 ISPY2 DWI and DCE MRI Data Descriptions_20210520.pdf**
  
Description of DICOM data tags and organization of I-SPY-2 dataset.

* **Analysis-mask-files-description.v20211020.docx**
  
Full description of analysis mask processing for FOV calculation.

* **ISPY2/ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx**

Excel file with all clinical and demographic data

### Metadata file created from metadata and dicom file

* **ISPY2/metadata_spy2_vis1_for_modeling.csv**

Data collected from dicom files and TCIA metadata fdiles:

| Name           | Description                                                                                              |
|----------------|----------------------------------------------------------------------------------------------------------|
| pid            | Unique patient identifier.                                                                               |
| n_xy           | Number of pixels in the x-y plane (e.g., image resolution in axial dimensions).                          |
| n_z            | Number of slices in the z-direction (depth of the image volume).                                         |
| n_times        | Number of time points or repeated imaging sessions.                                          |
| pre            | Indicator for pre-treatment imaging data (baseline scan).                                               |
| post_early     | Indicator for early post-treatment imaging (scan acquired soon after treatment).                         |
| post_late      | Indicator for late post-treatment imaging (scan acquired after an extended interval post-treatment).     |
| pix_type       | Pixel type or imaging modality, specifying the nature of the image pixels.                               |
| slice_thick    | Thickness of each slice in the imaging modality (measured in millimeters).                               |
| slice_space    | Spacing between consecutive slices (measured in millimeters).                                           |
| xy_spacing     | Spacing between pixels in the x-y plane (measured in millimeters).                                       |
| mask_start     | Starting slice with tumor segmentation.                                       |
| mask_end       | Ending slice with tumor segmentation.                                           |
| age            | Age of the patient at the time of imaging.                                                               |
| r_white        | Race White.                                    |
| r_black        | Race Black or African American.                                    |
| mask_count     | Number of slices with segmentation masks.                                                  |
| mask_max       | The slice index of the Maximum value of voxels in segmentation mask (e.g., highest intensity or largest area).           |
| pCR            | Pathologic complete response status (binary indicator: 1 for complete response, 0 otherwise).            |
| HR             | Hormone receptor status (binary, indicating positive or negative status).                       |
| MP             | Molecular profile             |
| HRposHER2neg   | Indicator for HR-positive/HER2-negative status (binary indicator).                                       |
| HER2pos        | Indicator for HER2-positive status (binary indicator).                                                 |
| TripleNeg      | Indicator for triple-negative status (binary indicator).                                               |
| test           | train/validation or test.                                   |



## ![MRI Scan](../images/spy2_data_organization.png)
MRI Breast image from Breast MRI - Mayo Clinic https://www.mayoclinic.org/tests-procedures/breast-mri/about/pac-20384809

### I-SPY-2 clinical metadata
![clinical_data_spy2](../images/spy2_drag_data.png)

Monticciolo, D. L., Newell, M. S., Moy, L., Niell, B., Monsees, B., & Sickles, E. A. (2018). Breast Cancer Screening in Women at Higher-Than-Average Risk: Recommendations From the ACR. Journal of the American College of Radiology, 15(3), 408–414. https://doi.org/10.1016/j.jacr.2017.11.034
QuantumLeap Healthcare Collaborative. (2024). I-SPY Trial (Investigation of Serial Studies to Predict Your Therapeutic Response With Imaging And moLecular Analysis 2) (Clinical Trial Registration No. NCT01042379). clinicaltrials.gov. https://clinicaltrials.gov/study/NCT01042379
