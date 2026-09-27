# Dataset source and provenance

- File: `heart.csv`, included unmodified from the downloaded archive.
- Source named in section 4 of the selected paper: John Smith (johnsmith88),
  Heart Disease Dataset, Kaggle.
- Page: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- Download: https://www.kaggle.com/api/v1/datasets/download/johnsmith88/heart-disease-dataset
- Retrieved: 26 September 2026.
- Kaggle API metadata: version 2, updated 6 June 2019; licence `Unknown`.
- File SHA-256: `ddb2996b2f4db2e00aad13f4518200179ff69f79093838e3c21ffa672ebec0f1`
- Shape: 1,025 rows, 13 predictors and 1 binary target.
- No synthetic records were created for this study. The source already contains
  723 extra exact duplicate rows. The raw file was not deduplicated or edited.

The original Heart Disease resource is attributed to A. Janosi, W. Steinbrunn,
M. Pfisterer and R. Detrano, UCI Machine Learning Repository (1988),
DOI https://doi.org/10.24432/C52P4X. UCI describes its resource as CC BY 4.0.
The Kaggle metadata does not state the same licence, so this package does not
claim that a licence for the original resource resolves the Kaggle copy's status.

The paper describes 0 as no disease and 1 as disease. Experiments preserve the
CSV's target values and evaluate class 1 as positive for comparability. Clinical
label meanings and exact transformations from original UCI records have not
been independently verified. UCI uses different category codes in some fields;
do not silently apply its codebook to this transformed Kaggle CSV. `ca=4` and
`thal=0` occur even though they are outside ranges shown in paper Table 9.
They are retained as observed values. There are no NaN values to impute.

There are no patient IDs, source-hospital IDs or future-event dates in this CSV.
Distinct rows must not be described as verified distinct patients, and the
recorded target should not be represented as a confirmed future heart attack.
