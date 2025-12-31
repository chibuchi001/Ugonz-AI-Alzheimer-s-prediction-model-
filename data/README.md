# Data Directory

This directory is for storing input data files.

## Quick Start (Synthetic Data)

Run the demo script to generate synthetic data:

```bash
python run_demo.py
```

This will create `data/clinical_imaging_genetic.csv` with 1000 synthetic patient records.

## Real Data Sources

For real-world validation, the pipeline supports:

### Clinical Data (CSV/TSV)
- Format: One row per patient
- Required columns: `sample_id`, `age`, `education_years`, `mmse_score`, `cdr_score`, `gender`
- Target column: `AD_status` (0=CN, 1=AD)

### Neuroimaging Data (NPZ)
- Format: NumPy compressed archive
- Contents: `mri_features` array (n_samples × n_features), `sample_ids` array

### Genetic Data (NPZ or PLINK BED)
- Format: NumPy archive or PLINK binary
- Contents: SNP genotypes (0, 1, 2 encoding), sample IDs, SNP IDs

## Recommended Datasets

- **ADNI** (Alzheimer's Disease Neuroimaging Initiative): https://adni.loni.usc.edu/
- **UK Biobank**: https://www.ukbiobank.ac.uk/

## Note

Large data files are excluded from version control via `.gitignore`.
