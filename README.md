# EviFus-CRC

We developed a framework synthesizing heterogeneous data streams from computed tomography and whole slide images using Dempster Shafer theory for robust uncertainty quantification.

## Files

- `config.py`: configuration and command-line arguments
- `dataset.py`: CSV-driven feature loading and missing-modality handling
- `model.py`: EviFus-CRC model
- `train.py`: training
- `evaluate.py`: inference and evaluation
- `extract_ctfm_features.py`: CT feature extraction
- `extract_plip_wsi_features.py`: pathology feature extraction
- `docs/Code_Documentation.md`: code documentation required by the lab
- `docs/GitHub_Upload_Checklist.md`: upload checklist

## Minimal CSV format

```csv
patient_name,path_ct,path_bingli,duration,event,split
```

`path_ct` and `path_bingli` are feature paths (`.npy`). Empty cells indicate missing modalities.

## Train

```bash
python train.py --split-csv ./example_split.csv
```

## Evaluate

```bash
python evaluate.py --split-csv ./example_split.csv --checkpoint-path ./outputs/your_run/best_model.pt
```

## Feature extraction

CT:
```bash
python extract_ctfm_features.py --input-path ct_manifest.csv --model-dir ./ct_fm --output-dir ./features/ct
```

WSI:
```bash
python extract_plip_wsi_features.py --input-path wsi_manifest.csv --output-dir ./features/pathology
```

## Note

The repository is organized in two independent stages:
1. feature extraction using CT-FM / PLIP;
2. feature-level fusion training using the extracted `.npy` features.

Relative paths are recommended for GitHub release.
