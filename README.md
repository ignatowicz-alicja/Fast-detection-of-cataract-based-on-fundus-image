# Fundus Cataract LBP Pipeline

MATLAB pipeline for fundus-image analysis in binary cataract classification experiments. The repository contains code for:

- fundus ROI extraction,
- LBP feature extraction,
- intensity histogram analysis,
- classical classification using SVM/RF,
- optional transfer learning with ResNet-50 and MobileNetV2,
- metric export and publication-style plots.

The code was reorganized from research scripts into a GitHub-ready project structure. Private absolute paths were removed and moved to `config/config_default.m`.

## Repository structure

```text
.
├── main.m
├── config/
│   └── config_default.m
├── src/
│   └── pipelines/
│       ├── run_binary_classification.m
│       ├── run_lbp_extraction.m
│       ├── run_intensity_histogram_plots.m
│       └── run_lbp_global_plots.m
├── data/
│   └── README.md
├── features/
│   └── README.md
├── results/
├── docs/
│   └── MATLAB_TOOLBOXES.md
└── examples/
    └── run_example.m
```

## Setup

1. Open the project folder in MATLAB.
2. Edit `config/config_default.m`.
3. Put datasets under `data/` or set custom absolute paths in the config file.
4. Run:

```matlab
main
```

## Configuration

The main run switches are in `config/config_default.m`:

```matlab
cfg.run.extractLBP              = false;
cfg.run.plotIntensityHistograms = false;
cfg.run.plotGlobalLBP           = false;
cfg.run.binaryClassification    = true;
```

Enable only the stages you need.

## Expected data layout

For raw images:

```text
data/<dataset_name>/<class_name>/<image_file>
```

Example:

```text
data/ODIR-5K_NORM/cataract/*.jpg
data/ODIR-5K_NORM/normal/*.jpg
```

For precomputed LBP features:

```text
features/lbp/<class_name>/<lbp_config>/X_LBP_LOCAL.mat
```

Example:

```text
features/lbp/cataract/lbp_local_grid4x4_P8_R1_riu2/X_LBP_LOCAL.mat
features/lbp/normal/lbp_local_grid4x4_P8_R1_riu2/X_LBP_LOCAL.mat
```

For histogram features:

```text
features/hist_features.csv
```

The CSV should contain at least `Filename` and `Label` columns, plus numeric histogram feature columns.

## Notes

- Dataset files, generated `.mat` feature files, result folders, and model outputs are ignored by Git.
- If paths saved inside `.mat` or `.csv` files still point to an old disk location, use `cfg.binary.pathRemapFrom` and `cfg.binary.pathRemapTo`.
- The deep learning stage requires the relevant MATLAB support packages for pretrained networks.

## License

Add your preferred license before publishing the repository.
