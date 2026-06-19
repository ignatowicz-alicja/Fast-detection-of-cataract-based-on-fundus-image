# Fundus Cataract LBP Pipeline — flat file version

This is a flat GitHub-ready MATLAB version of the fundus cataract analysis pipeline.  
There are no source-code subfolders: place all `.m` files in one directory and run `main.m`.

## Files

| File | Purpose |
|---|---|
| `main.m` | Main entry point. Runs selected stages from the configuration. |
| `config_default.m` | One place for all paths, labels, run switches and model settings. Edit this first. |
| `run_lbp_extraction.m` | Extracts ROI-based global/local LBP features and saves `.mat` feature files. |
| `run_intensity_histogram_plots.m` | Creates RGB/gray intensity histogram plots inside the fundus ROI. |
| `run_lbp_global_plots.m` | Creates global LBP histogram plots from precomputed LBP matrices. |
| `run_binary_classification.m` | Runs binary cataract-vs-normal experiments using LBP, HIST, LBP+HIST and optional CNN transfer learning. |
| `run_example.m` | Minimal example showing how to enable selected stages. |
| `MATLAB_TOOLBOXES.md` | Lists recommended MATLAB toolboxes/support packages. |
| `.gitignore` | Prevents committing datasets, results, temporary files and large generated outputs. |

## How to run

1. Put all files in one directory, for example:

```text
fundus_cataract_lbp_pipeline/
main.m
config_default.m
run_lbp_extraction.m
run_intensity_histogram_plots.m
run_lbp_global_plots.m
run_binary_classification.m
run_example.m
MATLAB_TOOLBOXES.md
.gitignore
README.md
```

2. Open this directory in MATLAB.

3. Edit:

```matlab
config_default.m
```

4. Set paths to your local data/features, for example:

```matlab
cfg.dataRoot    = fullfile(cfg.projectRoot, "data");
cfg.featureRoot = fullfile(cfg.projectRoot, "features");
cfg.outputRoot  = fullfile(cfg.projectRoot, "results");
```

You may also use absolute paths if your datasets are stored elsewhere.

5. Choose which stages to run:

```matlab
cfg.run.extractLBP              = false;
cfg.run.plotIntensityHistograms = false;
cfg.run.plotGlobalLBP           = false;
cfg.run.binaryClassification    = true;
```

6. Run:

```matlab
main
```

## Expected data layout

The code files are flat, but your datasets and generated results can still be stored in folders.

Raw images:

```text
data/<dataset_name>/<class_name>/<image_file>
```

Example:

```text
data/ODIR-5K_NORM/cataract/*.jpg
data/ODIR-5K_NORM/normal/*.jpg
```

Precomputed LBP features:

```text
features/lbp/<class_name>/<lbp_config>/X_LBP_LOCAL.mat
```

Histogram CSV:

```text
features/hist_features.csv
```

The histogram CSV should contain at least:

```text
Filename, Label, numeric feature columns...
```

## Before uploading to GitHub

Keep code files in GitHub, but do not upload datasets, generated `.mat` files, large CSV files, trained networks or result folders unless the data license allows it.  
The included `.gitignore` already excludes common generated files and folders.

## Notes

Private absolute paths from the original scripts were removed. Configure local paths only in `config_default.m`.
