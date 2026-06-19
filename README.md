# Fast Detection of Cataract Based on Fundus Image

This repository contains MATLAB code accompanying the conference paper:

**Fast Detection of Cataract Based on Fundus Image**  
Alicja A. Ignatowicz, Tomasz Marciniak

The repository is intended to support reproducibility of the experiments presented in the article. The paper is prepared for publication/presentation at **MMAR 2026 – 30th International Conference on Methods and Models in Automation and Robotics**, held on **18–21 August 2026** in Międzyzdroje, Poland. Full bibliographic details will be added after the official conference publication.

## Project overview

The aim of this project is to evaluate fast and interpretable machine learning methods for binary cataract detection from fundus images. The proposed pipeline focuses on handcrafted image descriptors, especially intensity histograms and Local Binary Patterns (LBP), and compares them with transfer-learning-based convolutional neural networks.

The main workflow includes:

- fundus region-of-interest (ROI) extraction,
- RGB and grayscale histogram analysis,
- Local Binary Pattern feature extraction,
- feature fusion using LBP and histogram descriptors,
- binary classification of fundus images into `cataract` and `normal`,
- comparison of classical machine learning models with deep learning models,
- generation of plots, metrics, prediction files, and timing summaries.

## Repository structure

The repository is organized as a set of MATLAB scripts. The main entry point is:

```matlab
main
```

Typical files included in this repository:

```text
main.m
config_default.m
run_lbp_extraction.m
run_intensity_histogram_plots.m
run_lbp_global_plots.m
run_binary_classification.m
run_example.m
MATLAB_TOOLBOXES.md
README.md
.gitignore
```

## Methods

The implemented pipeline follows two complementary classification paths.

The first path uses classical machine learning. Fundus images are preprocessed by extracting the retinal ROI. Then, histogram-based features and LBP texture descriptors are extracted and used independently or jointly. The resulting feature vectors can be classified using models such as Support Vector Machine with an RBF kernel and Random Forest.

The second path uses deep learning models based on transfer learning. The extracted fundus ROI is resized to `224 × 224` pixels and processed by CNN architectures such as ResNet50 and MobileNetV2. These models are included mainly as reference baselines for comparison with the lightweight feature-based approach.

## Datasets

The experiments described in the paper use publicly available fundus image datasets, including:

- ODIR-5K,
- Eye Diseases Classification (EDC).

The datasets are not included in this repository due to size and licensing limitations. They should be downloaded separately from their original sources and placed in the paths defined in `config_default.m`.

Before running the code, update the configuration file according to your local dataset structure.

## Requirements

The code was prepared for MATLAB and uses functions from standard image processing and machine learning workflows. Depending on the selected experiment variant, the following MATLAB toolboxes may be required:

- Image Processing Toolbox,
- Statistics and Machine Learning Toolbox,
- Deep Learning Toolbox,
- Computer Vision Toolbox.

A detailed list of recommended toolboxes is provided in `MATLAB_TOOLBOXES.md`.

## How to run

1. Clone the repository:

```bash
git clone https://github.com/ignatowicz-alicja/Fast-detection-of-cataract-based-on-fundus-image.git
```

2. Open the project folder in MATLAB.

3. Edit the configuration file:

```matlab
config_default.m
```

Set the paths to your local datasets, feature files, and output directory.

4. Run the main script:

```matlab
main
```

The results will be saved in the output directory specified in the configuration file.

## Outputs

Depending on the selected options, the scripts may generate:

- extracted LBP feature files,
- histogram feature files,
- classification metrics,
- prediction tables,
- timing summaries,
- intensity histogram plots,
- LBP histogram plots,
- summary figures for publication-oriented analysis.

## Publication status

This GitHub repository is associated with the article **“Fast Detection of Cataract Based on Fundus Image”**. The article is intended for publication/presentation at **MMAR 2026 – 30th International Conference on Methods and Models in Automation and Robotics**, scheduled for **18–21 August 2026**.

The repository will be updated after the official publication with the final citation details.

## Citation

If you use this code, please cite the related paper. The final citation will be added after publication.

Temporary citation format:

```bibtex
@inproceedings{ignatowicz2026fast,
  title        = {Fast Detection of Cataract Based on Fundus Image},
  author       = {Ignatowicz, Alicja A. and Marciniak, Tomasz},
  booktitle    = {Proceedings of the 30th International Conference on Methods and Models in Automation and Robotics (MMAR)},
  year         = {2026},
  note         = {To appear}
}
```

## Disclaimer

This repository is intended for research and reproducibility purposes only. The code and models are not intended for clinical diagnosis or direct medical decision-making without further validation by qualified medical specialists.
