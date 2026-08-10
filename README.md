# StegoCkavas — Image Steganography & Steganalysis Toolkit

An experimental Python toolkit for **generating steganographic images, preparing steganalysis datasets, extracting forensic image features, training cover-vs-stego classifiers, comparing embedding algorithms, and visualising model behaviour**.

The project supports a full experimental workflow rather than a single encoder/decoder: cover images can be transformed using multiple embedding methods and payload levels, organised into reproducible train/test datasets, analysed using handcrafted residual features, and classified with several machine-learning models.

## Research questions

The repository is structured around questions such as:

- How detectable are different image steganography methods?
- How does payload size in bits per pixel (BPP) affect detectability?
- Can residual statistics reveal embedding artifacts that are visually imperceptible?
- How do classical classifiers compare on the same steganalysis features?
- Can a classifier distinguish not only cover/stego images but also the embedding algorithm?
- Where does a CNN focus when it predicts that an image contains hidden information?

## End-to-end pipeline

```text
Cover images
     │
     ├──► optional normalization
     │
     ▼
Steganographic embedding
LSB / PVD / DCT / DWT
     │
     ▼
Binary + multiclass dataset generation
     │
     ▼
Feature extraction
raw / DCT / residual histogram / residual co-occurrence
     │
     ▼
Model training
LogReg / SVM / Random Forest / optional XGBoost / LightGBM
     │
     ▼
Evaluation + saved metrics/models
     │
     ├──► qualitative analysis
     ├──► payload sweeps
     └──► heatmaps / report figures
```

## Features

### Multiple steganography algorithms

The experiment pipeline can generate stego datasets using several embedding strategies, including:

- LSB
- PVD
- DCT
- DWT

Payload is configurable in **bits per pixel**, allowing experiments to measure the trade-off between hidden-data capacity and detectability.

### Reproducible dataset preparation

The pipeline can normalize source covers, generate stego variants, create binary and multiclass classification datasets, and split data reproducibly using a configurable random seed.

Binary experiments classify:

```text
cover vs stego
```

Multiclass experiments can distinguish classes such as:

```text
cover
stego_lsb
stego_pvd
stego_dct
...
```

### Handcrafted steganalysis features

`models.py` contains several feature extractors:

- `raw` — resized RGB pixels as a baseline;
- `dct` — block-wise DCT coefficient statistics;
- `residual_hist` — histograms of horizontal/vertical pixel residuals;
- `residual_cooc` — compact SPAM-like residual co-occurrence features.

The residual co-occurrence representation uses horizontal and vertical residual relationships to produce a compact forensic feature vector designed to emphasize subtle local pixel dependencies.

### Classical ML baselines

The training layer supports standard steganalysis baselines including:

- Logistic Regression
- SVM
- Random Forest
- XGBoost when installed
- LightGBM when installed

Optional scaling/PCA can be incorporated through scikit-learn pipelines.

### CNN analysis and heatmaps

The repository also contains CNN-oriented analysis utilities and heatmap generation scripts for investigating spatial evidence used by learned steganalysis models.

### Experiment automation

Payload-sweep and reporting utilities make it possible to compare algorithms and BPP settings without manually rebuilding every dataset.

For example, `run.py` is structured to iterate over payload levels, embedding algorithms and classifier families while keeping experiment outputs separated by run.

## Repository highlights

```text
stegoCkavas/
├── pipeline.py                 # Main CLI experiment pipeline
├── dataset_preparation.py      # Cover/stego generation and dataset splitting
├── models.py                   # Feature extraction + ML training/evaluation
├── backend_api.py              # Backend-facing project API
├── gui.py                      # Desktop GUI
├── run.py                      # Experiment matrix runner
├── run_bpp_sweep.py            # Payload/BPP experiments
├── get_cnn_heatmap.py          # CNN interpretation helper
├── heatmaps_cnn.py             # Heatmap analysis
├── qualitative_report.py       # Qualitative error/result analysis
├── make_report_figures.py      # Figure generation
├── requirements.txt
└── README.md
```

## Installation

Python 3 is required. A virtual environment is recommended.

```bash
python -m venv .venv
```

Activate it and install dependencies:

```bash
pip install -r requirements.txt
```

## Running experiments

The primary experimental entry point is `pipeline.py`, which exposes command-line arguments for dataset paths, algorithms, BPP, feature extraction, models, random seed, dataset splits and saved outputs.

The pipeline performs four major stages:

1. optional cover normalization;
2. stego-image generation;
3. binary/multiclass dataset preparation;
4. classifier training and evaluation.

The exact available flags can be inspected with:

```bash
python pipeline.py --help
```

If the repository is installed/arranged under the original `stego_ai` package layout, the module form used by the experiment scripts is:

```bash
python -m stego_ai.pipeline --help
```

## Experiment outputs

A work directory can contain generated stego images, prepared classification datasets and saved model artifacts. When model saving is enabled, the pipeline writes evaluation metadata and fitted estimators beneath `saved_models/`, including `metrics.json` and Joblib model bundles.

This separation makes it possible to retain the results of multiple payload/algorithm experiments for later comparison.

## Example experiment design

A useful steganalysis experiment is:

```text
Payloads:    0.1 / 0.4 / 0.8 BPP
Algorithms:  LSB / PVD / DCT
Features:    residual_hist or residual_cooc
Models:      Random Forest / XGBoost / SVM
```

For each configuration, evaluate binary cover-vs-stego performance and then inspect whether multiclass classification can identify the embedding method.

The important quantity is not merely raw accuracy: false-positive/false-negative behaviour and performance degradation at low payloads are especially informative in steganalysis.

## Why residual features?

Hidden data is designed to avoid obvious visual changes. Raw RGB pixels therefore contain enormous amounts of image-content information unrelated to steganography.

Residual-based features instead emphasize local differences between neighbouring pixels. Embedding operations can perturb the statistical relationships of these residuals even when the resulting image looks unchanged to a human observer.

That makes residual histograms and co-occurrence statistics useful interpretable baselines before moving to specialized deep steganalysis networks.

## Limitations

- The project is research/ coursework-oriented rather than a hardened forensic product.
- Classical features are intentionally compact and do not reproduce full high-dimensional Spatial Rich Models.
- Detection performance depends heavily on dataset source, preprocessing, embedding algorithm and payload.
- JPEG/spatial-domain assumptions differ between algorithms and should not be compared without controlling the image pipeline.
- Optional XGBoost/LightGBM functionality requires those packages to be installed.
- Some experiment scripts retain project-specific paths that should be configured before execution.
- Deep-learning utilities are experimental and separate from the main classical-ML pipeline.

## Possible extensions

Future work could include SRNet/Ye-Net baselines, JPEG-specific steganalysis features, cross-dataset evaluation, calibration curves, ROC/PR analysis, experiment configuration files, MLflow/W&B tracking, adversarial embedding experiments and a packaged CLI.

## Portfolio note

This project spans **image processing, information hiding, statistical feature engineering, machine learning, experimental design, explainability and model evaluation**. Its strongest aspect is the complete experiment pipeline: it can create the manipulated data being studied and then measure how detectable those manipulations are.
