# Sketch2Image: Sketch-to-Photoreal Translation with Pix2Pix

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Model-pix2pix-EE4C2C?logo=pytorch&logoColor=white)](https://phillipi.github.io/pix2pix/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-ready-brightgreen)](.github/workflows/ci.yml)
[![Security](https://img.shields.io/badge/Security-Policy-blue)](SECURITY.md)

Sketch2Image is an end-to-end computer vision project that translates hand-drawn sketches into realistic images using a conditional GAN (Pix2Pix). It includes reproducible data preparation, quality evaluation (FID/LPIPS/SSIM), benchmark automation, and an interactive Streamlit demo.

## Why This Repo Is Flagship-Ready

- Complete workflow: data prep, training, inference, evaluation, and demo.
- Reproducibility-first scripts with deterministic split controls.
- Works in both full-dataset mode and single before/after image mode.
- CI checks, security policy, contribution guide, and citation metadata included.

## Architecture (Mermaid)

```mermaid
flowchart TB
    A[Raw Image Dataset<br/>dataset/class_x/*.jpg] --> B[Data Prep CLI<br/>script/process_dataset.py]
    B --> C[Sketch Dataset<br/>dataset_sketch]
    B --> D[Pix2Pix A/B Splits<br/>dataset_for_pix2pix/A,B]
    D --> E[AB Concatenated Dataset<br/>dataset_ab]
    E --> F[Pix2Pix Dataset Registry<br/>pytorch-CycleGAN-and-pix2pix/datasets/sketch2image]

    F --> G[Model Training<br/>sketch2image_model_train.ipynb]
    G --> H[Trained Checkpoints]
    H --> I[Inference Script<br/>pix2pix test pipeline]
    I --> J[Generated Images<br/>output_images]

    J --> K[Metrics Engine<br/>script/evaluate_metrics.py]
    F --> K
    K --> L[Metrics JSON<br/>metrics/latest_metrics.json]
    L --> M[Benchmark Updater<br/>script/update_benchmark_table.py]
    M --> N[README Benchmark Table]

    J --> O[Streamlit Demo<br/>demo/app.py]
    C --> O
```

## Quick Start (Fastest Path)

### 1) Install

```bash
git clone https://github.com/prashantsingh5/Sketch_2_image.git
cd Sketch_2_image
python -m venv .venv
```

Activate venv:

```powershell
. .\.venv\Scripts\Activate.ps1
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
python script/doctor.py
```

### 2) Run Single-Pair Evaluation (No Dataset Setup Needed)

Put your images at:

- `assets/sample/before.jpg`
- `assets/sample/after.jpg`

Then run:

```bash
python script/run_full_eval.py --real-image assets/sample/before.jpg --generated-image assets/sample/after.jpg --experiment "Single Pair Smoke Test" --split test
```

This computes `FID`, `LPIPS`, `SSIM` and auto-updates the benchmark table in this README.

### 3) Launch Live Demo

```bash
streamlit run demo/app.py
```

Demo modes:

- `Folders`: browse full before/after datasets.
- `Upload single pair`: drag and drop one before/after pair.

## Full Pipeline (Dataset Mode)

### 1) Input dataset format

```text
dataset/
|-- bedroom/
|   |-- image_001.jpg
|   `-- image_002.jpg
`-- kitchen/
    |-- image_101.jpg
    `-- image_102.jpg
```

### 2) Build Pix2Pix dataset artifacts

```bash
python script/process_dataset.py --base-folder . --dataset-dir dataset --pix2pix-repo pytorch-CycleGAN-and-pix2pix --dataset-name sketch2image
```

Optional split control:

```bash
python script/process_dataset.py --val-ratio 0.1 --test-ratio 0.05 --seed 42
```

### 3) Train and inference

- Run `sketch2image_model_train.ipynb` in order.
- Run inference from `pytorch-CycleGAN-and-pix2pix/` (for example via your test script).

### 4) Evaluate and benchmark

```bash
python script/run_full_eval.py --real-dir dataset_for_pix2pix/B/val --generated-dir pytorch-CycleGAN-and-pix2pix/output_images --experiment "Baseline Pix2Pix (default config)" --split val
```

Metric interpretation:

- `FID`: lower is better.
- `LPIPS`: lower is better.
- `SSIM`: higher is better.

## Benchmark Table

| Experiment | Dataset Split | FID ↓ | LPIPS ↓ | SSIM ↑ | Notes |
|---|---|---:|---:|---:|---|
| Baseline Pix2Pix (default config) | val | TBD | TBD | TBD | Run `script/run_full_eval.py` in dataset mode |
| Single Pair Smoke Test | test | TBD | TBD | TBD | Run `script/run_full_eval.py` in single-pair mode |
| Tuned Run (example) | val | TBD | TBD | TBD | Add hyperparameter notes |

## Example Result

### Input Sketch
![sketch_flickr_cat_000627](https://github.com/user-attachments/assets/c9e29508-bdd9-4869-88f2-c0dcf56bb53e)

### Generated Image
![flickr_cat_000627](https://github.com/user-attachments/assets/c5d22024-7ab8-432b-8c27-78adebcc292b)

## Repository Structure

```text
.
|-- README.md
|-- CONTRIBUTING.md
|-- PROJECT_ROADMAP.md
|-- SECURITY.md
|-- CITATION.cff
|-- requirements.txt
|-- sketch2image_model_train.ipynb
|-- assets/
|   `-- sample/
|       `-- README.md
|-- demo/
|   `-- app.py
|-- script/
|   |-- process_dataset.py
|   |-- evaluate_metrics.py
|   |-- update_benchmark_table.py
|   |-- run_full_eval.py
|   `-- doctor.py
`-- pytorch-CycleGAN-and-pix2pix/
```

## Project Hygiene

- Security policy: `SECURITY.md`
- Citation metadata: `CITATION.cff`
- Contribution guide: `CONTRIBUTING.md`
- Planned milestones: `PROJECT_ROADMAP.md`

## Acknowledgments

- Pix2Pix paper and ecosystem.
- PyTorch community.
- `pytorch-CycleGAN-and-pix2pix` maintainers.

## Author

Prashant Singh  
Contact: `prashantsingha96@gmail.com`
