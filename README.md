# 3D Kidney Segmentation Using Mask-Guided Self-Supervision

## Overview
This project implements a lightweight MASS-style prototype for 3D kidney segmentation using reference CT patches, reference kidney masks, and query CT patches.

It includes baseline and V2 models, training and evaluation scripts, visualizations, and a Streamlit web application.

This is not an official reproduction of MASS and is not a clinically validated medical device.

## Web App
The app accepts preprocessed NumPy patches of shape (64, 64, 64):
- Reference CT patch
- Reference kidney mask
- Query CT patch

It generates a predicted binary mask, displays a slice and overlay, and lets users download the prediction. An optional query ground-truth mask can be used to calculate patch-level Dice and IoU.

The app does not accept raw full CT scans.

## Installation
Install project dependencies from the project root using:
python -m pip install -r requirements.txt

## Run the App
From the project root in Windows PowerShell:

$env:PYTHONPATH = (Resolve-Path .\our_projectdir).Path
.\.venv\Scripts\python.exe -m streamlit run .\our_projectdir\app.py

Open http://localhost:8501 in your browser.

The V2 checkpoint is expected at:
data/real/best_mass_kits23_model_v2.pth

## Recorded V2 Results

## Final Evaluation Results

### Baseline vs V2

Both models were evaluated on the same 10 patches from held-out case 00002 using a prediction threshold of 0.50.

| Metric | Baseline | V2 |
|---|---:|---:|
| Mean Dice | 0.2758 | 0.8400 |
| Mean IoU | 0.1667 | 0.7294 |
| Mean Precision | 0.3365 | 0.7479 |
| Mean Recall | 0.2550 | 0.9668 |
| Mean Specificity | 0.9072 | 0.9548 |

These results show that V2 performed better than our initial baseline in this experiment. The architecture and training procedure changed together, so these results do not identify the contribution of each individual change.

### Validation-Based Threshold Selection

We evaluated prediction thresholds from 0.30 to 0.70 on the validation subset. A threshold of 0.60 achieved the highest mean validation Dice score.

| Threshold | Mean Dice | Mean IoU |
|---:|---:|---:|
| 0.30 | 0.7977 | 0.6860 |
| 0.40 | 0.8427 | 0.7478 |
| 0.50 | 0.8718 | 0.7887 |
| 0.60 | 0.8851 | 0.8064 |
| 0.70 | 0.8842 | 0.8015 |

### Held-Out Test Results at Threshold 0.60

After selecting the threshold using validation data, V2 was evaluated on the held-out case 00002 patches.

| Metric | V2 at threshold 0.60 |
|---|---:|
| Mean Dice | 0.8680 |
| Mean IoU | 0.7712 |
| Mean Precision | 0.8056 |
| Mean Recall | 0.9473 |
| Mean Specificity | 0.9686 |

Compared with threshold 0.50 on the same test patches, threshold 0.60 increased mean Dice from 0.8400 to 0.8680. This is a promising result for threshold selection, but it is not evidence that the model itself was retrained or improved.

### Evaluation Limitations

- The evaluation used 10 patches from one held-out patient.
- The training validation split was performed at pair level, not patient level.
- Patch-level results do not establish generalization to other patients.
- This project is a lightweight MASS-inspired prototype, not an exact reproduction of the official MASS method.
- No direct benchmark against the official MASS implementation has been performed.
- The results are preliminary and are not evidence of clinical effectiveness.


These are preliminary patch-level results from one held-out patient, not evidence of generalization across patients or clinical effectiveness. The training and validation split was pair-level.

## Limitations and Safety
- Research and educational use only; not for clinical decisions.
- This is a MASS-style prototype, not the official MASS/Iris model.
- The experiment uses a limited number of patches and cases.
- Broader patient-level evaluation is needed.

## Main Files
- our_projectdir/mass_model.py: baseline model
- our_projectdir/mass_model_v2.py: V2 model
- our_projectdir/train_real_mass.py: baseline training
- our_projectdir/train_real_mass_v2.py: V2 training
- our_projectdir/evaluate_case02.py: baseline evaluation
- our_projectdir/evaluate_case02_v2.py: V2 evaluation
- our_projectdir/make_project_figures_v2.py: visualizations
- our_projectdir/app.py: Streamlit application
- requirements.txt: dependencies
- .gitignore: excluded files
