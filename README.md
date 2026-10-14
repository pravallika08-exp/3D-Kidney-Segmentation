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
Held-out case 00002, 10 patches:
- Mean Dice: 0.8400
- Mean IoU: 0.7294
- Mean precision: 0.7479
- Mean recall: 0.9668
- Mean specificity: 0.9548
- Best recorded validation Dice: approximately 0.8718

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
