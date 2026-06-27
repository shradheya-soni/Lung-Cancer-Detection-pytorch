# Lung Cancer Detection using PyTorch

A deep learning pipeline for automated lung nodule detection and classification on the LUNA16 dataset, implemented in **PyTorch**.
---

## Features

* 3D CT scan preprocessing using **SimpleITK**
* Lung nodule candidate extraction
* U-Net based lung nodule segmentation
* CNN-based nodule classification
* Automatic checkpoint management
* Optional fallback training when pretrained checkpoints are unavailable
* End-to-end inference pipeline
* LUNA16 evaluation support

---

## Project Structure

```text
dataset_processing/
│
├── dsets.py
└── vis.py

training_models/
│
├── model_seg.py
├── model_conv.py
├── training.py
├── classification_training.py
├── checkpoint_manager.py
├── nodule_analysis.py
└── ...

util/

evaluationScript/

models/
```

---

## Dataset

This project uses the **LUNA16** dataset.

Download the dataset from the official LUNA16 website and place it in the appropriate directories before training or inference.

The dataset is **not included** in this repository.

---

## Training

The project supports two workflows.

### Using pretrained checkpoints

If pretrained checkpoints are present inside the `models/` directory, they are automatically loaded during inference.

### Training from scratch

If checkpoints are unavailable, the repository supports training the segmentation and classification models before running inference.

Example:

```bash
python training_models/nodule_analysis.py --train-if-missing --epochs 15
```

---

## Running Inference

Run:

```bash
python training_models/nodule_analysis.py
```

The application automatically detects whether pretrained checkpoints are available and loads them before performing inference.

---

## Technologies Used

* Python
* PyTorch
* NumPy
* SimpleITK
* TensorBoard
* Pandas
* SciPy

---

## Repository Notes

This repository contains only the project source code.

Large datasets, pretrained model checkpoints, TensorBoard logs, and cache files are intentionally excluded from version control.

---

## Future Improvements

* Performance benchmarking
* TensorBoard visualizations
* Quantitative evaluation metrics
* Docker support
* FastAPI inference service
* Model deployment

---

## Acknowledgements

This project is inspired by the book **Deep Learning with PyTorch** by Eli Stevens, Luca Antiga, and Thomas Viehmann.

The implementation builds upon concepts presented in the book while adapting them into a modular project structure with additional compatibility improvements and reproducible training workflows.

The evaluation utilities included in this repository originate from the official **LUNA16** evaluation framework.
