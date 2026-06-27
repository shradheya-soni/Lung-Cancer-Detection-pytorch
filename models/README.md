# Pretrained Model Checkpoints

This directory holds the pretrained model checkpoint files required by Chapter 14's
end-to-end nodule analysis pipeline (`training_models/nodule_analysis.py`).

## Required Files

| Filename | Description | Source |
|----------|-------------|--------|
| `seg.best.state` | Trained UNet segmentation model (Chapter 13) | Book download or `training_models/training.py` output |
| `cls.best.state` | Trained LunaModel classification model (Chapter 11-12) | Book download or classification training output |
| `mal.best.state` | *(Optional)* Trained malignancy classifier (Chapter 14) | Book download or malignancy training output |

## Downloading from the Book

The official pretrained checkpoints can be downloaded from the book's data repository:

```
https://github.com/deep-learning-with-pytorch/dlwpt-code
```

After downloading, rename the files and place them in this directory:

```
models/
├── seg.best.state
├── cls.best.state
└── mal.best.state  (optional)
```

## Using Your Own Trained Models

If you trained models using `training_models/training.py`, copy the `.best.state`
checkpoint file here and rename it to `seg.best.state`.

## Custom Paths

You can also specify checkpoint paths directly via command-line arguments:

```bash
python training_models/nodule_analysis.py \
    --segmentation-path path/to/seg.state \
    --classification-path path/to/cls.state \
    --malignancy-path path/to/mal.state \
    --run-validation
```
