"""Compatibility re-export.

training.py imports from training_models.model, but the actual implementation
was renamed to model_seg.py. This module re-exports the classes so both
import paths work.

Deviation from book: The book has a single model.py per chapter containing all
model classes. In this repository, the segmentation model lives in model_seg.py
and the classification model in model_conv.py. This re-export preserves the
original import path that training.py expects.
"""
from training_models.model_seg import UNetWrapper, SegmentationAugmentation
