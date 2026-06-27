import os
import sys
import argparse
from util.logconf import logging

log = logging.getLogger(__name__)

def ensure_checkpoints(cli_args):
    """
    Checkpoint management module for Workflow B (Fallback).
    Checks if required models exist. If not, trains them in-process and returns
    control to the inference pipeline.
    """
    missing_models = []
    if not os.path.exists(cli_args.segmentation_path):
        missing_models.append(('segmentation', cli_args.segmentation_path))
    if not os.path.exists(cli_args.classification_path):
        missing_models.append(('classification', cli_args.classification_path))
    
    if not missing_models:
        return # All required models exist

    if not getattr(cli_args, 'train_if_missing', False):
        log.error("Missing pretrained checkpoints!")
        for name, path in missing_models:
            log.error(f"  {name} checkpoint missing at: {path}")
        log.error("Please download the official checkpoints and place them in the models/ directory.")
        log.error("Alternatively, pass --train-if-missing to train them from scratch.")
        sys.exit(1)
    
    # Train missing models in-process
    for name, path in missing_models:
        log.info(f"Training {name} model from scratch to generate {path}...")
        
        epochs = getattr(cli_args, 'epochs', 15)
        # Create minimal args list
        train_args = ['--epochs', str(epochs), '--save-path', path]
        
        if name == 'segmentation':
            from training_models.training import SegmentationTrainingApp
            app = SegmentationTrainingApp(train_args)
            app.main()
            
        elif name == 'classification':
            from training_models.classification_training import LunaTrainingApp
            app = LunaTrainingApp(train_args)
            app.main()
            
    log.info("Checkpoint generation complete. Resuming inference pipeline.")
