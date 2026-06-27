import argparse
import datetime
import os
import sys
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from util.util import enumerateWithEstimate
from dataset_processing.dsets import MalignantLunaDataset
from training_models.model_conv import LunaModel
from util.logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', help='Batch size to use for training', default=32, type=int)
        parser.add_argument('--num-workers', help='Number of worker processes', default=4, type=int)
        parser.add_argument('--epochs', help='Number of epochs to train for', default=15, type=int)
        parser.add_argument('--save-path', help='Explicit path to save the best model checkpoint', type=str, default=None)
        
        self.cli_args, _ = parser.parse_known_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.initModel()
        self.optimizer = self.initOptimizer()

    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)

    def initTrainDl(self):
        # We use MalignantLunaDataset as in the book, providing candidate tuples
        ds = MalignantLunaDataset(val_stride=10, isValSet_bool=False, ratio_int=1)
        return DataLoader(ds, batch_size=self.cli_args.batch_size, num_workers=self.cli_args.num_workers, pin_memory=self.use_cuda)

    def initValDl(self):
        ds = MalignantLunaDataset(val_stride=10, isValSet_bool=True)
        return DataLoader(ds, batch_size=self.cli_args.batch_size, num_workers=self.cli_args.num_workers, pin_memory=self.use_cuda)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))
        
        train_dl = self.initTrainDl()
        val_dl = self.initValDl()
        
        best_score = 0.0
        
        for epoch_ndx in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}".format(epoch_ndx, self.cli_args.epochs))
            
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)
            
            valMetrics_t = self.doValidation(epoch_ndx, val_dl)
            score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
            
            isBest = score > best_score
            best_score = max(score, best_score)
            
            self.saveModel('cls', epoch_ndx, isBest)

    def doTraining(self, epoch_ndx, train_dl):
        self.model.train()
        trnMetrics_g = torch.zeros(3, len(train_dl.dataset), device=self.device)
        
        batch_iter = enumerateWithEstimate(train_dl, "E{} Training".format(epoch_ndx), start_ndx=train_dl.num_workers)
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()
            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)
            loss_var.backward()
            self.optimizer.step()
            
        return trnMetrics_g.to('cpu')

    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(3, len(val_dl.dataset), device=self.device)
            
            batch_iter = enumerateWithEstimate(val_dl, "E{} Validation".format(epoch_ndx), start_ndx=val_dl.num_workers)
            for batch_ndx, batch_tup in batch_iter:
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)
                
        return valMetrics_g.to('cpu')

    def computeBatchLoss(self, batch_ndx, batch_tup, batch_size, metrics_g):
        input_t, label_t, _series_list, _center_list = batch_tup
        
        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
        
        logits_g, probability_g = self.model(input_g)
        
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss_g = loss_func(logits_g, label_g[:, 1]) # label_g is usually one-hot or index depending on dataset
        
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + input_t.size(0)
        
        metrics_g[0, start_ndx:end_ndx] = loss_g
        metrics_g[1, start_ndx:end_ndx] = (probability_g[:,1] > 0.5).to(torch.float32) # pred
        metrics_g[2, start_ndx:end_ndx] = label_g[:, 1] # truth
        
        return loss_g.mean()

    def logMetrics(self, epoch_ndx, mode_str, metrics_t):
        metrics_a = metrics_t.detach().numpy()
        
        neg_mask = metrics_a[2] == 0
        pos_mask = metrics_a[2] == 1
        
        tp = (metrics_a[1][pos_mask] == 1).sum()
        fn = (metrics_a[1][pos_mask] == 0).sum()
        fp = (metrics_a[1][neg_mask] == 1).sum()
        tn = (metrics_a[1][neg_mask] == 0).sum()
        
        precision = tp / ((tp + fp) or 1)
        recall = tp / ((tp + fn) or 1)
        f1 = 2 * (precision * recall) / ((precision + recall) or 1)
        
        loss = metrics_a[0].mean()
        
        log.info(f"E{epoch_ndx} {mode_str} loss: {loss:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1: {f1:.4f}")
        return f1

    def saveModel(self, type_str, epoch_ndx, isBest=False):
        if not isBest:
            return
            
        file_path = self.cli_args.save_path
        if not file_path:
            file_path = os.path.join(
                'data-unversioned', 'models',
                f'{type_str}_{self.time_str}.best.state'
            )
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state': self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
        }
        torch.save(state, file_path)
        log.info(f"Saved best model params to {file_path}")

if __name__ == '__main__':
    LunaTrainingApp().main()
