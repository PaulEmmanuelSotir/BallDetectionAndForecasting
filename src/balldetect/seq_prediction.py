#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball detector - Deeplearning Exercice 1 - Part 1
.. moduleauthor:: Paul-Emmanuel Sotir
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting
"""
import os
from typing import Tuple, Optional
from pathlib import Path
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import balldetect.torch_utils as tu
import balldetect.datasets as datasets

__all__ = ['SeqPredictor', 'train']
__author__ = 'Paul-Emmanuel SOTIR <paul-emmanuel@outlook.com>'

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class SeqPredictor(nn.Module):
    """ Ball position sequence forecasting pytorch module (fully connected neural net).
    .. class:: SeqPredictor
    """

    __constants__ = ['_input_shape', '_n_classes']

    def __init__(self, input_shape: torch.Size, n_classes: int, fc_params: list, act_fn: type = nn.ReLU, dropout_prob: float = 0., batch_norm: Optional[dict] = None):
        super(SeqPredictor, self).__init__()
        self._input_shape = input_shape
        self._n_classes = n_classes
        fc_params = list(fc_params)

        # Define fullyconnected neural network architecture
        layers = []
        for prev_params, (i, params) in zip([None] + fc_params[:-1], enumerate(fc_params)):
            params['in_features'] = prev_params['out_features'] if prev_params is not None else self._input_shape
            layers.append((f'fc_layer_{i}', tu.fc_layer(params, act_fn, dropout_prob, batch_norm)))
        # Append last fully connected inference layer (no dropout nor batch normalization for this layer)
        layers.append(('fc_layer_log_softmax', tu.fc_layer({'in_features': fc_params[-1]['out_features'],
                                                            'out_features': self._n_classes})))
        self._net = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._net(x)  # Apply fully connected head neural net


def train(batch_size: int, architecture: dict, optimizer_params: dict, scheduler_params: dict, epochs: int, early_stopping: Optional[int] = None, pbar: bool = True) -> Tuple[float, float, int]:
    """ Initializes and train seq forecasting model """
    # TODO: refactor this to avoid some duplicated code with ball_detector.init_training()
    # Create balls dataset
    dataset = datasets.BallsCFSeq(tu.source_dir() / r'../../datasets/mini_balls_seq')

    # Create ball detector model and dataloaders
    trainset, validset = datasets.create_dataloaders(dataset, batch_size)
    input_bb_sequence, colors, target_bb = dataset[0]  # Nescessary to retreive input image resolution (assumes all dataset images are of the same size)
    model = SeqPredictor(np.prod(input_bb_sequence.shape) + np.prod(colors.shape), np.prod(target_bb.shape), **architecture)
    if batch_size > 64:
        model = tu.parrallelize(model)
    model.train(True).to(DEVICE)

    # Define optimizer, loss and LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    mse = torch.nn.MSELoss()
    scheduler_params['step_size'] *= len(trainset)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(trainset), epochs=hp['epochs'], **scheduler_params)
    
    # Weight xavier initialization
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            # TODO: adapt this line according to act fn in hyperprameteres (like in BallDetector model)
            nn.init.xavier_normal_(module.weight.data, gain=nn.init.calculate_gain(tu.get_gain_name(nn.Tanh)))
    model.apply(_initialize_weights)

    best_valid_mse, best_train_mse = float("inf"), float("inf")
    best_run_epoch = -1
    epochs_since_best_loss = 0

    # Main training loop
    for epoch in range(1, epochs + 1):
        print("\nEpoch %03d/%03d\n" % (epoch, epochs) + '-' * 15)
        train_mse = 0

        trange, update_bar = tu.progess_bar(trainset, '> Training on trainset', trainset.batch_size, custom_vars=True, disable=not pbar)
        for i, (input_bb_sequence, colors, target_bb) in enumerate(trange):
            batch_x = torch.cat((tu.flatten_batch(input_bb_sequence.to(DEVICE)).T, colors.to(DEVICE).T)).T.requires_grad_(True)
            target_bb = tu.flatten_batch(target_bb.to(DEVICE))

            def closure():
                optimizer.zero_grad()
                output = model(batch_x)
                loss = mse(output, target_bb)
                loss.backward()
                return loss
            loss = float(optimizer.step(closure).clone().detach())
            scheduler.step()
            train_mse += loss / len(trainset)
            update_bar(trainMSE=f'{len(trainset) * train_mse / (trange.n + 1):.7f}', lr=f'{float(scheduler.get_lr()[0]):.3E}')

        print(f'\tDone: TRAIN_MSE = {train_mse:.7f}')
        valid_loss = evaluate(model, validset, pbar=pbar)
        print(f'\tDone: TEST_MSE = {valid_loss:.7f}')

        if best_valid_mse > valid_loss:
            print('>\tBest valid_loss found so far, saving model...')  # TODO: save model
            best_valid_mse, best_train_mse = valid_loss, train_mse
            best_run_epoch = epoch
            epochs_since_best_loss = 0
        else:
            epochs_since_best_loss += 1
            if early_stopping is not None and early_stopping > 0 and epochs_since_best_loss >= early_stopping:
                print(f'>\tModel not improving: Ran {epochs_since_best_loss} training epochs without improvement. Early stopping training loop...')
                break

    print(f'>\tBest training results obtained at {best_run_epoch}nth epoch (best_valid_mse={best_valid_mse:.7f}, best_train_mse={best_train_mse:.7f}).')
    return best_train_mse, best_valid_mse, best_run_epoch


def evaluate(model, validset, pbar: bool = True) -> float:
    model.eval()
    with torch.no_grad():
        metric = torch.nn.MSELoss()
        valid_mse = 0.

        for (input_bb_sequence, colors, target_bb) in tu.progess_bar(validset, '> Evaluation on validset', validset.batch_size, disable=not pbar):
            batch_x = torch.cat((tu.flatten_batch(input_bb_sequence.to(DEVICE)).T, colors.to(DEVICE).T)).T.requires_grad_(True)
            target_bb = tu.flatten_batch(target_bb.to(DEVICE))
            output = model(batch_x)
            valid_mse += metric(output, target_bb) / len(validset)
    return float(valid_mse)
