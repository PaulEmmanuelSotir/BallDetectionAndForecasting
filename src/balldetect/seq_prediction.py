#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball detector - Deeplearning Exercice 1 - Part 1
.. moduleauthor:: Paul-Emmanuel Sotir
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting
"""
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
INFERENCE_BATCH_SIZE = 16*1024  # Batch size used during inference (including validset evaluation)


class SeqPredictor(nn.Module):
    """ Ball position sequence forecasting pytorch module (fully connected neural net).
    .. class:: SeqPredictor
    """

    __constants__ = ['_input_shape', '_n_classes']

    def __init__(self, input_shape: torch.Size, n_classes: int, fc_params: list, act_fn: type = nn.ReLU, dropout_prob: float = 0., batch_norm: dict = {}):
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


def init_training(batch_size: int, architecture: dict, optimizer_params: dict, scheduler_params: dict) -> Tuple[DataLoader, DataLoader, nn.Module, Optimizer, _LRScheduler]:
    """ Initializes dataset, dataloaders, model, optimizer and lr_scheduler for future training """
    # TODO: refactor this to avoid some duplicated code with ball_detector.init_training()
    # Create balls dataset
    dataset = datasets.BallsCFSeq(Path("../datasets/mini_balls_seq"))

    # Create ball detector model and dataloaders
    trainset, validset = datasets.create_dataloaders(dataset, batch_size, INFERENCE_BATCH_SIZE)
    p, bb = dataset[0]  # Nescessary to retreive input image resolution (assumes all dataset images are of the same size)
    model = SeqPredictor(p.shape, np.prod(bb.shape), **architecture)
    if batch_size > 64:
        model = tu.parrallelize(model)

    # Define optimizer and LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(trainset), epochs=hp['epochs'], **scheduler_params)
    scheduler_params['step_size'] *= len(trainset)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    return trainset, validset, model, optimizer, scheduler


def train(trainset: DataLoader, validset: DataLoader, model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, epochs: int, early_stopping: Optional[int] = None, pbar: bool = True) -> Tuple[float, float, int]:
    """ Trains model on given dataset """
    # TODO: refactor this to avoid some duplicated code with ball_detector.train()
    model.train(True).to(DEVICE)
    mse = torch.nn.MSELoss()

    best_valid_mse, best_train_mse = float("inf"), float("inf")
    best_run_epoch = -1
    epochs_since_best_loss = 0

    # Weight xavier initialization
    def _initialize_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight.data, gain=1.)
    model.apply(_initialize_weights)

    # Main training loop
    for epoch in range(1, epochs + 1):
        print("\nEpoch %03d/%03d\n" % (epoch, epochs) + '-' * 15)
        train_mse = 0

        trange, update_bar = tu.progess_bar(trainset, '> Training on trainset', trainset.batch_size, custom_vars=True, disable=not pbar)
        for (batch_x, batch_y) in trange:
            batch_x, batch_y = tu.flatten(batch_x.to(DEVICE)), tu.flatten(batch_y.to(DEVICE).requires_grad_(True))

            def closure():
                optimizer.zero_grad()
                output = model(batch_x)
                loss = mse(output, batch_y)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            scheduler.step()
            train_mse += loss / len(trainset)
            update_bar(trainMSE=f'{len(trainset) * train_mse / trange.n:.7f}', lr=f'{scheduler.get_lr()[0]:.3E}')

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

        for (batch_x, _ps, batch_bbs) in tu.progess_bar(validset, '> Evaluation on validset', validset.batch_size, disable=not pbar):
            batch_x, batch_bbs = batch_x.to(DEVICE), tu.flatten(batch_bbs.to(DEVICE))
            output = model(batch_x)
            valid_mse += metric(output, batch_bbs) / len(validset)
    return float(valid_mse)
