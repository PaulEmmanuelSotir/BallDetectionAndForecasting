#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball detector - Deeplearning Exercice 1 - Part 1
.. moduleauthor:: Paul-Emmanuel Sotir
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import balldetect.torch_utils as tu

__all__ = ['SeqPredictor', 'train']
__author__ = 'Paul-Emmanuel SOTIR <paul-emmanuel@outlook.com>'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class SeqPredictor(nn.Module):
    """ Ball position sequence predictor pytorch module (fully connected neural net).
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


def train(trainset: DataLoader, testset: DataLoader, model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, epochs: int, pbar: bool = True):
    """ Trains model on given dataset """
    model.train(True).to(device)
    mse = torch.nn.MSELoss()

    # Weight xavier initialization
    def _initialize_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight.data, gain=1.)
    model.apply(_initialize_weights)

    # Main training loop
    for epoch in range(epochs):
        print("\nEpoch %03d/%03d\n" % (epoch + 1, epochs) + '-' * 15)

        # Preforms one epoch of training on trainset
        train_mse = 0

        trange, update_bar = tu.progess_bar(trainset, '> Training on trainset', trainset.batch_size, custom_vars=True, disable=not pbar)
        for (batch_x, batch_y) in trange:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).requires_grad_(True)
            batch_x = batch_x.view(batch_x.size(0), -1)  # Flattens input positions and bounding boxes
            batch_y = batch_y.view(batch_y.size(0), -1)  # Flattens target bounding boxes

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
        print(f'\tDone: TEST_MSE = {evaluate(model, testset):.7f}')
    return model


def evaluate(model, testset, pbar: bool = True) -> float:
    model.eval()
    with torch.no_grad():
        metric = torch.nn.MSELoss()
        test_mse = 0.

        for (batch_x, _ps, batch_bbs) in tu.progess_bar(testset, '> Evaluation on testset', testset.batch_size, disable=not pbar):
            batch_x, batch_bbs = batch_x.to(device), batch_bbs.to(device).requires_grad_(True)
            batch_bbs = batch_bbs.view(batch_bbs.size(0), -1)  # Flattens target bounding boxes
            output = model(batch_x)
            test_mse += metric(output, batch_bbs) / len(testset)
    return float(test_mse)
