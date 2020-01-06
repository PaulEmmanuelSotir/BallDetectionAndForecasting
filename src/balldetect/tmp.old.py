#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball detector - Deeplearning Session 1 Exercice 1
.. moduleauthor:: Paul-Emmanuel Sotir
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting
"""
import os
import types
import shutil
from typing import OrderedDict
import numpy as np
import pandas as pd
import torch
from torch.nn.modules.conv import Conv2d
import torchvision
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import balldetect.datasets as datasets

__all__ = ['ConvolutionBackbone', 'DenseHead', 'BallDetector', 'train']

INFERENCE_BATCH_SIZE = 8*1024
TEST_SET_SIZE = 0.015  # %
CPU_COUNT = os.cpu_count()

# Torch configuration
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
cudnn.benchmark = torch.cuda.is_available()  # Enable inbuilt CuDNN auto-tuner TODO: measure performances without this flag
cudnn.fastest = torch.cuda.is_available()  # Disable this if memory issues


class _Layer(nn.Module):
    def __init__(self, conv2d: dict, act_fn: types.FunctionType, dropout: nn.Module) -> None:
        super(_Layer, self).__init__()
        self._dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._dropout is not None:
            return self._dropout(act_fn(layer(x)))
        else:
            return self.act_fn(layer(x))
        # TODO: Add batch normalization


class ConvolutionBackbone(nn.Module):
    """ Convolution backbone block module.
    .. class:: ConvolutionBackbone
    """

    __constants__ = ['_hp', '_input_shape', '_n_classes']

    def __init__(self, input_shape: torch.Size, conv2d_params: list, act_fn: types.FunctionType, batch_norm_params: dict, dropout_prob: float = 0.) -> None:
        super(ConvolutionBackbone, self).__init__()
        d = nn.Dropout(p=dropout_prob) if dropout_prob != 0 else None
        self._layers = nn.Sequential(OrderedDict([('conv_layer_' + i, _Layer(dropout=d, act_fn=act_fn, **params)) for i, params in enumerate(conv2d_params)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)


class BallDetector(nn.Module):
    """ Ball detector pytorch module.
    .. class:: BallDetector
    """

    __constants__ = ['_hp', '_input_shape', '_n_classes']

    def __init__(self, input_shape: torch.Size, n_classes: int, hyperparameters: dict):
        super(BallDetector, self).__init__()
        self._hp = hyperparameters
        self._input_shape = input_shape
        self._n_classes = n_classes

        # Define NN layers
        self._dropout = nn.Dropout(p=self._hp['dropout'])
        (first_layer_type, first_layer_params) = self._hp['layers'][0]
        self._conv_layers = nn.ModuleList([first_layer_type(self._input_shape[0], **first_layer_params)]
                                          + [nn.Conv2d(prev_params['out_channels'], **params)
                                             for (prev_params, params) in zip(self._hp['layers'][:-1], self._hp['layers'][1:])])
# self._conv_layers[-1].shape
        test = nn.Conv2d(1, 4)
        test.in_channels
        self._dense_layers = nn.ModuleList([nn.Linear()]
                                           + [nn.Linear(prev_params['out_features'], **params)
                                              for (prev_params, params) in zip(self._hp['layers'][:-1], self._hp['layers'][1:])])
        self._output_logits = nn.Linear(self._hp['layers'][-1]['out_features'], self._n_classes)

        # Define batch normalization layers if needed
        if 'batch_norm' in self._hp and self._hp['batch_norm'] is not None:
            self._batch_norms = [torch.nn.BatchNorm2d(params['out_channels'], **self._hp['batch_norm']) for params in self._hp['conv_layers']]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act_fn = F.relu
        for layer in self._conv_layers:
            x = self._dropout(act_fn(layer(x)))
        x = x.flatten(-1, -3)
        for layer in self._dense_layers:
            x = self._dropout(act_fn(layer(x)))
        # TODO: Add batch normalization
        return F.log_softmax(self._output_logits(x))


# TODO: remove it
# def init(model_path, metadata):
""" Build and initialize model for prediction or training
Args:
    model_path: Local path to model file or directory if specified by user in API configuration, otherwise None.
    metadata: Custom dictionary specified by the user in API configuration ().
"""
#    pass

# def predict(sample, metadata):
""" Predicts balls position from given images
Args:
    sample: The JSON request payload (parsed as a Python object).
    metadata: Custom dictionary specified by the user in API configuration.
Returns:
    A prediction
"""
#   pass


def train(dataset: torch.utils.data.Dataset, hp: dict):
    """ Trains model on given dataset """
    indices = torch.randperm(len(dataset)).tolist()
    test_size = int(TEST_SET_SIZE * len(dataset))
    train_ds, test_ds = torch.utils.data.Subset(dataset, indices[:-test_size]), torch.utils.data.Subset(dataset, indices[-test_size:])

    num_workers = 0  # max(1, CPU_COUNT // 4) TODO:...
    trainset = torch.utils.data.DataLoader(train_ds, batch_size=hp['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True)
    testset = torch.utils.data.DataLoader(test_ds, batch_size=INFERENCE_BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    dummy_img, _p, _bb = dataset[0]  # Nescessary to retreive input image resolution (assumes all dataset images are of the same size)
    det = BallDetector(dummy_img.shape, len(datasets.COLORS), hp).train(True).to(device)

    cross_entropy = torch.nn.NLLLoss(weight=hp['class_weights'])
    optimizer = optim.SGD(det.parameters(), **hp['SGD'])

    # Weight xavier initialization
    def _initialize_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight.data, gain=1.)
    det.apply(_initialize_weights)

    # Main training loop
    for epoch in range(hp['epochs']):
        print("Epoch %03d/%03d\n" % (epoch, hp['epochs'] - 1) + '-' * 15)

        # Preforms one epoch of training on trainset
        for step, (img, p, bb) in enumerate(trainset):
            batch_x, batch_y = img.to(device), p.to(device).requires_grad_(True)

            def closure():
                optimizer.zero_grad()
                output = det(batch_x)
                loss = cross_entropy(output, batch_y)
                loss.backward()
                return loss
            optimizer.step(closure)

        # Evaluate neural network on testset
        # TODO: ...

    return det


def load(model_dir):
    # det = BallDetector(len(dataset[0][0]), len(datasets.COLORS), hp).to(device).train(False)
    # TODO:
    pass
