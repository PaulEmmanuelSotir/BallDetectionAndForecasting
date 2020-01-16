#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball detector - Deeplearning Session 1 Exercice 1
.. moduleauthor:: Paul-Emmanuel Sotir
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting
"""
import os
import argparse
import numpy as np
from typing import Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import balldetect.torch_utils as torch_utils
import balldetect.ball_detector as ball_detector
import balldetect.seq_prediction as seq_prediction

__author__ = 'Paul-Emmanuel SOTIR <paul-emmanuel@outlook.com>'

# Torch CuDNN configuration
torch.backends.cudnn.deterministic = True
cudnn.benchmark = torch.cuda.is_available()  # Enable builtin CuDNN auto-tuner, TODO: benchmarking isn't deterministic, disable this if this is an issue
cudnn.fastest = torch.cuda.is_available()  # Disable this if memory issues


# Previous hyperparameters obtained from first hyperparamter search
DETECTOR_HP_OLD = {
    'optimizer_params': {'amsgrad': False, 'betas': (0.9, 0.999), 'eps': 1e-08, 'lr': 5*5*9.9615e-05, 'weight_decay': 2.11924e-4},
    'scheduler_params': {'step_size': 20, 'gamma': 0.2},
    # 'scheduler_params': {'max_lr': 1e-2, 'pct_start': 0.3, 'anneal_strategy': 'cos'},
    'batch_size': 32,
    'bce_loss_scale': 0.1,
    'early_stopping': 30,
    'epochs': 350,
    'architecture': {
        'act_fn': nn.LeakyReLU,
        # 'batch_norm': {'eps': 1e-05, 'momentum': 0.1, 'affine': True},
        'dropout_prob': 0.7187,
        # Convolutional backbone block hyperparameters
        'conv2d_params': ({'kernel_size': (3, 3), 'out_channels': 4, 'padding': 1},
                          {'kernel_size': (3, 3), 'out_channels': 4, 'padding': 1},
                          {'kernel_size': (3, 3), 'out_channels': 8, 'padding': 1},
                          {'kernel_size': (3, 3), 'out_channels': 8, 'padding': 1, 'stride': 2},
                          {'kernel_size': (5, 5), 'out_channels': 16, 'padding': 2}),
        # Fully connected head block hyperparameters (a final FC inference layer with no dropout nor batchnorm will be added when ball detector model is instantiated)
        'fc_params': []
    }
}

# Detection hyperparameters obtained from last hyperparamter search
DETECTOR_HP_OLD2 = {
    'optimizer_params': {'amsgrad': False, 'betas': (0.9, 0.999), 'eps': 1e-08, 'lr': 5*9.40323e-05, 'weight_decay': 1e-3},
    'scheduler_params': {'step_size': 40, 'gamma': 0.2},
    'batch_size': 32,
    'bce_loss_scale': 0.1,
    'early_stopping': 30,
    'epochs': 350,
    'architecture': {
        'act_fn': nn.ReLU,
        'batch_norm': {'eps': 1e-05, 'momentum': 0.1, 'affine': True},
        'dropout_prob': 0.,
        'layers_param': (
            # Conv2d backbone layers
            ('conv2d', {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 0}),
            ('conv2d', {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 0}),
            ('conv2d', {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 0}),
            ('avg_pooling', {'kernel_size': (2, 2), 'stride': (2, 2)}),
            ('conv2d', {'out_channels': 16, 'kernel_size': (5, 5), 'padding': 0}),
            ('conv2d', {'out_channels': 16, 'kernel_size': (5, 5), 'padding': 0}),
            ('avg_pooling', {'kernel_size': (2, 2), 'stride': (2, 2)}),
            ('conv2d', {'out_channels': 32, 'kernel_size': (5, 5), 'padding': 2}),
            ('conv2d', {'out_channels': 32, 'kernel_size': (7, 7), 'padding': 3}),
            ('avg_pooling', {'kernel_size': (2, 2), 'stride': (2, 2)}),
            ('conv2d', {'out_channels': 64, 'kernel_size': (5, 5), 'padding': 2}),
            ('flatten', {}),
            ('fully_connected', {}))  # Last logits layer parameterized int balldetect.BallDetector.__init__ (no droupout, batch_norm nor activation function)
    }
}

DETECTOR_HP = {
    'optimizer_params': {'amsgrad': False, 'betas': (0.9, 0.999), 'eps': 1e-08, 'lr': 2*9.40323e-05, 'weight_decay': 1e-6},
    'scheduler_params': {'step_size': 20, 'gamma': 0.5},
    'batch_size': 64,
    'bce_loss_scale': 0.1,
    'early_stopping': 30,
    'epochs': 350,
    'architecture': {
        'act_fn': nn.ReLU,
        'batch_norm': {'eps': 1e-05, 'momentum': 0.1, 'affine': True},
        'dropout_prob': 0.,
        'layers_param': (
            # Conv2d backbone layers
            ('conv2d', {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 0}),
            ('conv2d', {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 0}),
            ('conv2d', {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 0}),
            ('avg_pooling', {'kernel_size': (2, 2), 'stride': (2, 2)}),
            ('conv2d', {'out_channels': 16, 'kernel_size': (5, 5), 'padding': 0}),
            ('conv2d', {'out_channels': 16, 'kernel_size': (5, 5), 'padding': 0}),
            ('avg_pooling', {'kernel_size': (2, 2), 'stride': (2, 2)}),
            ('conv2d', {'out_channels': 32, 'kernel_size': (5, 5), 'padding': 2}),
            ('conv2d', {'out_channels': 32, 'kernel_size': (7, 7), 'padding': 3}),
            ('avg_pooling', {'kernel_size': (2, 2), 'stride': (2, 2)}),
            ('conv2d', {'out_channels': 64, 'kernel_size': (5, 5), 'padding': 2}),
            ('flatten', {}),
            ('fully_connected', {}))  # Last logits layer parameterized int balldetect.BallDetector.__init__ (no droupout, batch_norm nor activation function)
        # Linear head layers
        # ('fully_connected', {'out_features': 64}),
        # ('fully_connected', {'out_features': 128}))
    }
}

# Forecasting hyperparameters obtained from hyperparamter search
SEQ_PRED_HP = {
    'optimizer_params': {'amsgrad': False, 'betas': (0.9, 0.999), 'eps': 1e-08, 'lr': 5*9.066e-05, 'weight_decay': 2.636e-06},
    'scheduler_params': {'step_size': 30, 'gamma': 0.3},
    'batch_size': 32,
    'early_stopping': 30,
    'epochs': 350,
    'architecture': {
        'act_fn': nn.Tanh,
        'dropout_prob': 0.,
        'fc_params': ({'out_features': 512}, {'out_features': 256}, {'out_features': 128}, {'out_features': 128})}
}


def main():
    # Parse arguments
    _DETECT, _FORECAST = 'detect', 'forecast'
    parser = argparse.ArgumentParser(description='Train ball detector or ball position forecasting model(s).')
    parser.add_argument('--model', nargs=1, type=str, required=True, choices=[_DETECT, _FORECAST],
                        help=f'Determines model to train ("{_DETECT}" or "{_FORECAST}").')
    train_model = parser.parse_args().model[0]

    torch_utils.set_seeds()  # Set seeds for better repducibility

    if train_model == _DETECT:
        print('> Initializing and training ball detection model (mini_balls dataset)...')
        ball_detector.train(**DETECTOR_HP)
        # TODO: save model: ball_detector.save_experiment(Path(r'../models/task1_experiments/train_0001/'), model, hp)

    elif train_model == _FORECAST:
        print('> Initializing and training ball position forecasting model (mini_balls dataset)...')
        seq_prediction.train(**SEQ_PRED_HP)
        # TODO: save model


if __name__ == "__main__":
    main()
