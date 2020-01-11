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

# Define hyperparameters
EPOCHS = 350
EARLY_STOPPING = 30
DETECTOR_HP = {
    'optimizer_params': {'amsgrad': False, 'betas': (0.9, 0.999), 'eps': 1e-08, 'lr': 9.9615e-05, 'weight_decay': 2.11924e-4},
    'scheduler_params': {'step_size': 40, 'gamma': 0.2},
    # 'scheduler_params': {'max_lr': 1e-2, 'pct_start': 0.3, 'anneal_strategy': 'cos'},
    'batch_size': 32,
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

SEQ_PRED_HP = {
    'optimizer_params': {'lr': 8e-5, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 1e-5, 'amsgrad': False},
    'scheduler_params': {'step_size': 40, 'gamma': 0.2},
    # 'scheduler_params': {'max_lr': 1e-2, 'pct_start': 0.3, 'anneal_strategy': 'cos'},
    'batch_size': 64,
    'architecture': {
        'act_fn': nn.LeakyReLU,
        # TODO: Avoid enabling both dropout and batch normalization at the same time: see ...
        'dropout_prob': 0.55,
        # 'batch_norm': {'eps': 1e-05, 'momentum': 0.1, 'affine': True},
        # Fully connected network hyperparameters (a final FC inference layer with no dropout nor batchnorm will be added when ball position predictor model is instantiated)
        'fc_params': [{'out_features': 512}, {'out_features': 256}, {'out_features': 128}, {'out_features': 128}]}
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
        training_objs = ball_detector.init_training(**DETECTOR_HP)
        _best_train_loss, _best_valid_loss, _best_epoch = ball_detector.train(*training_objs, epochs=EPOCHS, early_stopping=EARLY_STOPPING)
        # TODO: ball_detector.save_experiment(Path(r'../models/task1_experiments/train_0001/'), model, hp)

    elif train_model == _FORECAST:
        print('> Initializing and training ball position forecasting model (mini_balls dataset)...')
        training_objs = seq_prediction.init_training(**SEQ_PRED_HP)
        _best_train_loss, _best_valid_loss, _best_epoch = seq_prediction.train(*training_objs, epochs=EPOCHS, early_stopping=EARLY_STOPPING)
        # TODO: save pred_seq


if __name__ == "__main__":
    main()
