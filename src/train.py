#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball detector - Deeplearning Session 1 Exercice 1
.. moduleauthor:: Paul-Emmanuel Sotir
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting
"""
import argparse

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

DETECTOR_HP = {
    'batch_size': 16,
    'bce_loss_scale': 0.1,
    'early_stopping': 30,
    'epochs': 400,
    'optimizer_params': {'amsgrad': False, 'betas': (0.9, 0.999), 'eps': 1e-08, 'lr': 6.537177808319479e-4, 'weight_decay': 6.841231983628692e-06},
    'scheduler_params': {'gamma': 0.3, 'step_size': 40},
    'architecture': {
        'act_fn': nn.ReLU,
        'batch_norm': {'affine': True, 'eps': 1e-05, 'momentum': 0.07359778246238029},
        'dropout_prob': 0.0,
        'layers_param': (('conv2d', {'kernel_size': (3, 3), 'out_channels': 4, 'padding': 0}),
                         ('conv2d', {'kernel_size': (3, 3), 'out_channels': 4, 'padding': 0}),
                         ('conv2d', {'kernel_size': (3, 3), 'out_channels': 4, 'padding': 0}),
                         ('avg_pooling', {'kernel_size': (2, 2), 'stride': (2, 2)}),
                         ('conv2d', {'kernel_size': (5, 5), 'out_channels': 16, 'padding': 0}),
                         ('conv2d', {'kernel_size': (5, 5), 'out_channels': 16, 'padding': 0}),
                         ('avg_pooling', {'kernel_size': (2, 2), 'stride': (2, 2)}),
                         ('conv2d', {'kernel_size': (5, 5), 'out_channels': 32, 'padding': 2}),
                         ('conv2d', {'kernel_size': (7, 7), 'out_channels': 32, 'padding': 3}),
                         ('avg_pooling', {'kernel_size': (2, 2), 'stride': (2, 2)}),
                         ('conv2d', {'kernel_size': (5, 5), 'out_channels': 64, 'padding': 2}),
                         ('flatten', {}),
                         ('fully_connected', {}))
    }
}

SEQ_PRED_HP = {
    'batch_size': 16,
    'early_stopping': 30,
    'epochs': 400,
    'optimizer_params': {'amsgrad': False, 'betas': (0.9, 0.999), 'eps': 1e-08, 'lr': 9.891933484569264e-05, 'weight_decay': 2.0217734556558288e-4},
    'scheduler_params': {'gamma': 0.3, 'step_size': 30},
    'architecture': {
        'act_fn': nn.Tanh,
        'dropout_prob': 0.44996724122672166,
        'fc_params': ({'out_features': 512}, {'out_features': 256}, {'out_features': 128}, {'out_features': 128})
    }
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
