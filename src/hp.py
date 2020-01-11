#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball detector - Deeplearning Session 1 Exercice 1
.. moduleauthor:: Paul-Emmanuel Sotir
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting
"""
import math
import types
import argparse

from hyperopt import fmin, tpe, space_eval, Trials, hp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.modules.activation import ReLU, Tanh
from balldetect.ball_detector import train

import balldetect.torch_utils as torch_utils
import balldetect.ball_detector as ball_detector
import balldetect.seq_prediction as seq_prediction

__author__ = 'Paul-Emmanuel SOTIR <paul-emmanuel@outlook.com>'

EPOCHS = 90
EARLY_STOPPING = 12
HP_SEARCH_EVALS = 100
HP_SEARCH_ALGO = tpe.suggest

# Torch CuDNN configuration
torch.backends.cudnn.deterministic = True
cudnn.benchmark = torch.cuda.is_available()  # Enable builtin CuDNN auto-tuner, TODO: benchmarking isn't deterministic, disable this if this is an issue
cudnn.fastest = torch.cuda.is_available()  # Disable this if memory issues


def get_objective(model_module: types.ModuleType, train_model_name: str):
    def _objective(params: dict) -> float:
        print('\n' + '#' * 20 + f' {train_model_name.upper()} HYPERPARAMETERS TRIAL  ' + '#' * 20 + f'\n{params}')
        torch_utils.set_seeds()  # Set seeds for better repducibility

        # Train ball detector model
        training_objs = model_module.init_training(**params)
        _, valid_loss, _ = model_module.train(*training_objs, epochs=EPOCHS, early_stopping=EARLY_STOPPING, pbar=False)
        return valid_loss
    return _objective


def forecast_objective(params: dict) -> float:
    print('\n' + '#' * 20 + ' FORECAST HYPERPARAMETERS TRIAL  ' + '#' * 20)
    torch_utils.set_seeds()  # Set seeds for better repducibility

    # Train ball detector model
    training_objs = seq_prediction.init_training(**params)
    _, valid_loss, _ = seq_prediction.train(*training_objs, epochs=EPOCHS, early_stopping=EARLY_STOPPING, pbar=False)
    return valid_loss


def main():
    # Parse arguments
    _DETECT, _FORECAST = 'detect', 'forecast'
    parser = argparse.ArgumentParser(description='Train ball detector or ball position forecasting model(s).')
    parser.add_argument('--model', nargs=1, type=str, required=True, choices=[_DETECT, _FORECAST],
                        help=f'Determines model to train ("{_DETECT}" or "{_FORECAST}").')
    train_model = parser.parse_args().model[0]

    # Define hyperparameter search space for ball detector (task 1)
    detect_hp_space = {
        'optimizer_params': {'lr': hp.uniform('lr', 5e-6, 1e-4), 'betas': (0.9, 0.999), 'eps': 1e-8,
                             'weight_decay': hp.loguniform('weight_decay', math.log(1e-7), math.log(1e-3)), 'amsgrad': False},
        'scheduler_params': {'step_size': EPOCHS, 'gamma': 1.},
        # 'scheduler_params': {'max_lr': 1e-2, 'pct_start': 0.3, 'anneal_strategy': 'cos'},
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
        'architecture': {
            'act_fn': nn.LeakyReLU,
            # TODO: Avoid enabling both dropout and batch normalization at the same time: see ...
            'dropout_prob': hp.choice('dropout_prob', [1., hp.uniform('nonzero_dropout_prob', 0.45, 0.8)]),
            # 'batch_norm': {'eps': 1e-05, 'momentum': 0.1, 'affine': True},
            # Convolutional backbone block hyperparameters
            'conv2d_params': hp.choice('conv2d_params', [
                [{'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1},
                 {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1},
                 {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1, 'stride': 2},
                 {'out_channels': 8, 'kernel_size': (5, 5), 'padding': 2},
                 {'out_channels': 8, 'kernel_size': (7, 7), 'padding': 3}],

                [{'out_channels': 2, 'kernel_size': (3, 3), 'padding': 1},
                 {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1},
                 {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1, 'stride': 2},
                 {'out_channels': 8, 'kernel_size': (5, 5), 'padding': 2},
                 {'out_channels': 8, 'kernel_size': (5, 5), 'padding': 2, 'stride': 2},
                 {'out_channels': 16, 'kernel_size': (7, 7), 'padding': 3}],

                [{'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1},
                 {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1},
                 {'out_channels': 8, 'kernel_size': (3, 3), 'padding': 1},
                 {'out_channels': 8, 'kernel_size': (3, 3), 'padding': 1, 'stride': 2},
                 {'out_channels': 16, 'kernel_size': (5, 5), 'padding': 2}],

                [{'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1},
                 {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1},
                 {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1},
                 {'out_channels': 4, 'kernel_size': (3, 3), 'padding': 1, 'stride': 4},
                 {'out_channels': 8, 'kernel_size': (3, 3), 'padding': 1},
                 {'out_channels': 8, 'kernel_size': (3, 3), 'padding': 1}]
            ]),
            # Fully connected head block hyperparameters (a final FC inference layer with no dropout nor batchnorm will be added when ball detector model is instantiated)
            'fc_params': hp.choice('fc_params', [[{'out_features': 64}],
                                                 [{'out_features': 64}, {'out_features': 128}],
                                                 []])}
    }

    # Define hyperparameter search space for ball position forecasting (task 2)
    forecast_hp_space = {
        'optimizer_params': {'lr': hp.uniform('lr', 5e-6, 1e-4), 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': hp.loguniform('weight_decay', math.log(1e-7), math.log(1e-2)), 'amsgrad': False},
        'scheduler_params': {'step_size': EPOCHS, 'gamma': 1.},
        # 'scheduler_params': {'max_lr': 1e-2, 'pct_start': 0.3, 'anneal_strategy': 'cos'},
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
        'architecture': {
            'act_fn': hp.choice('act_fn', [nn.LeakyReLU, nn.ReLU, nn.Tanh]),
            # TODO: Avoid enabling both dropout and batch normalization at the same time: see ...
            'dropout_prob': hp.choice('dropout_prob', [1., hp.uniform('nonzero_dropout_prob', 0.45, 0.8)]),
            # 'batch_norm': {'eps': 1e-05, 'momentum': 0.1, 'affine': True},
            # Fully connected network hyperparameters (a final FC inference layer with no dropout nor batchnorm will be added when ball position predictor model is instantiated)
            'fc_params': hp.choice('fc_params', [[{'out_features': 512}, {'out_features': 256}] + [{'out_features': 128}] * 2,
                                                 [{'out_features': 128}] + [{'out_features': 256}] * 2 + [{'out_features': 512}],
                                                 [{'out_features': 128}] + [{'out_features': 256}] * 3,
                                                 [{'out_features': 128}] * 2 + [{'out_features': 256}] * 3,
                                                 [{'out_features': 128}] * 2 + [{'out_features': 256}] * 4,
                                                 [{'out_features': 128}] * 3 + [{'out_features': 256}] * 4])}
    }

    if train_model == _DETECT:
        print('Running hyperparameter search for ball detection model (mini_balls dataset)...')
        hp_space = detect_hp_space
        objective = get_objective(ball_detector, train_model)
    elif train_model == _FORECAST:
        print('Running hyperparameter search for ball position forecasting model (mini_balls_seq dataset)...')
        hp_space = forecast_hp_space
        objective = get_objective(seq_prediction, train_model)
    else:
        exit(-1)

    trials = Trials()
    best_parameters = fmin(objective,
                           algo=HP_SEARCH_ALGO,
                           max_evals=HP_SEARCH_EVALS,
                           space=hp_space,
                           trials=trials)

    print('\n\n' + '#' * 20 + '  BEST HYPERPARAMETERS  ' + '#' * 20)
    print(best_parameters)
    print('\n\n' + '#' * 20 + '  BEST HYPERPARAMETERS  ' + '#' * 20)
    print(space_eval(hp_space, best_parameters))


if __name__ == '__main__':
    main()
