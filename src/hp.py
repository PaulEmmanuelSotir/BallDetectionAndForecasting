#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball detector - Deeplearning Session 1 Exercice 1
.. moduleauthor:: Paul-Emmanuel Sotir
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting
"""
from hyperopt import fmin, tpe, space_eval, Trials, hp
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import balldetect.ball_detector as ball_detector

__author__ = 'Paul-Emmanuel SOTIR <paul-emmanuel@outlook.com>'

EPOCHS = 70
EARLY_STOPPING = 12
HP_SEARCH_EVALS = 200
HP_SEARCH_ALGO = tpe.suggest

# Torch CuDNN configuration
cudnn.benchmark = torch.cuda.is_available()  # Enable builtin CuDNN auto-tuner
cudnn.fastest = torch.cuda.is_available()  # Disable this if memory issues


def objective(params: dict) -> float:
    print('\n' + '#' * 20 + '  HYPERPARAMETERS TRIAL  ' + '#' * 20 + f'\n{params}')

    # Train ball detector model
    training_objs = ball_detector.init_training(**params)
    _, valid_loss, _ = ball_detector.train(EPOCHS, *training_objs, early_stopping=EARLY_STOPPING, pbar=False)
    return valid_loss


def main():
    # Define hyperparameter search space
    hp_space = {
        'optimizer_params': {'lr': hp.uniform('lr', 5e-6, 1e-4), 'betas': (0.9, 0.999), 'eps': 1e-8,
                             'weight_decay': hp.loguniform('weight_decay', math.log(1e-7), math.log(1e-3)), 'amsgrad': False},
        # 'scheduler_params': {'max_lr': 1e-2, 'pct_start': 0.3, 'anneal_strategy': 'cos'},
        'batch_size': hp.choice('batch_size', [32, 64, 128]),
        'architecture': {
            'act_fn': nn.LeakyReLU,
            # TODO: Avoid enabling both dropout and batch normalization at the same time: 1ee ...
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
    # ball_detector.save_experiment(Path(r'../models/task1_experiments/train_0001/'), model, hp)


if __name__ == '__main__':
    main()
