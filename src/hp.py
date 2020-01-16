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

import balldetect.torch_utils as tu
import balldetect.ball_detector as ball_detector
import balldetect.seq_prediction as seq_prediction
pickle = tu.import_pickle()

__author__ = 'Paul-Emmanuel SOTIR <paul-emmanuel@outlook.com>'

HP_SEARCH_EVALS = 100
HP_SEARCH_ALGO = tpe.suggest

# Torch CuDNN configuration
torch.backends.cudnn.deterministic = True
cudnn.benchmark = torch.cuda.is_available()  # Enable builtin CuDNN auto-tuner, TODO: benchmarking isn't deterministic, disable this if this is an issue
cudnn.fastest = torch.cuda.is_available()  # Disable this if memory issues


def main():
    # Parse arguments
    _DETECT, _FORECAST = 'detect', 'forecast'
    parser = argparse.ArgumentParser(description='Train ball detector or ball position forecasting model(s).')
    parser.add_argument('--model', nargs=1, type=str, required=True, choices=[_DETECT, _FORECAST],
                        help=f'Determines model to train ("{_DETECT}" or "{_FORECAST}").')
    train_model = parser.parse_args().model[0]
    TRIALS_FILEPATH = tu.source_dir(__file__) / f'../hp_trials_{train_model}.pkl'

    # Ball detector Conv2d backbone layers
    conv_backbone = (
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
        ('flatten', {}))

    # Define hyperparameter search space (second hp search space iteration) for ball detector (task 1)
    detect_hp_space = {
        'optimizer_params': {'lr': hp.uniform('lr', 1e-6, 1e-3), 'betas': (0.9, 0.999), 'eps': 1e-8,
                             'weight_decay': hp.loguniform('weight_decay', math.log(1e-7), math.log(3e-3)), 'amsgrad': False},
        'scheduler_params': {'step_size': 40, 'gamma': .3},
        # 'scheduler_params': {'max_lr': 1e-2, 'pct_start': 0.3, 'anneal_strategy': 'cos'},
        'batch_size': hp.choice('batch_size', [16, 32, 64]),
        'bce_loss_scale': 0.1,
        'early_stopping': 12,
        'epochs': 90,
        'architecture': {
            'act_fn': nn.ReLU,
            'batch_norm': {'eps': 1e-05, 'momentum': hp.uniform('momentum', 0.05, 0.15), 'affine': True},
            'dropout_prob': hp.choice('dropout_prob', [0., hp.uniform('nonzero_dropout_prob', 0.1, 0.45)]),
            'layers_param': hp.choice('layers_param', [(*conv_backbone, ('fully_connected', {'out_features': 64}),
                                                        ('fully_connected', {})),
                                                       (*conv_backbone, ('fully_connected', {'out_features': 64}),
                                                        ('fully_connected', {'out_features': 128}),
                                                        ('fully_connected', {})),
                                                       (*conv_backbone, ('fully_connected', {'out_features': 128}),
                                                        ('fully_connected', {'out_features': 128}),
                                                        ('fully_connected', {})),
                                                       (*conv_backbone, ('fully_connected', {}))])
        }
    }

    # Define hyperparameter search space for ball position forecasting (task 2)
    forecast_hp_space = {
        'optimizer_params': {'lr': hp.uniform('lr', 5e-6, 1e-4), 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': hp.loguniform('weight_decay', math.log(1e-7), math.log(1e-2)), 'amsgrad': False},
        'scheduler_params': {'step_size': 30, 'gamma': .3},
        # 'scheduler_params': {'max_lr': 1e-2, 'pct_start': 0.3, 'anneal_strategy': 'cos'},
        'batch_size': hp.choice('batch_size', [16, 32, 64]),
        'early_stopping': 12,
        'epochs': 90,
        'architecture': {
            'act_fn': nn.Tanh,
            'dropout_prob': hp.choice('dropout_prob', [0., hp.uniform('nonzero_dropout_prob', 0.1, 0.45)]),
            # Fully connected network hyperparameters (a final FC inference layer with no dropout nor batchnorm will be added when ball position predictor model is instantiated)
            'fc_params': hp.choice('fc_params', [[{'out_features': 512}, {'out_features': 256}] + [{'out_features': 128}] * 2,
                                                 [{'out_features': 128}] + [{'out_features': 256}] * 2 + [{'out_features': 512}],
                                                 [{'out_features': 128}] + [{'out_features': 256}] * 3,
                                                 [{'out_features': 128}] * 2 + [{'out_features': 256}] * 3,
                                                 [{'out_features': 128}] * 2 + [{'out_features': 256}] * 4,
                                                 [{'out_features': 128}] * 3 + [{'out_features': 256}] * 4])}
    }

    if train_model == _DETECT:
        hp_space = detect_hp_space
        model_module = ball_detector
    elif train_model == _FORECAST:
        hp_space = forecast_hp_space
        model_module = seq_prediction
    else:
        print('ERROR: bad model_name provided')  # TODO: logging.error
        exit(-1)

    # Define hp search objective (runs one hyperparameter trial)
    def _objective(params: dict) -> float:
        print('\n' + '#' * 20 + f' {train_model.upper()} HYPERPARAMETERS TRIAL  ' + '#' * 20 + f'\n{params}')
        # Set seeds for better repducibility
        tu.set_seeds()
        # Train ball detector model
        _, valid_loss, _ = model_module.train(**params, pbar=False)
        return valid_loss

    print(f'Running hyperparameter search for "{train_model}" model (mini_balls_seq dataset)...')
    trials = Trials()
    best_parameters = fmin(_objective,
                           algo=HP_SEARCH_ALGO,
                           max_evals=HP_SEARCH_EVALS,
                           space=hp_space,
                           trials=trials)

    print('\n\n' + '#' * 20 + f'  BEST HYPERPARAMETERS ({train_model.upper()})  ' + '#' * 20)
    print(space_eval(hp_space, best_parameters))

    print('\n\n' + '#' * 20 + f'  TRIALS  ({train_model.upper()})  ' + '#' * 20)
    print(trials)

    print('\n\n' + '#' * 20 + f'  TRIALS.results  ({train_model.upper()})  ' + '#' * 20)
    print(trials.results)

    print('\n\n' + '#' * 20 + f'  TRIALS.results  ({train_model.upper()})  ' + '#' * 20)
    print(trials.best_trial)

    print('Saving trials with pickle...')
    with open(TRIALS_FILEPATH, 'wb') as f:
        pickle.dump(trials, f)


if __name__ == '__main__':
    main()
