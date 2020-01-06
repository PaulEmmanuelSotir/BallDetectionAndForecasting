#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball detector - Deeplearning Exercice 1 - Part 1
.. moduleauthor:: Paul-Emmanuel Sotir
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting
"""
import os
from pathlib import Path
from collections import OrderedDict

import json
import logging
import numpy as np
from tqdm import tqdm

try:
    import cPickle as pickle
except ImportError:
    import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

__all__ = ['BallDetector', 'train']

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class BallDetector(nn.Module):
    """ Ball detector pytorch module.
    .. class:: BallDetector
    """

    __constants__ = ['_input_shape', '_n_classes', '_conv_features_shape', '_conv_out_features']

    def __init__(self, input_shape: torch.Size, n_classes: int, conv2d_params: list, fc_params: list, act_fn: type = nn.ReLU, dropout_prob: float = 0., batch_norm: dict = {}):
        super(BallDetector, self).__init__()
        self._input_shape = input_shape
        self._n_classes = n_classes

        # Define neural network architecture (convolution backbone followed by fullyconnected head)
        layers = []
        for prev_params, (i, params) in zip([None] + conv2d_params[:-1], enumerate(conv2d_params)):
            params['in_channels'] = prev_params['out_channels'] if prev_params is not None else self._input_shape[-3]
            layers.append((f'conv_layer_{i}', _ConvLayer(params, act_fn, dropout_prob, batch_norm)))
        self._conv_layers = nn.Sequential(OrderedDict(layers))
        self._conv_features_shape = self._conv_layers(torch.zeros(torch.Size((1, *self._input_shape)))).shape
        self._conv_out_features = np.prod(self._conv_features_shape[-3:])

        layers = []
        for prev_params, (i, params) in zip([None] + fc_params[:-1], enumerate(fc_params)):
            params['in_features'] = prev_params['out_features'] if prev_params is not None else self._conv_out_features
            layers.append((f'fc_layer_{i}', _FCLayer(params, act_fn, dropout_prob, batch_norm)))
        # Append last fully connected inference layer (no dropout nor batch normalization for this layer)
        layers.append(('fc_layer_log_softmax', _FCLayer({'in_features': fc_params[-1]['out_features'] if len(fc_params) > 0 else self._conv_out_features,
                                                         'out_features': self._n_classes})))
        self._fc_layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv_layers(x)  # Process backbone features
        x = x.view(x.size(0), -1)  # Flatten features from convolution backbone (-1 == self._conv_features_shape)
        return self._fc_layers(x)  # Apply fully connected head neural net


def train(trainset: DataLoader, testset: DataLoader, model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, epochs: int):
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
        print("\nEpoch %03d/%03d\n" % (epoch, epochs - 1) + '-' * 15)

        # Preforms one epoch of training on trainset
        train_mse = 0

        trange, update_tqdm = _tqdm(trainset, '> Training on trainset', trainset.batch_size, custom_vars=True)
        for (batch_x, _ps, batch_bbs) in trange:
            batch_x, batch_bbs = batch_x.to(device), batch_bbs.to(device).requires_grad_(True)
            batch_bbs = batch_bbs.view(batch_bbs.size(0), -1)  # Flattens target bounding boxes

            def closure():
                optimizer.zero_grad()
                output = model(batch_x)
                loss = mse(output, batch_bbs)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            scheduler.step()
            train_mse += loss / len(trainset)
            update_tqdm(trainMSE=f'{len(trainset) * train_mse / trange.n:.7f}', lr=f'{scheduler.get_lr()[0]:.3E}')

        print(f'\tDone: TRAIN_MSE = {train_mse:.7f}')
        print(f'\tDone: TEST_MSE = {evaluate(model, testset):.7f}')
    return model


def evaluate(model, testset) -> float:
    model.eval()
    with torch.no_grad():
        metric = torch.nn.MSELoss()
        test_mse = 0.

        for (batch_x, _ps, batch_bbs) in _tqdm(testset, '> Evaluation on testset', testset.batch_size):
            batch_x, batch_bbs = batch_x.to(device), batch_bbs.to(device).requires_grad_(True)
            batch_bbs = batch_bbs.view(batch_bbs.size(0), -1)  # Flattens target bounding boxes
            output = model(batch_x)
            test_mse += metric(output, batch_bbs) / len(testset)
    return float(test_mse)


def save_experiment(save_dir: Path, model: nn.Module, hyperparameters: dict = {}, eval_metrics: dict = {}, train_metrics: dict = {}):
    """ Saves a pytorch model along with experiment information like hyperparameters, test metrics and training metrics """
    if not save_dir.is_dir():
        logging.warning(f'Replacing existing directory during model snaphshot saving process. (save_dir="{save_dir}")')
        save_dir.rmdir()
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    model_name = model._get_name()
    meta = {'model_name': model_name,
            'model_filename': f'{model_name}.pt',
            'absolute_save_path': save_dir.resolve(),
            'metadata_filename': f'{model_name}_metadata.json'}

    torch.save(model.state_dict(), save_dir.joinpath(meta['model_filename']))

    if eval_metrics is not None and len(eval_metrics) > 0:
        meta['eval_metrics'] = json.dumps(eval_metrics)
    if train_metrics is not None and len(train_metrics) > 0:
        meta['train_metrics'] = json.dumps(train_metrics)
    if hyperparameters is not None and len(hyperparameters) > 0:
        meta['hp_json__repr'] = json.dumps(hyperparameters)  # Only stores a json representation of hyperparameters for convenience, use pickel
        meta['hp_pkl_filename'] = f'{model_name}_hyperparameters.pkl'
        with open(save_dir.joinpath(meta['hyperparameters_filename']), 'rb') as f:
            pickle.dump(hyperparameters, f)

    with open(save_dir.joinpath(meta['metadata_filename'])) as f:
        json.dump(meta, f)

    # Store training and evaluation metrics and some informatations about the snapshot into json file


def _Layer(layer_op: nn.Module, act_fn: nn.Module, dropout_prob: float, batch_norm: dict) -> tuple:
    if dropout_prob is not None and dropout_prob != 0.:
        return (nn.Dropout(p=dropout_prob), layer_op, act_fn)
    # elif batch_norm is not None and batch_norm:
    #    return (layer_op, act_fn, nn.BatchNorm2d(, **batch_norm))  # TODO: fix batch_norm: handle num_features
    return (layer_op, act_fn)


def _ConvLayer(conv2d: dict, act_fn: type = nn.Identity, dropout_prob: float = 0., batch_norm: dict = {}) -> nn.Module:
    return nn.Sequential(*_Layer(nn.Conv2d(**conv2d), act_fn(), dropout_prob, batch_norm))


def _FCLayer(linear: dict, act_fn: type = nn.Identity, dropout_prob: float = 0., batch_norm: dict = {}) -> nn.Module:
    return nn.Sequential(*_Layer(nn.Linear(**linear), act_fn(), dropout_prob, batch_norm))


def _tqdm(iterable, desc, batch_size, custom_vars=False):
    t = tqdm(iterable, unit='batch', ncols=240, desc=desc, postfix=f'BatchSize={batch_size}',
             bar_format='{desc} {percentage:3.0f}%|'
             '{bar}'
             '| {n_fmt}/{total_fmt} [Elapsed={elapsed}, Remaining={remaining}, Speed={rate_fmt}{postfix}]')
    if custom_vars:
        def callback(**kwargs):
            t.set_postfix(**kwargs)
        return t, callback
    return t

# def _log_eval_results(results, type, JSON_log, JSON_log_template='./eval_log_template.json'):
#     print("Storing evaluation results to " + JSON_log)

#     if not os.path.isfile(JSON_log):
#         # Copy empty JSON evaluation log template
#         shutil.copy(JSON_log_template, JSON_log)

#     # Append results to evaluation log
#     with open(JSON_log, "w") as f:
#         log = json.load(f)
#         log[type].append(results)
#         json.dump(log, f)


def load(save_path):
    # det = BallDetector(len(dataset[0][0]), len(datasets.COLORS), hp).to(device).train(False)

    # load model pickle
    # with open(model_path, 'rb') as model_pkl:
        #model = pickle.load(model_pkl)

    # try:
    #     import cPickle as pickle
    # except ImportError:
    #     import pickle
    # TODO:
    pass
