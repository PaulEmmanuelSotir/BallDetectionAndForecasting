#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball detector - Deeplearning Exercice 1 - Part 1
.. moduleauthor:: Paul-Emmanuel Sotir
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting
"""
import os
import re
import types
import random
import importlib
import pandas as pd
import pprint as pp
import operator as op
from pathlib import Path
from typing import Iterable, Optional, List, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

__all__ = ['layer', 'conv_layer', 'fc_layer', 'Flatten', 'flatten_batch', 'get_gain_name', 'parrallelize', 'set_seeds',
           'progess_bar', 'import_pickle', 'source_dir', 'extract_from_hp_search_log', 'summarize_hp_search']
__author__ = 'Paul-Emmanuel SOTIR <paul-emmanuel@outlook.com>'


def layer(layer_op: nn.Module, act_fn: nn.Module, dropout_prob: float, batch_norm: Optional[dict] = None) -> Iterable[nn.Module]:
    ops = (layer_op, act_fn)
    weight_rank = len(layer_op.weight.data.shape)

    if dropout_prob is not None and dropout_prob != 0.:
        ops = (nn.Dropout(p=dropout_prob),) + ops
    if batch_norm is not None:
        # Applies Batch_narm after activation function : see reddit thread about it : https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
        if weight_rank == 4:
            ops += (nn.BatchNorm2d(layer_op.out_channels, **batch_norm),)
        elif weight_rank < 4:
            ops += (nn.BatchNorm1d(layer_op.out_features, **batch_norm),)
    return ops


def conv_layer(conv2d: dict, act_fn: type = nn.Identity, dropout_prob: float = 0., batch_norm: Optional[dict] = None) -> nn.Module:
    return nn.Sequential(*layer(nn.Conv2d(**conv2d), act_fn(), dropout_prob, batch_norm))


def fc_layer(linear: dict, act_fn: type = nn.Identity, dropout_prob: float = 0., batch_norm: Optional[dict] = None) -> nn.Module:
    return nn.Sequential(*layer(nn.Linear(**linear), act_fn(), dropout_prob, batch_norm))


class Flatten(nn.Module):
    def __init__(self) -> None:
        super(Flatten, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return flatten_batch(x)


def flatten_batch(tensor):
    return tensor.view(tensor.size(0), -1)  # Flattens target bounding boxes and positions


def get_gain_name(act_fn: type) -> str:
    """ Intended to be used with nn.init.calculate_gain(str):
    .. Example: nn.init.calculate_gain(get_gain_act_name(nn.ReLU))
    """
    if act_fn is nn.ReLU:
        return 'relu'
    elif act_fn is nn.LeakyReLU:
        return 'leaky_relu'
    elif act_fn is nn.Tanh:
        return 'tanh'
    elif act_fn is nn.Identity:
        return 'linear'
    else:
        raise Exception("Unsupported activation function, can't initialize it.")


def parrallelize(model: nn.Module) -> nn.Module:
    """ Make use of all available GPU using nn.DataParallel
    NOTE: ensure to be using different random seeds for each process if you use techniques like data-augmentation or any other techniques which needs random numbers different for each steps. TODO: make sure this isn't already done by Pytorch?
    """
    if torch.cuda.device_count() > 1:
        print(f'> Using "nn.DataParallel(model)" on {torch.cuda.device_count()} GPUs.')
        model = nn.DataParallel(model)
    return model


def set_seeds(all_seeds: int = 345349):
    set_seeds(torch_seed=all_seeds, cuda_seed=all_seeds, np_seed=all_seeds, python_seed=all_seeds)


def set_seeds(torch_seed: Optional[int] = None, cuda_seed: Optional[int] = None, np_seed: Optional[int] = None, python_seed: Optional[int] = None):
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
    if cuda_seed is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(cuda_seed)
    if np_seed is not None:
        np.random.seed(np_seed)
    if python_seed is not None:
        random.seed(python_seed)


def progess_bar(iterable, desc, batch_size, custom_vars: bool = False, disable: bool = False):
    t = tqdm(iterable, unit='batch', ncols=180, desc=desc, postfix=f'BatchSize={batch_size}', ascii=False, position=0, disable=disable,
             bar_format='{desc} {percentage:3.0f}%|'
             '{bar}'
             '| {n_fmt}/{total_fmt} [Elapsed={elapsed}, Remaining={remaining}, Speed={rate_fmt}{postfix}]')
    if custom_vars:
        def callback(**kwargs):
            t.set_postfix(batch_size=batch_size, **kwargs)
        return t, callback
    return t


def import_pickle() -> types.ModuleType:
    """ Returns cPickle module if available, returns imported pickle module otherwise """
    try:
        pickle = importlib.import_module('cPickle')
    except ImportError:
        pickle = importlib.import_module('pickle')
    return pickle


def source_dir(source_file: str = __file__) -> Path:
    return Path(os.path.dirname(os.path.realpath(source_file)))


def extract_from_hp_search_log(log_path: Path) -> Tuple[List[dict], int, dict]:
    def _to_float(iterable): return list(map(float, iterable))
    with open(log_path, 'r') as f:
        log = f.read()

    # Split hyperparameter search into trials
    experiments = re.split('HYPERPARAMETERS TRIAL', log)
    del log

    flags = re.MULTILINE + re.IGNORECASE
    trials = []
    for i, exp in enumerate(experiments):
        hp_match = re.findall(r'#+\s*\n\r?\s*(\{.*\})\s*\n\r?', exp, flags)
        if hp_match is not None and len(hp_match) > 0:
            epoch_matches = list(map(int, re.findall(r'Epoch\s+([0-9]+)/[0-9]+\s*\n\r?', exp, flags)))
            train_matches = _to_float(re.findall(r'TRAIN_LOSS\s*=\s*([\.0-9]+)\n\r?', exp, flags))
            valid_matches = _to_float(re.findall(r'VALID_LOSS\s*=\s*([\.0-9]+)\n\r?', exp, flags))

            if epoch_matches is not None and train_matches is not None and valid_matches is not None:
                trials.append({'hyperparameters': hp_match[0], 'train_losses': _to_float(train_matches), 'valid_losses': _to_float(valid_matches),
                               'best_epoch': np.argmin(valid_matches), 'epochs': np.max(epoch_matches)})
            else:
                print(f"WARNING: Can't parse resulting losses of hyperparameter search trial NO#{i}.")
        else:
            print(f"WARNING: Can't parse hyperparameter search trial NO#{i}.")
    best_idx = np.argmin([np.min(t['valid_losses']) for t in trials])
    return trials, best_idx, trials[best_idx]


def summarize_hp_search(trials: List[dict], best_idx: int, hp_search_name: str = ''):
    best_trial = trials[best_idx]
    hp_search_name = hp_search_name.upper()

    valid_losses = list(map(op.itemgetter('valid_losses'), trials))
    pd.DataFrame(valid_losses).T.plot(figsize=(20, 10), legend=False, logy=True, title=f'{hp_search_name} HYPERPARAMETER SEARCH - VALID LOSSES')
    train_losses = list(map(op.itemgetter('train_losses'), trials))
    pd.DataFrame(train_losses).T.plot(figsize=(20, 10), legend=False, logy=True, title=f'{hp_search_name} HYPERPARAMETER SEARCH - TRAIN LOSSES')

    best_epoch = best_trial['best_epoch']
    print('#' * 10 + f'  {hp_search_name} HYPERPARAMETER SEARCH RESULTS  ' + '#' * 10)
    print(f'Hyperparameter search ran {len(trials)} trials. Best trial ({best_idx}th trial) results:')
    print('Best_valid_loss=' + str(best_trial['valid_losses'][best_epoch]) + f' at epoch={best_epoch}')
    print(f'Hyperparameters:')
    pp.pprint(best_trial['hyperparameters'])

    pd.DataFrame(best_trial).filter(('valid_losses', 'train_losses')) \
        .plot(figsize=(20, 10), logy=True, title='BALL DETECTOR: BEST TRIAL LOSSES')


if __name__ == '__main__':
    extract_from_hp_search_log(source_dir() / r'../../hp_search_logs/hp_detect3.log')
