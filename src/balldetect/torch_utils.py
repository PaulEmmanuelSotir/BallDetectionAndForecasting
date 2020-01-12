#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball detector - Deeplearning Exercice 1 - Part 1
.. moduleauthor:: Paul-Emmanuel Sotir
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting
"""
import types
import random
import importlib
from typing import Tuple, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

__all__ = ['layer', 'conv_layer', 'fc_layer', 'flatten_batch', 'parralelize', 'set_seeds', 'progess_bar', 'import_pickle']
__author__ = 'Paul-Emmanuel SOTIR <paul-emmanuel@outlook.com>'


def layer(layer_op: nn.Module, act_fn: nn.Module, dropout_prob: float, batch_norm: Optional[dict] = None) -> Tuple[nn.Module, nn.Module]:
    if dropout_prob is not None and dropout_prob != 0.:
        return (nn.Dropout(p=dropout_prob), layer_op, act_fn)
    # elif batch_norm is not None and batch_norm:
        # Applies Batch_narm after ReLu : see reddit thread about it : https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
    #    return (layer_op, act_fn, nn.BatchNorm2d(, **batch_norm))  # TODO: fix batch_norm: handle num_features
    return (layer_op, act_fn)


def conv_layer(conv2d: dict, act_fn: type = nn.Identity, dropout_prob: float = 0., batch_norm: Optional[dict] = None) -> nn.Module:
    return nn.Sequential(*layer(nn.Conv2d(**conv2d), act_fn(), dropout_prob, batch_norm))


def fc_layer(linear: dict, act_fn: type = nn.Identity, dropout_prob: float = 0., batch_norm: Optional[dict] = None) -> nn.Module:
    return nn.Sequential(*layer(nn.Linear(**linear), act_fn(), dropout_prob, batch_norm))


def flatten_batch(tensor):
    return tensor.view(tensor.size(0), -1)  # Flattens target bounding boxes and positions


def parrallelize(model: nn.Module) -> nn.Module:
    """ Make use of all available GPU using nn.DataParallel
    NOTE: ensure to be using different random seeds for each process if you use techniques like data-augmentation or any other techniques which needs random numbers different for each steps. TODO: make sure this isn't already done by Pytorch?
    """
    if torch.cuda.device_count() > 1:
        print(f'> Using "nn.DataParallel(model)" on {torch.cuda.device_count()} GPUs.')
        model = nn.DataParallel(model)
    return model


def set_seeds(all_seeds: int = 3453493):
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
