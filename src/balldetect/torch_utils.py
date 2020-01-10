#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball detector - Deeplearning Exercice 1 - Part 1
.. moduleauthor:: Paul-Emmanuel Sotir
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting
"""
from typing import Tuple, Optional

from tqdm import tqdm

import torch
import torch.nn as nn

__all__ = ['layer', 'conv_layer', 'fc_layer', 'flatten', 'parralelize', 'progess_bar']
__author__ = 'Paul-Emmanuel SOTIR <paul-emmanuel@outlook.com>'


def layer(layer_op: nn.Module, act_fn: nn.Module, dropout_prob: float, batch_norm: Optional[dict] = None) -> Tuple[nn.Module, nn.Module]:
    if dropout_prob is not None and dropout_prob != 0.:
        return (nn.Dropout(p=dropout_prob), layer_op, act_fn)
    # elif batch_norm is not None and batch_norm:
    #    return (layer_op, act_fn, nn.BatchNorm2d(, **batch_norm))  # TODO: fix batch_norm: handle num_features
    return (layer_op, act_fn)


def conv_layer(conv2d: dict, act_fn: type = nn.Identity, dropout_prob: float = 0., batch_norm: Optional[dict] = None) -> nn.Module:
    return nn.Sequential(*layer(nn.Conv2d(**conv2d), act_fn(), dropout_prob, batch_norm))


def fc_layer(linear: dict, act_fn: type = nn.Identity, dropout_prob: float = 0., batch_norm: Optional[dict] = None) -> nn.Module:
    return nn.Sequential(*layer(nn.Linear(**linear), act_fn(), dropout_prob, batch_norm))


def flatten(tensor):
    return tensor.view(tensor.size(0), -1)  # Flattens target bounding boxes and positions


def parrallelize(model: nn.Module) -> nn.Module:
    # Make use of all available GPU using nn.DataParallel
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    return model


def progess_bar(iterable, desc, batch_size, custom_vars: bool = False, disable: bool = False):
    t = tqdm(iterable, unit='batch', ncols=180, desc=desc, postfix=f'BatchSize={batch_size}', ascii=False, position=0, disable=disable,
             bar_format='{desc} {percentage:3.0f}%|'
             '{bar}'
             '| {n_fmt}/{total_fmt} [Elapsed={elapsed}, Remaining={remaining}, Speed={rate_fmt}{postfix}]')
    if custom_vars:
        def callback(**kwargs):
            t.set_postfix(**kwargs)
        return t, callback
    return t
