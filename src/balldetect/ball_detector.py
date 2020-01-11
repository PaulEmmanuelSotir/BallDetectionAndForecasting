#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball detector - Deeplearning Exercice 1 - Part 1
.. moduleauthor:: Paul-Emmanuel Sotir
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting
"""
import os
from typing import Tuple, Optional
from pathlib import Path
from collections import OrderedDict

import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import balldetect.datasets as datasets
import balldetect.torch_utils as tu
pickle = tu.import_pickle()

__all__ = ['BallDetector', 'init_training', 'train']
__author__ = 'Paul-Emmanuel SOTIR <paul-emmanuel@outlook.com>'

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
INFERENCE_BATCH_SIZE = 8*1024  # Batch size used during inference (including validset evaluation)


class BallDetector(nn.Module):
    """ Ball detector pytorch module.
    .. class:: BallDetector
    """

    __constants__ = ['_input_shape', '_p_output_size', '_bb_output_size', '_conv_features_shape', '_conv_out_features']

    def __init__(self, input_shape: torch.Size, output_sizes: tuple, conv2d_params: list, fc_params: list, act_fn: type = nn.ReLU, dropout_prob: float = 0., batch_norm: Optional[dict] = None):
        super(BallDetector, self).__init__()
        self._input_shape = input_shape
        self._p_output_size, self._bb_output_size = output_sizes
        conv2d_params, fc_params = list(conv2d_params), list(fc_params)

        # Define neural network architecture (convolution backbone followed by fullyconnected head)
        layers = []
        for prev_params, (i, params) in zip([None] + conv2d_params[:-1], enumerate(conv2d_params)):
            params['in_channels'] = prev_params['out_channels'] if prev_params is not None else self._input_shape[-3]
            layers.append((f'conv_layer_{i}', tu.conv_layer(params, act_fn, dropout_prob, batch_norm)))
        self._conv_layers = nn.Sequential(OrderedDict(layers))
        self._conv_features_shape = self._conv_layers(torch.zeros(torch.Size((1, *self._input_shape)))).shape
        self._conv_out_features = np.prod(self._conv_features_shape[-3:])

        layers = []
        for prev_params, (i, params) in zip([None] + fc_params[:-1], enumerate(fc_params)):
            params['in_features'] = prev_params['out_features'] if prev_params is not None else self._conv_out_features
            layers.append((f'fc_layer_{i}', tu.fc_layer(params, act_fn, dropout_prob, batch_norm)))
        # Append last fully connected inference layer (no dropout nor batch normalization for this layer)
        layers.append(('fc_layer', tu.fc_layer({'in_features': fc_params[-1]['out_features'] if len(fc_params) > 0 else self._conv_out_features,
                                                'out_features': self._p_output_size + self._bb_output_size})))
        self._fc_layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._conv_layers(x)  # Process backbone features
        x = tu.flatten(x)
        x = self._fc_layers(x)  # Apply fully connected head neural net

        pos_logits = x[:, :self._p_output_size]  # infered positions: logits from a log_softmax layer
        bbs = x[:, self._p_output_size:]  # bounding boxes (no activation function)

        # TODO: use data à-priori/redoundancy for better inference results. E.g. by zeroing bounding boxes for which position is 0...
        return pos_logits, bbs


# TODO: add path parameter for dataset dir
def init_training(batch_size: int, architecture: dict, optimizer_params: dict, scheduler_params: dict) -> Tuple[DataLoader, DataLoader, nn.Module, Optimizer, _LRScheduler]:
    """ Initializes dataset, dataloaders, model, optimizer and lr_scheduler for future training """
    # Create balls dataset
    dataset = datasets.BallsCFDetection(Path("../datasets/mini_balls"), img_transform=F.normalize)

    # Create ball detector model and dataloaders
    trainset, validset = datasets.create_dataloaders(dataset, batch_size, INFERENCE_BATCH_SIZE)
    dummy_img, p, bb = dataset[0]  # Nescessary to retreive input image resolution (assumes all dataset images are of the same size)
    model = BallDetector(dummy_img.shape, (np.prod(p.shape), np.prod(bb.shape)), **architecture)
    if batch_size > 64:
        model = tu.parrallelize(model)

    # Define optimizer and LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(trainset), epochs=hp['epochs'], **scheduler_params)
    scheduler_params['step_size'] *= len(trainset)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    return trainset, validset, model, optimizer, scheduler


def train(trainset: DataLoader, validset: DataLoader, model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler._LRScheduler, epochs: int, early_stopping: Optional[int] = None, pbar: bool = True) -> Tuple[float, float, int]:
    """ Trains model on given dataset """
    model.train(True).to(DEVICE)
    bb_metric, pos_metric = torch.nn.MSELoss(), torch.nn.BCEWithLogitsLoss()

    # Weight xavier initialization
    def _initialize_weights(module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_normal_(module.weight.data, gain=1.)
    model.apply(_initialize_weights)

    best_valid_loss, best_train_loss = float("inf"), float("inf")
    best_run_epoch = -1
    epochs_since_best_loss = 0

    # Main training loop
    for epoch in range(1, epochs + 1):
        print("\nEpoch %03d/%03d\n" % (epoch, epochs) + '-' * 15)
        train_loss = 0

        trange, update_bar = tu.progess_bar(trainset, '> Training on trainset', trainset.batch_size, custom_vars=True, disable=not pbar)
        for (batch_x, ps, bbs) in trange:
            batch_x, ps, bbs = batch_x.to(DEVICE).requires_grad_(True), tu.flatten(ps.to(DEVICE)), tu.flatten(bbs.to(DEVICE))

            def closure():
                optimizer.zero_grad()
                output_ps, output_bbs = model(batch_x)
                loss = pos_metric(output_ps, ps) + bb_metric(output_bbs, bbs)
                loss.backward()
                return loss
            loss = optimizer.step(closure).detach()
            scheduler.step()
            train_loss += loss / len(trainset)
            update_bar(trainLoss=f'{len(trainset) * train_loss / trange.n:.7f}', lr=f'{scheduler.get_lr()[0]:.3E}')

        print(f'>\tDone: TRAIN_LOSS = {train_loss:.7f}')
        valid_loss = evaluate(model, validset, pbar=pbar)
        print(f'>\tDone: VALID_LOSS = {valid_loss:.7f}')

        if best_valid_loss > valid_loss:
            print('>\tBest valid_loss found so far, saving model...')  # TODO: save model
            best_valid_loss, best_train_loss = valid_loss, train_loss
            best_run_epoch = epoch
            epochs_since_best_loss = 0
        else:
            epochs_since_best_loss += 1
            if early_stopping is not None and early_stopping > 0 and epochs_since_best_loss >= early_stopping:
                print(f'>\tModel not improving: Ran {epochs_since_best_loss} training epochs without improvement. Early stopping training loop...')
                break

    print(f'>\tBest training results obtained at {best_run_epoch}nth epoch (best_valid_loss={best_valid_loss:.7f}, best_train_loss={best_train_loss:.7f}).')
    return best_train_loss, best_valid_loss, best_run_epoch


def evaluate(model: nn.Module, validset: DataLoader, pbar: bool = True) -> float:
    model.eval()
    with torch.no_grad():
        bb_metric, pos_metric = torch.nn.MSELoss(), torch.nn.BCEWithLogitsLoss()
        valid_loss = 0.

        for (batch_x, ps, bbs) in tu.progess_bar(validset, '> Evaluation on validset', validset.batch_size, disable=not pbar):
            batch_x, ps, bbs = batch_x.to(DEVICE), tu.flatten(ps.to(DEVICE)), tu.flatten(bbs.to(DEVICE))
            output_ps, output_bbs = model(batch_x)
            valid_loss += (pos_metric(output_ps, ps) + bb_metric(output_bbs, bbs)) / len(validset)
    return float(valid_loss)


# def save_experiment(save_dir: Path, model: nn.Module, hyperparameters: Optional[dict] = None, eval_metrics: Optional[dict] = None, train_metrics: Optional[dict] = None):
#     """ Saves a pytorch model along with experiment information like hyperparameters, test metrics and training metrics """
#     if not save_dir.is_dir():
#         logging.warning(f'Replacing existing directory during model snaphshot saving process. (save_dir="{save_dir}")')
#         save_dir.rmdir()
#     save_dir.parent.mkdir(parents=True, exist_ok=True)

#     model_name = model._get_name()
#     meta = {'model_name': model_name,
#             'model_filename': f'{model_name}.pt',
#             'absolute_save_path': save_dir.resolve(),
#             'metadata_filename': f'{model_name}_metadata.json'}

#     torch.save(model.state_dict(), save_dir.joinpath(meta['model_filename']))

#     if eval_metrics is not None and len(eval_metrics) > 0:
#         meta['eval_metrics'] = json.dumps(eval_metrics)
#     if train_metrics is not None and len(train_metrics) > 0:
#         meta['train_metrics'] = json.dumps(train_metrics)
#     if hyperparameters is not None and len(hyperparameters) > 0:
#         meta['hp_json__repr'] = json.dumps(hyperparameters)  # Only stores a json representation of hyperparameters for convenience, use pickel
#         meta['hp_pkl_filename'] = f'{model_name}_hyperparameters.pkl'
#         with open(save_dir.joinpath(meta['hyperparameters_filename']), 'rb') as f:
#             pickle.dump(hyperparameters, f)

#     with open(save_dir.joinpath(meta['metadata_filename'])) as f:
#         json.dump(meta, f)

    # Store training and evaluation metrics and some informatations about the snapshot into json file


# def _log_eval_results(results, type, JSON_log, JSON_log_template='./eval_log_template.json'):
#     print("> Storing evaluation results to " + JSON_log)

#     if not os.path.isfile(JSON_log):
#         # Copy empty JSON evaluation log template
#         shutil.copy(JSON_log_template, JSON_log)

#     # Append results to evaluation log
#     with open(JSON_log, "w") as f:
#         log = json.load(f)
#         log[type].append(results)
#         json.dump(log, f)


# def load(save_path):
#     # det = BallDetector(len(dataset[0][0]), len(datasets.COLORS), hp).to(DEVICE).train(False)

#     # load model pickle
#     # with open(model_path, 'rb') as model_pkl:
#         # model = pickle.load(model_pkl)

#     # try:
#     #     import cPickle as pickle
#     # except ImportError:
#     #     import pickle
#     # TODO:
#     pass
