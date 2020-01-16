#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball detector - Deeplearning Exercice 1 - Part 1
.. moduleauthor:: Paul-Emmanuel Sotir
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting
"""
import os
import shutil
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
import balldetect.vis as vis
pickle = tu.import_pickle()

__all__ = ['BallDetector', 'init_training', 'train']
__author__ = 'Paul-Emmanuel SOTIR <paul-emmanuel@outlook.com>'

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
VIS_DIR = tu.source_dir() / f'../../visualization_imgs/detector2'


class BallDetector(nn.Module):
    """ Ball detector pytorch module.
    .. class:: BallDetector
    """

    __constants__ = ['_input_shape', '_p_output_size', '_bb_output_size', '_xavier_gain', '_conv_features_shapes', '_conv_out_features']

    def __init__(self, input_shape: torch.Size, output_sizes: tuple, layers_param: list, act_fn: type = nn.ReLU, dropout_prob: float = 0., batch_norm: Optional[dict] = None):
        super(BallDetector, self).__init__()
        self._input_shape = input_shape
        self._p_output_size, self._bb_output_size = output_sizes
        self._xavier_gain = nn.init.calculate_gain(tu.get_gain_name(act_fn))
        self._conv_features_shapes, self._conv_out_features = [], None
        layers_param = list(layers_param)

        # Define neural network architecture
        layers = []
        in_conv_backbone, prev_out = True, self._input_shape[-3]

        for name, params in layers_param:
            if name == 'avg_pooling':
                layers.append(nn.AvgPool2d(**params) if in_conv_backbone else nn.AvgPool1d(**params))

            elif name == 'conv2d':
                params['in_channels'] = prev_out
                layers.append(tu.conv_layer(params, act_fn, dropout_prob, batch_norm))
                prev_out = params['out_channels']
                # Get convolution output features shape by performing a dummy forward
                with torch.no_grad():
                    net = nn.Sequential(*layers)
                    dummy_batch_x = torch.zeros(torch.Size((1, *self._input_shape)))
                    self._conv_features_shapes.append(net(dummy_batch_x).shape)

            elif name == 'fully_connected':
                # Determine in_features for this fully connected layer
                if in_conv_backbone:  # First FC layer following convolution backbone
                    self._conv_out_features = np.prod(self._conv_features_shapes[-1][-3:])
                    params['in_features'] = self._conv_out_features
                    in_conv_backbone = False
                else:
                    params['in_features'] = prev_out
                if 'out_features' not in params:
                    # Handle last fully connected layer (no dropout nor batch normalization for this layer)
                    params['out_features'] = self._p_output_size + self._bb_output_size
                    layers.append(tu.fc_layer(params))
                else:
                    layers.append(tu.fc_layer(params, act_fn, dropout_prob, batch_norm))
                prev_out = params['out_features']

            elif name == 'flatten':
                layers.append(tu.Flatten())
        self._layers = nn.Sequential(*layers)

    def init_params(self):
        def _initialize_weights(module: nn.Module):
            mtype = type(module).__module__
            if mtype == nn.Conv2d.__module__:
                nn.init.xavier_normal_(module.weight.data, gain=self._xavier_gain)
                module.bias.data.fill_(0.)
            elif issubclass(type(module), nn.Linear):
                nn.init.xavier_uniform_(module.weight.data, gain=self._xavier_gain)
                module.bias.data.fill_(0.)
            elif mtype == nn.BatchNorm2d.__module__:
                nn.init.uniform_(module.weight.data)  # gamma == weight here
                module.bias.data.fill_(0.)  # beta == bias here
            elif list(module.parameters(recurse=False)) and list(module.children()):
                raise Exception("ERROR: Some module(s) which have parameter(s) havn't bee explicitly initialized.")
        self.apply(_initialize_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._layers(x)  # Apply whole neural net architecture

        pos_logits = x[:, :self._p_output_size]  # infered positions: logits from a log_softmax layer
        bbs = x[:, self._p_output_size:]  # bounding boxes (no activation function)
        return pos_logits, bbs


def train(batch_size: int, architecture: dict, optimizer_params: dict, scheduler_params: dict, bce_loss_scale: float, epochs: int, early_stopping: Optional[int] = None, pbar: bool = True) -> Tuple[float, float, int]:
    """ Initializes dataset, dataloaders, model, optimizer and lr_scheduler for future training """
    # TODO: refactor this to avoid some duplicated code with seq_prediction.init_training()
    # TODO: add path parameter for dataset dir
    # Create balls dataset
    dataset = datasets.BallsCFDetection(tu.source_dir() / r'../../datasets/mini_balls')

    # Create ball detector model and dataloaders
    trainset, validset = datasets.create_dataloaders(dataset, batch_size)
    dummy_img, p, bb = dataset[0]  # Nescessary to retreive input image resolution (assumes all dataset images are of the same size)
    model = BallDetector(dummy_img.shape, (np.prod(p.shape), np.prod(bb.shape)), **architecture)
    model.init_params()
    if batch_size > 64:
        model = tu.parrallelize(model)
    model.train(True).to(DEVICE)
    print(f'> MODEL ARCHITECTURE:\n{model.__repr__()}')
    print(f'> MODEL CONVOLUTION FEATURE SIZES: {model._conv_features_shapes}')

    # Define optimizer, loss and LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    bb_metric, pos_metric = torch.nn.MSELoss(), torch.nn.BCEWithLogitsLoss()
    scheduler_params['step_size'] *= len(trainset)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch = len(trainset), epochs = hp['epochs'], **scheduler_params)

    # Create directory for results visualization
    if VIS_DIR is not None:
        shutil.rmtree(VIS_DIR, ignore_errors=True)
        VIS_DIR.mkdir(parents=True)

    best_valid_loss, best_train_loss = float("inf"), float("inf")
    best_run_epoch = -1
    epochs_since_best_loss = 0

    # Main training loop
    for epoch in range(1, epochs + 1):
        print("\nEpoch %03d/%03d\n" % (epoch, epochs) + '-' * 15)
        train_loss = 0

        trange, update_bar = tu.progess_bar(trainset, '> Training on trainset', min(
            len(trainset.dataset), trainset.batch_size), custom_vars=True, disable=not pbar)
        for (batch_x, colors, bbs) in trange:
            batch_x, colors, bbs = batch_x.to(DEVICE).requires_grad_(True), tu.flatten_batch(colors.to(DEVICE)), tu.flatten_batch(bbs.to(DEVICE))

            def closure():
                optimizer.zero_grad()
                output_colors, output_bbs = model(batch_x)
                loss = bce_loss_scale * pos_metric(output_colors, colors) + bb_metric(output_bbs, bbs)
                loss.backward()
                return loss
            loss = float(optimizer.step(closure).clone().detach())
            scheduler.step()
            train_loss += loss / len(trainset)
            update_bar(trainLoss=f'{len(trainset) * train_loss / (trange.n + 1):.7f}', lr=f'{float(scheduler.get_lr()[0]):.3E}')

        print(f'>\tDone: TRAIN_LOSS = {train_loss:.7f}')
        valid_loss = evaluate(epoch, model, validset, bce_loss_scale, best_valid_loss, pbar=pbar)
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

    print(f'>\tBest training results obtained at {best_run_epoch}nth epoch (best_valid_loss = {best_valid_loss:.7f}, best_train_loss = {best_train_loss:.7f}).')
    return best_train_loss, best_valid_loss, best_run_epoch


def evaluate(epoch: int, model: nn.Module, validset: DataLoader, bce_loss_scale: float, best_valid_loss: float, pbar: bool = True) -> float:
    model.eval()
    with torch.no_grad():
        bb_metric, pos_metric = torch.nn.MSELoss(), torch.nn.BCEWithLogitsLoss()
        valid_loss, first_step = 0., True

        for (batch_x, colors, bbs) in tu.progess_bar(validset, '> Evaluation on validset', min(len(validset.dataset), validset.batch_size), disable=not pbar):
            batch_x, colors, bbs = batch_x.to(DEVICE), tu.flatten_batch(colors.to(DEVICE)), tu.flatten_batch(bbs.to(DEVICE))
            output_colors, output_bbs = model(batch_x)
            valid_loss += (bce_loss_scale * pos_metric(output_colors, colors) + bb_metric(output_bbs, bbs)) / len(validset)

            if first_step and VIS_DIR is not None and best_valid_loss >= valid_loss:
                first_step = False
                print(f"> ! Saving visualization images of inference on some validset values...")
                for idx in np.random.permutation(range(validset.batch_size))[:8]:
                    img, bbs, _cols = datasets.retrieve_data(batch_x[idx], output_bbs[idx], output_colors[idx])
                    vis.show_bboxes(img, bbs, datasets.COLORS, out_fn=VIS_DIR / f'vis_valid_{idx}.png')
    return float(valid_loss)

# TODO: finalize saving implementation
# def save_experiment(save_dir: Path, model: nn.Module, hyperparameters: Optional[dict] = None, eval_metrics: Optional[dict] = None, train_metrics: Optional[dict] = None):
#     """ Saves a pytorch model along with experiment information like hyperparameters, test metrics and training metrics """
#     if not save_dir.is_dir():
#         logging.warning(f'Replacing existing directory during model snaphshot saving process. (save_dir = "{save_dir}")')
#         save_dir.rmdir()
#     save_dir.parent.mkdir(parents = True, exist_ok = True)

#     model_name = model._get_name()
#     meta = {'model_name': model_name,
#             'model_filename': f'{model_name}.pt',
#             'absolute_save_path': save_dir.resolve(),
#             'metadata_filename': f'{model_name}_metadata.json'}

#     torch.save(model.state_dict(), save_dir / meta['model_filename']))

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


# def _log_eval_results(results, type, JSON_log, JSON_log_template = './eval_log_template.json'):
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
