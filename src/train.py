#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball detector - Deeplearning Session 1 Exercice 1
.. moduleauthor:: Paul-Emmanuel Sotir
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting
"""
import os
import numpy as np
from typing import Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import balldetect.ball_detector as ball_detector
import balldetect.seq_prediction as seq_prediction

__author__ = 'Paul-Emmanuel SOTIR <paul-emmanuel@outlook.com>'

# Torch configuration
cudnn.benchmark = torch.cuda.is_available()  # Enable inbuilt CuDNN auto-tuner TODO: measure performances without this flag
cudnn.fastest = torch.cuda.is_available()  # Disable this if memory issues

# Define hyperparameters
EPOCHS = 250
EARLY_STOPPING = 30
DETECTOR_HP = {
    'optimizer_params': {'lr': 5e-5, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 5e-6, 'amsgrad': False},
    # 'scheduler_params': {'max_lr': 1e-2, 'pct_start': 0.3, 'anneal_strategy': 'cos'},
    'batch_size': 32,
    'architecture': {
        'act_fn': nn.LeakyReLU,
        # TODO: Avoid enabling both dropout and batch normalization at the same time: see ...
        'dropout_prob': 0.5,
        # 'batch_norm': {'eps': 1e-05, 'momentum': 0.1, 'affine': True},
        # Convolutional backbone block hyperparameters
        'conv2d_params': [{'out_channels': 16, 'kernel_size': (3, 3), 'padding': 1},
                          {'out_channels': 8, 'kernel_size': (3, 3), 'padding': 1, 'stride': 2},
                          {'out_channels': 8, 'kernel_size': (3, 3), 'padding': 1},
                          {'out_channels': 8, 'kernel_size': (3, 3), 'padding': 1},
                          {'out_channels': 4, 'kernel_size': (5, 5), 'padding': 2}],
        # Fully connected head block hyperparameters (a final FC inference layer with no dropout nor batchnorm will be added when ball detector model is instantiated)
        'fc_params': [{'out_features': 64}]}
}
SEQ_PRED_HP = {
    'optimizer_params': {'lr': 8e-5, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 1e-5, 'amsgrad': False},
    # 'scheduler_params': {'max_lr': 1e-2, 'pct_start': 0.3, 'anneal_strategy': 'cos'},
    'batch_size': 32,
    'architecture': {
        'act_fn': nn.ReLU,
        # TODO: Avoid enabling both dropout and batch normalization at the same time: see ...
        'dropout_prob': 0.55,
        # 'batch_norm': {'eps': 1e-05, 'momentum': 0.1, 'affine': True},
        # Fully connected network hyperparameters (a final FC inference layer with no dropout nor batchnorm will be added when ball position predictor model is instantiated)
        'fc_params': [{'out_features': 512}, {'out_features': 256}, {'out_features': 128}, {'out_features': 128}]}
}


# TODO: refactor this like I did for ball detector
# def train_pred_seq():
#     dataset = datasets.BallsCFSeq(Path("./datasets/mini_balls_seq"))

#     # Create ball position sequence prediction model
#     trainset, validset = datasets.create_dataloaders(dataset, hp['batch_size'], INFERENCE_BATCH_SIZE)
#     p, bb = dataset[0]  # Nescessary to retreive input image resolution (assumes all dataset images are of the same size)
#     model = seq_prediction.SeqPredictor(p.shape, np.prod(bb.shape), **hp['architecture'])
#     model = tu.parrallelize(model)

#     # Define optimizer and LR scheduler
#     optimizer = torch.optim.Adam(model.parameters(), **hp['optimizer_params'])
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, len(trainset), gamma=1.)

#     # Train and save ball detector model
#     model = seq_prediction.train(trainset, validset, model, optimizer, scheduler, hp['epochs'])
#     # ball_detector.save_experiment(Path(r'../models/task2_experiments/train_0001/'), model, hp)


if __name__ == "__main__":
    # Train and save ball detector model
    training_objs = ball_detector.init_training(**DETECTOR_HP)
    _best_train_loss, _best_valid_loss, _best_epoch = ball_detector.train(EPOCHS, *training_objs, early_stopping=EARLY_STOPPING)

    # training_objs = seq_prediction.init_training(**SEQ_PRED_HP)
    # ball_detector.save_experiment(Path(r'../models/task1_experiments/train_0001/'), model, hp)

    # train_pred_seq(seq_pred_hp)
