#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball detector - Deeplearning Session 1 Exercice 1
.. moduleauthor:: Paul-Emmanuel Sotir
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting
"""
import os
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import balldetect.datasets as datasets
import balldetect.ball_detector as ball_detector
import balldetect.seq_prediction as seq_prediction

INFERENCE_BATCH_SIZE = 8*1024  # Batch size used during inference (including testset evaluation)
TEST_SET_SIZE = 0.015  # ~0.015% of dataset size
CPU_COUNT = os.cpu_count()

# Torch configuration
cudnn.benchmark = torch.cuda.is_available()  # Enable inbuilt CuDNN auto-tuner TODO: measure performances without this flag
cudnn.fastest = torch.cuda.is_available()  # Disable this if memory issues


# 🎓🏅🏆🎯🧬🔬🧰📟💻⌨💽💾📡🔦💡📚📉📈⏲⏳⌛
# 🙍‍♂️🙎‍♂️🙅‍♂️🙆‍♂️🧏‍♂️💁‍♂️🙋‍♂️🤦‍♂️🤷‍♂️💆‍♂️💇‍♂️🙇‍♂️
# 👇👈👆👉
# 👍
# 😶😗😕😐😙😚🙂😊😉😀😃😄😂😁

def train_detector():
    # Create balls dataset
    bbox_scale = [[[89., 88., 99., 99.]]]
    dataset = datasets.BallsCFDetection(Path("./datasets/mini_balls"), img_transform=F.normalize, bbox_scale=bbox_scale)

    # Define hyperparameters
    hp = {
        'optimizer_params': {'lr': 5e-5, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 5e-6, 'amsgrad': False},
        'scheduler_params': {'max_lr': 1e-2, 'pct_start': 0.3, 'anneal_strategy': 'cos'},
        'bbox_scale': bbox_scale,
        'batch_size': 32,
        'epochs': 500,
        # Classes weights/importance used in NLLLoss. Change these values in case of unbalanced classes:
        # TODO: remove it: 'class_weights': torch.tensor([1.] * len(datasets.COLORS)),
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

    # Split dataset into trainset and testset
    indices = torch.randperm(len(dataset)).tolist()
    test_size = int(TEST_SET_SIZE * len(dataset))
    train_ds, test_ds = torch.utils.data.Subset(dataset, indices[:-test_size]), torch.utils.data.Subset(dataset, indices[-test_size:])

    # Create testset and trainset dataloaders
    num_workers = 0 if __debug__ else min(CPU_COUNT - 1, max(1, CPU_COUNT // 4) * max(1, torch.cuda.device_count()))
    print(f'DEBUG = {__debug__} - Using {num_workers} workers in each DataLoader...')
    trainset = DataLoader(train_ds, batch_size=hp['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True)
    testset = DataLoader(test_ds, batch_size=INFERENCE_BATCH_SIZE, num_workers=num_workers, pin_memory=True)

    # Create ball detector model
    dummy_img, _p, bb = dataset[0]  # Nescessary to retreive input image resolution (assumes all dataset images are of the same size)
    model = ball_detector.BallDetector(dummy_img.shape, np.prod(bb.shape), **hp['architecture'])

    # Make use of all available GPU using nn.DataParallel
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Define optimizer and LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), **hp['optimizer_params'])
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(trainset), epochs=hp['epochs'], **hp['scheduler_params'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, len(trainset), gamma=1.)

    # Train and save ball detector model
    model = ball_detector.train(trainset, testset, model, optimizer, scheduler, hp['epochs'])
    # ball_detector.save_experiment(Path(r'../models/task1_experiments/train_0001/'), model, hp)


def train_pred_seq():

    # Create balls dataset
    bbox_scale = [[[89., 88., 99., 99.]]]
    dataset = datasets.BallsCFSeq(Path("./datasets/mini_balls_seq"), bbox_scale=bbox_scale)

    # Define hyperparameters
    hp = {
        'optimizer_params': {'lr': 8e-5, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 1e-5, 'amsgrad': False},
        'scheduler_params': {'max_lr': 1e-2, 'pct_start': 0.3, 'anneal_strategy': 'cos'},
        'bbox_scale': bbox_scale,
        'batch_size': 32,
        'epochs': 500,
        # Classes weights/importance used in NLLLoss. Change these values in case of unbalanced classes:
        # TODO: remove it: 'class_weights': torch.tensor([1.] * len(datasets.COLORS)),
        'architecture': {
            'act_fn': nn.ReLU,
            # TODO: Avoid enabling both dropout and batch normalization at the same time: see ...
            'dropout_prob': 0.55,
            # 'batch_norm': {'eps': 1e-05, 'momentum': 0.1, 'affine': True},
            # Fully connected network hyperparameters (a final FC inference layer with no dropout nor batchnorm will be added when ball position predictor model is instantiated)
            'fc_params': [{'out_features': 512}, {'out_features': 256}, {'out_features': 128}, {'out_features': 128}]}
    }

    # Split dataset into trainset and testset
    indices = torch.randperm(len(dataset)).tolist()
    test_size = int(TEST_SET_SIZE * len(dataset))
    train_ds, test_ds = torch.utils.data.Subset(dataset, indices[:-test_size]), torch.utils.data.Subset(dataset, indices[-test_size:])

    # Create testset and trainset dataloaders
    num_workers = 0 if __debug__ else min(CPU_COUNT - 1, max(1, CPU_COUNT // 4) * max(1, torch.cuda.device_count()))
    print(f'DEBUG = {__debug__} - Using {num_workers} workers in each DataLoader...')
    trainset = DataLoader(train_ds, batch_size=hp['batch_size'], shuffle=True, num_workers=num_workers, pin_memory=True)
    testset = DataLoader(test_ds, batch_size=INFERENCE_BATCH_SIZE, num_workers=num_workers, pin_memory=True)

    # Create ball position sequence prediction model
    dummy_img, _p, bb = dataset[0]  # Nescessary to retreive input image resolution (assumes all dataset images are of the same size)
    model = seq_prediction.SeqPredictor(dummy_img.shape, np.prod(bb.shape), **hp['architecture'])

    # Make use of all available GPU using nn.DataParallel
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    # Define optimizer and LR scheduler
    optimizer = torch.optim.Adam(model.parameters(), **hp['optimizer_params'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, len(trainset), gamma=1.)

    # Train and save ball detector model
    model = seq_prediction.train(trainset, testset, model, optimizer, scheduler, hp['epochs'])
    # ball_detector.save_experiment(Path(r'../models/task2_experiments/train_0001/'), model, hp)


if __name__ == "__main__":
    train_detector()
    # train_pred_seq()
