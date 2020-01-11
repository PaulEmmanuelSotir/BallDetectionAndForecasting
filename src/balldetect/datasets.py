#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball synthetic dataset - Deeplearning Session 1  
.. moduleauthor:: Fabien Baradel, Paul-Emmanuel Sotir, Christian Wolf  
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting  
"""
import os
import fnmatch
import numpy as np
from skimage import io
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

__all__ = ['COLORS', 'BBOX_SCALE', 'VALID_SET_SIZE', 'BallsCFDetection', 'BallsCFSeq', 'create_dataloaders']
__author__ = 'Fabien Baradel, Paul-Emmanuel Sotir, Christian Wolf'

COLORS = ['red', 'green', 'blue', 'yellow', 'lime', 'purple', 'orange', 'cyan', 'magenta']
BBOX_SCALE = [[[89., 88., 99., 99.]]]
VALID_SET_SIZE = 0.015  # ~0.015% of dataset size, TODO: do cross_validation (dataset too small) and create a small testset in addition to validset and trainset
_DEFAULT_WORKERS = 0 if __debug__ else min(os.cpu_count() - 1, max(1, os.cpu_count() // 4) * max(1, torch.cuda.device_count()))


def create_dataloaders(dataset: Dataset, train_batch_size: int, valid_batch_size: int = 1024, validset_size_pct: float = VALID_SET_SIZE, num_workers: int = _DEFAULT_WORKERS) -> Tuple[DataLoader, DataLoader]:
    # Split dataset into trainset and testset
    indices = torch.randperm(len(dataset)).tolist()
    test_size = int(validset_size_pct * len(dataset))
    train_ds, valid_ds = torch.utils.data.Subset(dataset, indices[:-test_size]), torch.utils.data.Subset(dataset, indices[-test_size:])

    # Create validset and trainset dataloaders
    print(f'> __debug__ == {__debug__} - Using {num_workers} workers in each DataLoader...')
    trainset = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validset = DataLoader(valid_ds, batch_size=valid_batch_size, num_workers=num_workers, pin_memory=True)

    return trainset, validset


class BallsCFDetection(Dataset):
    """ BallsCFDetection Pytorch dataset.
    .. class:: BallsCFDetection
    """

    def __init__(self, path, img_transform=None, bbox_scale=1.):
        self.path = path
        self.img_transform = img_transform
        self.bbox_scale = bbox_scale
        self.image_count = len(fnmatch.filter(os.listdir(path), '*.npy'))

    # The access is _NOT_ shuffled. The Dataloader will need to do this.
    def __getitem__(self, index):
        if index >= self.image_count:
            raise IndexError()
        img = io.imread("%s/img_%05d.jpg" % (self.path, index))
        img = np.asarray(img)
        img = img.astype(np.float32)

        # Dims in: x, y, color
        # should be: color, x, y
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float32)
        if self.img_transform is not None:
            img = self.img_transform(img)

        # Load presence and bounding boxes and split it up
        p_bb = np.load("%s/p_bb_%05d.npy" % (self.path, index)).astype(dtype=np.float32, copy=False)
        p = torch.tensor(p_bb[:, 0], dtype=torch.float32)
        bb = torch.tensor(p_bb[:, 1:5] / BBOX_SCALE, dtype=torch.float32)
        return img, p, bb

    # Return the dataset size
    def __len__(self):
        return self.image_count


class BallsCFSeq(Dataset):
    """ BallsCFSeq Pytorch dataset.
    .. class:: BallsCFSeq
    """

    def __init__(self, path):
        self.path = path
        self.seq_count = len(fnmatch.filter(os.listdir(path), 'seq_bb_*.npy'))

    # The access is _NOT_ shuffled. The Dataloader will need
    # to do this.
    def __getitem__(self, index):
        # Load presence and bounding boxes
        p = np.load("%s/p_%05d.npy" % (self.path, index))
        bb = np.load("%s/seq_bb_%05d.npy" % (self.path, index))

        # split bounding boxes and create tensors from data
        return torch.tensor([p, bb[:5] / BBOX_SCALE], dtype=torch.float32), torch.tensor(bb[5:], dtype=torch.float32)

    # Return the dataset size
    def __len__(self):
        return self.seq_count
