#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball synthetic dataset - Deeplearning Session 1  
.. moduleauthor:: Fabien Baradel, Paul-Emmanuel Sotir, Christian Wolf  
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting  
"""
import os
import fnmatch
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
from skimage import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

__all__ = ['COLORS', 'BBOX_SCALE', 'VALID_SET_SIZE', 'BallsCFDetection', 'BallsCFSeq', 'create_dataloaders', 'retrieve_data']
__author__ = 'Fabien Baradel, Paul-Emmanuel Sotir, Christian Wolf'

COLORS = ['red', 'green', 'blue', 'yellow', 'lime', 'purple', 'orange', 'cyan', 'magenta']
BBOX_SCALE = [[[89., 88., 99., 99.]]]
# Validset size: ~10% of dataset size, TODO: make larger validset or do cross_validation (dataset too small) and create a small testset in addition to validset and trainset
VALID_SET_SIZE = 0.10
_DEFAULT_WORKERS = 0 if __debug__ else min(os.cpu_count() - 1, max(1, os.cpu_count() // 4) * max(1, torch.cuda.device_count()))
_REMOVE_ZERO_BBOX = False


def create_dataloaders(dataset: Dataset, batch_size: int, validset_size_pct: float = VALID_SET_SIZE, num_workers: int = _DEFAULT_WORKERS) -> Tuple[DataLoader, DataLoader]:
    # Split dataset into trainset and testset
    indices = torch.randperm(len(dataset)).tolist()
    test_size = int(validset_size_pct * len(dataset))
    train_ds, valid_ds = torch.utils.data.Subset(dataset, indices[:-test_size]), torch.utils.data.Subset(dataset, indices[-test_size:])

    # Create validset and trainset dataloaders
    print(f'> __debug__ == {__debug__} - Using {num_workers} workers in each DataLoader...')
    trainset = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    validset = DataLoader(valid_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True)

    return trainset, validset


def retrieve_data(img: Optional[torch.Tensor] = None, bbs: Optional[torch.Tensor] = None, colors: Optional[torch.Tensor] = None):
    # TODO: refactor data preprocessing/normalization and inverse transofrm (retreive_data / inference postprocessing)
    (retrieved_img, retrieved_bbs, retrieved_colors) = None, None, None

    if img is not None:
        retrieved_img = np.array(img.clone().detach().cpu()).reshape(3, 100, 100)
        retrieved_img *= (255 / retrieved_img.max(axis=(1, 2)))[:, np.newaxis, np.newaxis]
        retrieved_img = np.moveaxis(retrieved_img, 0, 2)
        retrieved_img = np.asarray(np.clip(np.round(retrieved_img), a_min=0, a_max=254), dtype=np.uint8)

    if colors is not None:
        colors = np.array(colors.clone().detach().cpu())
        retrieved_colors = np.zeros(colors.shape, dtype=np.bool)
        retrieved_colors[np.argsort(colors)[-3:]] = True

    if bbs is not None:
        retrieved_bbs = np.array(bbs.clone().detach().cpu()).reshape(3 if _REMOVE_ZERO_BBOX else 9, 4) * BBOX_SCALE[0]
        retrieved_bbs = np.clip(np.round(retrieved_bbs), a_min=0, a_max=100)
        if retrieved_colors is not None and _REMOVE_ZERO_BBOX:
            # Use colors to add missing empty/zero bounding boxes to bbs
            new_bbs = np.zeros((9, 4))
            new_bbs[retrieved_colors] = retrieved_bbs
            retrieved_bbs = new_bbs

    return retrieved_img, retrieved_bbs, retrieved_colors


class BallsCFDetection(Dataset):
    """ BallsCFDetection Pytorch dataset.
    .. class:: BallsCFDetection
    """

    def __init__(self, path: Path, img_transform: nn.Module = F.normalize, bbox_scale: List[List[List[float]]] = BBOX_SCALE, remove_zero_bbox: bool = _REMOVE_ZERO_BBOX):
        self.path = path
        self.img_transform = img_transform
        self.bbox_scale = bbox_scale
        self.remove_zero_bbox = remove_zero_bbox
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
        colors = torch.tensor(p_bb[:, 0], dtype=torch.float32)
        bbs = torch.tensor(p_bb[:, 1:5] / self.bbox_scale, dtype=torch.float32)

        # Remove zero vectors (remove information redoundancy) to simplify task to be learned by model (we don't need to infer ball colors twice, 'colors' vector will already be infered)
        if self.remove_zero_bbox:
            bbs = torch.tensor(bbs[:, colors != 0, :].clone().detach(), dtype=torch.float32)

        return img, colors, bbs

    # Return the dataset size
    def __len__(self):
        return self.image_count


class BallsCFSeq(Dataset):
    """ BallsCFSeq Pytorch dataset.
    .. class:: BallsCFSeq
    """

    def __init__(self, path: Path, bbox_scale: List[List[List[float]]] = BBOX_SCALE):
        self.path = path
        self.bbox_scale = bbox_scale
        self.seq_count = len(fnmatch.filter(os.listdir(path), 'seq_bb_*'))

    # The access is _NOT_ shuffled. The Dataloader will need to do this.
    def __getitem__(self, index):
        # Load presence and bounding boxes
        colors = np.load("%s/p_%05d.npy" % (self.path, index))
        bbs = np.load("%s/seq_bb_%05d.npy" % (self.path, index)) / self.bbox_scale

        # Remove zero vectors to simplify task to be learned by model (we don't need to infer ball color: 'p' color won't change during sequence)
        bbs = torch.tensor(bbs[:, colors != 0, :], dtype=torch.float32)

        # Return input bb sequence, target_bb (last balls positions) and colors vector
        return bbs[:-1], torch.tensor(colors, dtype=torch.float32), bbs[-1]

    # Return the dataset size
    def __len__(self):
        return self.seq_count
