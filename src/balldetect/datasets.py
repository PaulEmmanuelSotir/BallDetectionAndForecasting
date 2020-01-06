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

import torch
from torch.utils.data import Dataset

__all__ = ['COLORS', 'BallsCFDetection', 'BallsCFSeq']

COLORS = ['red', 'green', 'blue', 'yellow', 'lime', 'purple', 'orange', 'cyan', 'magenta']
BB_SEQ_SPLIT_INDEX = 5


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
        bb = torch.tensor(p_bb[:, 1:5] / self.bbox_scale, dtype=torch.float32)

        return img, p, bb

    # Return the dataset size
    def __len__(self):
        return self.image_count


class BallsCFSeq(Dataset):
    """ BallsCFSeq Pytorch dataset.
    .. class:: BallsCFSeq
    """

    def __init__(self, path, seq_count):
        self.path = path
        self.seq_count = seq_count

    # The access is _NOT_ shuffled. The Dataloader will need
    # to do this.
    def __getitem__(self, index):
        # Load presence
        p = np.load("%s/p_%05d.npy" % (self.path, index))

        # Load bounding boxes
        bb = np.load("%s/seq_bb_%05d.npy" % (self.path, index))

        # split bounding boxes and create tensors from data
        return torch.tensor([p, bb[:BB_SEQ_SPLIT_INDEX]]), torch.tensor(bb[BB_SEQ_SPLIT_INDEX:])

    # Return the dataset size
    def __len__(self):
        return self.seq_count
