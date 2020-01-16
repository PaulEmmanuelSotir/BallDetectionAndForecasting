#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Ball synthetic dataset - Deeplearning Session 1  
.. moduleauthor:: Fabien Baradel, Paul-Emmanuel Sotir, Christian Wolf  
.. See https://perso.liris.cnrs.fr/christian.wolf/teaching/deeplearning/tp.html and https://github.com/PaulEmmanuelSotir/BallDetectionAndForecasting  
"""
from PIL import Image, ImageDraw

import balldetect.torch_utils as tu
from balldetect.datasets import BallsCFDetection, COLORS

__all__ = ['show_img', 'show_bboxes']
__author__ = 'Fabien Baradel, Paul-Emmanuel Sotir, Christian Wolf'


def show_img(np_array_uint8, out_fn):
    if len(np_array_uint8.shape) == 3:
        img = Image.fromarray(np_array_uint8, 'RGB')
    elif len(np_array_uint8.shape) == 2:
        img = Image.fromarray(np_array_uint8)
    else:
        raise NameError('Unknown data type to show.')

    img.save(out_fn)
    img.show()


def show_bboxes(rgb_array, np_bbox, list_colors, out_fn='./bboxes_on_rgb.png'):
    """ Show the bounding box on a RGB image
    rgb_array: a np.array of shape (H,W,3) - it represents the rgb frame in uint8 type
    np_bbox: np.array of shape (9,4) and a bbox is of type [x1,y1,x2,y2]
    list_colors: list of string of length 9
    """
    assert np_bbox.shape[0] == len(list_colors)

    img_rgb = Image.fromarray(rgb_array, 'RGB')
    draw = ImageDraw.Draw(img_rgb)

    for i in range(len(list_colors)):
        color = COLORS[i]
        x_1, y_1, x_2, y_2 = np_bbox[i]
        draw.rectangle(((x_1, y_1), (x_2, y_2)), outline=color, fill=None)

    # img_rgb.show()  # TODO: make sure there is a runing graphical server before calling this?
    img_rgb.save(out_fn)


if __name__ == "__main__":
    dataset = BallsCFDetection(tu.source_dir() / r'../datasets/mini_balls/')

    # Get a single image from the dataset and display it
    img, pose, p = dataset.__getitem__(2)

    print(img.shape)
    print(pose.shape)

    show_bboxes(img, pose, COLORS, out_fn='_x.png')
