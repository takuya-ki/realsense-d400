#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2

def scale_to_width(img, width):
    scale = width / img.shape[1]
    return cv2.resize(img, dsize=None, fx=scale, fy=scale)
