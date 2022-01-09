#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import numpy as np


def make_rs_cfg_pickle(
        pklname,
        fps,
        w,
        h,
        exp,
        gain,
        bright,
        contrast,
        gamma,
        hue,
        sat,
        sharp,
        white):
    rs_cfg_dict = {
        'FPS': fps,
        'WIDTH': w,
        'HEIGHT': h,
        'EXPOSURE': exp,
        'GAIN': gain,
        'BRIGHTNESS': bright,
        'CONTRAST': contrast,
        'GAMMA': gamma,
        'HUE': hue,
        'SATURATION': sat,
        'SHARPNESS': sharp,
        'WHITE_BALANCE': white
    }
    save_path = os.path.join(os.path.dirname(__file__), pklname)
    with open(save_path, 'wb') as f:
        pickle.dump(rs_cfg_dict, f, protocol=4)


if __name__ == '__main__':
    make_rs_cfg_pickle(
        pklname="rsd435.pkl",
        fps=30,
        w=1280,
        h=720,
        exp=200,  # setting large value (e.g. 5000), FPS goes down than it
        gain=0,
        bright=-10,
        contrast=50,
        gamma=500,
        hue=0,
        sat=50,
        sharp=50,
        white=4500
    )
