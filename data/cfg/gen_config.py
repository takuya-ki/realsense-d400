#!/usr/bin/env python3

import pickle
import numpy as np
import os.path as osp


def save_rs_cfg_pickle_d435(
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
        white,
        auto_exp,
        auto_exp_prior,
        backlight_comp,
        enable_auto_exp,
        enable_auto_white,
        global_time_enabled,
        power_line_freq):
    """Saves a configuration parameter file for realsense d435."""

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
        'WHITE_BALANCE': white,
        'AUTO_EXP_PRIOR': auto_exp_prior,
        'BACKLIGHT_COMP': backlight_comp,
        'ENABLE_AUTO_EXP': enable_auto_exp,
        'ENABLE_AUTO_WHITE': enable_auto_white,
        'GLOBAL_TIME_ENABLED': global_time_enabled,
        'POWER_LINE_FREQ': power_line_freq
    }
    save_path = osp.join(osp.dirname(__file__), pklname)
    with open(save_path, 'wb') as f:
        pickle.dump(rs_cfg_dict, f, protocol=4)


if __name__ == '__main__':
    save_rs_cfg_pickle_d435(
        pklname="rsd435.pkl",
        fps=30,
        w=1280,
        h=720,
        exp=200.0,  # setting large value (e.g. 5000), FPS goes down than it
        gain=0.0,
        bright=0.0,
        contrast=50.0,
        gamma=100.0,
        hue=0.0,
        sat=50.0,
        sharp=50.0,
        white=4500.0,
        auto_exp_prior=False,
        backlight_comp=False,
        enable_auto_exp=False,
        enable_auto_white=False,
        global_time_enabled=False,
        power_line_freq=False)
