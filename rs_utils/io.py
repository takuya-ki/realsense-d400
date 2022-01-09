#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import argparse


def get_options():
    parser = argparse.ArgumentParser(
        description='Set options for all devices.')
    parser.add_argument(
        '--save_type', dest='save_type',
        type=str, default='RGB', choices=['RGB', 'D', 'IR', 'RGBD', 'RGBDIR'],
        help='set save type, RGB or Depth or IR or RGBD or RGBDIR')
    parser.add_argument(
        '--is_rsopt', dest='is_rsopt', action='store_true',
        help='use custom realsense options?')
    parser.add_argument(
        '--rectime', dest='record_time', type=int, default=10,
        help="set recording time [sec]")
    parser.add_argument(
        '--bag_path', dest='bag_path', type=str,
        help='set path to bag file')
    parser.add_argument(
        '--indir', dest='indir', type=str,
        help='set path to input directory')
    parser.add_argument(
        '--outdir', dest='outdir', type=str,
        help='set path to output directory')
    parser.add_argument(
        '--cfg_path', dest='cfg_path',
        type=str, default='data/cfg/rsd435.pkl',
        help='set path to realsense config file')
    parser.add_argument(
        '--save_mode', dest='save_mode',
        type=str, default='snapshot', choices=['snapshot', 'one-scene', 'all'],
        help='set save mode for bag2img')
    parser.add_argument(
        '--save_fps', dest='save_fps',
        type=float, default=1.0,
        help='set save fps for bag2img')
    return parser.parse_args()


def get_file_paths(file_dir, file_ext, is_show=False):
    path = os.path.join(file_dir, '*.'+file_ext)
    file_paths = glob.glob(path)
    file_names = [os.path.splitext(os.path.basename(r))[0] for r in file_paths]
    if is_show:
        print(file_names)
        print(file_paths)
    return file_paths, file_names