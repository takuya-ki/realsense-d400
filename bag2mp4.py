#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import rs_utils.io as ioutil
import rs_utils.realsense as rsutil


if __name__ == '__main__':
    args = ioutil.get_options()
    save_type = args.save_type

    bags_path = args.indir
    bags_dir = os.path.join(
        os.path.dirname(__file__),
        bags_path)
    if not os.path.exists(bags_dir):
        print("Not found directory for bag files...")
        exit()

    cfg_path = os.path.join(
        os.path.dirname(__file__),
        args.cfg_path)

    save_path = args.outdir
    save_dir = os.path.join(
        os.path.dirname(__file__),
        save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    bag_paths, bag_names = ioutil.get_file_paths(bags_dir, 'bag')
    for (bag_path, bag_name) in zip(bag_paths, bag_names):
        save_video_path = os.path.join(save_dir, bag_name+'.mp4')
        rs_d435 = rsutil.RealSenseD435(
            save_type=save_type,
            rs_cfg_path=cfg_path,
            in_bag_path=bag_path)
        rs_d435.bag2mp4(save_video_path, is_show=True)
