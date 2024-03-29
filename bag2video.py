#!/usr/bin/env python3

import os
import os.path as osp

from rs_utils.io import get_options, get_file_paths
from rs_utils.realsense import RealSenseD435


if __name__ == '__main__':
    args = get_options()

    save_type = args.save_type
    bags_dir = osp.join(osp.dirname(__file__), args.indir)
    cfg_path = osp.join(osp.dirname(__file__), args.cfg_path)
    save_dir = osp.join(osp.dirname(__file__), args.outdir)

    if not osp.exists(bags_dir):
        print("Not found directory for bag files...")
        exit()
    os.makedirs(save_dir, exist_ok=True)

    bag_paths, bag_names = get_file_paths(bags_dir, 'bag')
    save_ext = args.videoext
    for (bag_path, bag_name) in zip(bag_paths, bag_names):
        save_video_path = osp.join(save_dir, bag_name+'.'+save_ext)
        rs_d435 = RealSenseD435(
            save_type=save_type,
            rs_cfg_path=cfg_path,
            in_bag_path=bag_path)
        rs_d435.bag2video(
            save_video_path,
            save_ext,
            is_show=True)
