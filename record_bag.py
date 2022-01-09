#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from rs_utils.io import get_options
from rs_utils.realsense import RealSenseD435


if __name__ == '__main__':
    args = get_options()
    save_type = args.save_type
    custom_rs_options = args.is_rsopt
    rec_time = args.record_time
    save_bag_path = os.path.join(
        os.path.dirname(__file__),
        args.bag_path)
    cfg_path = os.path.join(
        os.path.dirname(__file__),
        args.cfg_path)

    rs_d435 = RealSenseD435(
        save_type,
        cfg_path,
        custom_rs_options)
    rs_d435.record_bag(save_bag_path, rec_time)
    print("saved "+save_bag_path)
