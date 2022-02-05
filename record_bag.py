#!/usr/bin/env python3

import os
import os.path as osp

from rs_utils.io import get_options
from rs_utils.realsense import RealSenseD435


if __name__ == '__main__':
    args = get_options()

    save_type = args.save_type
    custom_rs_options = args.is_rsopt
    rec_time = args.record_time
    save_bag_paths = [
        osp.join(osp.dirname(__file__),
                 osp.splitext(args.bag_path)[0] +
                 '_' + str(i) + '.bag')
        for i in range(args.num_camera)]
    cfg_path = osp.join(osp.dirname(__file__), args.cfg_path)
    os.makedirs(os.path.dirname(args.bag_path), exist_ok=True)

    rs_d435 = RealSenseD435(
        save_type,
        cfg_path,
        custom_rs_options,
        device_sn=args.device_sn,
        num_camera=args.num_camera)
    rs_d435.record_bag(save_bag_paths, rec_time, isShow=True)
    for savebagpath in save_bag_paths:
        print("saved " + savebagpath)
