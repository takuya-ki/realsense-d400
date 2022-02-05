#!/usr/bin/env python3

import open3d as o3d
import os.path as osp

from rs_utils.io import get_options
from rs_utils.realsense import RealSenseD435


def rotate_view(vis):
    """Rotates the view used in Open3D."""
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)
    return False


if __name__ == '__main__':
    args = get_options()

    custom_rs_options = args.is_rsopt
    save_pcd_path = osp.join(osp.dirname(__file__), args.pcd_path)
    cfg_path = osp.join(osp.dirname(__file__), args.cfg_path)

    rs_d435 = RealSenseD435(
        'RGBD',
        cfg_path,
        custom_rs_options)
    rs_d435.capture_pcd(save_pcd_path)
    print("saved "+save_pcd_path)

    # visualize saved pcd file
    pcd = o3d.io.read_point_cloud(save_pcd_path)
    o3d.visualization.draw_geometries_with_animation_callback(
        [pcd], rotate_view)
