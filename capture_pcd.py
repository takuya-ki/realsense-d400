#!/usr/bin/env python3

import os
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
    save_pcd_paths = [
        osp.join(osp.dirname(__file__),
                 osp.splitext(args.pcd_path)[0] +
                 '_' + str(i) + '.pcd')
        for i in range(args.num_camera)]
    cfg_path = osp.join(osp.dirname(__file__), args.cfg_path)
    os.makedirs(os.path.dirname(args.pcd_path), exist_ok=True)

    rs_d435 = RealSenseD435(
        'RGBD',
        cfg_path,
        custom_rs_options,
        device_sn=args.device_sn,
        num_camera=args.num_camera)
    rs_d435.capture_pcd(save_pcd_paths)

    for i, savepcdpath in enumerate(save_pcd_paths):
        print("displaying " + savepcdpath)
        # visualize saved pcd file
        pcd = o3d.io.read_point_cloud(save_pcd_paths[i])
        o3d.visualization.draw_geometries_with_animation_callback(
            [pcd], rotate_view)
