#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import numpy as np
import pyrealsense2 as rs

from sensor_config import *
from sensor_utils import *


def play_bag(bag_path):
    # set the stream (color/depth/infrared)
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

    # start streaming
    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    # align depth image with color image
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while True:
            # waiting for a frame (Color & Depth)
            try:
                frames = pipeline.wait_for_frames()
            except:
                pass

            # for alignment
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not depth_frame or not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # get depth image
            depth_color_frame = rs.colorizer().colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            # displaying
            images = np.hstack((color_image, depth_color_image))
            dst_images = scale_to_width(images, 800)

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow('RealSense', 100, 200)
            cv2.imshow('RealSense', dst_images)

            if cv2.waitKey(1) & 0xff == 27:
                png_dir_path = os.path.join(
                    bag_dir_path, "png"
                )
                if not os.path.exists(png_dir_path):
                    os.makedirs(png_dir_path)
                cv2.imwrite(os.path.join(
                    png_dir_path,
                    os.path.splitext(os.path.basename(bag_path))[0]+'_color.png'), 
                    color_image)
                cv2.imwrite(os.path.join(
                    png_dir_path, 
                    os.path.splitext(os.path.basename(bag_path))[0]+'_depth.png'), 
                    depth_color_image)
                break

    finally:
        # stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("bagname", help="bag file name")
    args = parser.parse_args()
    filename = args.bagname
    bag_dir_path = os.path.join(os.path.dirname(__file__), '../data/bag/')

    # set the file name recorded
    if not os.path.exists(bag_dir_path):
        os.makedirs(bag_dir_path)

    # add ext for bag file
    if '.bag' not in filename:
        filename += '.bag'

    bag_file_path = os.path.join(bag_dir_path, filename)
    play_bag(bag_path=bag_file_path)
