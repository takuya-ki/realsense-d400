#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import argparse
import numpy as np
import pyrealsense2 as rs

from sensor_config import *
from sensor_utils import *


def bag2mp4(bag_path, mp4_path):
    # setting of fourcc (for mp4)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    
    # specification of the video (file name, fourcc, FPS, size) 
    video = cv2.VideoWriter(mp4_path, fourcc, FPS, (WIDTH, HEIGHT)) 

    # set the stream (color/depth/infrared)
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, FPS)
    # config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)

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
            # depth_frame = aligned_frames.get_depth_frame()

            # if not depth_frame or not color_frame:
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())
            # depth_image = np.asanyarray(depth_frame.get_data())

            # get depth image
            # depth_color_frame = rs.colorizer().colorize(depth_frame)
            # depth_color_image = np.asanyarray(depth_color_frame.get_data())

            # displaying
            # images = np.hstack((color_image, depth_color_image))
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            video.write(color_image) 
            images = color_image.copy()
            dst_images = scale_to_width(images, 800)

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow('RealSense', 100, 200)
            cv2.imshow('RealSense', dst_images)

            if cv2.waitKey(1) & 0xff == 27:
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

    # remove ext in the input file
    filename = os.path.splitext(os.path.basename(filename))[0]

    bag_dir_path = os.path.join(
        os.path.dirname(__file__), '../data/bag/')
    if not os.path.exists(bag_dir_path):
        os.makedirs(bag_dir_path)
    bag_file_path = os.path.join(bag_dir_path, filename+'.bag')

    mp4_dir_path = os.path.join(
        os.path.dirname(__file__), '../data/mp4/')
    if not os.path.exists(mp4_dir_path):
        os.makedirs(mp4_dir_path)
    mp4_file_path = os.path.join(mp4_dir_path, filename+'.mp4')

    bag2mp4(bag_path=bag_file_path, mp4_path=mp4_file_path)
