#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import time
import argparse
import numpy as np
import pyrealsense2 as rs

from sensor_config import *
from sensor_utils import *


def setting_sensor_params():
    ctx = rs.context()
    device_list = ctx.query_devices()
    num_devices = device_list.size()
    print(num_devices)
    assert num_devices > 0
    device = device_list[0]
    sensors = device.query_sensors()
    color_idx = -1
    for i in range(len(sensors)):
        if not sensors[i].is_depth_sensor():
            color_idx = i
            break
    assert color_idx != -1

    sensor_color = sensors[color_idx]
    sensor_color.set_option(rs.option.enable_auto_exposure, 0)
    sensor_color.set_option(rs.option.enable_auto_white_balance, 0)

    print("\nTrying to set exposure")
    exp = sensor_color.get_option(rs.option.exposure)
    print("Exposure = %d" % exp)
    print("Setting exposure to new value")
    sensor_color.set_option(rs.option.exposure, EXPOSURE)
    exp = sensor_color.get_option(rs.option.exposure)
    print("New exposure = %d" % exp)

    print("\nTrying to set white balance")
    wb = sensor_color.get_option(rs.option.white_balance)
    print("White balance = %d" % wb)
    print("Setting white balance to new value")
    sensor_color.set_option(rs.option.white_balance, WHITE_BALANCE)
    wb = sensor_color.get_option(rs.option.white_balance)
    print("New white balance = %d" % wb)


def record_bag(save_bagpath, record_sec):
    # set the stream (color/depth/infrared)
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.infrared, WIDTH, HEIGHT, rs.format.y8, FPS)
    config.enable_record_to_file(save_bagpath)

    # start streaming
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    
    # align depth image with color image
    align_to = rs.stream.color
    align = rs.align(align_to)

    # set fixed sensor parameters
    setting_sensor_params()

    time.sleep(1)
    start = time.time()
    try:
        while True:
            # waiting for a frame (Color & Depth)
            frames = pipeline.wait_for_frames()
            
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
            cv2.waitKey(1)

            # save for sec
            elapsed_time = time.time() - start
            if elapsed_time > record_sec:
                break

    finally:
        # stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("bagname", help="bag file name")
    parser.add_argument("-rs", "--record_time_sec", type=float, default=5.0,
                        help="time set for recording a bag")
    args = parser.parse_args()
    filename = args.bagname
    record_time_sec = args.record_time_sec
    bag_dir_path = os.path.join(os.path.dirname(__file__), '../data/bag/')

    # set the file name recorded
    if not os.path.exists(bag_dir_path):
        os.makedirs(bag_dir_path)

    # add ext for bag file
    if '.bag' not in filename:
        filename += '.bag'

    bag_path = os.path.join(bag_dir_path, filename)
    record_bag(save_bagpath=bag_path, record_sec=record_time_sec)
