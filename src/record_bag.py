#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

RECORD_TIME_SEC = 20.0

FPS = 30
WIDTH = 1280
HEIGHT = 720


def record_bag():
    # set the stream (color/depth/infrared)
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.infrared, WIDTH, HEIGHT, rs.format.y8, FPS)

    # set the file name recorded
    save_dir_path = os.path.join(os.path.dirname(__file__), '../data/bag/')
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    filepath = os.path.join(save_dir_path, filename+'.bag')
    config.enable_record_to_file(filepath)

    # start streaming
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    
    # align depth image with color image
    align_to = rs.stream.color
    align = rs.align(align_to)

    sensor_dep = profile.get_device().first_depth_sensor()
    print("Trying to set Exposure")
    exp = sensor_dep.get_option(rs.option.exposure)
    print("exposure = %d" % exp)
    print("Setting exposure to new value")
    sensor_dep.set_option(rs.option.exposure, 25000)
    exp = sensor_dep.get_option(rs.option.exposure)
    print("New exposure = %d" % exp)

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
            # get depth image
            depth_color_frame = rs.colorizer().colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            # displaying
            images = np.hstack((cv2.resize(color_image, 
                                           (640, 360), 
                                           interpolation=cv2.INTER_LINEAR), \
                                cv2.resize(depth_color_image, 
                                           (640, 360), 
                                           interpolation=cv2.INTER_LINEAR) ))
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)

            # save for sec
            elapsed_time = time.time() - start
            if elapsed_time > RECORD_TIME_SEC:
                break

    finally:
        # stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    filename = "record"
    record_bag()
