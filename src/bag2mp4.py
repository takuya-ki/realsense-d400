#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import numpy as np
import cv2
import os

FPS = 30
WIDTH = 1280
HEIGHT = 720


def scale_to_width(img, width):
    scale = width / img.shape[1]
    return cv2.resize(img, dsize=None, fx=scale, fy=scale)


def bag2mp4():
    
    # setting of fourcc (for mp4)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # specification of the video (file name, fourcc, FPS, size) 
    save_dir_path = os.path.join(os.path.dirname(__file__), '../data/mp4/')
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    mp4_path = os.path.join(save_dir_path, filename+'.mp4')
    video = cv2.VideoWriter(mp4_path, fourcc, FPS, (WIDTH, HEIGHT)) 

    # set the stream (color/depth/infrared)
    config = rs.config()
    # set the file name recorded
    bag_dir = os.path.join(os.path.dirname(__file__), '../data/bag/')
    config.enable_device_from_file(os.path.join(bag_dir, filename+'.bag'), 
                                   repeat_playback=False)
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

            # get depth image
            depth_color_frame = rs.colorizer().colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            # displaying
            images = np.hstack((color_image, depth_color_image))
            video.write(color_image) 
            dst_images = scale_to_width(images, 800)

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow('RealSense', 100, 200)
            cv2.imshow('RealSense', dst_images)

            key = cv2.waitKey(1)
            if key & 0xff == 27:
                break

    finally:
        # stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    filename = "record"
    bag2mp4()
