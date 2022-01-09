#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import time
import pickle
import numpy as np
import pyrealsense2 as rs


class RealSenseD435(object):

    def __init__(self,
                 save_type,
                 rs_cfg_path,
                 custom_rs_options=False,
                 in_bag_path=None,
                 align_frames=True):

        self._save_type = save_type
        self._custom_rs_options = custom_rs_options
        with open(rs_cfg_path, 'rb') as f:
            self._rs_cfgs = pickle.load(f)
        self._in_bag_path = in_bag_path
        self._align_frames = align_frames
        self._realsense = rs.config()
        self._rs_setup()
        self._pipeline = rs.pipeline()

    def _rs_setup(self):
        if self._in_bag_path is not None:
            self._realsense.enable_device_from_file(
                self._in_bag_path, repeat_playback=False)
        if self._save_type in ['RGB', 'RGBD', 'RGBDIR']:
            self._realsense.enable_stream(
                rs.stream.color,
                self._rs_cfgs['WIDTH'],
                self._rs_cfgs['HEIGHT'],
                rs.format.bgr8,
                self._rs_cfgs['FPS'])
        if self._save_type in ['D', 'RGBD', 'RGBDIR']:
            self._realsense.enable_stream(
                rs.stream.depth,
                self._rs_cfgs['WIDTH'],
                self._rs_cfgs['HEIGHT'],
                rs.format.z16,
                self._rs_cfgs['FPS'])
        if self._save_type in ['IR', 'RGBDIR']:
            self._realsense.enable_stream(
                rs.stream.infrared,
                2,
                self._rs_cfgs['WIDTH'],
                self._rs_cfgs['HEIGHT'],
                rs.format.y8,
                self._rs_cfgs['FPS'])

    def _setting_sensor_params(self):
        ctx = rs.context()
        device_list = ctx.query_devices()
        num_devices = device_list.size()
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

        if self._custom_rs_options:
            sensor_color.set_option(
                rs.option.enable_auto_exposure, False)
            sensor_color.set_option(
                rs.option.enable_auto_white_balance, False)
            sensor_color.set_option(
                rs.option.exposure, self._rs_cfgs['EXPOSURE'])
            sensor_color.set_option(
                rs.option.gain, self._rs_cfgs['GAIN'])
            sensor_color.set_option(
                rs.option.brightness, self._rs_cfgs['BRIGHTNESS'])
            sensor_color.set_option(
                rs.option.contrast, self._rs_cfgs['CONTRAST'])
            sensor_color.set_option(
                rs.option.gamma, self._rs_cfgs['GAMMA'])
            sensor_color.set_option(
                rs.option.hue, self._rs_cfgs['HUE'])
            sensor_color.set_option(
                rs.option.saturation, self._rs_cfgs['SATURATION'])
            sensor_color.set_option(
                rs.option.sharpness, self._rs_cfgs['SHARPNESS'])
            sensor_color.set_option(
                rs.option.white_balance, self._rs_cfgs['WHITE_BALANCE'])
        else:
            sensor_color.set_option(
                rs.option.enable_auto_exposure, True)
            sensor_color.set_option(
                rs.option.enable_auto_white_balance, True)

    def scale_to_width(self, img, width):
        scale = width / img.shape[1]
        return cv2.resize(img, dsize=None, fx=scale, fy=scale)

    def close(self):
        self._pipeline.stop()
        cv2.destroyAllWindows()
        print("finish pipeline for realsense")

    def start_saving_bag(self, out_bag_path):
        # set the file name recorded
        self._realsense.enable_record_to_file(out_bag_path)
        # start streaming
        self._pipeline.start(self._realsense)
        # set fixed sensor parameters
        self._setting_sensor_params()
        time.sleep(1)
        print("start pipeline for realsense...")

    def record_bag(self, out_bag_path, rec_time, isShow=False):
        try:
            # capture the image not including an object
            self.start_saving_bag(out_bag_path)
            start = time.time()
            try:
                while True:
                    if isShow:
                        # waiting for a frame (Color & Depth)
                        frames = self._pipeline.wait_for_frames()
                        color_frame = frames.get_color_frame()
                        depth_frame = frames.get_depth_frame()
                        if not depth_frame or not color_frame:
                            continue
                        color_image = np.asanyarray(color_frame.get_data())
                        depth_color_frame = \
                            rs.colorizer().colorize(depth_frame)
                        depth_color_image = \
                            np.asanyarray(depth_color_frame.get_data())
                        images = np.hstack((color_image, depth_color_image))
                        dst_images = self.scale_to_width(images, 1200)

                        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                        cv2.moveWindow('RealSense', 50, 100)
                        cv2.imshow('RealSense', dst_images)
                        cv2.waitKey(1)
                    # save for sec
                    elapsed_time = time.time() - start
                    if elapsed_time > rec_time:
                        break
            finally:
                # stop streaming
                self.close()
        except KeyboardInterrupt:
            return

    def play_bag(self):
        # start streaming
        profile = self._pipeline.start(self._realsense)
        playback = profile.get_device().as_playback()
        playback.set_real_time(True)

        # align depth image with color image
        if self._align_frames:
            align_to = rs.stream.color
            align = rs.align(align_to)

        try:
            while True:
                # waiting for a frame (Color & Depth)
                try:
                    frames = self._pipeline.wait_for_frames(5000)
                    if self._align_frames:
                        frames = align.process(frames)
                except RuntimeError:
                    break
                color_frame = frames.get_color_frame()

                if self._save_type in ['D', 'RGBD', 'RGBDIR']:
                    depth_frame = frames.get_depth_frame()
                if self._save_type in ['IR', 'RGBDIR']:
                    infrared_frame = frames.get_infrared_frame()

                if not color_frame:
                    continue
                if (self._save_type in ['D', 'RGBD', 'RGBDIR']) and \
                   not depth_frame:
                    continue
                if (self._save_type in ['IR', 'RGBDIR']) and \
                   not infrared_frame:
                    continue
                images = np.asanyarray(color_frame.get_data())
                if self._save_type in ['D', 'RGBD', 'RGBDIR']:
                    depth_color_frame = rs.colorizer().colorize(depth_frame)
                    depth_color_image = np.asanyarray(
                        depth_color_frame.get_data())
                    images = np.hstack((images, depth_color_image))
                if self._save_type in ['IR', 'RGBDIR']:
                    infrared_image = np.asanyarray(
                        infrared_frame.get_data())
                    infrared_3c_image = cv2.cvtColor(
                        infrared_image, cv2.COLOR_GRAY2BGR)
                    images = np.hstack((images, infrared_3c_image))

                # displaying
                dst_images = self.scale_to_width(images, 800)
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow('RealSense', 100, 200)
                cv2.imshow('RealSense', dst_images)
                if cv2.waitKey(1) & 0xff == 27:
                    break

        finally:
            # stop streaming
            self._pipeline.stop()
            cv2.destroyAllWindows()

    # mode = "one-scene" or "snapshot"
    # fps = float value
    def bag2img(self,
                save_img_path_noext,
                mode="snapshot",
                fps=0.5,
                is_show=False):
        # start streaming
        profile = self._pipeline.start(self._realsense)
        playback = profile.get_device().as_playback()
        playback.set_real_time(True)

        # align depth image with color image
        if self._align_frames:
            align_to = rs.stream.color
            align = rs.align(align_to)

        cnt = 0
        start = time.time()
        prev_time = 0
        stack_diff = 0
        try:
            while True:
                # waiting for a frame (Color & Depth)
                try:
                    frames = self._pipeline.wait_for_frames(5000)
                    if self._align_frames:
                        frames = align.process(frames)
                except RuntimeError:
                    break

                color_frame = frames.get_color_frame()
                if self._save_type in ['D', 'RGBD', 'RGBDIR']:
                    depth_frame = frames.get_depth_frame()
                if self._save_type in ['IR', 'RGBDIR']:
                    infrared_frame = frames.get_infrared_frame()

                if not color_frame:
                    continue
                if (self._save_type in ['D', 'RGBD', 'RGBDIR']) and \
                   not depth_frame:
                    continue
                if (self._save_type in ['IR', 'RGBDIR']) and \
                   not infrared_frame:
                    continue
                images = np.asanyarray(color_frame.get_data())
                if self._save_type in ['D', 'RGBD', 'RGBDIR']:
                    depth_color_frame = rs.colorizer().colorize(depth_frame)
                    depth_color_image = np.asanyarray(
                        depth_color_frame.get_data())
                    images = np.hstack((images, depth_color_image))
                if self._save_type in ['IR', 'RGBDIR']:
                    infrared_image = np.asanyarray(
                        infrared_frame.get_data())
                    infrared_3c_image = cv2.cvtColor(
                        infrared_image, cv2.COLOR_GRAY2BGR)
                    images = np.hstack((images, infrared_3c_image))

                # displaying
                if is_show:
                    dst_images = self.scale_to_width(images, 800)
                    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                    cv2.moveWindow('RealSense', 100, 200)
                    cv2.imshow('RealSense', dst_images)
                    cv2.waitKey(0)

                # saving
                save_img_path = save_img_path_noext+'_'+str(cnt)+'.png'
                if self._save_type == 'D':
                    if depth_color_image is None:
                        continue
                    save_img = depth_color_image.copy()
                elif self._save_type == 'IR':
                    if infrared_3c_image is None:
                        continue
                    save_img = infrared_3c_image.copy()
                elif self._save_type in ['RGB', 'RGBD', 'RGBDIR']:
                    if images is None:
                        continue
                    save_img = images.copy()
                if mode == "one-scene":
                    cv2.imwrite(save_img_path, save_img)
                    break
                elif mode == "snapshot":
                    elapsed_time = time.time() - start
                    stack_diff += elapsed_time - prev_time
                    if stack_diff > 1.0/float(fps):
                        print("Save image in " + save_img_path +
                              " at " + str(elapsed_time))
                        cv2.imwrite(save_img_path, save_img)
                        stack_diff = 0
                    prev_time = elapsed_time
                else:
                    cv2.imwrite(save_img_path, save_img)
                cnt += 1

        finally:
            # stop streaming
            self._pipeline.stop()
            cv2.destroyAllWindows()

    def bag2mp4(self, save_video_path, is_show=False):
        # setting of fourcc (for mp4)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        # specification of the video (file name, fourcc, FPS, size)
        if self._save_type in ['RGB', 'D', 'IR']:
            video_width = self._rs_cfgs['WIDTH']
        elif self._save_type == 'RGBD':
            video_width = self._rs_cfgs['WIDTH']*2
        elif self._save_type == 'RGBDIR':
            video_width = self._rs_cfgs['WIDTH']*3

        video = cv2.VideoWriter(
            save_video_path,
            fourcc,
            20,
            (video_width, self._rs_cfgs['HEIGHT']))

        # start streaming
        profile = self._pipeline.start(self._realsense)
        playback = profile.get_device().as_playback()
        playback.set_real_time(True)

        # align depth image with color image
        if self._align_frames:
            align_to = rs.stream.color
            align = rs.align(align_to)

        try:
            while True:
                # waiting for a frame (Color & Depth)
                try:
                    frames = self._pipeline.wait_for_frames(5000)
                    if self._align_frames:
                        frames = align.process(frames)
                except RuntimeError:
                    break

                color_frame = frames.get_color_frame()
                if self._save_type in ['D', 'RGBD', 'RGBDIR']:
                    depth_frame = frames.get_depth_frame()
                if self._save_type in ['IR', 'RGBDIR']:
                    infrared_frame = frames.get_infrared_frame()

                if not color_frame:
                    continue
                if (self._save_type in ['D', 'RGBD', 'RGBDIR']) and \
                   not depth_frame:
                    continue
                if (self._save_type in ['IR', 'RGBDIR']) and \
                   not infrared_frame:
                    continue
                images = np.asanyarray(color_frame.get_data())
                if self._save_type in ['D', 'RGBD', 'RGBDIR']:
                    depth_color_frame = rs.colorizer().colorize(depth_frame)
                    depth_color_image = np.asanyarray(
                        depth_color_frame.get_data())
                    images = np.hstack((images, depth_color_image))
                if self._save_type in ['IR', 'RGBDIR']:
                    infrared_image = np.asanyarray(
                        infrared_frame.get_data())
                    infrared_3c_image = cv2.cvtColor(
                        infrared_image, cv2.COLOR_GRAY2BGR)
                    images = np.hstack((images, infrared_3c_image))

                # displaying
                if is_show:
                    dst_images = self.scale_to_width(images, 800)
                    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                    cv2.moveWindow('RealSense', 100, 200)
                    cv2.imshow('RealSense', dst_images)
                if cv2.waitKey(1) & 0xff == 27:
                    break

                # saving
                if self._save_type == 'D':
                    if depth_color_image is not None:
                        video.write(depth_color_image)
                elif self._save_type == 'IR':
                    if infrared_3c_image is not None:
                        video.write(infrared_3c_image)
                elif self._save_type in ['RGB', 'RGBD', 'RGBDIR']:
                    if images is not None:
                        video.write(images)

        finally:
            # stop streaming
            self._pipeline.stop()
            cv2.destroyAllWindows()
