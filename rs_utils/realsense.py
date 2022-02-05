#!/usr/bin/env python3

import cv2
import time
import pickle
import numpy as np
import open3d as o3d
from enum import IntEnum
import pyrealsense2 as rs


class Preset(IntEnum):
    """Defines preset for setting realsense's resolution."""
    Custom = 0
    Default = 1
    Hand = 2
    HighAccuracy = 3
    HighDensity = 4
    MediumDensity = 5


class RealSenseD435(object):
    """Class of realsense d435 handler."""

    def __init__(self,
                 save_type,
                 rs_cfg_path,
                 custom_rs_options=False,
                 in_bag_path=None,
                 align_frames=True,
                 device_sn=None,
                 num_camera=1):
        """Initializes the camera configuration and pipeline."""

        self._save_type = save_type
        self._custom_rs_options = custom_rs_options
        with open(rs_cfg_path, 'rb') as f:
            self._rs_cfgs = pickle.load(f)
        self._in_bag_path = in_bag_path
        self._align_frames = align_frames
        self._device_sn = device_sn
        self._num_camera = num_camera
        self._pipelines = [rs.pipeline() for _ in range(num_camera)]
        self._realsenses = [rs.config() for _ in range(num_camera)]
        self._devices = [None for _ in range(num_camera)]
        for i in range(num_camera):
            self._rs_setup(i)

    def _rs_setup(self, tid):
        """Enables device and stream with the input specification."""
        if self._in_bag_path is not None:
            self._realsenses[0].enable_device_from_file(
                self._in_bag_path, repeat_playback=False)
        else:
            ctx = rs.context()
            dev_list = ctx.query_devices()
            num_devices = dev_list.size()
            assert num_devices > 0

            # select sensor devices used
            devnames = []
            devsns = []
            for d in dev_list:
                devnames.append(d.get_info(rs.camera_info.name))
                devsns.append(d.get_info(rs.camera_info.serial_number))
                print('found device: ', devnames[-1], ' ', devsns[-1])

            target_id = None
            if num_devices == 1:
                print("only found %s" % dev_list[0])
                target_id = 0
            else:
                if self._device_sn is not None:
                    for i in range(num_devices):
                        if devsns[i] == self._device_sn:
                            target_id = i
                    if target_id is None:
                        print("error: seleceted device not found")
                        return None
                else:
                    for i in range(num_devices):
                        print("input %3d: open %s %s"
                              % (i, devnames[i], devsns[i]))
                    print("input number of target device >> ", end="")
                    num = int(input())
                    target_id = num

            # enable target device
            self._devices[tid] = dev_list[target_id]
            targetdev = devnames[target_id]
            targetsn = devsns[target_id]
            print(' -> open device: ', targetdev, ' ', targetsn)
            self._realsenses[tid].enable_device(targetsn)

        if self._save_type in ['RGB', 'RGBD', 'RGBDIR']:
            self._realsenses[tid].enable_stream(
                rs.stream.color,
                self._rs_cfgs['WIDTH'],
                self._rs_cfgs['HEIGHT'],
                rs.format.bgr8,
                self._rs_cfgs['FPS'])
        if self._save_type in ['D', 'RGBD', 'RGBDIR']:
            self._realsenses[tid].enable_stream(
                rs.stream.depth,
                self._rs_cfgs['WIDTH'],
                self._rs_cfgs['HEIGHT'],
                rs.format.z16,
                self._rs_cfgs['FPS'])
        if self._save_type in ['IR', 'RGBDIR']:
            self._realsenses[tid].enable_stream(
                rs.stream.infrared,
                2,
                self._rs_cfgs['WIDTH'],
                self._rs_cfgs['HEIGHT'],
                rs.format.y8,
                self._rs_cfgs['FPS'])

    def _setting_sensor_params(self, tid):
        """Sets camera configuration options."""
        sensors = self._devices[tid].query_sensors()
        color_idx = -1
        for i in range(len(sensors)):
            if not sensors[i].is_depth_sensor():
                color_idx = i
                break
        assert color_idx != -1
        sensor_color = sensors[color_idx]

        if self._custom_rs_options:
            sensor_color.set_option(
                rs.option.exposure,
                self._rs_cfgs['EXPOSURE'])
            sensor_color.set_option(
                rs.option.gain,
                self._rs_cfgs['GAIN'])
            sensor_color.set_option(
                rs.option.brightness,
                self._rs_cfgs['BRIGHTNESS'])
            sensor_color.set_option(
                rs.option.contrast,
                self._rs_cfgs['CONTRAST'])
            sensor_color.set_option(
                rs.option.gamma,
                self._rs_cfgs['GAMMA'])
            sensor_color.set_option(
                rs.option.hue,
                self._rs_cfgs['HUE'])
            sensor_color.set_option(
                rs.option.saturation,
                self._rs_cfgs['SATURATION'])
            sensor_color.set_option(
                rs.option.sharpness,
                self._rs_cfgs['SHARPNESS'])
            sensor_color.set_option(
                rs.option.white_balance,
                self._rs_cfgs['WHITE_BALANCE'])
            sensor_color.set_option(
                rs.option.auto_exposure_priority,
                self._rs_cfgs['AUTO_EXP_PRIOR'])
            sensor_color.set_option(
                rs.option.backlight_compensation,
                self._rs_cfgs['BACKLIGHT_COMP'])
            sensor_color.set_option(
                rs.option.enable_auto_exposure,
                self._rs_cfgs['ENABLE_AUTO_EXP'])
            sensor_color.set_option(
                rs.option.enable_auto_white_balance,
                self._rs_cfgs['ENABLE_AUTO_WHITE'])
            sensor_color.set_option(
                rs.option.global_time_enabled,
                self._rs_cfgs['GLOBAL_TIME_ENABLED'])
            sensor_color.set_option(
                rs.option.power_line_frequency,
                self._rs_cfgs['POWER_LINE_FREQ'])

    def scale_to_width(self, img, width):
        """Resizes an OpenCV image with the specified image width."""
        scale = width / img.shape[1]
        return cv2.resize(img, dsize=None, fx=scale, fy=scale)

    def close(self):
        """Closes the realsense stream and image windows."""
        for p in self._pipelines:
            p.stop()
        cv2.destroyAllWindows()
        print("finish pipeline for realsense")

    def show_frames(self, end_time=None):
        """Gets and shows frames obtained from the stream."""
        # align depth image with color image
        if self._align_frames:
            align_to = rs.stream.color
            align = rs.align(align_to)

        if end_time is not None:
            start = time.time()
        try:
            while True:
                dst_image_list = []
                for i in range(self._num_camera):
                    # waiting for a frame (Color & Depth)
                    try:
                        frames = self._pipelines[i].wait_for_frames(5000)
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
                    if (self._save_type in ['D', 'RGBD', 'RGBDIR']) \
                    and not depth_frame:
                        continue
                    if (self._save_type in ['IR', 'RGBDIR']) \
                    and not infrared_frame:
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
                    dst_image = self.scale_to_width(images, 800)
                    dst_image_list.append(dst_image)

                # displaying
                dst_images = np.hstack(dst_image_list)
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow('RealSense', 100, 200)
                cv2.imshow('RealSense', dst_images)
                if cv2.waitKey(1) & 0xff == 27:
                    break
                if end_time is not None:
                    # save for sec
                    elapsed_time = time.time() - start
                    if elapsed_time > end_time:
                        break
        finally:
            # stop streaming
            self.close()

    def start_saving_bag(self, out_bag_path, tid):
        """Starts the pipeline to save a bag file."""
        # set the file name recorded
        self._realsenses[tid].enable_record_to_file(out_bag_path)
        # start streaming
        self._pipelines[tid].start(self._realsenses[tid])
        # set fixed sensor parameters
        self._setting_sensor_params(tid)
        time.sleep(1)
        print("start pipeline for realsense...")

    def record_bag(self, out_bag_path, rec_time, isShow=False):
        """Starts recording to save a bag file."""
        try:
            # capture the image not including an object
            for i in range(self._num_camera):
                self.start_saving_bag(out_bag_path, i)
            if isShow:
                self.show_frames(end_time=rec_time)
        except KeyboardInterrupt:
            return

    def play_bag(self):
        """Starts the pipeline."""
        # start streaming
        profile = self._pipelines[0].start(self._realsenses[0])
        playback = profile.get_device().as_playback()
        playback.set_real_time(True)
        self.show_frames()

    def bag2img(self,
                save_img_path_noext,
                save_ext='png',
                mode="snapshot",
                fps=0.5,
                is_show=False):
        """Converts a bag file to images."""

        # start streaming
        profile = self._pipelines[0].start(self._realsenses[0])
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
                    frames = self._pipelines[0].wait_for_frames(5000)
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
                save_img_path = save_img_path_noext+'_'+str(cnt)+'.'+save_ext
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
            self._pipelines[0].stop()
            cv2.destroyAllWindows()

    def bag2video(self,
                  save_video_path,
                  save_ext='mp4',
                  is_show=False):
        """Converts a bag file to a video file."""

        if save_ext == 'mp4':
            fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
        elif save_ext == 'wmv':
            fourcc = cv2.VideoWriter_fourcc('w', 'm', 'v', '1')
        elif save_ext == 'mov':
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        elif save_ext == 'avi':
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

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
        profile = self._pipelines[0].start(self._realsenses[0])
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
                    frames = self._pipelines[0].wait_for_frames(5000)
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
            self._pipelines[0].stop()
            cv2.destroyAllWindows()

    def _get_intrinsic_matrix(self, frame):
        """Gets camera intrinsic parameters from the stream."""
        intr = frame.profile.as_video_stream_profile().intrinsics
        out = o3d.camera.PinholeCameraIntrinsic(
            self._rs_cfgs['WIDTH'], self._rs_cfgs['HEIGHT'],
            intr.fx, intr.fy, intr.ppx, intr.ppy)
        return out

    def capture_pcd(self, pcdpath):
        """Captures point cloud created from RGBD frames."""

        self._pcdpath = pcdpath
        self._return_cmd = False
        def save_pcd(_vis):
            """Saves point cloud data."""
            o3d.io.write_point_cloud(self._pcdpath, self._pcd)
        def return_with_q(_vis):
            """Finishes capturing."""
            o3d.io.write_point_cloud(self._pcdpath, self._pcd)
            self._return_cmd = True
        flip_transform = [
            [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        # Create an align object
        align_to = rs.stream.depth
        align = rs.align(align_to)

        depth_scales = []
        clipping_distances = []
        vises = [o3d.visualization.VisualizerWithKeyCallback()
                 for _ in range(self._num_camera)]
        for i in range(self._num_camera):
            # start streaming
            profile = self._pipelines[i].start(self._realsenses[i])
            depth_sensor = profile.get_device().first_depth_sensor()
            # using preset HighAccuracy for recording
            depth_sensor.set_option(
                rs.option.visual_preset, Preset.HighAccuracy)
            # set fixed sensor parameters
            self._setting_sensor_params(i)
            time.sleep(1)
            # getting depth scale (see rs-align example for explanation)
            depth_scales.append(depth_sensor.get_depth_scale())
            # not display the background of objects more than
            # clipping_distance_in_meters meters away
            clipping_distance_in_meters = 5  # meter
            clipping_distances.append(clipping_distance_in_meters / depth_scales[-1])

            vises[i].create_window()
            vises[i].register_key_callback(ord("S"), save_pcd)
            vises[i].register_key_callback(ord("Q"), return_with_q)

        self._pcd = [o3d.geometry.PointCloud()
                     for _ in range(self._num_camera)]
        # streaming loop
        geometry_added = False
        start = time.time()
        try:
            while True:
                for i in range(self._num_camera):
                    try:
                        frames = self._pipelines[i].wait_for_frames(5000)
                        frames = align.process(frames)
                    except RuntimeError:
                        break

                    # Get aligned frames
                    depth_frame = frames.get_depth_frame()
                    color_frame = frames.get_color_frame()
                    intrinsic = o3d.camera.PinholeCameraIntrinsic(
                        self._get_intrinsic_matrix(color_frame))

                    # Validate that both frames are valid
                    if not depth_frame or not color_frame:
                        continue

                    depth_image = o3d.geometry.Image(
                        np.array(depth_frame.get_data()))
                    color_temp = np.asarray(color_frame.get_data())
                    color_image = o3d.geometry.Image(
                        cv2.cvtColor(color_temp, cv2.COLOR_BGR2RGB))

                    rgbd_image = \
                        o3d.geometry.RGBDImage.create_from_color_and_depth(
                            color_image,
                            depth_image,
                            depth_scale=1.0 / depth_scales[i],
                            depth_trunc=clipping_distances[i],
                            convert_rgb_to_intensity=False)
                    temp = o3d.geometry.PointCloud.create_from_rgbd_image(
                        rgbd_image, intrinsic)
                    temp.transform(flip_transform)
                    self._pcd.points = temp.points
                    self._pcd.colors = temp.colors

                    if not geometry_added:
                        vises[i].add_geometry(self._pcd)
                        geometry_added = True
                    vises[i].update_geometry(self._pcd)
                    vises[i].poll_events()
                    vises[i].update_renderer()

                    cv2.imshow('bgr', color_temp)
                    cv2.waitKey(1)
                elapsed_time = time.time() - start
                print("FPS: " + str(1.0 / elapsed_time))

                if self._return_cmd:
                    break

        finally:
            # stop streaming
            self.close()
            for vis in vises:
                vis.destroy_window()
