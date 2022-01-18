# realsense-d400

RealSense D400 series utilities

## Requirements

- Python 3.7.3
  - numpy 1.21.5
  - open3d 0.9.0.0
  - pickle-mixin 1.0.2
  - opencv-python 4.5.1.48
  - pyrealsense2 2.34.0.1470

## Installation

	$ git clone https://github.com/takuya-ki/realsense-d400.git; cd realsense-d400
	$ python setup.py install

## Usage

##### Recording a bag with specified options
    $ python record_bag.py --save_type [save_type] --is_rsopt --rectime 10 --bag_path [path_to_bag_file] --cfg_path [path_to_cfg_file]
    $ python record_bag.py --save_type RGBD --is_rsopt --rectime 5 --bag_path data/bag/record/record.bag --cfg_path data/cfg/rsd435.pkl

##### Playing a recorded bag
    $ python play_bag.py --save_type [save_type] --is_rsopt --bag_path [path_to_bag_file] --cfg_path [path_to_cfg_file]
    $ python play_bag.py --save_type RGBD --is_rsopt --bag_path data/bag/record.bag --cfg_path data/cfg/rsd435.pkl

##### Converting bag files to images
    $ python bag2img.py --save_type [save_type] --indir [path_to_bags_dir] --outdir [path_to_imgs_dir] --cfg_path [path_to_cfg_file] --save_mode [save_mode] (--save_fps [float less than 1.0])
    $ python bag2img.py --save_type RGBD --is_rsopt --indir data/bag/record --outdir data/img/record --cfg_path data/cfg/rsd435.pkl --save_mode one-scene --img_ext png

##### Converting bag files to videos
    $ python bag2video.py --save_type [save_type] --indir [path_to_bags_dir] --outdir [path_to_mp4s_dir] --cfg_path [path_to_cfg_file]
    $ python bag2video.py --save_type RGBD --is_rsopt --indir data/bag/record --outdir data/video/record --cfg_path data/cfg/rsd435.pkl --video_ext mp4

##### Capturing point clouds
    $ python capture_pcd.py --is_rsopt --pcd_path [path_to_pcd_file] --cfg_path [path_to_cfg_file]
    $ python capture_pcd.py --is_rsopt --pcd_path data/pcd/test.pcd --cfg_path data/cfg/rsd435.pkl

##### change camera configurations

1. Modify the parameters written in data/cfg/gen_config.py
2. Generate configuration file `$ python data/cfg/gen_config.py`

## Author / Contributor

[Takuya Kiyokawa](https://takuya-ki.github.io/)

## License

This software is released under the MIT License, see [LICENSE](./LICENSE).
