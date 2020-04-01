# realsense-d400

RealSense D400 series utilities.

## Requirements

- Python 3.6.9 (recommended)
- opencv-contrib-python
- pyrealsense2
- numpy

## Usage

1. Prepare a sample bag file using record_bag.py  
`$ python src/record_bag.py record --record_time_sec 5.0`

2. To play the recorded bag and to display the images in the bag  
`$ python src/play_bag.py record`

3. To convert .bag into .mp4  
`$ python src/bag2mp4.py record`

4. To snap from bag file (set directory name instead of file name)
`$ python src/bagsnap.py bag`

## Author

[Takuya Kiyokawa](https://takuya-ki.github.io/)

## License

MIT
