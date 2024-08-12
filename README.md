# Run yolov5-sort on RK3588
![](Figures/video.gif)

## Installation
*Tips: If you want to integrate yolov5-sort with ROS/ROS2, using conda env is not recommended.* 

We use ROS2 Humble on Ubuntu 22.04 with default python version 3.10. You should be concerned about the conflict between different package versions.

### Step 1: Install [RKNN-Toolkit-Lite](https://github.com/airockchip/rknn-toolkit2) and Runtime package
```
cd Installation
pip install rknn_toolkit_lite2-1.5.2-cp310-cp310-linux_aarch64.whl
cp librknnrt.so /usr/lib
```

### Step 2: Install essential package for [Sort](https://github.com/abewley/sort/tree/master)
```
pip install -r requirements.txt
```
*If there are errors, follow the error instruction to install other packages*

### Step 3: Connect the stereo camera and run yolov5-sort on RK3588|RK3588s
```
cd ..
cd Code
python3 main.py
```
*If you want to check the NPU load, open another shell and run*
```
sudo bash rkcat.sh
```

### Parameters Setting
| Name  | Description |
| ----- | ----------- |
| OBJ_THRESH | Threshold for detecting |
| NMS_THRESH | Non-maximum Suppression |
| TPEs | The number of threads |
| modelPath | *.rknn* model |

## Reference
[rknn-multi-threaded](https://github.com/leafqycc/rknn-multi-threaded)\
[rknn-multi-threaded-3588](https://github.com/thanhtantran/rknn-multi-threaded-3588?tab=readme-ov-file)