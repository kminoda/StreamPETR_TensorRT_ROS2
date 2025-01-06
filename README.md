# TensorRT StreamPETR

## Prerequisites
- OS: Ubuntu (only tested on 22.04)
- ROS 2 (only tested on ROS 2 Humble)
- TensorRT (only tested on 8.6.1)

## Setup

Clone this repository.
```bash
git clone https://github.com/kminoda/StreamPETR_TensorRT_ROS2.git
```

### Step 1: Prepare ONNX

> [!TIP]
> You may bypass this step by directly downloading ONNX file from [google drive](https://drive.google.com/drive/folders/1GgsCzlHh0W6_kbCiGf-sIUCcuR1ediyh?usp=sharing).

Follow [ONNX conversion instruction from official repository and NVIDIA](https://github.com/NVIDIA/DL4AGX/blob/9a4f60c2847d32e81372b9a2165299a3b65eabf1/AV-Solutions/streampetr-trt/conversion/README.md).

For StreamPETR repository, please use this one: [./conversion/pth2onnx.py](./conversion/pth2onnx.py)

You may also use a [Dockerfile](https://github.com/kminoda/StreamPETR/blob/main/Dockerfile) created for this project.

### Step 2: Prepare this repository

```bash
cd StreamPETR_TensorRT_ROS2
rosdep install --from-paths . -iry --rosdistro $ROS_DISTRO
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

## How to run

```bash
ros2 launch tensorrt_stream_petr tensorrt_stream_petr.launch.xml
```

## Useful scripts

### Play Nuscenes in ROS 2
You can use this script when launching the TensorRT StreamPETR node.
```bash
python3 scripts/nuscenes_to_ros.py --nuscenes_data_root <PATH_TO_YOUR_NUSCENES_DATASET>
```

### Visualization scripts
We have also provided some scripts for visualization for debugging purpose. Execute this alongside with the TensorRT StreamPETR node.

For 3D visualization on front camera:
```bash
python3 scripts/debug_visualize_3d.py
```

For BEV visualization:
```bash
python3 scripts/debug_visualize_bev.py
```
