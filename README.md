# TensorRT StreamPETR

- [x] Inference from bin files
- [x] Inference from bin files with position embedding computed online
- [] Inference from bin files with position embedding computed at the initial step
- [ ] Inference from ROS topics for NuScenes dataset

## Setup

### Step 1: Prepare ONNX

Follow [ONNX conversion instruction from official repository and NVIDIA](https://github.com/NVIDIA/DL4AGX/blob/9a4f60c2847d32e81372b9a2165299a3b65eabf1/AV-Solutions/streampetr-trt/conversion/README.md).

For StreamPETR repository, please use this one: https://github.com/kminoda/StreamPETR/blob/main/tools/pth2onnx.py

### Step 1.5: Prepare temporary bin files

Follow [the instruction from NVIDIA](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/streampetr-trt/inference_app#data-preparation) to produce the bin files from NuScenes beforehand. Note that this feature will soon be removed and will support input from ROS 2 topics instead.

### Step 2: Prepare this repository

```bash
git clone https://github.com/kminoda/StreamPETR_TensorRT_ROS2.git
cd StreamPETR_TensorRT_ROS2
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
```

Note that currently this repository heavily depends on [Autoware Universe](https://github.com/autowarefoundation/autoware.universe). Thus, you need to build and source the `setup.bash` beforehand. For this, please follow the instruction from Autoware.

## How to run

```bash
ros2 launch tensorrt_stream_petr tensorrt_stream_petr.launch.xml
```
