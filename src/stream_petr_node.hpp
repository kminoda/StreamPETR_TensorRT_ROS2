// Copyright 2024 Koji Minoda
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORRT_STREAM_PETR__STREAM_PETR_NODE_HPP__
#define TENSORRT_STREAM_PETR__STREAM_PETR_NODE_HPP__


#include <image_transport/image_transport.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <map>

#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_msgs/msg/tf_message.hpp>

// From NVIDIA/DL4AGX
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <unordered_map>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <NvInferRuntime.h>
#include "memory.cuh"
// From NVIDIA/DL4AGX

#include "camera_data_store.hpp"

namespace tensorrt_stream_petr
{
// Taken from NVIDIA/DL4AGX
#define LASTERR()   { \
  auto code = cudaGetLastError(); \
  if( code != cudaSuccess) { \
    std::cout << cudaGetErrorString(code) << std::endl; \
  } \
}

using namespace nvinfer1;

class Logger : public ILogger {
public:
void log(Severity severity, const char* msg) noexcept override {
  // Only print error messages
  if (severity == Severity::kERROR) {
    std::cerr << msg << std::endl;
  }
}
};

inline unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kFP8:
      throw std::logic_error("Unsupported DataType encountered: " + std::to_string(static_cast<int>(t)));
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

struct Tensor {
  std::string name;
  void* ptr;
  Dims dim;
  int32_t volume = 1;
  DataType dtype;
  TensorIOMode iomode;

  Tensor(std::string name, Dims dim, DataType dtype): 
    name(name), dim(dim), dtype(dtype) 
  {
    if( dim.nbDims == 0 ) {
      volume = 0;
    } else {
      volume = 1;
      for(int i=0; i<dim.nbDims; i++) {
        volume *= dim.d[i];
      }
    }
    cudaMalloc(&ptr, volume * getElementSize(dtype));
  }

  int32_t nbytes() const {
    return volume * getElementSize(dtype);
  }

  void mov(std::shared_ptr<Tensor> other, cudaStream_t stream) {
    // copy from 'other'
    cudaMemcpyAsync(
      ptr, other->ptr, 
      nbytes(), 
      cudaMemcpyHostToDevice,
      stream);
  }

  // template<class Htype=float, class Dtype=float>
  template<class Htype=float>
  void load_from_vector(const std::vector<Htype> &data) {
    std::cerr << "KOJI!!! load_from_vector started, size = " << data.size() << ", volume = " << volume << std::endl;
    if (data.size() != static_cast<size_t>(volume)) {
      std::cerr << "Data size mismatch! Expected " << volume << " elements." << std::endl;
      return;
    }

    size_t dsize = volume * getElementSize(dtype);
    // std::vector<char> b2(dsize);

    // // Convert and copy data from Htype to Dtype
    // Dtype* dbuffer = reinterpret_cast<Dtype*>(b2.data());
    // for (int i = 0; i < volume; i++) {
    //   dbuffer[i] = static_cast<Dtype>(data[i]);
    // }

    // // Copy to CUDA device memory
    // cudaMemcpy(ptr, b2.data(), dsize, cudaMemcpyHostToDevice);
    cudaMemcpy(ptr, data.data(), dsize, cudaMemcpyHostToDevice);
  }

  std::vector<float> cpu() {
    std::vector<float> buffer(volume);
    cudaMemcpy(buffer.data(), ptr, volume * sizeof(float), cudaMemcpyDeviceToHost);
    return buffer;
  }

  std::vector<char> load_ref(std::string fname) {
    size_t bsize = volume * sizeof(float);
    std::vector<char> buffer(bsize);
    std::ifstream file_(fname, std::ios::binary);
    file_.read(buffer.data(), bsize);
    return buffer;
  }
}; // struct Tensor

std::ostream& operator<<(std::ostream& os, Tensor& t) {
  os << "[" << (int)(t.iomode) << "] ";
  os << t.name << ", [";
  
  for( int nd=0; nd<t.dim.nbDims; nd++ ) {
    if( nd == 0 ) {
      os << t.dim.d[nd];
    } else {
      os << ", " << t.dim.d[nd];
    }
  }
  std::cout << "]";
  std::cout << ", type = " << int(t.dtype);
  return os;
}

Logger gLogger;

class SubNetwork {
  ICudaEngine* engine;
  IExecutionContext* context; 
public:
  std::unordered_map<std::string, std::shared_ptr<Tensor>> bindings;
  bool use_cuda_graph = false;
  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;

  SubNetwork(std::string engine_path, IRuntime* runtime) {
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file) {
      throw std::runtime_error("Error opening engine file: " + engine_path);
    }
    engine_file.seekg(0, engine_file.end);
    long int fsize = engine_file.tellg();
    engine_file.seekg(0, engine_file.beg);

    // Read the engine file into a buffer
    std::vector<char> engineData(fsize);

    engine_file.read(engineData.data(), fsize);
    engine = runtime->deserializeCudaEngine(engineData.data(), fsize);
    context = engine->createExecutionContext(); 

    int nb = engine->getNbIOTensors();  

    for( int n=0; n<nb; n++ ) {
      std::string name = engine->getIOTensorName(n);
      Dims d = engine->getTensorShape(name.c_str());            
      DataType dtype = engine->getTensorDataType(name.c_str());
      bindings[name] = std::make_shared<Tensor>(name, d, dtype);
      bindings[name]->iomode = engine->getTensorIOMode(name.c_str());
      std::cout << *(bindings[name]) << std::endl;
      context->setTensorAddress(name.c_str(), bindings[name]->ptr);
    }
  }

  void Enqueue(cudaStream_t stream) {
    if( this->use_cuda_graph ) {
      cudaGraphLaunch(graph_exec, stream);
    } else {
      context->enqueueV3(stream);
    }  
  }

  ~SubNetwork() {
  }

  void EnableCudaGraph(cudaStream_t stream) {        
    // run first time to avoid allocation
    this->Enqueue(stream);
    cudaStreamSynchronize(stream);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    this->Enqueue(stream);
    cudaStreamEndCapture(stream, &graph);
    this->use_cuda_graph = true;
#if CUDART_VERSION < 12000
    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
#else
    cudaGraphInstantiate(&graph_exec, graph, 0);
#endif
  }
}; // class SubNetwork

class Duration {
  // stat
  std::vector<float> stats;
  cudaEvent_t b, e;
  std::string m_name;
public:
  Duration(std::string name): m_name(name) {
    cudaEventCreate(&b);
    cudaEventCreate(&e);
  }

  void MarkBegin(cudaStream_t s) {
    cudaEventRecord(b, s);
  }

  void MarkEnd(cudaStream_t s) {
    cudaEventRecord(e, s);
  }

  float Elapsed() {
    float val;
    cudaEventElapsedTime(&val, b, e);
    stats.push_back(val);
    return val;
  }
}; // class 
// Taken from NVIDIA/DL4AGX

class StreamPetrNode : public rclcpp::Node
{
  using Odometry = nav_msgs::msg::Odometry;
  using Image = sensor_msgs::msg::Image;
  using CameraInfo = sensor_msgs::msg::CameraInfo;
  using DetectedObjects = autoware_perception_msgs::msg::DetectedObjects;
  using DetectedObject = autoware_perception_msgs::msg::DetectedObject;

public:
  explicit StreamPetrNode(const rclcpp::NodeOptions & node_options);

private:
  void inference(const int f, const std::string & data_dir);
  void odometry_callback(Odometry::ConstSharedPtr input_msg);
  void camera_info_callback(
    CameraInfo::ConstSharedPtr input_camera_info_msg,
    const std::size_t camera_id);
  void camera_image_callback(
    Image::ConstSharedPtr input_camera_image_msg,
    const std::size_t camera_id);
  std::pair<std::vector<float>, std::vector<float>> get_ego_pose_vector() const;
  void inference_position_embedding(
    const std::vector<int> & img_metas_pad,
    const std::vector<float> & intrinsics,
    const std::vector<float> & img2lidar);
  void inference_detector(
    const std::vector<float> & imgs,
    const std::vector<float> & ego_pose,
    const std::vector<float> & ego_pose_inv,
    const double stamp);
  std::vector<float> get_camera_extrinsics_vector(
    const std::vector<std::string> & camera_links);
  rclcpp::Subscription<Odometry>::SharedPtr localization_sub_;
  std::vector<rclcpp::Subscription<CameraInfo>::SharedPtr> camera_info_subs_;
  std::vector<rclcpp::Subscription<Image>::SharedPtr> camera_image_subs_;
  rclcpp::Publisher<DetectedObjects>::SharedPtr pub_objects_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  const size_t rois_number_;
  const double confidence_threshold_;
  std::vector<float> point_cloud_range_;
  Odometry::ConstSharedPtr initial_kinematic_state_ = nullptr;
  Odometry::ConstSharedPtr latest_kinematic_state_ = nullptr;
  bool is_inference_initialized_ = false;

  std::unique_ptr<CameraDataStore> data_store_;
  std::unique_ptr<SubNetwork> backbone_;
  std::unique_ptr<SubNetwork> pts_head_;
  std::unique_ptr<SubNetwork> pos_embed_;
  std::unique_ptr<Duration> dur_backbone_;
  std::unique_ptr<Duration> dur_ptshead_;
  std::unique_ptr<Duration> dur_pos_embed_;
  Memory mem_;
  cudaStream_t stream_; 
};

}  // namespace tensorrt_stream_petr

#endif  // TENSORRT_STREAM_PETR__STREAM_PETR_NODE_HPP__
