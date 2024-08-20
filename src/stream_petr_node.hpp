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

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

#include <image_transport/image_transport.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>

#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <sensor_msgs/msg/image.hpp>

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

#include "non_maximum_suppression.hpp"

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

  int32_t nbytes() {
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

  template<class Htype=float, class Dtype=float>
  void load(std::string fname) {
    size_t hsize = volume * sizeof(Htype);
    size_t dsize = volume * getElementSize(dtype);
    std::vector<char> b1(hsize);
    std::vector<char> b2(dsize);
    std::ifstream file_(fname, std::ios::binary);
    if( file_.fail() ) {
      std::cerr << fname << " missing!" << std::endl;
      return;
    }
    file_.read(b1.data(), hsize);
    Htype* hbuffer = reinterpret_cast<Htype*>(b1.data());
    Dtype* dbuffer = reinterpret_cast<Dtype*>(b2.data());
    // in some cases we want to load from different dtype
    for( int i=0; i<volume; i++ ) {
      dbuffer[i] = (Dtype)hbuffer[i];
    }

    cudaMemcpy(ptr, b2.data(), dsize, cudaMemcpyHostToDevice);
  }

  template<class Htype=float, class Dtype=float>
  void save(std::string fname) {
    size_t hsize = volume * sizeof(Htype);
    size_t dsize = volume * getElementSize(dtype);
    std::vector<char> b1(hsize);
    std::vector<char> b2(dsize);
    std::ofstream file_(fname, std::ios::binary);
    if( file_.fail() ) {
      std::cerr << fname << " can't open!" << std::endl;
      return;
    }
    // file_.read(b1.data(), hsize);
    Htype* hbuffer = reinterpret_cast<Htype*>(b1.data());
    Dtype* dbuffer = reinterpret_cast<Dtype*>(b2.data());
    cudaMemcpy(b2.data(), ptr, dsize, cudaMemcpyDeviceToHost);
    // in some cases we want to load from different dtype
    for( int i=0; i<volume; i++ ) {
      hbuffer[i] = (Htype)dbuffer[i];
    }
    file_.write(b2.data(), hsize);
    file_.close();
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
  using Image = sensor_msgs::msg::Image;
  using DetectedObjects = autoware_perception_msgs::msg::DetectedObjects;
  using DetectedObject = autoware_perception_msgs::msg::DetectedObject;

public:
  explicit StreamPetrNode(const rclcpp::NodeOptions & node_options);

private:
  void inference(const int f, const std::string & data_dir);
  void on_image(const Image & msg);

  rclcpp::Subscription<Image>::SharedPtr sub_image_;
  rclcpp::Publisher<DetectedObjects>::SharedPtr pub_objects_;

  const double confidence_threshold_;
  std::vector<float> point_cloud_range_;

  NonMaximumSuppression iou_bev_nms_;

  std::unique_ptr<SubNetwork> backbone_;
  std::unique_ptr<SubNetwork> pts_head_;
  std::unique_ptr<Duration> dur_backbone_;
  std::unique_ptr<Duration> dur_ptshead_;
  Memory mem_;
  cudaStream_t stream_; 
  bool is_first_frame_ = false;
};

}  // namespace tensorrt_stream_petr

#endif  // TENSORRT_STREAM_PETR__STREAM_PETR_NODE_HPP__
