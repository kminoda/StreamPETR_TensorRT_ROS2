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

namespace tensorrt_stream_petr
{
class StreamPetrNode : public rclcpp::Node
{
public:
  explicit StreamPetrNode(const rclcpp::NodeOptions & node_options);

  struct NodeParam
  {
    std::string onnx_backbone_path{};
    std::string onnx_head_path{};
    std::string precision{};
  };
};

}  // namespace tensorrt_stream_petr

#endif  // TENSORRT_STREAM_PETR__STREAM_PETR_NODE_HPP__
