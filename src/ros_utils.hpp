// Copyright 2022 TIER IV, Inc.
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

#ifndef TENSORRT_STREAM_PETR__ROS_UTILS_HPP_
#define TENSORRT_STREAM_PETR__ROS_UTILS_HPP_

// ros packages cannot be included from cuda.

#include "utils.hpp"

#include <autoware_perception_msgs/msg/detected_objects.hpp>

#include <string>
#include <vector>

namespace tensorrt_stream_petr
{

void box3DToDetectedObject(
  const Box3D & box3d, const std::vector<std::string> & class_names, const bool has_twist,
  const bool has_variance, autoware_perception_msgs::msg::DetectedObject & obj);

uint8_t getSemanticType(const std::string & class_name);

std::array<double, 36> convertPoseCovarianceMatrix(const Box3D & box3d);

std::array<double, 36> convertTwistCovarianceMatrix(const Box3D & box3d);

bool isCarLikeVehicleLabel(const uint8_t label);

}  // namespace tensorrt_stream_petr

#endif  // TENSORRT_STREAM_PETR__ROS_UTILS_HPP_