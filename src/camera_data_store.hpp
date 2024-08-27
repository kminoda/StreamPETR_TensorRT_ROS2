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

#ifndef TENSORRT_STREAM_PETR__CAMERA_DATA_STORE_HPP__
#define TENSORRT_STREAM_PETR__CAMERA_DATA_STORE_HPP__

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <vector>
#include <memory>
#include <string>

namespace tensorrt_stream_petr
{
class CameraDataStore
{
  using Image = sensor_msgs::msg::Image;
  using CameraInfo = sensor_msgs::msg::CameraInfo;

public:
  CameraDataStore(rclcpp::Node * node, const int rois_number, const int image_height, const int image_width);
  void update_camera_image(const int camera_id, const Image::ConstSharedPtr & input_camera_image_msg);
  void update_camera_info(const int camera_id, const CameraInfo::ConstSharedPtr & input_camera_info_msg);
  bool check_if_all_camera_image_received() const;
  bool check_if_all_camera_info_received() const;
  bool check_if_all_images_synced() const;
  std::vector<float> get_camera_info_vector() const;
  std::vector<float> get_camera_images_vector() const;
  std::vector<int> get_image_shape() const;
  double get_timestamp() const;
  std::vector<std::string> get_camera_link_names() const;
  void reset_camera_images();

private:
  const int rois_number_;
  const int image_height_;
  const int image_width_;
  rclcpp::Logger logger_;
  std::vector<CameraInfo::ConstSharedPtr> camera_info_list_;
  std::vector<Image::ConstSharedPtr> camera_image_list_;
};

}  // namespace tensorrt_stream_petr

#endif  // TENSORRT_STREAM_PETR__CAMERA_DATA_STORE_HPP__
