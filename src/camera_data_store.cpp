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

#include "camera_data_store.hpp"

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

#include <opencv2/opencv.hpp>

#include <cmath>
#include <algorithm>

namespace tensorrt_stream_petr
{
CameraDataStore::CameraDataStore(rclcpp::Node * node, const int rois_number, const int image_height, const int image_width)
: rois_number_(rois_number),
  image_height_(image_height),
  image_width_(image_width),
  logger_(node->get_logger())
{
  camera_info_list_.resize(rois_number_);
  camera_image_list_.resize(rois_number_);
}

void CameraDataStore::update_camera_image(const int camera_id, const Image::ConstSharedPtr & input_camera_image_msg)
{
  camera_image_list_[camera_id] = input_camera_image_msg;
}

void CameraDataStore::update_camera_info(const int camera_id, const CameraInfo::ConstSharedPtr & input_camera_info_msg)
{
  std::cout << "KOJI update_camera_info size = " << camera_info_list_.size() << std::endl;
  camera_info_list_[camera_id] = input_camera_info_msg;
}

bool CameraDataStore::check_if_all_camera_info_received() const
{
  for (const auto & camera_info: camera_info_list_)
  {
    if (!camera_info) return false;
  }

  return true;
}

bool CameraDataStore::check_if_all_camera_image_received() const
{
  for (const auto & camera_image: camera_image_list_)
  {
    if (!camera_image) return false;
  }

  return true;
}

bool CameraDataStore::check_if_all_images_synced() const
{
  if (camera_image_list_.size() != rois_number_) {
    return false;
  }

  rclcpp::Time min_time(INT64_MAX, RCL_ROS_TIME); // Maximum possible time
  rclcpp::Time max_time(INT64_MIN, RCL_ROS_TIME); // Minimum possible time

  for (size_t camera_id = 0; camera_id < camera_image_list_.size(); ++camera_id)
  {
    const auto & image_msg = camera_image_list_[camera_id];
    rclcpp::Time timestamp(image_msg->header.stamp);

    if (timestamp < min_time) {
      min_time = timestamp;
    }
    if (timestamp > max_time) {
      max_time = timestamp;
    }
  }

  // Check if the difference is within 0.05 seconds (50 milliseconds)
  const double time_diff = (max_time - min_time).seconds();
  RCLCPP_INFO(logger_, "time_diff: %f", time_diff);
  return (max_time - min_time).seconds() <= 0.05;
}

std::vector<float> CameraDataStore::get_camera_info_vector() const
{
  std::vector<float> intrinsics;

  for (size_t camera_id = 0; camera_id < camera_info_list_.size(); ++camera_id)
  {
    const auto & camera_info_msg = camera_info_list_[camera_id];
    if (!camera_info_msg) {
      throw std::runtime_error("CameraInfo message not received for camera ID: " + std::to_string(camera_id));
    }

    std::vector<float> K = {
      static_cast<float>(camera_info_msg->k[0]), static_cast<float>(camera_info_msg->k[1]), static_cast<float>(camera_info_msg->k[2]), 0.0f,
      static_cast<float>(camera_info_msg->k[3]), static_cast<float>(camera_info_msg->k[4]), static_cast<float>(camera_info_msg->k[5]), 0.0f,
      static_cast<float>(camera_info_msg->k[6]), static_cast<float>(camera_info_msg->k[7]), static_cast<float>(camera_info_msg->k[8]), 0.0f,
      0.0f,                                       0.0f,                                       0.0f,                                       1.0f
    };

    intrinsics.insert(intrinsics.end(), K.begin(), K.end());
  }

  return intrinsics;
}

std::vector<int> CameraDataStore::get_image_shape() const
{
  std::vector<int> vec{image_height_, image_width_, 3};
  return vec;
}

std::vector<float> CameraDataStore::get_camera_images_vector() const
{
  std::vector<float> vec;
  for (size_t camera_id = 0; camera_id < camera_image_list_.size(); ++camera_id)
  {
    const auto & image_msg = camera_image_list_[camera_id];
    if (!image_msg) {
      throw std::runtime_error("Image message not received for camera ID: " + std::to_string(camera_id));
    }
    cv::Mat img;
    try {
      img = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::TYPE_32FC3)->image;
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_WARN(logger_, "Image conversion failed: %s", e.what());
 
      if (image_msg->encoding == sensor_msgs::image_encodings::BGR8) {
        img = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8)->image;
        img.convertTo(img, CV_32FC3); // Convert to float with 3 channels
      } else {
        throw std::runtime_error("Unsupported image encoding for 3-channel conversion: " + image_msg->encoding);
      }
    }

    if (img.empty()) {
      throw std::runtime_error("Failed to convert image.");
    }

    cv::resize(img, img, cv::Size(image_width_, image_height_));
    RCLCPP_INFO(logger_, "Camera ID %zu: Image shape: (%d, %d, %d)",
                camera_id, img.rows, img.cols, img.channels());

    const size_t old_size = vec.size();
    vec.resize(old_size + img.total() * img.channels());
    std::memcpy(vec.data() + old_size, img.data, img.total() * img.channels() * sizeof(float));
  }

  return vec;
}

double CameraDataStore::get_timestamp() const
{
  double min_timestamp = std::numeric_limits<double>::max();

  for (size_t camera_id = 0; camera_id < camera_image_list_.size(); ++camera_id)
  {
    const auto & image_msg = camera_image_list_[camera_id];
    if (!image_msg) {
      throw std::runtime_error("Image message not received for camera ID: " + std::to_string(camera_id));
    }
    double timestamp = image_msg->header.stamp.sec + image_msg->header.stamp.nanosec * 1e-9;
    if (timestamp < min_timestamp) {
      min_timestamp = timestamp;
    }
  }

  return min_timestamp;
}

std::vector<std::string> CameraDataStore::get_camera_link_names() const
{
  std::vector<std::string> result(rois_number_);
  for (size_t camera_id = 0; camera_id < camera_image_list_.size(); ++camera_id)
  {
    const auto & image_msg = camera_image_list_[camera_id];
    if (!image_msg) {
      throw std::runtime_error("Image message not received for camera ID: " + std::to_string(camera_id));
    }
    result[camera_id] = image_msg->header.frame_id;
  }
  return result;
}

void CameraDataStore::reset_camera_images()
{
  for (size_t camera_id = 0; camera_id < camera_image_list_.size(); ++camera_id)
  {
    camera_image_list_[camera_id] = nullptr;
  }
}
}  // namespace tensorrt_stream_petr