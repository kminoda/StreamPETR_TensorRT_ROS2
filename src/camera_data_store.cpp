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
#include <Eigen/Dense>

#include <cmath>
#include <algorithm>

namespace tensorrt_stream_petr
{
static void updateIntrinsics(float *K_4x4, const Eigen::Matrix3f &ida_mat)
{
  Eigen::Matrix3f K;
  K << K_4x4[0], K_4x4[1], K_4x4[2],
       K_4x4[4], K_4x4[5], K_4x4[6],
       K_4x4[8], K_4x4[9], K_4x4[10];

  Eigen::Matrix3f K_new = ida_mat * K;

  K_4x4[0]  = K_new(0,0); // fx'
  K_4x4[1]  = K_new(0,1);
  K_4x4[2]  = K_new(0,2);
  K_4x4[4]  = K_new(1,0);
  K_4x4[5]  = K_new(1,1); // fy'
  K_4x4[6]  = K_new(1,2);
  K_4x4[8]  = K_new(2,0);
  K_4x4[9]  = K_new(2,1);
  K_4x4[10] = K_new(2,2);
}

static cv::Mat inferenceAugmentImage(
    const cv::Mat &img_in,
    int final_H,      // fH
    int final_W       // fW
)
{
  int H = img_in.rows;
  int W = img_in.cols;

  float scaleH = static_cast<float>(final_H) / static_cast<float>(H);
  float scaleW = static_cast<float>(final_W) / static_cast<float>(W);
  float resize = std::max(scaleH, scaleW);

  int newW = static_cast<int>(W * resize);
  int newH = static_cast<int>(H * resize);

  cv::Mat resized;
  cv::resize(img_in, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

  float bot_mean = 0.0f;  // for test
  int crop_h = static_cast<int>((1.0f - bot_mean) * newH) - final_H; 
  if (crop_h < 0) crop_h = 0;
  int crop_w = std::max(0, (newW - final_W) / 2);

  int x1 = std::max(0, crop_w);
  int y1 = std::max(0, crop_h);
  int x2 = std::min(x1 + final_W, newW);
  int y2 = std::min(y1 + final_H, newH);

  cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
  cv::Mat cropped = resized(roi).clone();

  return cropped;
}


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
  return (max_time - min_time).seconds() <= 0.05;
}

std::vector<float> CameraDataStore::get_camera_info_vector() const
{
  std::vector<float> intrinsics_all;

  int fH = image_height_;
  int fW = image_width_;

  for (size_t camera_id = 0; camera_id < camera_info_list_.size(); ++camera_id)
  {
    const auto & camera_info_msg = camera_info_list_[camera_id];
    if (!camera_info_msg) {
      throw std::runtime_error(
        "CameraInfo not received for camera ID: " + std::to_string(camera_id));
    }

    int rawW = camera_info_msg->width;
    int rawH = camera_info_msg->height;
    std::vector<float> K_4x4 = {
      static_cast<float>(camera_info_msg->k[0]), static_cast<float>(camera_info_msg->k[1]), static_cast<float>(camera_info_msg->k[2]), 0.f,
      static_cast<float>(camera_info_msg->k[3]), static_cast<float>(camera_info_msg->k[4]), static_cast<float>(camera_info_msg->k[5]), 0.f,
      static_cast<float>(camera_info_msg->k[6]), static_cast<float>(camera_info_msg->k[7]), static_cast<float>(camera_info_msg->k[8]), 0.f,
      0.f, 0.f, 0.f, 1.f
    };

    float scaleH = float(fH)/float(rawH);
    float scaleW = float(fW)/float(rawW);
    float resize = std::max(scaleH, scaleW);
    int newW = int(rawW * resize);
    int newH = int(rawH * resize);

    float bot_mean = 0.0f;
    int crop_h = int((1.0f - bot_mean)* newH) - fH;
    if(crop_h<0) crop_h=0;
    int crop_w = std::max(0, newW - fW)/2;

    Eigen::Matrix3f S = Eigen::Matrix3f::Identity();
    S(0,0) = resize;
    S(1,1) = resize;

    Eigen::Matrix3f T = Eigen::Matrix3f::Identity();
    T(0,2) = -float(crop_w);
    T(1,2) = -float(crop_h);

    Eigen::Matrix3f ida_mat = T * S;

    updateIntrinsics(K_4x4.data(), ida_mat);

    intrinsics_all.insert(intrinsics_all.end(), K_4x4.begin(), K_4x4.end());
  }

  return intrinsics_all;
}

std::vector<int> CameraDataStore::get_image_shape() const
{
  std::vector<int> vec{image_height_, image_width_, 3};
  return vec;
}

std::vector<float> CameraDataStore::get_camera_images_vector() const
{
  std::vector<float> vec;

  // Normalization parameters
  const cv::Scalar mean(123.675, 116.28, 103.53);
  const cv::Scalar std(58.395, 57.12, 57.375);
  const bool to_rgb = true;

  int fH = image_height_;
  int fW = image_width_;

  for (size_t camera_id = 0; camera_id < camera_image_list_.size(); ++camera_id)
  {
    const auto & image_msg = camera_image_list_[camera_id];
    if (!image_msg) {
      throw std::runtime_error(
        "Image msg not received for camera ID: " + std::to_string(camera_id));
    }

    // 1) ROS->cv::Mat (BGR8)
    cv::Mat rawImg;
    try {
      rawImg = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8)->image;
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_WARN(logger_, "Image conversion failed: %s", e.what());
      throw std::runtime_error("Unsupported image encoding for conversion");
    }
    if (rawImg.empty()) {
      throw std::runtime_error("Failed to convert image for camera ID: " + std::to_string(camera_id));
    }

    // 2) "inferenceAugmentImage"
    cv::Mat augImg = inferenceAugmentImage(rawImg, fH, fW);

    // 3) to RGB if needed
    if (to_rgb) {
      cv::cvtColor(augImg, augImg, cv::COLOR_BGR2RGB);
    }

    // 4) to float
    augImg.convertTo(augImg, CV_32FC3);

    // 5) normalize
    cv::subtract(augImg, mean, augImg);
    cv::divide(augImg, std, augImg);

    // 6) (C, H, W) order
    vec.reserve(vec.size() + 3 * augImg.rows * augImg.cols);
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < augImg.rows; ++y) {
        for (int x = 0; x < augImg.cols; ++x) {
          vec.push_back(augImg.at<cv::Vec3f>(y, x)[c]);
        }
      }
    }
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
