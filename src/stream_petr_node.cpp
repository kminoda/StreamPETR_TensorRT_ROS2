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

#include "stream_petr_node.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <algorithm>

namespace tensorrt_stream_petr
{

std::pair<std::vector<size_t>, std::vector<float>> sort_with_indices_in_descending_order(const std::vector<float>& vec) {
  std::vector<std::pair<float, size_t>> paired_vec(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    paired_vec[i] = std::make_pair(vec[i], i);
  }

  std::sort(paired_vec.begin(), paired_vec.end(), [](const auto& a, const auto& b) {
    return a.first > b.first;
  });

  std::vector<size_t> sorted_indices(vec.size());
  std::vector<float> sorted_values(vec.size());
  for (size_t i = 0; i < paired_vec.size(); ++i) {
    sorted_values[i] = paired_vec[i].first;
    sorted_indices[i] = paired_vec[i].second;
  }

  return std::make_pair(sorted_indices, sorted_values);
}

float compute_rotation(float sine, float cosine) {
  return std::atan2(sine, cosine);
}

float exp_value(float value) {
  return std::exp(value);
}

std::vector<float> denormalize_bbox(const std::vector<float>& bbox) {
  bool has_velocity = bbox.size() > 8;

  float rot_sine = bbox[6];
  float rot_cosine = bbox[7];
  float rot = compute_rotation(rot_sine, rot_cosine);

  float cx = bbox[0];
  float cy = bbox[1];
  float cz = bbox[2];

  float w = exp_value(bbox[3]);
  float l = exp_value(bbox[4]);
  float h = exp_value(bbox[5]);

  std::vector<float> denormalized_bboxes;

  denormalized_bboxes.push_back(cx);
  denormalized_bboxes.push_back(cy);
  denormalized_bboxes.push_back(cz);
  denormalized_bboxes.push_back(w);
  denormalized_bboxes.push_back(l);
  denormalized_bboxes.push_back(h);
  denormalized_bboxes.push_back(rot);

  if (has_velocity) {
    float vx = bbox[8];
    float vy = bbox[9];
    denormalized_bboxes.push_back(vx);
    denormalized_bboxes.push_back(vy);
  }

  return denormalized_bboxes;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

std::vector<float> apply_sigmoid(const std::vector<float>& all_cls_scores) {
  std::vector<float> sigmoid_scores;
  sigmoid_scores.reserve(all_cls_scores.size());

  for (const auto& score : all_cls_scores) {
    sigmoid_scores.push_back(sigmoid(score));
  }

  return sigmoid_scores;
}

autoware_perception_msgs::msg::DetectedObject bbox_to_ros_msg(const std::vector<float> & bbox)
{
  // cx, cy, cz, w, l, h, rot, vx, vy
  autoware_perception_msgs::msg::DetectedObject object;
  object.kinematics.pose_with_covariance.pose.position.x = bbox[0];
  object.kinematics.pose_with_covariance.pose.position.y = bbox[1];
  object.kinematics.pose_with_covariance.pose.position.z = bbox[2];
  object.shape.dimensions.x = bbox[3];
  object.shape.dimensions.y = bbox[4];
  object.shape.dimensions.z = bbox[5];
  const double yaw = bbox[6];
  object.kinematics.pose_with_covariance.pose.orientation.w = cos(yaw * 0.5);
  object.kinematics.pose_with_covariance.pose.orientation.x = 0;
  object.kinematics.pose_with_covariance.pose.orientation.y = 0;
  object.kinematics.pose_with_covariance.pose.orientation.z = sin(yaw * 0.5);

  object.kinematics.has_position_covariance = false;
  object.kinematics.has_twist = false;
  object.shape.type = 0;
  return object;
}

std::tuple<std::vector<float>, std::vector<int>, std::vector<std::vector<float>>> decode_results(
  const std::vector<float> & all_bbox_preds,
  const std::vector<float> & all_cls_scores,
  const int max_num)
{
  const int NUM_CLASSES = 10;
  const auto all_cls_scores_sigmoid = apply_sigmoid(all_cls_scores);

  auto [sorted_indices, scores] = sort_with_indices_in_descending_order(all_cls_scores_sigmoid);

  std::vector<int> labels;
  std::vector<std::vector<float>> bboxes;
  for (int i = 0; i < max_num; ++i) {
    labels.push_back(sorted_indices[i] % NUM_CLASSES);
    const int bbox_index = sorted_indices[i] / NUM_CLASSES;
    const std::vector<float> bbox(
      all_bbox_preds.begin() + bbox_index * 10,
      all_bbox_preds.begin() + bbox_index * 10 + 10
    );
    bboxes.push_back(denormalize_bbox(bbox));
  }
  return std::make_tuple(scores, labels, bboxes);
}

std::vector<float> cast_to_float(const std::vector<double>& double_vector) {
  std::vector<float> float_vector(double_vector.size());
  std::transform(double_vector.begin(), double_vector.end(), float_vector.begin(),
                [](double value) { return static_cast<float>(value); });
  return float_vector;
}

StreamPetrNode::StreamPetrNode(const rclcpp::NodeOptions & node_options)
: Node("tensorrt_stream_petr", node_options),
  tf_buffer_(this->get_clock()),
  tf_listener_(tf_buffer_),
  rois_number_(static_cast<size_t>(declare_parameter<int>("rois_number", 6))),
  confidence_threshold_(declare_parameter<double>("post_process_params.confidence_threshold")),
  camera_order_remapping_(initialize_camera_order_remapping())
{
  RCLCPP_INFO(get_logger(), "nvinfer: %d.%d.%d\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
  cudaSetDevice(0);

  using std::placeholders::_1;

  // Initialize parameters
  const std::string engine_backbone_path = declare_parameter<std::string>("model_params.engine_backbone_path");
  const std::string engine_head_path = declare_parameter<std::string>("model_params.engine_head_path");
  const std::string engine_position_embedding_path = declare_parameter<std::string>("model_params.engine_position_embedding_path");
  const std::string precision_backbone = declare_parameter<std::string>("model_params.precision_backbone");
  const std::string precision_head = declare_parameter<std::string>("model_params.precision_head");
  const std::string precision_position_embedding = declare_parameter<std::string>("model_params.precision_position_embedding");
  const std::vector<double> point_cloud_range_double = declare_parameter<std::vector<double>>("model_params.point_cloud_range");
  point_cloud_range_ = cast_to_float(point_cloud_range_double);

  localization_sub_ = this->create_subscription<Odometry>(
      "/localization/kinematic_state", rclcpp::QoS{1},
      [this](const Odometry::ConstSharedPtr msg) {
        this->odometry_callback(msg);
      }
    );
  camera_info_subs_.resize(rois_number_);
  camera_image_subs_.resize(rois_number_);
  for (size_t roi_i = 0; roi_i < rois_number_; ++roi_i) {
    const std::string camera_info_topic = declare_parameter<std::string>(
      "input/camera_info" + std::to_string(roi_i),
      "/sensing/camera/camera" + std::to_string(roi_i) + "/camera_info");
    const std::string camera_image_topic = declare_parameter<std::string>(
      "input/image" + std::to_string(roi_i),
      "/sensing/camera/camera" + std::to_string(roi_i) + "/image_rect_color");

    camera_info_subs_.at(roi_i) = this->create_subscription<CameraInfo>(
      camera_info_topic, rclcpp::QoS{1}.best_effort(),
      [this, roi_i](const CameraInfo::ConstSharedPtr msg) {
        this->camera_info_callback(msg, roi_i);
      }
    );
    camera_image_subs_.at(roi_i) = this->create_subscription<Image>(
      camera_image_topic, rclcpp::QoS{1}.best_effort(),
      [this, roi_i](const Image::ConstSharedPtr msg) {
        this->camera_image_callback(msg, roi_i);
      }
    );
  }
  
  // Publishers
  pub_objects_ = this->create_publisher<DetectedObjects>("~/output/objects", rclcpp::QoS{1});

  // Data store
  data_store_ = std::make_unique<CameraDataStore>(
    this, rois_number_,
    declare_parameter<int>("model_params.input_image_height"),
    declare_parameter<int>("model_params.input_image_width")
  );

  // TensorRT
  auto runtime_deleter = [](IRuntime *runtime) { (void)runtime; /* runtime->destroy(); */ };
  std::unique_ptr<IRuntime, decltype(runtime_deleter)> runtime{createInferRuntime(gLogger), runtime_deleter};
  backbone_ = std::make_unique<SubNetwork>(engine_backbone_path, runtime.get());
  pts_head_ = std::make_unique<SubNetwork>(engine_head_path, runtime.get());
  pos_embed_ = std::make_unique<SubNetwork>(engine_position_embedding_path, runtime.get());

  cudaStreamCreate(&stream_);
  backbone_->EnableCudaGraph(stream_);
  pts_head_->EnableCudaGraph(stream_);
  pos_embed_->EnableCudaGraph(stream_);

  mem_.mem_stream = stream_;
  mem_.pre_buf = (float*)pts_head_->bindings["pre_memory_timestamp"]->ptr;
  mem_.post_buf = (float*)pts_head_->bindings["post_memory_timestamp"]->ptr;

  // events for measurement
  dur_backbone_ = std::make_unique<Duration>("backbone");
  dur_ptshead_ = std::make_unique<Duration>("ptshead");
  dur_pos_embed_ = std::make_unique<Duration>("pos_embed");

  // const std::filesystem::path data_dir{this->declare_parameter<std::string>("temporary_params.bins_directory_path")};
  // const int n_frames = std::distance(std::filesystem::directory_iterator(data_dir), std::filesystem::directory_iterator{});
  // RCLCPP_INFO(get_logger(), "Total frames: %d\n", n_frames);

  // for (int i = 0; i < n_frames; ++i) {
  //   inference(i, data_dir);
  //   std::this_thread::sleep_for(std::chrono::milliseconds(100));
  // }
}

std::map<int, int> StreamPetrNode::initialize_camera_order_remapping()
{
  std::map<int, int> remapping;
  for (int i = 0; i < static_cast<int>(rois_number_); ++i) {
    std::string param_name = "camera_order_remapping." + std::to_string(i);
    int remapped_value = declare_parameter<int>(param_name);
    remapping[i] = remapped_value;
  }
  return remapping;
}

void StreamPetrNode::odometry_callback(
  Odometry::ConstSharedPtr input_msg)
{
  if (!initial_kinematic_state_) {
    initial_kinematic_state_ = input_msg;
  }
  latest_kinematic_state_ = input_msg;
  return;
}

void StreamPetrNode::camera_info_callback(
  CameraInfo::ConstSharedPtr input_camera_info_msg,
  const std::size_t camera_id)
{
  std::cout << "KOJI!!!! camera info callback @" << camera_id << std::endl;
  data_store_->update_camera_info(camera_order_remapping_.at(camera_id), input_camera_info_msg);
}

void StreamPetrNode::camera_image_callback(
  Image::ConstSharedPtr input_camera_image_msg,
  const std::size_t camera_id)
{
  std::cout << "KOJI!!!! camera image callback @" << camera_id << std::endl;
  data_store_->update_camera_image(camera_order_remapping_.at(camera_id), input_camera_image_msg);
  RCLCPP_INFO(get_logger(), "received camera %d", static_cast<int>(camera_id));

  if (!data_store_->check_if_all_camera_image_received()) {
    RCLCPP_WARN(get_logger(), "skipping since not all camera is received yet");
    return;
  }
  if (!data_store_->check_if_all_camera_info_received()) {
    RCLCPP_WARN(get_logger(), "skipping since not all camera info is received yet");
    return;
  }

  if (data_store_->check_if_all_images_synced()) {
    RCLCPP_INFO(get_logger(), "All images are synchronized and all camera info received.");
    if (!is_inference_initialized_) {
      inference_position_embedding(
        data_store_->get_image_shape(),
        data_store_->get_camera_info_vector(),
        get_camera_extrinsics_vector(data_store_->get_camera_link_names())
      );
      is_inference_initialized_ = true;
    }

    const auto [ego_pose, ego_pose_inv] = get_ego_pose_vector();
    inference_detector(
      data_store_->get_camera_images_vector(),
      ego_pose, ego_pose_inv,
      data_store_->get_timestamp()
    );
    data_store_->reset_camera_images();
  }
}

std::vector<float> StreamPetrNode::get_camera_extrinsics_vector(
  const std::vector<std::string> & camera_links)
{
  std::vector<float> res;
  for (const std::string & camera_link : camera_links) {
    try {
      geometry_msgs::msg::TransformStamped transform = tf_buffer_.lookupTransform("base_link", camera_link, tf2::TimePointZero);
      std::cout << camera_link << ": " << transform.transform.rotation.x << ", " << transform.transform.rotation.y << ", " << transform.transform.rotation.z << ", " << transform.transform.rotation.w << std::endl;
      tf2::Quaternion quat(
        transform.transform.rotation.x,
        transform.transform.rotation.y,
        transform.transform.rotation.z,
        transform.transform.rotation.w
      );
      tf2::Matrix3x3 rotation_matrix;
      rotation_matrix.setRotation(quat);

      std::vector<float> extrinsics = {
        static_cast<float>(rotation_matrix[0][0]), static_cast<float>(rotation_matrix[0][1]), static_cast<float>(rotation_matrix[0][2]), static_cast<float>(transform.transform.translation.x),
        static_cast<float>(rotation_matrix[1][0]), static_cast<float>(rotation_matrix[1][1]), static_cast<float>(rotation_matrix[1][2]), static_cast<float>(transform.transform.translation.y),
        static_cast<float>(rotation_matrix[2][0]), static_cast<float>(rotation_matrix[2][1]), static_cast<float>(rotation_matrix[2][2]), static_cast<float>(transform.transform.translation.z),
        0.0f, 0.0f, 0.0f, 1.0f
      };

      res.insert(res.end(), extrinsics.begin(), extrinsics.end());
    } catch (const tf2::TransformException & ex) {
      throw std::runtime_error("Could not transform from " + camera_link + " to base_link: " + std::string(ex.what()));
    }
  }

  return res;
}

std::pair<std::vector<float>, std::vector<float>> StreamPetrNode::get_ego_pose_vector() const
{
  if (!latest_kinematic_state_ || !initial_kinematic_state_) {
    throw std::runtime_error("Kinematic states have not been received.");
  }

  const auto& latest_pose = latest_kinematic_state_->pose.pose;
  // const auto& initial_pose = initial_kinematic_state_->pose.pose;

  // tf2::Quaternion latest_quat(latest_pose.orientation.x, latest_pose.orientation.y, latest_pose.orientation.z, latest_pose.orientation.w);
  // tf2::Quaternion initial_quat(initial_pose.orientation.x, initial_pose.orientation.y, initial_pose.orientation.z, initial_pose.orientation.w);

  // tf2::Matrix3x3 latest_rot(latest_quat);
  // tf2::Matrix3x3 initial_rot(initial_quat);

  // tf2::Matrix3x3 relative_rot = initial_rot.inverse() * latest_rot;

  // tf2::Vector3 latest_translation(latest_pose.position.x, latest_pose.position.y, latest_pose.position.z);
  // tf2::Vector3 initial_translation(initial_pose.position.x, initial_pose.position.y, initial_pose.position.z);
  // tf2::Vector3 relative_translation = latest_translation - initial_translation;

  // relative_translation = initial_rot.inverse() * relative_translation;

  tf2::Quaternion latest_quat(latest_pose.orientation.x, latest_pose.orientation.y, latest_pose.orientation.z, latest_pose.orientation.w);
  tf2::Matrix3x3 latest_rot(latest_quat);
  tf2::Matrix3x3 relative_rot = latest_rot;
  tf2::Vector3 latest_translation(latest_pose.position.x, latest_pose.position.y, latest_pose.position.z);
  tf2::Vector3 relative_translation = latest_translation;

  std::vector<float> egopose = {
    static_cast<float>(relative_rot[0][0]), static_cast<float>(relative_rot[0][1]), static_cast<float>(relative_rot[0][2]), static_cast<float>(relative_translation.x()),
    static_cast<float>(relative_rot[1][0]), static_cast<float>(relative_rot[1][1]), static_cast<float>(relative_rot[1][2]), static_cast<float>(relative_translation.y()),
    static_cast<float>(relative_rot[2][0]), static_cast<float>(relative_rot[2][1]), static_cast<float>(relative_rot[2][2]), static_cast<float>(relative_translation.z()),
    0.0f, 0.0f, 0.0f, 1.0f
  };

  tf2::Matrix3x3 inverse_rot = relative_rot.transpose();
  tf2::Vector3 inverse_translation = -(inverse_rot * relative_translation);

  std::vector<float> inverse_egopose = {
    static_cast<float>(inverse_rot[0][0]), static_cast<float>(inverse_rot[0][1]), static_cast<float>(inverse_rot[0][2]), static_cast<float>(inverse_translation.x()),
    static_cast<float>(inverse_rot[1][0]), static_cast<float>(inverse_rot[1][1]), static_cast<float>(inverse_rot[1][2]), static_cast<float>(inverse_translation.y()),
    static_cast<float>(inverse_rot[2][0]), static_cast<float>(inverse_rot[2][1]), static_cast<float>(inverse_rot[2][2]), static_cast<float>(inverse_translation.z()),
    0.0f, 0.0f, 0.0f, 1.0f
  };

  return std::make_pair(egopose, inverse_egopose);
}

void StreamPetrNode::inference_position_embedding(
  const std::vector<int> & img_metas_pad,
  const std::vector<float> & intrinsics,
  const std::vector<float> & img2lidar)
{
  RCLCPP_INFO(get_logger(), "intrinsics size: %ld", intrinsics.size());
  RCLCPP_INFO(get_logger(), "img2lidar size: %ld", img2lidar.size());

  pos_embed_->bindings["img_metas_pad"]->load_from_vector<int>(img_metas_pad);
  pos_embed_->bindings["intrinsics"]->load_from_vector(intrinsics);
  pos_embed_->bindings["img2lidar"]->load_from_vector(img2lidar);

  dur_pos_embed_->MarkBegin(stream_);
  pos_embed_->Enqueue(stream_);
  dur_pos_embed_->MarkEnd(stream_);

  cudaMemcpyAsync(
    pts_head_->bindings["pos_embed"]->ptr,
    pos_embed_->bindings["pos_embed"]->ptr, 
    pos_embed_->bindings["pos_embed"]->nbytes(), 
    cudaMemcpyDeviceToDevice, stream_);
  cudaMemcpyAsync(
    pts_head_->bindings["cone"]->ptr,
    pos_embed_->bindings["cone"]->ptr, 
    pos_embed_->bindings["cone"]->nbytes(), 
    cudaMemcpyDeviceToDevice, stream_);
}

void StreamPetrNode::inference_detector(
  const std::vector<float> & imgs,
  const std::vector<float> & ego_pose,
  const std::vector<float> & ego_pose_inv,
  const double stamp)
{
  RCLCPP_INFO(get_logger(), "imgs size: %ld", imgs.size());
  RCLCPP_INFO(get_logger(), "ego_pose size: %ld", ego_pose.size());
  RCLCPP_INFO(get_logger(), "stamp: %f", stamp);
  backbone_->bindings["img"]->load_from_vector(imgs);

  { // feature extraction execution
    dur_backbone_->MarkBegin(stream_);
    // inference
    backbone_->Enqueue(stream_);
    dur_backbone_->MarkEnd(stream_);

    cudaMemcpyAsync(
      pts_head_->bindings["x"]->ptr,
      backbone_->bindings["img_feats"]->ptr,
      backbone_->bindings["img_feats"]->nbytes(),
      cudaMemcpyDeviceToDevice, stream_);
  }

  pts_head_->bindings["data_ego_pose"]->load_from_vector(ego_pose);
  pts_head_->bindings["data_ego_pose_inv"]->load_from_vector(ego_pose_inv);

  { // backbone execution
    // TODO: Properly initialize the first weights for the first frame
    mem_.StepPre(stamp);

    // inference
    dur_ptshead_->MarkBegin(stream_);
    pts_head_->Enqueue(stream_);
    dur_ptshead_->MarkEnd(stream_);
    mem_.StepPost(stamp);

    // copy mem_post to mem_pre for next round
    pts_head_->bindings["pre_memory_embedding"]->mov(pts_head_->bindings["post_memory_embedding"], stream_);
    pts_head_->bindings["pre_memory_reference_point"]->mov(pts_head_->bindings["post_memory_reference_point"], stream_);
    pts_head_->bindings["pre_memory_egopose"]->mov(pts_head_->bindings["post_memory_egopose"], stream_);
    pts_head_->bindings["pre_memory_velo"]->mov(pts_head_->bindings["post_memory_velo"], stream_);
  }

  cudaStreamSynchronize(stream_);

  std::cout << "KOJI!!! FINISHED INFERENCE!!!! " << std::endl;

  ////////////////////////////// TODO MOVE THIS TO ANOTHER FUNC //////////////////////////////
  // cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy
  const std::vector<float> all_bbox_preds = pts_head_->bindings["all_bbox_preds"]->cpu();
  const std::vector<float> all_cls_scores = pts_head_->bindings["all_cls_scores"]->cpu();

  // TODO(kminoda): resize beforehand
  std::vector<float> scores;
  std::vector<int> labels;
  std::vector<std::vector<float>> bboxes;
  std::tie(scores, labels, bboxes) = decode_results(all_bbox_preds, all_cls_scores, 300);

  std::vector<autoware_perception_msgs::msg::DetectedObject> raw_objects;
  for (size_t i = 0; i < bboxes.size(); ++i) {
    const float score = scores[i];
    if (score > confidence_threshold_) {
      raw_objects.push_back(bbox_to_ros_msg(bboxes[i]));
    }
  }

  DetectedObjects output_msg;
  output_msg.objects = raw_objects;
  output_msg.header.frame_id = "base_link";
  pub_objects_->publish(output_msg);
  ////////////////////////////// TODO MOVE THIS TO ANOTHER FUNC //////////////////////////////
}

}  // namespace tensorrt_stream_petr


#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(tensorrt_stream_petr::StreamPetrNode)