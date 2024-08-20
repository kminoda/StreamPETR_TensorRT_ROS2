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
  confidence_threshold_(declare_parameter<double>("post_process_params.confidence_threshold"))
{
  RCLCPP_INFO(get_logger(), "nvinfer: %d.%d.%d\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
  cudaSetDevice(0);

  using std::placeholders::_1;

  // Initialize parameters
  const std::string engine_backbone_path = declare_parameter<std::string>("model_params.engine_backbone_path");
  const std::string engine_head_path = declare_parameter<std::string>("model_params.engine_head_path");
  const std::string precision_backbone = declare_parameter<std::string>("model_params.precision_backbone");
  const std::string precision_head = declare_parameter<std::string>("model_params.precision_head");
  const std::vector<double> point_cloud_range_double = declare_parameter<std::vector<double>>("model_params.point_cloud_range");
  point_cloud_range_ = cast_to_float(point_cloud_range_double);

  // Subscriber
  sub_image_ = create_subscription<Image>(
    "~/input/image_raw", 10, std::bind(&StreamPetrNode::on_image, this, _1));
  pub_objects_ = this->create_publisher<DetectedObjects>("~/output/objects", rclcpp::QoS{1});
  
  // NMS
  {  // IoU NMS
    NMSParams p;
    p.nms_type_ = NMS_TYPE::IoU_BEV;
    p.target_class_names_ =
      this->declare_parameter<std::vector<std::string>>("post_process_params.iou_nms_target_class_names");
    p.search_distance_2d_ =
      this->declare_parameter<double>("post_process_params.iou_nms_search_distance_2d");
    p.iou_threshold_ = this->declare_parameter<double>("post_process_params.iou_nms_threshold");
    iou_bev_nms_.setParameters(p);
  }

  // TensorRT
  auto runtime_deleter = [](IRuntime *runtime) { (void)runtime; /* runtime->destroy(); */ };
  std::unique_ptr<IRuntime, decltype(runtime_deleter)> runtime{createInferRuntime(gLogger), runtime_deleter};
  backbone_ = std::make_unique<SubNetwork>(engine_backbone_path, runtime.get());
  pts_head_ = std::make_unique<SubNetwork>(engine_head_path, runtime.get());

  cudaStreamCreate(&stream_);
  backbone_->EnableCudaGraph(stream_);
  pts_head_->EnableCudaGraph(stream_);

  mem_.mem_stream = stream_;
  mem_.pre_buf = (float*)pts_head_->bindings["pre_memory_timestamp"]->ptr;
  mem_.post_buf = (float*)pts_head_->bindings["post_memory_timestamp"]->ptr;

  // events for measurement
  dur_backbone_ = std::make_unique<Duration>("backbone");
  dur_ptshead_ = std::make_unique<Duration>("ptshead");

  const std::filesystem::path data_dir{this->declare_parameter<std::string>("temporary_params.bins_directory_path")};
  const int n_frames = std::distance(std::filesystem::directory_iterator(data_dir), std::filesystem::directory_iterator{});
  RCLCPP_INFO(get_logger(), "Total frames: %d\n", n_frames);

  for (int i = 0; i < n_frames; ++i) {
    inference(i, data_dir);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

void StreamPetrNode::on_image(const Image & msg){
  (void)msg;
}

void StreamPetrNode::inference(const int f, const std::string & data_dir) {
  // load data
  char buf[5] = {0};
  sprintf(buf, "%04d", f);
  std::string frame_dir = data_dir + std::string(buf) + "/";
  std::cout << frame_dir << std::endl;
  // img
  backbone_->bindings["img"]->load(frame_dir + "img.bin");

  dur_backbone_->MarkBegin(stream_);
  // inference
  backbone_->Enqueue(stream_);
  dur_backbone_->MarkEnd(stream_);

  cudaMemcpyAsync(
    pts_head_->bindings["x"]->ptr,
    backbone_->bindings["img_feats"]->ptr, 
    backbone_->bindings["img_feats"]->nbytes(), 
    cudaMemcpyDeviceToDevice, stream_);

  pts_head_->bindings["pos_embed"]->load(frame_dir + "pos_embed.bin");
  pts_head_->bindings["cone"]->load(frame_dir + "cone.bin");
  
  // load double timestamp from file
  double stamp_current = 0.0;
  char stamp_buf[8];
  std::ifstream file_(frame_dir + "data_timestamp.bin", std::ios::binary);
  file_.read(stamp_buf, sizeof(double));
  stamp_current = reinterpret_cast<double*>(stamp_buf)[0];
  std::cout << "stamp: " << stamp_current << std::endl;

  if( is_first_frame_ ) {
    // binary is stored as double
    pts_head_->bindings["pre_memory_timestamp"]->load<double, float>(frame_dir + "prev_memory_timestamp.bin");

    // start from dumped values
    pts_head_->bindings["pre_memory_embedding"]->load(frame_dir + "init_memory_embedding.bin");
    pts_head_->bindings["pre_memory_reference_point"]->load(frame_dir + "init_memory_reference_point.bin");
    pts_head_->bindings["pre_memory_egopose"]->load(frame_dir + "init_memory_egopose.bin");
    pts_head_->bindings["pre_memory_velo"]->load(frame_dir + "init_memory_velo.bin");

    is_first_frame_ = false;
  } else {
    // update timestamp
    mem_.StepPre(stamp_current);
  }        

  pts_head_->bindings["data_ego_pose"]->load(frame_dir + "data_ego_pose.bin");
  pts_head_->bindings["data_ego_pose_inv"]->load(frame_dir + "data_ego_pose_inv.bin");

  // inference
  dur_ptshead_->MarkBegin(stream_);
  pts_head_->Enqueue(stream_);
  dur_ptshead_->MarkEnd(stream_);
  mem_.StepPost(stamp_current);

  // copy mem_post to mem_pre for next round
  pts_head_->bindings["pre_memory_embedding"]->mov(pts_head_->bindings["post_memory_embedding"], stream_);
  pts_head_->bindings["pre_memory_reference_point"]->mov(pts_head_->bindings["post_memory_reference_point"], stream_);
  pts_head_->bindings["pre_memory_egopose"]->mov(pts_head_->bindings["post_memory_egopose"], stream_);
  pts_head_->bindings["pre_memory_velo"]->mov(pts_head_->bindings["post_memory_velo"], stream_);
  
  cudaStreamSynchronize(stream_);

  std::cout << "backbone: " << dur_backbone_->Elapsed() 
            << ", ptshead: " << dur_ptshead_->Elapsed() 
            << std::endl;

  // cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy
  const std::vector<float> all_bbox_preds = pts_head_->bindings["all_bbox_preds"]->cpu();
  const std::vector<float> all_cls_scores = pts_head_->bindings["all_cls_scores"]->cpu();

  // TODO(kminoda): resize beforehand
  std::vector<float> scores;
  std::vector<int> labels;
  std::vector<std::vector<float>> bboxes;
  std::tie(scores, labels, bboxes) = decode_results(all_bbox_preds, all_cls_scores, 300);

  std::vector<autoware_perception_msgs::msg::DetectedObject> raw_objects;
  size_t counter = 0;
  for (size_t i = 0; i < bboxes.size(); ++i) {
    const auto bbox = bboxes[i];
    const float score = scores[i];
    if (score < confidence_threshold_) continue;

    // cx, cy, cz, w, l, h, rot, vx, vy
    DetectedObject object;
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
    raw_objects.push_back(object);
    ++counter;
  }

  DetectedObjects output_msg;
  // output_msg.objects = iou_bev_nms_.apply(raw_objects);
  output_msg.objects = raw_objects;
  pub_objects_->publish(output_msg);
}

}  // namespace tensorrt_stream_petr


#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(tensorrt_stream_petr::StreamPetrNode)