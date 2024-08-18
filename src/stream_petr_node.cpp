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

#include <memory>
#include <string>
#include <vector>
#include <thread>

namespace tensorrt_stream_petr
{
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

  // cx, cy, cz, w, l, h, rot, vx, vy, vz???
  const std::vector<float> all_bbox_preds = pts_head_->bindings["all_bbox_preds"]->cpu();
  const std::vector<float> all_cls_scores = pts_head_->bindings["all_cls_scores"]->cpu();

  // TODO(kminoda): resize beforehand
  std::vector<autoware_perception_msgs::msg::DetectedObject> raw_objects;
  size_t counter = 0;
  while ((counter * 10) + 9 < all_bbox_preds.size()) {
    DetectedObject object;
    object.kinematics.pose_with_covariance.pose.position.x = all_bbox_preds[counter * 10 + 0];
    object.kinematics.pose_with_covariance.pose.position.y = all_bbox_preds[counter * 10 + 1];
    object.kinematics.pose_with_covariance.pose.position.z = all_bbox_preds[counter * 10 + 2];
    object.shape.dimensions.x = all_bbox_preds[counter * 10 + 3];
    object.shape.dimensions.y = all_bbox_preds[counter * 10 + 4];
    object.shape.dimensions.z = all_bbox_preds[counter * 10 + 5];

    const double yaw = all_bbox_preds[counter * 10 + 6];
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
  output_msg.objects = iou_bev_nms_.apply(raw_objects);
  pub_objects_->publish(output_msg);
}

}  // namespace tensorrt_stream_petr


#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(tensorrt_stream_petr::StreamPetrNode)