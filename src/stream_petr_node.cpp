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

namespace tensorrt_stream_petr
{
StreamPetrNode::StreamPetrNode(const rclcpp::NodeOptions & node_options)
: Node("tensorrt_stream_petr", node_options)
{
  using std::placeholders::_1;
  throw std::runtime_error("hoge");

  RCLCPP_INFO(get_logger(), "HEY");
  // Initialize parameters
  const std::string onnx_backbone_path = declare_parameter<std::string>("onnx_backbone_path");
  const std::string onnx_head_path = declare_parameter<std::string>("onnx_head_path");
  const std::string precision_backbone = declare_parameter<std::string>("precision_backbone");
  const std::string precision_head = declare_parameter<std::string>("precision_head");

  // Subscriber
  sub_image_ = create_subscription<Image>(
    "~/input/image_raw", 10, std::bind(&StreamPetrNode::on_image, this, _1));

  RCLCPP_INFO(get_logger(), "nvinfer: %d.%d.%d\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
  // cudaSetDevice(0);

  // auto runtime_deleter = [](IRuntime *runtime) { /* runtime->destroy(); */ };
  // std::unique_ptr<IRuntime, decltype(runtime_deleter)> runtime{createInferRuntime(gLogger), runtime_deleter};
  // backbone_ = std::make_unique<SubNetwork>(onnx_backbone_path, runtime.get());
  // pts_head_ = std::make_unique<SubNetwork>(onnx_head_path, runtime.get());

  // cudaStreamCreate(&stream_);
  // backbone_->EnableCudaGraph(stream_);
  // pts_head_->EnableCudaGraph(stream_);

  // mem_.mem_stream = stream_;
  // mem_.pre_buf = (float*)pts_head_->bindings["pre_memory_timestamp"]->ptr;
  // mem_.post_buf = (float*)pts_head_->bindings["post_memory_timestamp"]->ptr;

  // // events for measurement
  // dur_backbone_ = std::make_unique<Duration>("backbone");
  // dur_ptshead_ = std::make_unique<Duration>("ptshead");

  // const std::filesystem::path data_dir{"data"};

  // int n_frames = 0;
  // for (auto const& dir_entry : std::filesystem::directory_iterator{data_dir}) {
  //   n_frames ++;
  // }
  // RCLCPP_INFO(get_logger(), "Total frames: %d\n", n_frames);

  // for (int i = 0; i < n_frames; ++i) {
  //   inference(i);
  // }
}

void StreamPetrNode::on_image(const Image & msg){
  (void)msg;
}

void StreamPetrNode::inference(const int f) {
  // load data
  char buf[5] = {0};
  sprintf(buf, "%04d", f);
  std::string frame_dir = "./data/" + std::string(buf) + "/";
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

  // validate(pts_head, frame_dir, "all_bbox_preds"); 
  // validate(pts_head, frame_dir, "all_cls_scores"); 

  // dump preds and labels
  pts_head_->bindings["all_bbox_preds"]->save(frame_dir + "/all_bbox_preds_trt.bin");
  pts_head_->bindings["all_cls_scores"]->save(frame_dir + "/all_cls_scores_trt.bin");
}


}  // namespace tensorrt_stream_petr
