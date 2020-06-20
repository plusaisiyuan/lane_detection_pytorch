#!/bin/bash

python src/convert_to_tensorrt.py \
  --resume trained/20200608162559/_erfnet_model_best.pth.tar \
  --onnx_file frozen_onnx/lane_detector_cls_ego_exist_20200612170525.onnx \
  --calib_dir /media/jiangzh/zhihao-2TB/calibration_data_lane \
  --tensorrt_max_batch 1 \
  --calib_batch 1 \
  --not_use_int8 false \
  --cache_file frozen_onnx/lane_detector_cls_ego_exist_20200612170525.calibration_cache \
  --tensorrt_file frozen_engine/lane_detector_cls_ego_exist_20200612170525.in8.engine \
  --gpus 0