#!/bin/bash

rm -rf frozen_engine/lane_detector_cls_ego_exist_20200825160726.in8.engine

python src/convert_to_tensorrt.py \
  --onnx_file frozen_onnx/lane_detector_cls_ego_exist_20200616115322.onnx \
  --calib_dir /home/julian/data/calibration_data_lane \
  --tensorrt_max_batch 10 \
  --calib_batch 10 \
  --use_int8 false \
  --cache_file frozen_onnx/lane_detector_cls_ego_exist_20200616115322.calibration_cache \
  --engine_file frozen_onnx/lane_detector_cls_ego_exist_20200616115322.int32.engine \
  --gpus 0