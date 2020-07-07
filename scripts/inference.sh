#!/bin/bash

python src/inference.py   --gpus 0 \
                          --model_file frozen_onnx/unified_lane_detector_erf_ins_off_480x224_20200624_l4e.onnx