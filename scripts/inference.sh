#!/bin/bash

python src/inference.py   --gpus 0 \
                          --model_file frozen_engine/lane_detector_cls_ego_exist_20200612170525.fp32.engine