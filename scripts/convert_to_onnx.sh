#!/bin/bash

python src/convert_to_onnx.py --resume trained/20200616115322/_erfnet_model_best.pth.tar \
                          --gpus 0 \
                          --onnx_file frozen_onnx/lane_detector_cls_ego_exist_20200616115322.onnx \
                          --onnx_optim_passes fuse_bn_into_conv