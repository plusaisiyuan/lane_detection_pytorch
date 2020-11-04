#!/bin/bash

python src/convert_to_onnx.py --resume trained/20200825160726/_erfnet_checkpoint.pth.tar \
                          --gpus 0 \
                          --onnx_file frozen_onnx/lane_detector_cls_ego_exist_20200825160726.onnx \
                          --onnx_optim_passes fuse_bn_into_conv,fuse_add_bias_into_conv,eliminate_identity,eliminate_nop_pad,eliminate_nop_transpose,eliminate_unused_initializer,extract_constant_to_initializer,fuse_consecutive_squeezes,fuse_consecutive_transposes,fuse_transpose_into_gemm