#!/bin/bash

path="/media/jiangzh/zhihao-2TB/yy/"
for bag_path in `ls $path/*.bag`
do
  bag=${bag_path##*/}
  out_dir="$path/result/$bag"
  if [ ! -d ${out_dir} ];then
    mkdir -p ${out_dir}
  fi
  python src/run_rosbag.py   --bags ${bag_path} --out ${out_dir} \
                             --cam_topics /usb_cam_left/image_raw/compressed,/usb_cam_right/image_raw/compressed \
                             --odom_topics /navsat/odom \
                             --rate 0.1 \
                             --frame_limit 100000 \
                             --gpus 0 \
                             --resume trained/20200520223916/_erfnet_model_best.pth.tar
done

#            --cam_topics /front_left_camera/image_color/compressed,/front_right_camera/image_color/compressed \