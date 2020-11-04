#!/bin/bash

path="/media/jiangzh/zhihao-2TB/j7-00007"
for bag_path in `ls $path/*_2000to2080.bag`
do
  bag=${bag_path##*/}
  out_dir="$path/result/$bag"
  if [ ! -d ${out_dir} ];then
    mkdir -p ${out_dir}
  fi
  python src/run_rosbag.py   --bags ${bag_path} --out ${out_dir} \
                             --cam_topics /front_left_camera/image_color/compressed,/front_right_camera/image_color/compressed \
                             --odom_topics /navsat/odom \
                             --rate 0.1 \
                             --tolerance 0.05 \
                             --frame_limit 100000 \
                             --gpus 0 \
                             --resume trained/20200828170211/_erfnet_checkpoint.pth.tar
done

#            --cam_topics /front_left_camera/image_color/compressed,/front_right_camera/image_color/compressed \
#            --cam_topics /usb_cam_left/image_raw/compressed,/usb_cam_right/image_raw/compressed
